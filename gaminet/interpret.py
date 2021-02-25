# All the codes in this file are derived from interpret package by Microsoft Corporation

# Distributed under the MIT software license

import os
import struct
import numpy as np
import ctypes as ct
from sys import platform
from numpy.ctypeslib import ndpointer
from collections import OrderedDict
from pandas.core.generic import NDFrame
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
try:
    from pandas.api.types import is_numeric_dtype, is_string_dtype
except ImportError:  # pragma: no cover
    from pandas.core.dtypes.common import is_numeric_dtype, is_string_dtype

def gen_attributes(col_types, col_n_bins):
    # Create Python form of attributes
    # Undocumented.
    attributes = [None] * len(col_types)
    for col_idx, _ in enumerate(attributes):
        attributes[col_idx] = {
            # NOTE: Ordinal only handled at native, override.
            # 'type': col_types[col_idx],
            "type": "continuous",
            # NOTE: Missing not implemented at native, always set to false.
            "has_missing": False,
            "n_bins": col_n_bins[col_idx],
        }
    return attributes


def gen_attribute_sets(attribute_indices):
    attribute_sets = [None] * len(attribute_indices)
    for i, indices in enumerate(attribute_indices):
        attribute_set = {"n_attributes": len(indices), "attributes": indices}
        attribute_sets[i] = attribute_set
    return attribute_sets


def autogen_schema(X, ordinal_max_items=2, feature_names=None, feature_types=None):
    """ Generates data schema for a given dataset as JSON representable.
    Args:
        X: Dataframe/ndarray to build schema from.
        ordinal_max_items: If a numeric column's cardinality
            is at most this integer,
            consider it as ordinal instead of continuous.
        feature_names: Feature names
        feature_types: Feature types
    Returns:
        A dictionary - schema that encapsulates column information,
        such as type and domain.
    """
    schema = OrderedDict()
    col_number = 0
    if isinstance(X, np.ndarray):
        if feature_names is None:
            feature_names = ["col_" + str(i) for i in range(X.shape[1])]

        # NOTE: Use rolled out infer_objects for old pandas.
        # As used from SO:
        # https://stackoverflow.com/questions/47393134/attributeerror-dataframe-object-has-no-attribute-infer-objects
        X = pd.DataFrame(X, columns=feature_names)
        try:
            X = X.infer_objects()
        except AttributeError:
            for k in list(X):
                X[k] = pd.to_numeric(X[k], errors="ignore")

    if isinstance(X, NDFrame):
        for name, col_dtype in zip(X.dtypes.index, X.dtypes):
            schema[name] = {}
            if is_numeric_dtype(col_dtype):
                # schema[name]['type'] = 'continuous'
                # TODO: Fix this once we know it works.
                if len(set(X[name])) > ordinal_max_items:
                    schema[name]["type"] = "continuous"
                else:
                    # TODO: Work with ordinal later.
                    schema[name]["type"] = "categorical"
                    # schema[name]['type'] = 'ordinal'
                    # schema[name]['order'] = list(set(X[name]))
            elif is_string_dtype(col_dtype):
                schema[name]["type"] = "categorical"
            else:  # pragma: no cover
                warnings.warn("Unknown column: " + name, RuntimeWarning)
                schema[name]["type"] = "unknown"
            schema[name]["column_number"] = col_number
            col_number += 1

        # Override if feature_types is passed as arg.
        if feature_types is not None:
            for idx, name in enumerate(X.dtypes.index):
                schema[name]["type"] = feature_types[idx]
    else:  # pragma: no cover
        raise TypeError("GA2M only supports numpy arrays or pandas dataframes.")

    return schema


class EBMPreprocessor(BaseEstimator, TransformerMixin):
    """ Transformer that preprocesses data to be ready before EBM. """

    def __init__(
        self,
        schema=None,
        max_n_bins=255,
        missing_constant=0,
        unknown_constant=0,
        feature_names=None,
        binning_strategy="uniform",
    ):
        """ Initializes EBM preprocessor.
        Args:
            schema: A dictionary that encapsulates column information,
                    such as type and domain.
            max_n_bins: Max number of bins to process numeric features.
            missing_constant: Missing encoded as this constant.
            unknown_constant: Unknown encoded as this constant.
            feature_names: Feature names as list.
            binning_strategy: Strategy to compute bins according to density if "quantile" or equidistant if "uniform".
        """
        self.schema = schema
        self.max_n_bins = max_n_bins
        self.missing_constant = missing_constant
        self.unknown_constant = unknown_constant
        self.feature_names = feature_names
        self.binning_strategy = binning_strategy

    def fit(self, X):
        """ Fits transformer to provided instances.
        Args:
            X: Numpy array for training instances.
        Returns:
            Itself.
        """
        # self.col_bin_counts_ = {}
        self.col_bin_edges_ = {}

        self.hist_counts_ = {}
        self.hist_edges_ = {}

        self.col_mapping_ = {}
        self.col_mapping_counts_ = {}

        self.col_n_bins_ = {}

        self.col_names_ = []
        self.col_types_ = []
        self.has_fitted_ = False

        self.schema_ = (
            self.schema
            if self.schema is not None
            else autogen_schema(X, feature_names=self.feature_names)
        )
        schema = self.schema_

        for col_idx in range(X.shape[1]):
            col_name = list(schema.keys())[col_idx]
            self.col_names_.append(col_name)

            col_info = schema[col_name]
            assert col_info["column_number"] == col_idx
            col_data = X[:, col_idx]

            self.col_types_.append(col_info["type"])
            if col_info["type"] == "continuous":
                col_data = col_data.astype(float)

                uniq_vals = set(col_data[~np.isnan(col_data)])
                if len(uniq_vals) < self.max_n_bins:
                    bins = list(sorted(uniq_vals))
                else:
                    if self.binning_strategy == "uniform":
                        bins = self.max_n_bins
                    elif self.binning_strategy == "quantile":
                        bins = np.unique(
                            np.quantile(
                                col_data, q=np.linspace(0, 1, self.max_n_bins + 1)
                            )
                        )
                    else:
                        raise ValueError(
                            "Unknown binning_strategy: '{}'.".format(
                                self.binning_strategy
                            )
                        )

                _, bin_edges = np.histogram(col_data, bins=bins)

                hist_counts, hist_edges = np.histogram(col_data, bins="doane")
                self.col_bin_edges_[col_idx] = bin_edges

                self.hist_edges_[col_idx] = hist_edges
                self.hist_counts_[col_idx] = hist_counts
                self.col_n_bins_[col_idx] = len(bin_edges)
            elif col_info["type"] == "ordinal":
                mapping = {val: indx for indx, val in enumerate(col_info["order"])}
                self.col_mapping_[col_idx] = mapping
                self.col_n_bins_[col_idx] = len(col_info["order"])
            elif col_info["type"] == "categorical":
                uniq_vals, counts = np.unique(col_data, return_counts=True)

                non_nan_index = ~np.isnan(counts)
                uniq_vals = uniq_vals[non_nan_index]
                counts = counts[non_nan_index]

                mapping = {val: indx for indx, val in enumerate(uniq_vals)}
                self.col_mapping_counts_[col_idx] = counts
                self.col_mapping_[col_idx] = mapping

                # TODO: Review NA as we don't support it yet.
                self.col_n_bins_[col_idx] = len(uniq_vals)

        self.has_fitted_ = True
        return self

    def transform(self, X):
        """ Transform on provided instances.
        Args:
            X: Numpy array for instances.
        Returns:
            Transformed numpy array.
        """
        check_is_fitted(self, "has_fitted_")

        schema = self.schema
        X_new = np.copy(X)
        for col_idx in range(X.shape[1]):
            col_info = schema[list(schema.keys())[col_idx]]
            assert col_info["column_number"] == col_idx
            col_data = X[:, col_idx]
            if col_info["type"] == "continuous":
                col_data = col_data.astype(float)
                bin_edges = self.col_bin_edges_[col_idx].copy()

                digitized = np.digitize(col_data, bin_edges, right=False)
                digitized[digitized == 0] = 1
                digitized -= 1

                # NOTE: NA handling done later.
                # digitized[np.isnan(col_data)] = self.missing_constant
                X_new[:, col_idx] = digitized
            elif col_info["type"] == "ordinal":
                mapping = self.col_mapping_[col_idx]
                mapping[np.nan] = self.missing_constant
                vec_map = np.vectorize(
                    lambda x: mapping[x] if x in mapping else self.unknown_constant
                )
                X_new[:, col_idx] = vec_map(col_data)
            elif col_info["type"] == "categorical":
                mapping = self.col_mapping_[col_idx]
                mapping[np.nan] = self.missing_constant
                vec_map = np.vectorize(
                    lambda x: mapping[x] if x in mapping else self.unknown_constant
                )
                X_new[:, col_idx] = vec_map(col_data)

        return X_new.astype(np.int64)

    
class Native:
    """Layer/Class responsible for native function calls."""

    # enum FeatureType : int64_t
    # Ordinal = 0
    FeatureTypeOrdinal = 0
    # Nominal = 1
    FeatureTypeNominal = 1

    class EbmCoreFeature(ct.Structure):
        _fields_ = [
            # FeatureType featureType;
            ("featureType", ct.c_longlong),
            # int64_t hasMissing;
            ("hasMissing", ct.c_longlong),
            # int64_t countBins;
            ("countBins", ct.c_longlong),
        ]

    class EbmCoreFeatureCombination(ct.Structure):
        _fields_ = [
            # int64_t countFeaturesInCombination;
            ("countFeaturesInCombination", ct.c_longlong)
        ]

    LogFuncType = ct.CFUNCTYPE(None, ct.c_char, ct.c_char_p)

    # const signed char TraceLevelOff = 0;
    TraceLevelOff = 0
    # const signed char TraceLevelError = 1;
    TraceLevelError = 1
    # const signed char TraceLevelWarning = 2;
    TraceLevelWarning = 2
    # const signed char TraceLevelInfo = 3;
    TraceLevelInfo = 3
    # const signed char TraceLevelVerbose = 4;
    TraceLevelVerbose = 4

    def __init__(self):

        self.lib = ct.cdll.LoadLibrary(self.get_ebm_lib_path())
        self.harden_function_signatures()

    def harden_function_signatures(self):
        """ Adds types to function signatures. """

        self.lib.InitializeInteractionClassification.argtypes = [
            # int64_t countFeatures
            ct.c_longlong,
            # EbmCoreFeature * features
            ct.POINTER(self.EbmCoreFeature),
            # int64_t countTargetClasses
            ct.c_longlong,
            # int64_t countInstances
            ct.c_longlong,
            # int64_t * targets
            ndpointer(dtype=ct.c_longlong, flags="F_CONTIGUOUS", ndim=1),
            # int64_t * binnedData
            ndpointer(dtype=ct.c_longlong, flags="F_CONTIGUOUS", ndim=2),
            # double * predictorScores
            ndpointer(dtype=ct.c_double, flags="F_CONTIGUOUS", ndim=1),
        ]
        self.lib.InitializeInteractionClassification.restype = ct.c_void_p

        self.lib.InitializeInteractionRegression.argtypes = [
            # int64_t countFeatures
            ct.c_longlong,
            # EbmCoreFeature * features
            ct.POINTER(self.EbmCoreFeature),
            # int64_t countInstances
            ct.c_longlong,
            # double * targets
            ndpointer(dtype=ct.c_double, flags="F_CONTIGUOUS", ndim=1),
            # int64_t * binnedData
            ndpointer(dtype=ct.c_longlong, flags="F_CONTIGUOUS", ndim=2),
            # double * predictorScores
            ndpointer(dtype=ct.c_double, flags="F_CONTIGUOUS", ndim=1),
        ]
        self.lib.InitializeInteractionRegression.restype = ct.c_void_p

        self.lib.GetInteractionScore.argtypes = [
            # void * ebmInteraction
            ct.c_void_p,
            # int64_t countFeaturesInCombination
            ct.c_longlong,
            # int64_t * featureIndexes
            ndpointer(dtype=ct.c_longlong, flags="F_CONTIGUOUS", ndim=1),
            # double * interactionScoreReturn
            ct.POINTER(ct.c_double),
        ]
        self.lib.GetInteractionScore.restype = ct.c_longlong

        self.lib.FreeInteraction.argtypes = [
            # void * ebmInteraction
            ct.c_void_p
        ]

    def get_ebm_lib_path(self):
        """ Returns filepath of core EBM library.
        Returns:
            A string representing filepath.
        """
        bitsize = struct.calcsize("P") * 8
        is_64_bit = bitsize == 64

        script_path = os.path.dirname(os.path.abspath(__file__))
        package_path = script_path # os.path.join(script_path, "..", "..")

        debug_str = ""
        if platform == "linux" or platform == "linux2" and is_64_bit:
            return os.path.join(
                package_path, "lib", "lib_ebmcore_linux_x64{0}.so".format(debug_str)
            )
        elif platform == "win32" and is_64_bit:
            return os.path.join(
                package_path, "lib", "lib_ebmcore_win_x64{0}.dll".format(debug_str)
            )
        elif platform == "darwin" and is_64_bit:
            return os.path.join(
                package_path, "lib", "lib_ebmcore_mac_x64{0}.dylib".format(debug_str)
            )
        else:
            msg = "Platform {0} at {1} bit not supported for EBM".format(
                platform, bitsize
            )
            raise Exception(msg)


class NativeEBM:
    """Lightweight wrapper for EBM C code.
    """

    def __init__(
        self,
        attributes,
        attribute_sets,
        X_train,
        y_train,
        X_val,
        y_val,
        model_type="regression",
        num_inner_bags=0,
        num_classification_states=2,
        training_scores=None,
        validation_scores=None,
        random_state=1337,
    ):

        # TODO: Update documentation for training/val scores args.
        """ Initializes internal wrapper for EBM C code.
        Args:
            attributes: List of attributes represented individually as
                dictionary of keys ('type', 'has_missing', 'n_bins').
            attribute_sets: List of attribute sets represented as
                a dictionary of keys ('n_attributes', 'attributes')
            X_train: Training design matrix as 2-D ndarray.
            y_train: Training response as 1-D ndarray.
            X_val: Validation design matrix as 2-D ndarray.
            y_val: Validation response as 1-D ndarray.
            model_type: 'regression'/'classification'.
            num_inner_bags: Per feature training step, number of inner bags.
            num_classification_states: Specific to classification,
                number of unique classes.
            training_scores: Undocumented.
            validation_scores: Undocumented.
            random_state: Random seed as integer.
        """
        if not hasattr(self, "native"):
            self.native = Native()

        # Store args
        self.attributes = attributes
        self.attribute_sets = attribute_sets
        self.attribute_array, self.attribute_sets_array, self.attribute_set_indexes = self._convert_attribute_info_to_c(
            attributes, attribute_sets
        )

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model_type = model_type
        self.num_inner_bags = num_inner_bags
        self.num_classification_states = num_classification_states

        # # Set train/val scores to zeros if not passed.
        # if isinstance(intercept, numbers.Number) or len(intercept) == 1:
        #     score_vector = np.zeros(X.shape[0])
        #     else:
        # score_vector = np.zeros((X.shape[0], len(intercept)))

        self.training_scores = training_scores
        self.validation_scores = validation_scores
        if self.training_scores is None:
            if self.num_classification_states > 2:
                self.training_scores = np.zeros(
                    (X_train.shape[0], self.num_classification_states)
                ).reshape(-1)
            else:
                self.training_scores = np.zeros(X_train.shape[0])
        if self.validation_scores is None:
            if self.num_classification_states > 2:
                self.validation_scores = np.zeros(
                    (X_train.shape[0], self.num_classification_states)
                ).reshape(-1)
            else:
                self.validation_scores = np.zeros(X_train.shape[0])
        self.random_state = random_state

        # Convert n-dim arrays ready for C.
        self.X_train_f = np.asfortranarray(self.X_train)
        self.X_val_f = np.asfortranarray(self.X_val)

        # Define extra properties
        self.model_pointer = None
        self.interaction_pointer = None

        # Allocate external resources
        if self.model_type == "regression":
            self.y_train = self.y_train.astype("float64")
            self.y_val = self.y_val.astype("float64")
            self._initialize_interaction_regression()
        elif self.model_type == "classification":
            self.y_train = self.y_train.astype("int64")
            self.y_val = self.y_val.astype("int64")
            self._initialize_interaction_classification()

    def _convert_attribute_info_to_c(self, attributes, attribute_sets):
        # Create C form of attributes
        attribute_ar = (self.native.EbmCoreFeature * len(attributes))()
        for idx, attribute in enumerate(attributes):
            if attribute["type"] == "categorical":
                attribute_ar[idx].featureType = self.native.FeatureTypeNominal
            else:
                attribute_ar[idx].featureType = self.native.FeatureTypeOrdinal
            attribute_ar[idx].hasMissing = 1 * attribute["has_missing"]
            attribute_ar[idx].countBins = attribute["n_bins"]

        attribute_set_indexes = []
        attribute_sets_ar = (
            self.native.EbmCoreFeatureCombination * len(attribute_sets)
        )()
        for idx, attribute_set in enumerate(attribute_sets):
            attribute_sets_ar[idx].countFeaturesInCombination = attribute_set[
                "n_attributes"
            ]

            for attr_idx in attribute_set["attributes"]:
                attribute_set_indexes.append(attr_idx)

        attribute_set_indexes = np.array(attribute_set_indexes, dtype="int64")

        return attribute_ar, attribute_sets_ar, attribute_set_indexes

    def _initialize_interaction_regression(self):
        self.interaction_pointer = self.native.lib.InitializeInteractionRegression(
            len(self.attribute_array),
            self.attribute_array,
            self.X_train.shape[0],
            self.y_train,
            self.X_train_f,
            self.training_scores,
        )

    def _initialize_interaction_classification(self):
        self.interaction_pointer = self.native.lib.InitializeInteractionClassification(
            len(self.attribute_array),
            self.attribute_array,
            self.num_classification_states,
            self.X_train.shape[0],
            self.y_train,
            self.X_train_f,
            self.training_scores,
        )

    def close(self):
        """ Deallocates C objects used to train EBM. """
        self.native.lib.FreeInteraction(self.interaction_pointer)

    def fast_interaction_score(self, attribute_index_tuple):
        """ Provides score for an attribute interaction. Higher is better."""
        score = ct.c_double(0.0)
        self.native.lib.GetInteractionScore(
            self.interaction_pointer,
            len(attribute_index_tuple),
            np.array(attribute_index_tuple, dtype=np.int64),
            ct.byref(score),
        )
        return score.value
