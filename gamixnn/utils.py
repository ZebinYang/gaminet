import numpy as np
from contextlib import closing
from itertools import combinations
from sklearn.model_selection import train_test_split 

from interpret.glassbox.ebm.utils import EBMUtils
from interpret.utils import autogen_schema
from interpret.glassbox.ebm.internal import NativeEBM
from interpret.glassbox.ebm.ebm import EBMPreprocessor


def get_interaction_list(x, y, interactions, meta_info, task_type="Regression"):

    if task_type == "Regression":
        num_classes_ = -1
        model_type = "regression"
    elif task_type == "Classification":
        num_classes_ = 2
        model_type = "classification"

    schema_ = autogen_schema(x, feature_names=list(meta_info.keys())[:-1], feature_types=[item['type'] for key, item in meta_info.items()])
    preprocessor_ = EBMPreprocessor(schema=schema_)
    preprocessor_.fit(x)
    xt = preprocessor_.transform(x)

    attributes_ = EBMUtils.gen_attributes(preprocessor_.col_types_, preprocessor_.col_n_bins_)
    main_attr_sets = EBMUtils.gen_attribute_sets([[item] for item in range(len(attributes_))])

    x_train, x_val, y_train, y_val = train_test_split(xt, y, test_size=0.2, random_state=0, stratify=None)

    with closing(
        NativeEBM(
            attributes_,
            main_attr_sets,
            x_train,
            y_train.ravel(),
            x_val,
            y_val.ravel(),
            num_inner_bags=0,
            num_classification_states=num_classes_,
            model_type=model_type,
            training_scores=None,
            validation_scores=None,
        )
    ) as native_ebm:

        interaction_scores = []
        interaction_indices = [item for item in combinations(range(len(preprocessor_.col_types_)), 2)]
        for pair in interaction_indices:
            score = native_ebm.fast_interaction_score(pair)
            interaction_scores.append((pair, score))

        ranked_scores = list(
            sorted(interaction_scores, key=lambda item: item[1], reverse=True)
        )

    interaction_list = [ranked_scores[i][0] for i in range(interactions)]
    return interaction_list
