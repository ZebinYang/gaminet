import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .layers import *
from .utils import get_interaction_list


class GAMINet(tf.keras.Model):

    def __init__(self, meta_info,
                 interact_num=20,
                 subnet_arch=[40] * 5,
                 interact_arch=[40] * 5,
                 lr_bp=[1e-4, 1e-4, 1e-4],
                 batch_size=200,
                 task_type="Regression",
                 activation_func=tf.nn.relu,
                 main_effect_epochs=5000,
                 interaction_epochs=5000,
                 tuning_epochs=500,
                 early_stop_thres=[50, 50, 50],
                 heredity=True,
                 reg_clarity=0.1,
                 loss_threshold=0.01,
                 val_ratio=0.2,
                 mono_increasing_list=None,
                 mono_decreasing_list=None,
                 lattice_size=10,
                 verbose=False,
                 random_state=0):

        super(GAMINet, self).__init__()

        self.meta_info = meta_info
        self.subnet_arch = subnet_arch
        self.interact_arch = interact_arch

        self.lr_bp = lr_bp
        self.batch_size = batch_size
        self.task_type = task_type
        self.activation_func = activation_func
        self.tuning_epochs = tuning_epochs
        self.main_effect_epochs = main_effect_epochs
        self.interaction_epochs = interaction_epochs
        self.early_stop_thres = early_stop_thres
        self.early_stop_thres1 = early_stop_thres[0]
        self.early_stop_thres2 = early_stop_thres[1]
        self.early_stop_thres3 = early_stop_thres[2]

        self.heredity = heredity
        self.reg_clarity = reg_clarity
        self.loss_threshold = loss_threshold

        self.mono_increasing_list = [] if mono_increasing_list is None else mono_increasing_list
        self.mono_decreasing_list = [] if mono_decreasing_list is None else mono_decreasing_list
        self.mono_list = self.mono_increasing_list + self.mono_decreasing_list
        self.lattice_size = lattice_size
        
        self.verbose = verbose
        self.val_ratio = val_ratio
        self.random_state = random_state

        np.random.seed(random_state)
        tf.random.set_seed(random_state)

        self.dummy_values_ = {}
        self.nfeature_scaler_ = {}
        self.cfeature_num_ = 0
        self.nfeature_num_ = 0
        self.cfeature_list_ = []
        self.nfeature_list_ = []
        self.cfeature_index_list_ = []
        self.nfeature_index_list_ = []

        self.feature_list_ = []
        self.feature_type_list_ = []
        for idx, (feature_name, feature_info) in enumerate(meta_info.items()):
            if feature_info["type"] == "target":
                continue
            if feature_info["type"] == "categorical":
                self.cfeature_num_ += 1
                self.cfeature_list_.append(feature_name)
                self.cfeature_index_list_.append(idx)
                self.feature_type_list_.append("categorical")
                self.dummy_values_.update({feature_name: meta_info[feature_name]["values"]})
            else:
                self.nfeature_num_ += 1
                self.nfeature_list_.append(feature_name)
                self.nfeature_index_list_.append(idx)
                self.feature_type_list_.append("continuous")
                self.nfeature_scaler_.update({feature_name: meta_info[feature_name]["scaler"]})
            self.feature_list_.append(feature_name)

        # build
        self.interaction_list = []
        self.interact_num_added = 0
        self.interaction_status = False
        self.input_num = self.nfeature_num_ + self.cfeature_num_
        self.max_interact_num = int(round(self.input_num * (self.input_num - 1) / 2))
        self.interact_num = min(interact_num, self.max_interact_num)

        self.maineffect_blocks = MainEffectBlock(feature_list=self.feature_list_,
                                 dummy_values=self.dummy_values_,
                                 nfeature_index_list=self.nfeature_index_list_,
                                 cfeature_index_list=self.cfeature_index_list_,
                                 subnet_arch=self.subnet_arch,
                                 activation_func=self.activation_func,
                                 mono_list=self.mono_list,
                                 lattice_size=self.lattice_size)
        self.interact_blocks = InteractionBlock(interact_num=self.interact_num,
                                feature_list=self.feature_list_,
                                cfeature_index_list=self.cfeature_index_list_,
                                dummy_values=self.dummy_values_,
                                interact_arch=self.interact_arch,
                                activation_func=self.activation_func,
                                mono_list=self.mono_list,
                                lattice_size=self.lattice_size)
        self.output_layer = OutputLayer(input_num=self.input_num,
                              interact_num=self.interact_num,
                              mono_increasing_list=self.mono_increasing_list,
                              mono_decreasing_list=self.mono_decreasing_list)

        self.optimizer = tf.keras.optimizers.Adam()
        if self.task_type == "Regression":
            self.loss_fn = tf.keras.losses.MeanSquaredError()
        elif self.task_type == "Classification":
            self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        else:
            raise ValueError("The task type is not supported")

    def call(self, inputs, sample_weight=None, main_effect_training=False, interaction_training=False):

        self.clarity_loss = tf.constant(0.0)
        self.maineffect_outputs = self.maineffect_blocks(inputs, sample_weight, training=main_effect_training)
        if self.interaction_status:
            self.interact_outputs = self.interact_blocks(inputs, sample_weight, training=interaction_training)
            main_weights = tf.multiply(self.output_layer.main_effect_switcher, self.output_layer.main_effect_weights)
            interaction_weights = tf.multiply(self.output_layer.interaction_switcher, self.output_layer.interaction_weights)
            for i, (k1, k2) in enumerate(self.interaction_list):
                a1 = tf.multiply(tf.gather(self.maineffect_outputs, [k1], axis=1), tf.gather(main_weights, [k1], axis=0))
                a2 = tf.multiply(tf.gather(self.maineffect_outputs, [k2], axis=1), tf.gather(main_weights, [k2], axis=0))
                b = tf.multiply(tf.gather(self.interact_outputs, [i], axis=1), tf.gather(interaction_weights, [i], axis=0))
                if sample_weight is not None:
                    self.clarity_loss += tf.abs(tf.reduce_mean(tf.multiply(tf.multiply(a1, b), sample_weight)))
                    self.clarity_loss += tf.abs(tf.reduce_mean(tf.multiply(tf.multiply(a2, b), sample_weight)))
                else:
                    self.clarity_loss += tf.abs(tf.reduce_mean(tf.multiply(a1, b)))
                    self.clarity_loss += tf.abs(tf.reduce_mean(tf.multiply(a2, b)))
        else:
            self.interact_outputs = tf.zeros([inputs.shape[0], self.interact_num])

        concat_list = [self.maineffect_outputs]
        if self.interact_num > 0:
            concat_list.append(self.interact_outputs)

        if self.task_type == "Regression":
            output = self.output_layer(tf.concat(concat_list, 1))
        elif self.task_type == "Classification":
            output = tf.nn.sigmoid(self.output_layer(tf.concat(concat_list, 1)))
        else:
            raise ValueError("The task type is not supported")

        return output

    @tf.function
    def predict_graph(self, x, main_effect_training=False, interaction_training=False):
        return self.__call__(x, sample_weight=None,
                      main_effect_training=main_effect_training,
                      interaction_training=interaction_training)

    def predict(self, x):
        return self.predict_graph(tf.cast(x, tf.float32)).numpy()

    @tf.function
    def evaluate_graph_init(self, x, y, sample_weight=None, main_effect_training=False, interaction_training=False):
        return self.loss_fn(y, self.__call__(x, sample_weight,
                               main_effect_training=main_effect_training,
                               interaction_training=interaction_training), sample_weight=sample_weight)

    @tf.function
    def evaluate_graph_inter(self, x, y, sample_weight=None, main_effect_training=False, interaction_training=False):
        return self.loss_fn(y, self.__call__(x, sample_weight,
                               main_effect_training=main_effect_training,
                               interaction_training=interaction_training), sample_weight=sample_weight)

    def evaluate(self, x, y, sample_weight=None, main_effect_training=False, interaction_training=False):
        if self.interaction_status:
            return self.evaluate_graph_inter(tf.cast(x, tf.float32), tf.cast(y, tf.float32),
                                  tf.cast(sample_weight, tf.float32) if sample_weight is not None else None,
                                  main_effect_training=main_effect_training,
                                  interaction_training=interaction_training).numpy()
        else:
            return self.evaluate_graph_init(tf.cast(x, tf.float32), tf.cast(y, tf.float32),
                                  tf.cast(sample_weight, tf.float32) if sample_weight is not None else None,
                                  main_effect_training=main_effect_training,
                                  interaction_training=interaction_training).numpy()

    @tf.function
    def train_main_effect(self, inputs, labels, sample_weight=None):

        with tf.GradientTape() as tape:
            pred = self.__call__(inputs, sample_weight, main_effect_training=True, interaction_training=False)
            total_loss = self.loss_fn(labels, pred, sample_weight=sample_weight)

        train_weights_list = []
        train_weights = self.maineffect_blocks.weights
        train_weights.append(self.output_layer.main_effect_weights)
        train_weights.append(self.output_layer.output_bias)
        trainable_weights_names = [w.name for w in self.trainable_weights]
        for i in range(len(train_weights)):
            if train_weights[i].name in trainable_weights_names:
                train_weights_list.append(train_weights[i])
        grads = tape.gradient(total_loss, train_weights_list)
        self.optimizer.apply_gradients(zip(grads, train_weights_list))

    @tf.function
    def train_interaction(self, inputs, labels, sample_weight=None):

        with tf.GradientTape() as tape:
            pred = self.__call__(inputs, sample_weight, main_effect_training=False, interaction_training=True)
            total_loss = self.loss_fn(labels, pred, sample_weight=sample_weight) + self.reg_clarity * self.clarity_loss

        train_weights_list = []
        train_weights = self.interact_blocks.weights
        train_weights.append(self.output_layer.interaction_weights)
        train_weights.append(self.output_layer.output_bias)
        trainable_weights_names = [w.name for w in self.trainable_weights]
        for i in range(len(train_weights)):
            if train_weights[i].name in trainable_weights_names:
                train_weights_list.append(train_weights[i])
        grads = tape.gradient(total_loss, train_weights_list)
        self.optimizer.apply_gradients(zip(grads, train_weights_list))

    @tf.function
    def train_all(self, inputs, labels, sample_weight=None):

        with tf.GradientTape() as tape_maineffects:
            with tf.GradientTape() as tape_intearction:
                pred = self.__call__(inputs, sample_weight, main_effect_training=True, interaction_training=False)
                total_loss_maineffects = self.loss_fn(labels, pred, sample_weight=sample_weight)
                total_loss_interactions = self.loss_fn(labels, pred, sample_weight=sample_weight) + self.reg_clarity * self.clarity_loss

        train_weights_list = []
        train_weights = self.maineffect_blocks.weights
        train_weights.append(self.output_layer.main_effect_weights)
        train_weights.append(self.output_layer.output_bias)
        trainable_weights_names = [w.name for w in self.trainable_weights]
        for i in range(len(train_weights)):
            if train_weights[i].name in trainable_weights_names:
                train_weights_list.append(train_weights[i])
        grads = tape_maineffects.gradient(total_loss_maineffects, train_weights_list)
        self.optimizer.apply_gradients(zip(grads, train_weights_list))

        train_weights_list = []
        train_weights = self.interact_blocks.weights
        train_weights.append(self.output_layer.interaction_weights)
        train_weights.append(self.output_layer.output_bias)
        for i in range(len(train_weights)):
            if train_weights[i].name in trainable_weights_names:
                train_weights_list.append(train_weights[i])
        grads = tape_intearction.gradient(total_loss_interactions, train_weights_list)
        self.optimizer.apply_gradients(zip(grads, train_weights_list))

    def get_main_effect_rank(self):

        sorted_index = np.array([])
        componment_scales = [0 for i in range(self.input_num)]
        main_effect_norm = [self.maineffect_blocks.subnets[i].moving_norm.numpy()[0] for i in range(self.input_num)]
        beta = (self.output_layer.main_effect_weights.numpy() ** 2 * np.array([main_effect_norm]).reshape([-1, 1]))
        componment_scales = (np.abs(beta) / np.sum(np.abs(beta))).reshape([-1])
        sorted_index = np.argsort(componment_scales)[::-1]
        return sorted_index, componment_scales

    def get_interaction_rank(self):

        sorted_index = np.array([])
        componment_scales = [0 for i in range(self.interact_num_added)]
        if self.interact_num_added > 0:
            interaction_norm = [self.interact_blocks.interacts[i].moving_norm.numpy()[0] for i in range(self.interact_num_added)]
            gamma = (self.output_layer.interaction_weights.numpy()[:self.interact_num_added] ** 2
                  * np.array([interaction_norm]).reshape([-1, 1]))
            componment_scales = (np.abs(gamma) / np.sum(np.abs(gamma))).reshape([-1])
            sorted_index = np.argsort(componment_scales)[::-1]
        return sorted_index, componment_scales

    def get_all_active_rank(self):

        componment_scales = [0 for i in range(self.input_num + self.interact_num_added)]
        main_effect_norm = [self.maineffect_blocks.subnets[i].moving_norm.numpy()[0] for i in range(self.input_num)]
        beta = (self.output_layer.main_effect_weights.numpy() ** 2 * np.array([main_effect_norm]).reshape([-1, 1])
             * self.output_layer.main_effect_switcher.numpy())

        interaction_norm = [self.interact_blocks.interacts[i].moving_norm.numpy()[0] for i in range(self.interact_num_added)]
        gamma = (self.output_layer.interaction_weights.numpy()[:self.interact_num_added] ** 2
              * np.array([interaction_norm]).reshape([-1, 1])
              * self.output_layer.interaction_switcher.numpy()[:self.interact_num_added])
        gamma = np.vstack([gamma, np.zeros((self.interact_num - self.interact_num_added, 1))])

        componment_coefs = np.vstack([beta, gamma])
        componment_scales = (np.abs(componment_coefs) / np.sum(np.abs(componment_coefs))).reshape([-1])
        sorted_index = np.argsort(componment_scales)[::-1]
        return sorted_index, componment_scales

    def estimate_density(self, x, sample_weight):

        n_samples = x.shape[0]
        self.data_dict_density = {}
        for indice in range(self.input_num):
            feature_name = self.feature_list_[indice]
            if indice in self.nfeature_index_list_:
                sx = self.nfeature_scaler_[feature_name]
                density, bins = np.histogram(sx.inverse_transform(x[:,[indice]]), bins=10, weights=sample_weight.reshape(-1, 1), density=True)
                self.data_dict_density.update({feature_name: {"density": {"names": bins,"scores": density}}})
            elif indice in self.cfeature_index_list_:
                unique, counts = np.unique(x[:, indice], return_counts=True)
                density = np.zeros((len(self.dummy_values_[feature_name])))
                for val in unique:
                    density[val.round().astype(int)] = np.sum((x[:, indice] == val).astype(int) * sample_weight) / sample_weight.sum()
                self.data_dict_density.update({feature_name: {"density": {"names": np.arange(len(self.dummy_values_[feature_name])),
                                                     "scores": density}}})

    def center_main_effects(self):

        output_bias = self.output_layer.output_bias
        main_weights = tf.multiply(self.output_layer.main_effect_switcher, self.output_layer.main_effect_weights)
        for idx, subnet in enumerate(self.maineffect_blocks.subnets):
            if idx in self.nfeature_index_list_:
                if idx in self.mono_list:
                    subnet_bias = subnet.lattice_layer_bias - subnet.moving_mean
                    subnet.lattice_layer_bias.assign(subnet_bias)
                else:
                    subnet_bias = subnet.output_layer.bias - subnet.moving_mean
                    subnet.output_layer.bias.assign(subnet_bias)
            elif idx in self.cfeature_index_list_:
                subnet_bias = subnet.output_layer_bias - subnet.moving_mean
                subnet.output_layer_bias.assign(subnet_bias)

            output_bias = output_bias + tf.multiply(subnet.moving_mean, tf.gather(main_weights, idx, axis=0))
        self.output_layer.output_bias.assign(output_bias)

    def center_interactions(self):

        output_bias = self.output_layer.output_bias
        interaction_weights = tf.multiply(self.output_layer.interaction_switcher, self.output_layer.interaction_weights)
        for idx, interact in enumerate(self.interact_blocks.interacts):
            if idx >= len(self.interaction_list):
                break

            if (interact.interaction[0] in self.mono_list) or (interact.interaction[1] in self.mono_list):
                interact_bias = interact.lattice_layer_bias - interact.moving_mean
                interact.lattice_layer_bias.assign(interact_bias)
            else:
                interact_bias = interact.output_layer.bias - interact.moving_mean
                interact.output_layer.bias.assign(interact_bias)
            output_bias = output_bias + tf.multiply(interact.moving_mean, tf.gather(interaction_weights, idx, axis=0))
        self.output_layer.output_bias.assign(output_bias)

    def fit_main_effect(self, tr_x, tr_y, val_x, val_y, sample_weight=None):

        last_improvement = 0
        best_validation = np.inf
        train_size = tr_x.shape[0]
        tr_sw = sample_weight[self.tr_idx]
        for epoch in range(self.main_effect_epochs):
            shuffle_index = np.arange(tr_x.shape[0])
            np.random.shuffle(shuffle_index)
            tr_x = tr_x[shuffle_index]
            tr_y = tr_y[shuffle_index]
            tr_sw = tr_sw[shuffle_index]
            for iterations in range(train_size // self.batch_size):
                offset = (iterations * self.batch_size) % train_size
                batch_xx = tr_x[offset:(offset + self.batch_size), :]
                batch_yy = tr_y[offset:(offset + self.batch_size)]
                batch_sw = tr_sw[offset:(offset + self.batch_size)]
                self.train_main_effect(tf.cast(batch_xx, tf.float32), tf.cast(batch_yy, tf.float32), tf.cast(batch_sw, tf.float32))
            
            self.err_train_main_effect_training.append(self.evaluate(tr_x, tr_y, tr_sw,
                                                 main_effect_training=False, interaction_training=False))
            self.err_val_main_effect_training.append(self.evaluate(val_x, val_y, sample_weight[self.val_idx],
                                                 main_effect_training=False, interaction_training=False))
            if self.verbose & (epoch % 1 == 0):
                print("Main effects training epoch: %d, train loss: %0.5f, val loss: %0.5f" %
                      (epoch + 1, self.err_train_main_effect_training[-1], self.err_val_main_effect_training[-1]))

            if self.err_val_main_effect_training[-1] < best_validation:
                best_validation = self.err_val_main_effect_training[-1]
                last_improvement = epoch
            if epoch - last_improvement > self.early_stop_thres1:
                if self.verbose:
                    print("Early stop at epoch %d, with validation loss: %0.5f" % (epoch + 1, self.err_val_main_effect_training[-1]))
                break
        self.evaluate(tr_x, tr_y, sample_weight[self.tr_idx], main_effect_training=True, interaction_training=False)
        self.center_main_effects()

    def prune_main_effect(self, val_x, val_y, sample_weight=None):

        self.main_effect_val_loss = []
        sorted_index, componment_scales = self.get_main_effect_rank()
        self.output_layer.main_effect_switcher.assign(tf.constant(np.zeros((self.input_num, 1)), dtype=tf.float32))
        self.main_effect_val_loss.append(self.evaluate(val_x, val_y, sample_weight[self.val_idx],
                                        main_effect_training=False, interaction_training=False))
        for idx in range(self.input_num):
            selected_index = sorted_index[:(idx + 1)]
            main_effect_switcher = np.zeros((self.input_num, 1))
            main_effect_switcher[selected_index] = 1
            self.output_layer.main_effect_switcher.assign(tf.constant(main_effect_switcher, dtype=tf.float32))
            val_loss = self.evaluate(val_x, val_y, sample_weight[self.val_idx], main_effect_training=False, interaction_training=False)
            self.main_effect_val_loss.append(val_loss)

        best_idx = np.argmin(self.main_effect_val_loss)
        best_loss = np.min(self.main_effect_val_loss)
        if best_loss > 0:
            if np.sum((self.main_effect_val_loss / best_loss - 1) < self.loss_threshold) > 0:
                best_idx = np.where((self.main_effect_val_loss / best_loss - 1) < self.loss_threshold)[0][0]
            
        self.active_main_effect_index = sorted_index[:best_idx]
        main_effect_switcher = np.zeros((self.input_num, 1))
        main_effect_switcher[self.active_main_effect_index] = 1
        self.output_layer.main_effect_switcher.assign(tf.constant(main_effect_switcher, dtype=tf.float32))

    def add_interaction(self, tr_x, tr_y, val_x, val_y, sample_weight=None):

        if sample_weight is not None:
            tr_resample = np.random.choice(tr_x.shape[0], size=(tr_x.shape[0], ),
                                  p=sample_weight[self.tr_idx] / sample_weight[self.tr_idx].sum())
            tr_x = tr_x[tr_resample]
            tr_y = tr_y[tr_resample]
            val_resample = np.random.choice(val_x.shape[0], size=(val_x.shape[0], ),
                                  p=sample_weight[self.val_idx] / sample_weight[self.val_idx].sum())
            val_x = val_x[val_resample]
            val_y = val_y[val_resample]

        tr_pred = self.__call__(tf.cast(tr_x, tf.float32), sample_weight[self.tr_idx],
                        main_effect_training=False, interaction_training=False).numpy().astype(np.float64)
        val_pred = self.__call__(tf.cast(val_x, tf.float32), sample_weight[self.val_idx],
                         main_effect_training=False, interaction_training=False).numpy().astype(np.float64)
        if self.heredity:
            interaction_list_all = get_interaction_list(tr_x, val_x, tr_y.ravel(), val_y.ravel(),
                                      tr_pred.ravel(), val_pred.ravel(),
                                      self.feature_list_,
                                      self.feature_type_list_,
                                      task_type=self.task_type,
                                      active_main_effect_index=self.active_main_effect_index)
        else:
            interaction_list_all = get_interaction_list(tr_x, val_x, tr_y.ravel(), val_y.ravel(),
                          tr_pred.ravel(), val_pred.ravel(),
                          self.feature_list_,
                          self.feature_type_list_,
                          task_type=self.task_type,
                          active_main_effect_index=np.arange(self.input_num))

        self.interaction_list = interaction_list_all[:self.interact_num]
        self.interact_num_added = len(self.interaction_list)
        self.interact_blocks.set_interaction_list(self.interaction_list)
        self.output_layer.set_interaction_list(self.interaction_list)

    def fit_interaction(self, tr_x, tr_y, val_x, val_y, sample_weight=None):

        last_improvement = 0
        best_validation = np.inf
        train_size = tr_x.shape[0]
        self.interaction_status = True
        tr_sw = sample_weight[self.tr_idx]
        for epoch in range(self.interaction_epochs):
            shuffle_index = np.arange(tr_x.shape[0])
            np.random.shuffle(shuffle_index)
            tr_x = tr_x[shuffle_index]
            tr_y = tr_y[shuffle_index]
            tr_sw = tr_sw[shuffle_index]
            for iterations in range(train_size // self.batch_size):
                offset = (iterations * self.batch_size) % train_size
                batch_xx = tr_x[offset:(offset + self.batch_size), :]
                batch_yy = tr_y[offset:(offset + self.batch_size)]
                batch_sw = tr_sw[offset:(offset + self.batch_size)]
                self.train_interaction(tf.cast(batch_xx, tf.float32), tf.cast(batch_yy, tf.float32), tf.cast(batch_sw, tf.float32))

            self.err_train_interaction_training.append(self.evaluate(tr_x, tr_y, tr_sw,
                                                 main_effect_training=False, interaction_training=False))
            self.err_val_interaction_training.append(self.evaluate(val_x, val_y, sample_weight[self.val_idx],
                                                 main_effect_training=False, interaction_training=False))
            if self.verbose & (epoch % 1 == 0):
                print("Interaction training epoch: %d, train loss: %0.5f, val loss: %0.5f" %
                      (epoch + 1, self.err_train_interaction_training[-1], self.err_val_interaction_training[-1]))

            if self.err_val_interaction_training[-1] < best_validation:
                best_validation = self.err_val_interaction_training[-1]
                last_improvement = epoch
            if epoch - last_improvement > self.early_stop_thres2:
                if self.verbose:
                    print("Early stop at epoch %d, with validation loss: %0.5f" % (epoch + 1, self.err_val_interaction_training[-1]))
                break
        self.evaluate(tr_x, tr_y, sample_weight[self.tr_idx], main_effect_training=False, interaction_training=True)
        self.center_interactions()

    def prune_interaction(self, val_x, val_y, sample_weight=None):

        self.interaction_val_loss = []
        sorted_index, componment_scales = self.get_interaction_rank()
        self.output_layer.interaction_switcher.assign(tf.constant(np.zeros((self.interact_num, 1)), dtype=tf.float32))
        self.interaction_val_loss.append(self.evaluate(val_x, val_y, sample_weight[self.val_idx],
                                        main_effect_training=False, interaction_training=False))
        for idx in range(self.interact_num_added):
            selected_index = sorted_index[:(idx + 1)]
            interaction_switcher = np.zeros((self.interact_num, 1))
            interaction_switcher[selected_index] = 1
            self.output_layer.interaction_switcher.assign(tf.constant(interaction_switcher, dtype=tf.float32))
            val_loss = self.evaluate(val_x, val_y, sample_weight[self.val_idx], main_effect_training=False, interaction_training=False)
            self.interaction_val_loss.append(val_loss)

        best_idx = np.argmin(self.interaction_val_loss)
        best_loss = np.min(self.interaction_val_loss)
        if best_loss > 0:
            if np.sum((self.interaction_val_loss / best_loss - 1) < self.loss_threshold) > 0:
                best_idx = np.where((self.interaction_val_loss / best_loss - 1) < self.loss_threshold)[0][0]
            
        self.active_interaction_index = sorted_index[:best_idx]
        interaction_switcher = np.zeros((self.interact_num, 1))
        interaction_switcher[self.active_interaction_index] = 1
        self.output_layer.interaction_switcher.assign(tf.constant(interaction_switcher, dtype=tf.float32))

    def fine_tune_all(self, tr_x, tr_y, val_x, val_y, sample_weight=None):

        last_improvement = 0
        best_validation = np.inf
        train_size = tr_x.shape[0]
        tr_sw = sample_weight[self.tr_idx]
        for epoch in range(self.tuning_epochs):
            shuffle_index = np.arange(train_size)
            np.random.shuffle(shuffle_index)
            tr_x = tr_x[shuffle_index]
            tr_y = tr_y[shuffle_index]
            tr_sw = tr_sw[shuffle_index]
            for iterations in range(train_size // self.batch_size):
                offset = (iterations * self.batch_size) % train_size
                batch_xx = tr_x[offset:(offset + self.batch_size), :]
                batch_yy = tr_y[offset:(offset + self.batch_size)]
                batch_sw = tr_sw[offset:(offset + self.batch_size)]
                self.train_all(tf.cast(batch_xx, tf.float32), tf.cast(batch_yy, tf.float32), tf.cast(batch_sw, tf.float32))

            self.err_train_tuning.append(self.evaluate(tr_x, tr_y, tr_sw,
                                         main_effect_training=False, interaction_training=False))
            self.err_val_tuning.append(self.evaluate(val_x, val_y, sample_weight[self.val_idx],
                                        main_effect_training=False, interaction_training=False))
            if self.verbose & (epoch % 1 == 0):
                print("Fine tuning epoch: %d, train loss: %0.5f, val loss: %0.5f" %
                      (epoch + 1, self.err_train_tuning[-1], self.err_val_tuning[-1]))

            if self.err_val_tuning[-1] < best_validation:
                best_validation = self.err_val_tuning[-1]
                last_improvement = epoch
            if epoch - last_improvement > self.early_stop_thres3:
                if self.verbose:
                    print("Early stop at epoch %d, with validation loss: %0.5f" % (epoch + 1, self.err_val_tuning[-1]))
                break
        self.evaluate(tr_x, tr_y, sample_weight[self.tr_idx], main_effect_training=True, interaction_training=True)
        self.center_main_effects()
        self.center_interactions()

    def init_fit(self, train_x, train_y, sample_weight=None):

        # initialization
        self.data_dict_density = {}
        self.err_train_main_effect_training = []
        self.err_val_main_effect_training = []
        self.err_train_interaction_training = []
        self.err_val_interaction_training = []
        self.err_train_tuning = []
        self.err_val_tuning = []

        self.interaction_list = []
        self.active_main_effect_index = []
        self.active_interaction_index = []
        self.main_effect_val_loss = []
        self.interaction_val_loss = []

        # data loading
        n_samples = train_x.shape[0]
        indices = np.arange(n_samples)
        if self.task_type == "Regression":
            tr_x, val_x, tr_y, val_y, tr_idx, val_idx = train_test_split(train_x, train_y, indices, test_size=self.val_ratio,
                                          random_state=self.random_state)
        elif self.task_type == "Classification":
            tr_x, val_x, tr_y, val_y, tr_idx, val_idx = train_test_split(train_x, train_y, indices, test_size=self.val_ratio,
                                      stratify=train_y, random_state=self.random_state)
        self.tr_idx = tr_idx
        self.val_idx = val_idx
        self.estimate_density(tr_x, sample_weight[self.tr_idx])
        return tr_x, val_x, tr_y, val_y

    def fit(self, train_x, train_y, sample_weight=None):

        n_samples = train_x.shape[0]
        if sample_weight is None:
            sample_weight = np.ones(n_samples)
        else:
            sample_weight = n_samples * sample_weight.ravel() / np.sum(sample_weight)

        tr_x, val_x, tr_y, val_y = self.init_fit(train_x, train_y, sample_weight)
        if self.verbose:
            print("#" * 20 + "GAMI-Net training start." + "#" * 20)
        # step 1: main effects
        if self.verbose:
            print("#" * 10 + "Stage 1: main effect training start." + "#" * 10)
        
        self.optimizer.lr.assign(self.lr_bp[0])
        self.fit_main_effect(tr_x, tr_y, val_x, val_y, sample_weight)
        if self.verbose:
            print("#" * 10 + "Stage 1: main effect training stop." + "#" * 10)
        self.prune_main_effect(val_x, val_y, sample_weight)
        if len(self.active_main_effect_index) == 0:
            if self.verbose:
                print("#" * 10 + "No main effect is selected, training stop." + "#" * 10)
            return

        # step2: interaction
        if self.interact_num == 0:
            if self.verbose:
                print("#" * 10 + "Max interaction is specified to zero, training stop." + "#" * 10)
            return
        if self.verbose:
            print("#" * 10 + "Stage 2: interaction training start." + "#" * 10)
        self.add_interaction(tr_x, tr_y, val_x, val_y, sample_weight)
        self.optimizer.lr.assign(self.lr_bp[1])
        self.fit_interaction(tr_x, tr_y, val_x, val_y, sample_weight)
        if self.verbose:
            print("#" * 10 + "Stage 2: interaction training stop." + "#" * 10)
        self.prune_interaction(val_x, val_y, sample_weight)

        self.optimizer.lr.assign(self.lr_bp[2])
        self.fine_tune_all(tr_x, tr_y, val_x, val_y, sample_weight)
        self.active_indice = 1 + np.hstack([-1, self.active_main_effect_index, self.input_num + self.active_interaction_index]).astype(int)
        self.effect_names = np.hstack(["Intercept", np.array(self.feature_list_), [self.feature_list_[self.interaction_list[i][0]] + " x "
                          + self.feature_list_[self.interaction_list[i][1]] for i in range(len(self.interaction_list))]])
        if self.verbose:
            print("#" * 20 + "GAMI-Net training finished." + "#" * 20)

    def summary_logs(self, save_dict=False, folder="./", name="summary_logs"):

        data_dict_log = {}
        data_dict_log.update({"err_train_main_effect_training": self.err_train_main_effect_training,
                       "err_val_main_effect_training": self.err_val_main_effect_training,
                       "err_train_interaction_training": self.err_train_interaction_training,
                       "err_val_interaction_training": self.err_val_interaction_training,
                       "err_train_tuning": self.err_train_tuning,
                       "err_val_tuning": self.err_val_tuning,
                       "interaction_list": self.interaction_list,
                       "active_main_effect_index": self.active_main_effect_index,
                       "active_interaction_index": self.active_interaction_index,
                       "main_effect_val_loss": self.main_effect_val_loss,
                       "interaction_val_loss": self.interaction_val_loss})
        
        if save_dict:
            if not os.path.exists(folder):
                os.makedirs(folder)
            save_path = folder + name
            np.save("%s.npy" % save_path, data_dict_log)

        return data_dict_log

    def global_explain(self, main_grid_size=100, interact_grid_size=100, save_dict=False, folder="./", name="global_explain"):

        # By default, we use the same main_grid_size and interact_grid_size as that of the zero mean constraint
        # Alternatively, we can also specify it manually, e.g., when we want to have the same grid size as EBM (256).        
        data_dict_global = self.data_dict_density
        sorted_index, componment_scales = self.get_all_active_rank()
        for indice in range(self.input_num):
            feature_name = self.feature_list_[indice]
            subnet = self.maineffect_blocks.subnets[indice]
            if indice in self.nfeature_index_list_:
                sx = self.nfeature_scaler_[feature_name]
                main_effect_inputs = np.linspace(0, 1, main_grid_size).reshape([-1, 1])
                main_effect_inputs_original = sx.inverse_transform(main_effect_inputs)
                main_effect_outputs = (self.output_layer.main_effect_weights.numpy()[indice]
                            * self.output_layer.main_effect_switcher.numpy()[indice]
                            * subnet.__call__(tf.cast(tf.constant(main_effect_inputs), tf.float32)).numpy())
                data_dict_global[feature_name].update({"type":"continuous",
                                      "importance":componment_scales[indice],
                                      "inputs":main_effect_inputs_original.ravel(),
                                      "outputs":main_effect_outputs.ravel()})

            elif indice in self.cfeature_index_list_:
                main_effect_inputs_original = self.dummy_values_[feature_name]
                main_effect_inputs = np.arange(len(main_effect_inputs_original)).reshape([-1, 1])
                main_effect_outputs = (self.output_layer.main_effect_weights.numpy()[indice]
                            * self.output_layer.main_effect_switcher.numpy()[indice]
                            * subnet.__call__(tf.cast(main_effect_inputs, tf.float32)).numpy())

                main_effect_input_ticks = (main_effect_inputs.ravel().astype(int) if len(main_effect_inputs_original) <= 6 else
                              np.linspace(0.1 * len(main_effect_inputs_original), len(main_effect_inputs_original) * 0.9, 4).astype(int))
                main_effect_input_labels = [main_effect_inputs_original[i] for i in main_effect_input_ticks]
                if len("".join(list(map(str, main_effect_input_labels)))) > 30:
                    main_effect_input_labels = [str(main_effect_inputs_original[i])[:4] for i in main_effect_input_ticks]

                data_dict_global[feature_name].update({"type": "categorical",
                                      "importance": componment_scales[indice],
                                      "inputs": main_effect_inputs_original,
                                      "outputs": main_effect_outputs.ravel(),
                                      "input_ticks": main_effect_input_ticks,
                                      "input_labels": main_effect_input_labels})

        for indice in range(self.interact_num_added):

            inter_net = self.interact_blocks.interacts[indice]
            feature_name1 = self.feature_list_[self.interaction_list[indice][0]]
            feature_name2 = self.feature_list_[self.interaction_list[indice][1]]
            feature_type1 = "categorical" if feature_name1 in self.cfeature_list_ else "continuous"
            feature_type2 = "categorical" if feature_name2 in self.cfeature_list_ else "continuous"
            
            axis_extent = []
            interact_input_list = []
            if feature_name1 in self.cfeature_list_:
                interact_input1_original = self.dummy_values_[feature_name1]
                interact_input1 = np.arange(len(interact_input1_original), dtype=np.float32)
                interact_input1_ticks = (interact_input1.astype(int) if len(interact_input1) <= 6 else 
                                 np.linspace(0.1 * len(interact_input1), len(interact_input1) * 0.9, 4).astype(int))
                interact_input1_labels = [interact_input1_original[i] for i in interact_input1_ticks]
                if len("".join(list(map(str, interact_input1_labels)))) > 30:
                    interact_input1_labels = [str(interact_input1_original[i])[:4] for i in interact_input1_ticks]
                interact_input_list.append(interact_input1)
                axis_extent.extend([-0.5, len(interact_input1_original) - 0.5])
            else:
                sx1 = self.nfeature_scaler_[feature_name1]
                interact_input1 = np.array(np.linspace(0, 1, interact_grid_size), dtype=np.float32)
                interact_input1_original = sx1.inverse_transform(interact_input1.reshape([-1, 1])).ravel()
                interact_input1_ticks = []
                interact_input1_labels = []
                interact_input_list.append(interact_input1)
                axis_extent.extend([interact_input1_original.min(), interact_input1_original.max()])
            if feature_name2 in self.cfeature_list_:
                interact_input2_original = self.dummy_values_[feature_name2]
                interact_input2 = np.arange(len(interact_input2_original), dtype=np.float32)
                interact_input2_ticks = (interact_input2.astype(int) if len(interact_input2) <= 6 else
                                 np.linspace(0.1 * len(interact_input2), len(interact_input2) * 0.9, 4).astype(int))
                interact_input2_labels = [interact_input2_original[i] for i in interact_input2_ticks]
                if len("".join(list(map(str, interact_input2_labels)))) > 30:
                    interact_input2_labels = [str(interact_input2_original[i])[:4] for i in interact_input2_ticks]
                interact_input_list.append(interact_input2)
                axis_extent.extend([-0.5, len(interact_input2_original) - 0.5])
            else:
                sx2 = self.nfeature_scaler_[feature_name2]
                interact_input2 = np.array(np.linspace(0, 1, interact_grid_size), dtype=np.float32)
                interact_input2_original = sx2.inverse_transform(interact_input2.reshape([-1, 1])).ravel()
                interact_input2_ticks = []
                interact_input2_labels = []
                interact_input_list.append(interact_input2)
                axis_extent.extend([interact_input2_original.min(), interact_input2_original.max()])

            x1, x2 = np.meshgrid(interact_input_list[0], interact_input_list[1][::-1])
            input_grid = np.hstack([np.reshape(x1, [-1, 1]), np.reshape(x2, [-1, 1])])

            interact_outputs = (self.output_layer.interaction_weights.numpy()[indice]
                        * self.output_layer.interaction_switcher.numpy()[indice]
                        * inter_net.__call__(input_grid, training=False).numpy().reshape(x1.shape))
            data_dict_global.update({feature_name1 + " vs. " + feature_name2:{"type": "pairwise",
                                                       "xtype": feature_type1,
                                                       "ytype": feature_type2,
                                                       "importance": componment_scales[self.input_num + indice],
                                                       "input1": interact_input1_original,
                                                       "input2": interact_input2_original,
                                                       "outputs": interact_outputs,
                                                       "input1_ticks": interact_input1_ticks,
                                                       "input2_ticks": interact_input2_ticks,
                                                       "input1_labels": interact_input1_labels,
                                                       "input2_labels": interact_input2_labels,
                                                       "axis_extent": axis_extent}})

        if save_dict:
            if not os.path.exists(folder):
                os.makedirs(folder)
            save_path = folder + name
            np.save("%s.npy" % save_path, data_dict_global)
            
        return data_dict_global
        
    def local_explain(self, x, y=None, save_dict=False, folder="./", name="local_explain"):

        predicted = self.predict(x)
        intercept = self.output_layer.output_bias.numpy()

        main_effect_output = self.maineffect_blocks.__call__(tf.cast(tf.constant(x), tf.float32)).numpy()
        if self.interact_num > 0:
            interaction_output = self.interact_blocks.__call__(tf.cast(tf.constant(x), tf.float32)).numpy()
        else:
            interaction_output = np.empty(shape=(x.shape[0], 0))

        main_effect_weights = ((self.output_layer.main_effect_weights.numpy()) * self.output_layer.main_effect_switcher.numpy()).ravel()
        interaction_weights = ((self.output_layer.interaction_weights.numpy()[:self.interact_num_added])
                              * self.output_layer.interaction_switcher.numpy()[:self.interact_num_added]).ravel()
        interaction_weights = np.hstack([interaction_weights, np.zeros((self.interact_num - self.interact_num_added))])
        scores = np.hstack([np.repeat(intercept[0], x.shape[0]).reshape(-1, 1), np.hstack([main_effect_weights, interaction_weights])
                                  * np.hstack([main_effect_output, interaction_output])])

        data_dict_local = [{"active_indice": self.active_indice,
                    "scores": scores[i],
                    "effect_names": self.effect_names,
                    "predicted": predicted[i],
                    "actual": y[i]} for i in range(x.shape[0])]

        if save_dict:
            if not os.path.exists(folder):
                os.makedirs(folder)
            save_path = folder + name
            np.save("%s.npy" % save_path, data_dict_local)

        return data_dict_local
    
    def load(self, folder="./", name="model"):
        
        save_path = folder + name + ".pickle"
        if not os.path.exists(save_path):
            raise "file not found!"

        with open(save_path, "rb") as input_file:
            model_dict = pickle.load(input_file)
        for key, item in model_dict.items():
            setattr(self, key, item)
        self.optimizer.lr = model_dict["lr_bp"][0]

    def save(self, folder="./", name="model"):

        self.__call__(np.random.uniform(0, 1, size=(1, len(self.meta_info) - 1)))

        model_dict = {}
        model_dict["meta_info"] = self.meta_info
        model_dict["subnet_arch"] = self.subnet_arch
        model_dict["interact_arch"] = self.interact_arch

        model_dict["lr_bp"] = self.lr_bp
        model_dict["batch_size"] = self.batch_size
        model_dict["task_type"] = self.task_type
        model_dict["activation_func"] = self.activation_func
        model_dict["tuning_epochs"] = self.tuning_epochs
        model_dict["main_effect_epochs"] = self.main_effect_epochs
        model_dict["interaction_epochs"] = self.interaction_epochs
        model_dict["early_stop_thres"] = self.early_stop_thres

        model_dict["heredity"] = self.heredity
        model_dict["reg_clarity"] = self.reg_clarity
        model_dict["loss_threshold"] = self.loss_threshold

        model_dict["mono_increasing_list"] = self.mono_increasing_list
        model_dict["mono_decreasing_list"] = self.mono_decreasing_list
        model_dict["lattice_size"] = self.lattice_size

        model_dict["verbose"] = self.verbose
        model_dict["val_ratio"]= self.val_ratio
        model_dict["random_state"] = self.random_state

        model_dict["dummy_values_"] = self.dummy_values_ 
        model_dict["nfeature_scaler_"] = self.nfeature_scaler_
        model_dict["cfeature_num_"] = self.cfeature_num_
        model_dict["nfeature_num_"] = self.nfeature_num_
        model_dict["feature_list_"] = self.feature_list_
        model_dict["cfeature_list_"] = self.cfeature_list_
        model_dict["nfeature_list_"] = self.nfeature_list_
        model_dict["feature_type_list_"] = self.feature_type_list_
        model_dict["cfeature_index_list_"] = self.cfeature_index_list_
        model_dict["nfeature_index_list_"] = self.nfeature_index_list_

        model_dict["interaction_list"] = self.interaction_list
        model_dict["interact_num_added"] = self.interact_num_added 
        model_dict["interaction_status"] = self.interaction_status
        model_dict["input_num"] = self.input_num
        model_dict["max_interact_num"] = self.max_interact_num
        model_dict["interact_num"] = self.interact_num

        model_dict["maineffect_blocks"] = self.maineffect_blocks
        model_dict["interact_blocks"] = self.interact_blocks
        model_dict["output_layer"] = self.output_layer
        model_dict["loss_fn"] = self.loss_fn

        model_dict["clarity_loss"] = self.clarity_loss
        model_dict["data_dict_density"] = self.data_dict_density

        model_dict["err_train_main_effect_training"] = self.err_train_main_effect_training
        model_dict["err_val_main_effect_training"] = self.err_val_main_effect_training
        model_dict["err_train_interaction_training"] = self.err_train_interaction_training
        model_dict["err_val_interaction_training"] = self.err_val_interaction_training
        model_dict["err_train_tuning"] = self.err_train_tuning
        model_dict["err_val_tuning"] = self.err_val_tuning
        model_dict["interaction_list"] = self.interaction_list
        model_dict["main_effect_val_loss"] = self.main_effect_val_loss
        model_dict["interaction_val_loss"] = self.interaction_val_loss

        model_dict["active_indice"] = self.active_indice
        model_dict["effect_names"] = self.effect_names
        model_dict["active_main_effect_index"] = self.active_main_effect_index
        model_dict["active_interaction_index"] = self.active_interaction_index

        model_dict["tr_idx"] = self.tr_idx
        model_dict["val_idx"] = self.val_idx
        
        if not os.path.exists(folder):
            os.makedirs(folder)
        save_path = folder + name + ".pickle"
        with open(save_path, 'wb') as handle:
            pickle.dump(model_dict, handle)
