import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class CategNet(tf.keras.layers.Layer):

    def __init__(self, category_num, cagetnet_id):
        super(CategNet, self).__init__()
        self.category_num = category_num
        self.cagetnet_id = cagetnet_id

        self.output_layer_bias = self.add_weight(name="output_layer_bias_" + str(self.cagetnet_id),
                                     shape=[1, 1],
                                     initializer=tf.zeros_initializer(),
                                     trainable=False)
        self.categ_bias = self.add_weight(name="cate_bias_" + str(self.cagetnet_id),
                                         shape=[self.category_num, 1],
                                         initializer=tf.zeros_initializer(),
                                         trainable=True)
        self.moving_mean = self.add_weight(name="mean" + str(self.cagetnet_id), shape=[1],
                                           initializer=tf.zeros_initializer(), trainable=False)
        self.moving_norm = self.add_weight(name="norm" + str(self.cagetnet_id), shape=[1],
                                           initializer=tf.ones_initializer(), trainable=False)

    def call(self, inputs, training=False):

        dummy = tf.one_hot(indices=tf.cast(inputs[:, 0], tf.int32), depth=self.category_num)
        self.output_original = tf.matmul(dummy, self.categ_bias) + self.output_layer_bias

        if training:
            self.subnet_mean = tf.reduce_mean(self.output_original, 0)
            self.subnet_norm = tf.math.reduce_variance(self.output_original, 0)
            self.moving_mean.assign(self.subnet_mean)
            self.moving_norm.assign(self.subnet_norm)
        else:
            self.subnet_mean = self.moving_mean
            self.subnet_norm = self.moving_norm

        output = self.output_original
        return output


class NumerNet(tf.keras.layers.Layer):

    def __init__(self, subnet_arch, activation_func, subnet_id):
        super(NumerNet, self).__init__()
        self.layers = []
        self.subnet_arch = subnet_arch
        self.activation_func = activation_func
        self.subnet_id = subnet_id

        for nodes in self.subnet_arch:
            self.layers.append(layers.Dense(nodes, activation=self.activation_func, kernel_initializer=tf.keras.initializers.Orthogonal()))
        self.output_layer = layers.Dense(1, activation=tf.identity, kernel_initializer=tf.keras.initializers.Orthogonal())

        self.min_value = self.add_weight(name="min" + str(self.subnet_id), shape=[1], initializer=tf.constant_initializer(np.inf), trainable=False)
        self.max_value = self.add_weight(name="max" + str(self.subnet_id), shape=[1], initializer=tf.constant_initializer(-np.inf), trainable=False)
        self.moving_mean = self.add_weight(name="mean" + str(self.subnet_id), shape=[1], initializer=tf.zeros_initializer(), trainable=False)
        self.moving_norm = self.add_weight(name="norm" + str(self.subnet_id), shape=[1], initializer=tf.ones_initializer(), trainable=False)

    def call(self, inputs, training=False):

        if training:
            self.min_value.assign(tf.minimum(self.min_value, tf.reduce_min(inputs)))
            self.max_value.assign(tf.maximum(self.max_value, tf.reduce_max(inputs)))

        x = tf.clip_by_value(inputs, self.min_value, self.max_value)
        for dense_layer in self.layers:
            x = dense_layer(x)
        self.output_original = self.output_layer(x)

        if training:
            self.subnet_mean = tf.reduce_mean(self.output_original, 0)
            self.subnet_norm = tf.math.reduce_variance(self.output_original, 0)
            self.moving_mean.assign(self.subnet_mean)
            self.moving_norm.assign(self.subnet_norm)
        else:
            self.subnet_mean = self.moving_mean
            self.subnet_norm = self.moving_norm

        output = self.output_original
        return output


class MainEffectBlock(tf.keras.layers.Layer):

    def __init__(self, feature_list, nfeature_index_list, cfeature_index_list, dummy_values,
                 subnet_arch, activation_func):
        super(MainEffectBlock, self).__init__()

        self.subnet_arch = subnet_arch
        self.activation_func = activation_func

        self.dummy_values = dummy_values
        self.feature_list = feature_list
        self.subnet_num = len(feature_list)
        self.nfeature_index_list = nfeature_index_list
        self.cfeature_index_list = cfeature_index_list

        self.subnets = []
        for i in range(self.subnet_num):
            if i in self.nfeature_index_list:
                self.subnets.append(NumerNet(self.subnet_arch, self.activation_func, subnet_id=i))
            elif i in self.cfeature_index_list:
                feature_name = self.feature_list[i]
                self.subnets.append(CategNet(category_num=len(self.dummy_values[feature_name]), cagetnet_id=i))

    def call(self, inputs, training=False):

        self.subnet_outputs = []
        for i in range(self.subnet_num):
            subnet = self.subnets[i]
            subnet_output = subnet(tf.gather(inputs, [i], axis=1), training=training)
            self.subnet_outputs.append(subnet_output)
        output = tf.reshape(tf.squeeze(tf.stack(self.subnet_outputs, 1)), [-1, self.subnet_num])

        return output


class Interactnetwork(tf.keras.layers.Layer):

    def __init__(self, feature_list, cfeature_index_list, dummy_values, interact_arch,
                 activation_func, interact_id):
        super(Interactnetwork, self).__init__()

        self.feature_list = feature_list
        self.dummy_values = dummy_values
        self.cfeature_index_list = cfeature_index_list

        self.layers = []
        self.interact_arch = interact_arch
        self.activation_func = activation_func
        self.interact_id = interact_id
        self.interaction = None

        for nodes in self.interact_arch:
            self.layers.append(layers.Dense(nodes, activation=self.activation_func,
                                  kernel_initializer=tf.keras.initializers.Orthogonal()))
        self.output_layer = layers.Dense(1, activation=tf.identity,
                              kernel_initializer=tf.keras.initializers.Orthogonal())
        self.moving_mean = self.add_weight(name="mean_" + str(self.interact_id),
                                shape=[1], initializer=tf.zeros_initializer(), trainable=False)
        self.moving_norm = self.add_weight(name="norm_" + str(self.interact_id),
                                shape=[1], initializer=tf.ones_initializer(), trainable=False)

    def set_interaction(self, interaction):

        self.interaction = interaction

    def onehot_encoding(self, inputs):

        interact_input_list = []
        if self.interaction[0] in self.cfeature_index_list:
            interact_input1 = tf.one_hot(indices=tf.cast(inputs[:, 0], tf.int32),
                               depth=len(self.dummy_values[self.feature_list[self.interaction[0]]]))
            interact_input_list.extend(tf.unstack(interact_input1, axis=-1))
        else:
            interact_input_list.append(inputs[:, 0])
        if self.interaction[1] in self.cfeature_index_list:
            interact_input2 = tf.one_hot(indices=tf.cast(inputs[:, 1], tf.int32),
                               depth=len(self.dummy_values[self.feature_list[self.interaction[1]]]))
            interact_input_list.extend(tf.unstack(interact_input2, axis=-1))
        else:
            interact_input_list.append(inputs[:, 1])
        return interact_input_list

    def call(self, inputs, training=False):

        x = tf.stack(self.onehot_encoding(inputs), 1)
        for dense_layer in self.layers:
            x = dense_layer(x)
        self.output_original = self.output_layer(x)

        if training:
            self.subnet_mean = tf.reduce_mean(self.output_original, 0)
            self.subnet_norm = tf.math.reduce_variance(self.output_original, 0)
            self.moving_mean.assign(self.subnet_mean)
            self.moving_norm.assign(self.subnet_norm)
        else:
            self.subnet_mean = self.moving_mean
            self.subnet_norm = self.moving_norm

        output = self.output_original
        return output


class InteractionBlock(tf.keras.layers.Layer):

    def __init__(self, interact_num, feature_list, cfeature_index_list, dummy_values,
                 interact_arch, activation_func):

        super(InteractionBlock, self).__init__()

        self.feature_list = feature_list
        self.dummy_values = dummy_values
        self.cfeature_index_list = cfeature_index_list

        self.interact_num = interact_num
        self.interact_num_filtered = 0
        self.interact_arch = interact_arch
        self.activation_func = activation_func

        self.interacts = []
        self.interaction_list = []
        for i in range(self.interact_num):
            self.interacts.append(Interactnetwork(self.feature_list,
                                      self.cfeature_index_list,
                                      self.dummy_values,
                                      self.interact_arch,
                                      self.activation_func,
                                      interact_id=i))

    def set_interaction_list(self, interaction_list):

        self.interaction_list = interaction_list
        self.interact_num_filtered = len(interaction_list)
        for i in range(self.interact_num_filtered):
            self.interacts[i].set_interaction(interaction_list[i])

    def call(self, inputs, training=False):

        self.interact_outputs = []
        for i in range(self.interact_num):
            if i >= self.interact_num_filtered:
                self.interact_outputs.append(tf.zeros([inputs.shape[0], 1]))
            else:
                interact = self.interacts[i]
                interact_input = tf.gather(inputs, self.interaction_list[i], axis=1)
                interact_output = interact(interact_input, training=training)
                self.interact_outputs.append(interact_output)

        if len(self.interact_outputs) > 0:
            output = tf.reshape(tf.squeeze(tf.stack(self.interact_outputs, 1)), [-1, self.interact_num])
        else:
            output = 0
        return output


class OutputLayer(tf.keras.layers.Layer):

    def __init__(self, input_num, interact_num):

        super(OutputLayer, self).__init__()
        self.input_num = input_num
        self.interact_num = interact_num

        self.main_effect_weights = self.add_weight(name="subnet_weights",
                                              shape=[self.input_num, 1],
                                              initializer=tf.keras.initializers.Orthogonal(),
                                              trainable=True)
        self.main_effect_switcher = self.add_weight(name="subnet_switcher",
                                              shape=[self.input_num, 1],
                                              initializer=tf.ones_initializer(),
                                              trainable=False)

        self.interaction_weights = self.add_weight(name="interaction_weights",
                                              shape=[self.interact_num, 1],
                                              initializer=tf.keras.initializers.Orthogonal(),
                                              trainable=True)
        self.interaction_switcher = self.add_weight(name="interaction_switcher",
                                              shape=[self.interact_num, 1],
                                              initializer=tf.ones_initializer(),
                                              trainable=False)
        self.output_bias = self.add_weight(name="output_bias",
                                           shape=[1],
                                           initializer=tf.zeros_initializer(),
                                           trainable=True)

    def call(self, inputs):
        self.input_main_effect = inputs[:, :self.input_num]
        if self.interact_num > 0:
            self.input_interaction = inputs[:, self.input_num:]
            output = (tf.matmul(self.input_main_effect, self.main_effect_switcher * self.main_effect_weights)
                   + tf.matmul(self.input_interaction, self.interaction_switcher * self.interaction_weights)
                   + self.output_bias)
        else:
            output = (tf.matmul(self.input_main_effect, self.main_effect_switcher * self.main_effect_weights)
                   + self.output_bias)

        return output
