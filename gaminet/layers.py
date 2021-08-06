import numpy as np
import tensorflow as tf
import tensorflow_lattice as tfl
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

    def call(self, inputs, sample_weight=None, training=False):

        dummy = tf.one_hot(indices=tf.cast(inputs[:, 0], tf.int32), depth=self.category_num)
        self.output_original = tf.matmul(dummy, self.categ_bias) + self.output_layer_bias

        if training:
            if sample_weight is None:
                if inputs.shape[0] is not None:
                    sample_weight = tf.ones([inputs.shape[0], 1])
                    self.subnet_mean, self.subnet_norm = tf.nn.weighted_moments(self.output_original,
                                                            frequency_weights=sample_weight, axes=0)
            else:
                sample_weight = tf.reshape(sample_weight, shape=(-1, 1))
                self.subnet_mean, self.subnet_norm = tf.nn.weighted_moments(self.output_original,
                                                        frequency_weights=sample_weight, axes=0)
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

    def call(self, inputs, sample_weight=None, training=False):

        if training:
            self.min_value.assign(tf.minimum(self.min_value, tf.reduce_min(inputs)))
            self.max_value.assign(tf.maximum(self.max_value, tf.reduce_max(inputs)))

        x = tf.clip_by_value(inputs, self.min_value, self.max_value)
        for dense_layer in self.layers:
            x = dense_layer(x)
        self.output_original = self.output_layer(x)

        if training:
            if sample_weight is None:
                if inputs.shape[0] is not None:
                    sample_weight = tf.ones([inputs.shape[0], 1])
                    self.subnet_mean, self.subnet_norm = tf.nn.weighted_moments(self.output_original,
                                                            frequency_weights=sample_weight, axes=0)
            else:
                sample_weight = tf.reshape(sample_weight, shape=(-1, 1))
                self.subnet_mean, self.subnet_norm = tf.nn.weighted_moments(self.output_original,
                                                        frequency_weights=sample_weight, axes=0)
            self.moving_mean.assign(self.subnet_mean)
            self.moving_norm.assign(self.subnet_norm)
        else:
            self.subnet_mean = self.moving_mean
            self.subnet_norm = self.moving_norm

        output = self.output_original
        return output

    
class MonoNumerNet(tf.keras.layers.Layer):

    def __init__(self, lattice_size, subnet_id):
        super(MonoNumerNet, self).__init__()

        self.subnet_id = subnet_id
        self.lattice_size = lattice_size
        self.lattice_layer = tfl.layers.Lattice(lattice_sizes=[self.lattice_size], monotonicities=['increasing'])
        self.lattice_layer_input = tfl.layers.PWLCalibration(input_keypoints=np.linspace(0, 1, num=8, dtype=np.float32),
                                    output_min=0.0, output_max=self.lattice_size - 1.0, monotonicity='increasing')
        self.lattice_layer_bias = self.add_weight(name="lattice_layer_bias_" + str(self.subnet_id), shape=[1],
                                    initializer=tf.zeros_initializer(), trainable=False)

        self.min_value = self.add_weight(name="min" + str(self.subnet_id), shape=[1], initializer=tf.constant_initializer(np.inf), trainable=False)
        self.max_value = self.add_weight(name="max" + str(self.subnet_id), shape=[1], initializer=tf.constant_initializer(-np.inf), trainable=False)
        self.moving_mean = self.add_weight(name="mean" + str(self.subnet_id), shape=[1], initializer=tf.zeros_initializer(), trainable=False)
        self.moving_norm = self.add_weight(name="norm" + str(self.subnet_id), shape=[1], initializer=tf.ones_initializer(), trainable=False)

    def call(self, inputs, sample_weight=None, training=False):

        if training:
            self.min_value.assign(tf.minimum(self.min_value, tf.reduce_min(inputs)))
            self.max_value.assign(tf.maximum(self.max_value, tf.reduce_max(inputs)))

        x = tf.clip_by_value(inputs, self.min_value, self.max_value)
        lattice_input = self.lattice_layer_input(x)
        self.output_original = self.lattice_layer(lattice_input) + self.lattice_layer_bias

        if training:
            if sample_weight is None:
                if inputs.shape[0] is not None:
                    sample_weight = tf.ones([inputs.shape[0], 1])
                    self.subnet_mean, self.subnet_norm = tf.nn.weighted_moments(self.output_original,
                                                            frequency_weights=sample_weight, axes=0)
            else:
                sample_weight = tf.reshape(sample_weight, shape=(-1, 1))
                self.subnet_mean, self.subnet_norm = tf.nn.weighted_moments(self.output_original,
                                                        frequency_weights=sample_weight, axes=0)
            self.moving_mean.assign(self.subnet_mean)
            self.moving_norm.assign(self.subnet_norm)
        else:
            self.subnet_mean = self.moving_mean
            self.subnet_norm = self.moving_norm

        output = self.output_original
        return output


class MainEffectBlock(tf.keras.layers.Layer):

    def __init__(self, feature_list, nfeature_index_list, cfeature_index_list, dummy_values,
                 subnet_arch, activation_func, mono_list, lattice_size):
        super(MainEffectBlock, self).__init__()

        self.subnet_arch = subnet_arch
        self.lattice_size = lattice_size
        self.activation_func = activation_func
        
        self.dummy_values = dummy_values
        self.feature_list = feature_list
        self.subnet_num = len(feature_list)
        self.nfeature_index_list = nfeature_index_list
        self.cfeature_index_list = cfeature_index_list
        self.mono_list = mono_list

        self.subnets = []
        for i in range(self.subnet_num):
            if i in self.nfeature_index_list:
                if i in self.mono_list:
                    self.subnets.append(MonoNumerNet(self.lattice_size, subnet_id=i))
                else:
                    self.subnets.append(NumerNet(self.subnet_arch, self.activation_func, subnet_id=i))
            elif i in self.cfeature_index_list:
                feature_name = self.feature_list[i]
                self.subnets.append(CategNet(category_num=len(self.dummy_values[feature_name]), cagetnet_id=i))

    def call(self, inputs, sample_weight=None, training=False):

        self.subnet_outputs = []
        for i in range(self.subnet_num):
            subnet = self.subnets[i]
            subnet_output = subnet(tf.gather(inputs, [i], axis=1), sample_weight=sample_weight, training=training)
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

    def set_interaction(self, interaction):

        self.interaction = interaction
        for nodes in self.interact_arch:
            self.layers.append(layers.Dense(nodes, activation=self.activation_func, kernel_initializer=tf.keras.initializers.Orthogonal()))
        self.output_layer = layers.Dense(1, activation=tf.identity, kernel_initializer=tf.keras.initializers.Orthogonal())
        
        self.min_value1 = self.add_weight(name="min1" + str(self.interact_id), shape=[1],
                                          initializer=tf.constant_initializer(np.inf), trainable=False)
        self.max_value1 = self.add_weight(name="max1" + str(self.interact_id), shape=[1],
                                          initializer=tf.constant_initializer(-np.inf), trainable=False)
        self.min_value2 = self.add_weight(name="min2" + str(self.interact_id), shape=[1],
                                          initializer=tf.constant_initializer(np.inf), trainable=False)
        self.max_value2 = self.add_weight(name="max2" + str(self.interact_id), shape=[1],
                                          initializer=tf.constant_initializer(-np.inf), trainable=False)

        self.moving_mean = self.add_weight(name="mean_" + str(self.interact_id),
                                shape=[1], initializer=tf.zeros_initializer(), trainable=False)
        self.moving_norm = self.add_weight(name="norm_" + str(self.interact_id),
                                shape=[1], initializer=tf.ones_initializer(), trainable=False)

    def preprocessing(self, inputs):

        interact_input_list = []
        if self.interaction[0] in self.cfeature_index_list:
            interact_input1 = tf.one_hot(indices=tf.cast(inputs[:, 0], tf.int32),
                               depth=len(self.dummy_values[self.feature_list[self.interaction[0]]]))
            interact_input_list.extend(tf.unstack(interact_input1, axis=-1))
        else:
            interact_input_list.append(tf.clip_by_value(inputs[:, 0], self.min_value1, self.max_value1))
        if self.interaction[1] in self.cfeature_index_list:
            interact_input2 = tf.one_hot(indices=tf.cast(inputs[:, 1], tf.int32),
                               depth=len(self.dummy_values[self.feature_list[self.interaction[1]]]))
            interact_input_list.extend(tf.unstack(interact_input2, axis=-1))
        else:
            interact_input_list.append(tf.clip_by_value(inputs[:, 1], self.min_value2, self.max_value2))
        return interact_input_list

    def call(self, inputs, sample_weight=None, training=False):

        if training:
            self.min_value1.assign(tf.minimum(self.min_value1, tf.reduce_min(inputs[:, 0])))
            self.max_value1.assign(tf.maximum(self.max_value1, tf.reduce_max(inputs[:, 0])))
            self.min_value2.assign(tf.minimum(self.min_value2, tf.reduce_min(inputs[:, 1])))
            self.max_value2.assign(tf.maximum(self.max_value2, tf.reduce_max(inputs[:, 1])))

        x = tf.stack(self.preprocessing(inputs), 1)
        for dense_layer in self.layers:
            x = dense_layer(x)
        self.output_original = self.output_layer(x)

        if training:
            if sample_weight is None:
                if inputs.shape[0] is not None:
                    sample_weight = tf.ones([inputs.shape[0], 1])
                    self.subnet_mean, self.subnet_norm = tf.nn.weighted_moments(self.output_original,
                                                            frequency_weights=sample_weight, axes=0)
            else:
                sample_weight = tf.reshape(sample_weight, shape=(-1, 1))
                self.subnet_mean, self.subnet_norm = tf.nn.weighted_moments(self.output_original,
                                                        frequency_weights=sample_weight, axes=0)
            self.moving_mean.assign(self.subnet_mean)
            self.moving_norm.assign(self.subnet_norm)
        else:
            self.subnet_mean = self.moving_mean
            self.subnet_norm = self.moving_norm

        output = self.output_original
        return output


class MonoInteractnetwork(tf.keras.layers.Layer):

    def __init__(self, feature_list, cfeature_index_list, dummy_values, lattice_size, increasing, interact_id):
        super(MonoInteractnetwork, self).__init__()

        self.feature_list = feature_list
        self.dummy_values = dummy_values
        self.cfeature_index_list = cfeature_index_list
        
        self.increasing = increasing
        self.lattice_size = lattice_size
        self.interact_id = interact_id
        self.interaction = None

    def set_interaction(self, interaction):

        self.interaction = interaction
        if self.interaction[0] in self.cfeature_index_list:
            depth = len(self.dummy_values[self.feature_list[self.interaction[0]]])
            self.lattice_layer_input1 = tfl.layers.CategoricalCalibration(num_buckets=depth, output_min=0.0, output_max=1.0)
        else:
            self.lattice_layer_input1 = tfl.layers.PWLCalibration(input_keypoints=np.linspace(0, 1, num=8, dtype=np.float32),
                                     output_min=0.0, output_max=self.lattice_size[0] - 1.0, monotonicity='increasing')

        if self.interaction[1] in self.cfeature_index_list:
            depth = len(self.dummy_values[self.feature_list[self.interaction[1]]])
            self.lattice_layer_input2 = tfl.layers.CategoricalCalibration(num_buckets=depth, output_min=0.0, output_max=1.0)
        else:
            self.lattice_layer_input2 = tfl.layers.PWLCalibration(input_keypoints=np.linspace(0, 1, num=8, dtype=np.float32),
                                     output_min=0.0, output_max=self.lattice_size[1] - 1.0, monotonicity='increasing')

        self.lattice_layer2d = tfl.layers.Lattice(lattice_sizes=self.lattice_size, monotonicities=self.increasing)
        self.lattice_layer_bias = self.add_weight(name="lattice_layer2d_bias_" + str(self.interact_id), shape=[1],
                                    initializer=tf.zeros_initializer(), trainable=False)

        self.min_value1 = self.add_weight(name="min1" + str(self.interact_id), shape=[1],
                                          initializer=tf.constant_initializer(np.inf), trainable=False)
        self.max_value1 = self.add_weight(name="max1" + str(self.interact_id), shape=[1],
                                          initializer=tf.constant_initializer(-np.inf), trainable=False)
        self.min_value2 = self.add_weight(name="min2" + str(self.interact_id), shape=[1],
                                          initializer=tf.constant_initializer(np.inf), trainable=False)
        self.max_value2 = self.add_weight(name="max2" + str(self.interact_id), shape=[1],
                                          initializer=tf.constant_initializer(-np.inf), trainable=False)

        self.moving_mean = self.add_weight(name="mean_" + str(self.interact_id),
                                shape=[1], initializer=tf.zeros_initializer(), trainable=False)
        self.moving_norm = self.add_weight(name="norm_" + str(self.interact_id),
                                shape=[1], initializer=tf.ones_initializer(), trainable=False)

    def preprocessing(self, inputs):

        interact_input_list = []
        if self.interaction[0] in self.cfeature_index_list:
            interact_input_list.append(tf.reshape(inputs[:, 0], (-1, 1)))
        else:
            interact_input_list.append(tf.reshape(tf.clip_by_value(inputs[:, 0], self.min_value1, self.max_value1), (-1, 1)))
        if self.interaction[1] in self.cfeature_index_list:
            interact_input_list.append(tf.reshape(inputs[:, 1], (-1, 1)))
        else:
            interact_input_list.append(tf.reshape(tf.clip_by_value(inputs[:, 1], self.min_value2, self.max_value2), (-1, 1)))
        return interact_input_list

    def call(self, inputs, sample_weight=None, training=False):

        if training:
            self.min_value1.assign(tf.minimum(self.min_value1, tf.reduce_min(inputs[:, 0])))
            self.max_value1.assign(tf.maximum(self.max_value1, tf.reduce_max(inputs[:, 0])))
            self.min_value2.assign(tf.minimum(self.min_value2, tf.reduce_min(inputs[:, 1])))
            self.max_value2.assign(tf.maximum(self.max_value2, tf.reduce_max(inputs[:, 1])))

        x = self.preprocessing(inputs)
        lattice_input2d = tf.keras.layers.Concatenate(axis=1)([self.lattice_layer_input1(x[0]), self.lattice_layer_input2(x[1])])
        self.output_original = self.lattice_layer2d(lattice_input2d) + self.lattice_layer_bias

        if training:
            if sample_weight is None:
                if inputs.shape[0] is not None:
                    sample_weight = tf.ones([inputs.shape[0], 1])
                    self.subnet_mean, self.subnet_norm = tf.nn.weighted_moments(self.output_original,
                                                            frequency_weights=sample_weight, axes=0)
            else:
                sample_weight = tf.reshape(sample_weight, shape=(-1, 1))
                self.subnet_mean, self.subnet_norm = tf.nn.weighted_moments(self.output_original,
                                                        frequency_weights=sample_weight, axes=0)
            self.moving_mean.assign(self.subnet_mean)
            self.moving_norm.assign(self.subnet_norm)
        else:
            self.subnet_mean = self.moving_mean
            self.subnet_norm = self.moving_norm

        output = self.output_original
        return output


class InteractionBlock(tf.keras.layers.Layer):

    def __init__(self, interact_num, feature_list, cfeature_index_list, dummy_values,
                 interact_arch, activation_func, mono_list, lattice_size):

        super(InteractionBlock, self).__init__()

        self.feature_list = feature_list
        self.dummy_values = dummy_values
        self.cfeature_index_list = cfeature_index_list

        self.interact_num_added = 0
        self.interact_num = interact_num
        self.interact_arch = interact_arch
        self.activation_func = activation_func
        self.lattice_size = lattice_size
        self.mono_list = mono_list

    def set_interaction_list(self, interaction_list):

        self.interacts = []
        self.interaction_list = interaction_list
        self.interact_num_added = len(interaction_list)
        for i in range(self.interact_num_added):
            if (interaction_list[i][0] in self.mono_list) or (interaction_list[i][1] in self.mono_list):
                lattice_size = [2, 2]
                increasing = ['none', 'none']
                if interaction_list[i][0] in self.mono_list:
                    increasing[0] = 'increasing'
                    lattice_size[0] = self.lattice_size
                if interaction_list[i][1] in self.mono_list:
                    increasing[1] = 'increasing'
                    lattice_size[1] = self.lattice_size

                interact = MonoInteractnetwork(self.feature_list,
                                      self.cfeature_index_list,
                                      self.dummy_values,
                                      increasing=increasing,
                                      lattice_size=lattice_size,
                                      interact_id=i)
            else:
                interact = Interactnetwork(self.feature_list,
                                          self.cfeature_index_list,
                                          self.dummy_values,
                                          self.interact_arch,
                                          self.activation_func,
                                          interact_id=i)
            interact.set_interaction(interaction_list[i])
            self.interacts.append(interact)
            
    def call(self, inputs, sample_weight=None, training=False):

        self.interact_outputs = []
        for i in range(self.interact_num):
            if i >= self.interact_num_added:
                self.interact_outputs.append(tf.zeros([inputs.shape[0], 1]))
            else:
                interact = self.interacts[i]
                interact_input = tf.gather(inputs, self.interaction_list[i], axis=1)
                interact_output = interact(interact_input, sample_weight=sample_weight, training=training)
                self.interact_outputs.append(interact_output)

        if len(self.interact_outputs) > 0:
            output = tf.reshape(tf.squeeze(tf.stack(self.interact_outputs, 1)), [-1, self.interact_num])
        else:
            output = 0
        return output


class NonNegative(tf.keras.constraints.Constraint):

    def __init__(self, mono_increasing_list, mono_decreasing_list):

        self.mono_increasing_list = mono_increasing_list
        self.mono_decreasing_list = mono_decreasing_list

    def __call__(self, w):

        mono_weights = []
        if len(self.mono_increasing_list) > 0:
            mono_increasing_weights = tf.abs(tf.gather(w, self.mono_increasing_list))
            w = tf.tensor_scatter_nd_update(w, [[item] for item in self.mono_increasing_list], mono_increasing_weights)
        if len(self.mono_decreasing_list) > 0:
            mono_decreasing_weights = - tf.abs(tf.gather(w, self.mono_decreasing_list))
            w = tf.tensor_scatter_nd_update(w, [[item] for item in self.mono_decreasing_list], mono_decreasing_weights)
        return w


class OutputLayer(tf.keras.layers.Layer):

    def __init__(self, input_num, interact_num,mono_increasing_list, mono_decreasing_list):

        super(OutputLayer, self).__init__()

        self.interaction = []
        self.input_num = input_num
        self.interact_num_added = 0
        self.interact_num = interact_num
        self.mono_increasing_list = mono_increasing_list
        self.mono_decreasing_list = mono_decreasing_list
        
        self.main_effect_weights = self.add_weight(name="subnet_weights",
                                              shape=[self.input_num, 1],
                                              initializer=tf.keras.initializers.Orthogonal(),
                                              constraint=NonNegative(self.mono_increasing_list, self.mono_decreasing_list),
                                              trainable=True)
        self.main_effect_switcher = self.add_weight(name="subnet_switcher",
                                              shape=[self.input_num, 1],
                                              initializer=tf.ones_initializer(),
                                              trainable=False)
        
        self.interaction_weights = self.add_weight(name="interaction_weights",
                                  shape=[self.interact_num, 1],
                                  initializer=tf.keras.initializers.Orthogonal(),
                                  constraint=NonNegative([], []),
                                  trainable=True)
        self.interaction_switcher = self.add_weight(name="interaction_switcher",
                                              shape=[self.interact_num, 1],
                                              initializer=tf.ones_initializer(),
                                              trainable=False)
        self.output_bias = self.add_weight(name="output_bias",
                                           shape=[1],
                                           initializer=tf.zeros_initializer(),
                                           trainable=True)

    def set_interaction_list(self, interaction_list):

        self.mono_increasing_interact_list = []
        self.mono_decreasing_interact_list = []
        self.interaction_list = interaction_list
        self.interact_num_added = len(interaction_list)
        for i, interaction in enumerate(self.interaction_list):
            if (interaction[0] in self.mono_increasing_list) or (interaction[1] in self.mono_increasing_list):
                self.mono_increasing_interact_list.append(i)
            if (interaction[0] in self.mono_decreasing_list) or (interaction[1] in self.mono_decreasing_list):
                self.mono_decreasing_interact_list.append(i)
        self.interaction_weights.constraint.mono_increasing_list = self.mono_increasing_interact_list
        self.interaction_weights.constraint.mono_decreasing_list = self.mono_decreasing_interact_list

    def call(self, inputs):

        self.input_main_effect = inputs[:, :self.input_num]
        if self.interact_num_added > 0:
            self.input_interaction = inputs[:, self.input_num:]
            output = (tf.matmul(self.input_main_effect, self.main_effect_switcher * self.main_effect_weights)
                   + tf.matmul(self.input_interaction, self.interaction_switcher * self.interaction_weights)
                   + self.output_bias)
        else:
            output = (tf.matmul(self.input_main_effect, self.main_effect_switcher * self.main_effect_weights)
                   + self.output_bias)
        return output
