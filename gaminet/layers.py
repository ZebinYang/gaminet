import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

    
class CategNet(tf.keras.layers.Layer):

    def __init__(self, category_num, cagetnet_id=0):
        super(CategNet, self).__init__()
        self.category_num = category_num
        self.cagetnet_id = cagetnet_id
        
        self.categ_bias = self.add_weight(name="cate_bias_" + str(self.cagetnet_id),
                                                 shape=[self.category_num, 1],
                                                 initializer=tf.zeros_initializer(),
                                                 trainable=True)
        self.moving_mean = self.add_weight(name="mean"+str(self.cagetnet_id), shape=[1], 
                                           initializer=tf.zeros_initializer(),trainable=False)
        self.moving_norm = self.add_weight(name="norm"+str(self.cagetnet_id), shape=[1], 
                                           initializer=tf.ones_initializer(),trainable=False)

    def set_pdf(self, input_grid, pdf_grid):
        
        self.input_grid = input_grid
        self.pdf_grid = pdf_grid

    def call(self, inputs, training=False):

        dummy = tf.one_hot(indices=tf.cast(inputs[:,0], tf.int32), depth=self.category_num)
        self.output_original = tf.matmul(dummy, self.categ_bias)

        if training:
                
            input_grid_dummy = tf.one_hot(indices=self.input_grid, depth=self.category_num)
            self.output_grid = tf.matmul(input_grid_dummy, self.categ_bias)

            self.subnet_mean = tf.reshape(tf.matmul(self.pdf_grid, tf.reshape(self.output_grid, [-1, 1])), [1])
            self.subnet_norm = tf.sqrt(tf.matmul(self.pdf_grid, tf.reshape(tf.square(self.output_grid - self.subnet_mean), [-1, 1])))
            self.subnet_norm = tf.reshape(tf.maximum(self.subnet_norm, 1e-10), [1])

            self.moving_mean.assign(self.subnet_mean)
            self.moving_norm.assign(self.subnet_norm)
        else:
            self.subnet_mean = self.moving_mean
            self.subnet_norm = self.moving_norm

        output = self.output_original - self.subnet_mean
        return output


class NumerNet(tf.keras.layers.Layer):

    def __init__(self, subnet_arch=[10, 6], activation_func=tf.tanh, grid_size=101, subnet_id=0):
        super(NumerNet, self).__init__()
        self.layers = []
        self.subnet_arch = subnet_arch
        self.activation_func = activation_func
        self.subnet_id = subnet_id
        self.grid_size = grid_size
        self.pdf_grid = None
        
        for nodes in self.subnet_arch:
            self.layers.append(layers.Dense(nodes, activation=self.activation_func, kernel_initializer=tf.keras.initializers.GlorotNormal()))
        self.output_layer = layers.Dense(1, activation=tf.identity, kernel_initializer=tf.keras.initializers.GlorotNormal())
        self.moving_mean = self.add_weight(name="mean"+str(self.subnet_id), shape=[1], initializer=tf.zeros_initializer(), trainable=False)
        self.moving_norm = self.add_weight(name="norm"+str(self.subnet_id), shape=[1], initializer=tf.ones_initializer(), trainable=False)

    def set_pdf(self, input_grid, pdf_grid):

        self.input_grid = input_grid
        self.pdf_grid = pdf_grid
        
    def call(self, inputs, training=False):
        
        x = inputs
        for dense_layer in self.layers:
            x = dense_layer(x)
        self.output_original = self.output_layer(x)

        if training:

            input_grid = self.input_grid
            for dense_layer in self.layers:
                input_grid = dense_layer(input_grid)
            self.output_grid = self.output_layer(input_grid)

            self.subnet_mean = tf.reshape(tf.matmul(self.pdf_grid, tf.reshape(self.output_grid, [-1, 1])), [1])
            self.subnet_norm = tf.sqrt(tf.matmul(self.pdf_grid, tf.reshape(tf.square(self.output_grid - self.subnet_mean), [-1, 1])))
            self.subnet_norm = tf.reshape(tf.maximum(self.subnet_norm, 1e-10), [1])

            self.moving_mean.assign(self.subnet_mean)
            self.moving_norm.assign(self.subnet_norm)
        else:
            self.subnet_mean = self.moving_mean
            self.subnet_norm = self.moving_norm

        output = self.output_original - self.subnet_mean
        return output


class MainEffectBlock(tf.keras.layers.Layer):

    def __init__(self, meta_info, numerical_index_list, categ_index_list=[], subnet_arch=[10, 6], activation_func=tf.tanh, grid_size=101):
        super(MainEffectBlock, self).__init__()

        self.meta_info = meta_info
        self.subnet_num = len(meta_info) - 1
        self.subnet_arch = subnet_arch
        self.activation_func = activation_func
        self.numerical_index_list = numerical_index_list
        self.categ_index_list = categ_index_list
        self.grid_size = grid_size

        self.subnets = []
        for i in range(self.subnet_num):
            if i in self.numerical_index_list:
                self.subnets.append(NumerNet(self.subnet_arch,
                               self.activation_func,
                               self.grid_size,
                               subnet_id=i))
            elif i in self.categ_index_list:
                key = list(self.meta_info.keys())[i]
                self.subnets.append(CategNet(category_num=len(self.meta_info[key]['values']), cagetnet_id=i))

    def call(self, inputs, training=False):
        
        self.subnet_outputs = []
        for i in range(self.subnet_num):
            subnet = self.subnets[i]
            subnet_output = subnet(tf.gather(inputs, [i], axis=1), training=training)
            self.subnet_outputs.append(subnet_output)
        output = tf.reshape(tf.squeeze(tf.stack(self.subnet_outputs, 1)), [-1, self.subnet_num])

        return output


class Interactnetwork(tf.keras.layers.Layer):

    def __init__(self, meta_info, categ_index_list, interact_arch=[100, 60], activation_func=tf.tanh, grid_size=101, interact_id=0):
        super(Interactnetwork, self).__init__()
        
        self.layers = []
        self.meta_info = meta_info
        self.categ_index_list = categ_index_list
        self.interact_arch = interact_arch
        self.activation_func = activation_func
        self.interact_id = interact_id
        self.grid_size = grid_size

        for nodes in self.interact_arch:
            self.layers.append(layers.Dense(nodes, activation=self.activation_func, 
                                  kernel_initializer=tf.keras.initializers.GlorotNormal()))
        self.output_layer = layers.Dense(1, activation=tf.identity,
                              kernel_initializer=tf.keras.initializers.GlorotNormal())
        self.interaction = None

    def onehot_encoding(self, inputs):

        interact_input_list = []
        if self.interaction[0] in self.categ_index_list:
            interact_input1 = tf.one_hot(indices=tf.cast(inputs[:,0], tf.int32), depth=self.length1)
            interact_input_list.extend(tf.unstack(interact_input1, axis=-1))
        else:
            interact_input_list.append(inputs[:, 0])
        if self.interaction[1] in self.categ_index_list:
            interact_input2 = tf.one_hot(indices=tf.cast(inputs[:,1], tf.int32), depth=self.length2)
            interact_input_list.extend(tf.unstack(interact_input2, axis=-1))
        else:
            interact_input_list.append(inputs[:, 1])
        return interact_input_list
        
    def set_interaction(self, interaction):

        self.interaction = interaction
        if self.interaction is not None:
            self.length1 = len(self.meta_info[list(self.meta_info.keys())[self.interaction[0]]]['values']) \
                        if self.interaction[0] in self.categ_index_list else self.grid_size
            self.length2 = len(self.meta_info[list(self.meta_info.keys())[self.interaction[1]]]['values']) \
                        if self.interaction[1] in self.categ_index_list else self.grid_size

            self.moving_mean1 = self.add_weight(name="mean1_"+str(self.interact_id), 
                        shape=[self.length1], initializer=tf.zeros_initializer(), trainable=False)
            self.moving_mean2 = self.add_weight(name="mean2_"+str(self.interact_id), 
                        shape=[self.length2], initializer=tf.zeros_initializer(), trainable=False)
            self.moving_mean = self.add_weight(name="mean_"+str(self.interact_id), 
                                    shape=[1], initializer=tf.zeros_initializer(), trainable=False)
            self.moving_norm = self.add_weight(name="norm_"+str(self.interact_id),
                                    shape=[1], initializer=tf.ones_initializer(), trainable=False)

    def set_pdf(self, input_grid, pdf_grid):

        self.input_grid = input_grid
        self.pdf_grid = pdf_grid

    def call(self, inputs, training=False):
        
        x = tf.stack(self.onehot_encoding(inputs), 1)
        for dense_layer in self.layers:
            x = dense_layer(x)
        self.output_original = self.output_layer(x)

        if training:

            input_grid = self.input_grid
            input_grid = tf.stack(self.onehot_encoding(input_grid), 1)
            for dense_layer in self.layers:
                input_grid = dense_layer(input_grid)
            self.output_grid = self.output_layer(input_grid)
            self.output_grid = tf.reshape(self.output_grid, [self.length2, self.length1])

            self.weighted_output_grid = tf.math.multiply(self.output_grid, self.pdf_grid)
            self.subnet_mean1 = tf.reduce_sum(self.weighted_output_grid, 0) / tf.maximum(tf.reduce_sum(self.pdf_grid, 0), 1e-10)
            self.subnet_mean2 = tf.reduce_sum(self.weighted_output_grid, 1) / tf.maximum(tf.reduce_sum(self.pdf_grid, 1), 1e-10)
            self.subnet_mean = tf.reshape(tf.reduce_sum(self.weighted_output_grid) / tf.maximum(tf.reduce_sum(self.pdf_grid), 1e-10), [1])
            self.output_grid_normalized = self.output_grid - self.subnet_mean1 - tf.reshape(self.subnet_mean2, [-1, 1]) + self.subnet_mean

            self.subnet_norm = tf.sqrt(tf.reduce_sum(tf.math.multiply(self.pdf_grid, tf.square(self.output_grid_normalized))))
            self.subnet_norm = tf.reshape(tf.maximum(self.subnet_norm, 1e-10), [1])

            self.moving_mean1.assign(self.subnet_mean1)
            self.moving_mean2.assign(self.subnet_mean2)
            self.moving_mean.assign(self.subnet_mean)
            self.moving_norm.assign(self.subnet_norm)
        else:
            self.subnet_mean1 = self.moving_mean1
            self.subnet_mean2 = self.moving_mean2
            self.subnet_mean = self.moving_mean
            self.subnet_norm = self.moving_norm
            
        if self.interaction[0] in self.categ_index_list:
            value1 = tf.gather(self.subnet_mean1, tf.cast(inputs[:, 0], tf.int32), axis=0)
        else:
            ratio1 = tf.cast(tf.math.mod((inputs[:, 0] + 0) / (1 / (self.length1 - 1)), 1), tf.float32)
            loclow1 = tf.cast((tf.math.floor(((inputs[:, 0] + 0) / (1 / (self.length1 - 1))))), tf.int32)
            lochigh1 = tf.minimum(tf.cast((tf.math.ceil(((inputs[:, 0] + 0) / (1 / (self.length1 - 1))))), tf.int32), self.length1 - 1)
            value1 = (1 - ratio1) * tf.gather(self.subnet_mean1, loclow1, axis=0) + ratio1 * tf.gather(self.subnet_mean1, lochigh1, axis=0)
        if self.interaction[1] in self.categ_index_list:
            value2 = tf.gather(self.subnet_mean2, tf.cast(inputs[:, 1], tf.int32), axis=0)
        else:
            ratio2 = tf.cast(tf.math.mod((inputs[:, 1] + 0) / (1 / (self.length2 - 1)), 1), tf.float32)
            loclow2 = tf.cast((tf.math.floor(((inputs[:, 1] + 0) / (1 / (self.length2 - 1))))), tf.int32)
            lochigh2 = tf.minimum(tf.cast((tf.math.ceil(((inputs[:, 1] + 0) / (1 / (self.length2 - 1))))), tf.int32), self.length2 - 1)
            value2 = (1 - ratio2) * tf.gather(self.subnet_mean2, loclow2, axis=0) + ratio2 * tf.gather(self.subnet_mean2, lochigh2, axis=0)
        output = (self.output_original - tf.reshape(value1, [-1, 1]) - tf.reshape(value2, [-1, 1]) + self.subnet_mean)           
        return output


class InteractionBlock(tf.keras.layers.Layer):

    def __init__(self, interact_num, meta_info, interact_arch=[10, 6], activation_func=tf.tanh, grid_size=101):

        super(InteractionBlock, self).__init__()
        self.interact_num = interact_num
        self.interact_num_filtered = interact_num
        self.interact_arch = interact_arch
        self.activation_func = activation_func
        self.grid_size = grid_size
        
        self.meta_info = meta_info
        self.categ_variable_num = 0
        self.categ_variable_list = []
        self.categ_index_list = []
        self.interaction_list = []
        for i, (key, item) in enumerate(self.meta_info.items()):
            if key == "target":
                continue
            if item['type'] == "categorical":
                self.categ_variable_num += 1
                self.categ_variable_list.append(key)
                self.categ_index_list.append(i)

        self.interacts = []
        for i in range(self.interact_num):
            self.interacts.append(Interactnetwork(self.meta_info, 
                                      self.categ_index_list, 
                                      self.interact_arch,
                                      self.activation_func,
                                      self.grid_size,
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
                                              initializer=tf.keras.initializers.GlorotNormal(),
                                              trainable=True)
        self.main_effect_switcher = self.add_weight(name="subnet_switcher",
                                              shape=[self.input_num, 1],
                                              initializer=tf.ones_initializer(),
                                              trainable=False)
        
        self.interaction_weights = self.add_weight(name="interaction_weights",
                                              shape=[self.interact_num, 1],
                                              initializer=tf.keras.initializers.GlorotNormal(),
                                              trainable=True)
        self.interaction_switcher = self.add_weight(name="interaction_switcher",
                                              shape=[self.interact_num, 1],
                                              initializer=tf.ones_initializer(),
                                              trainable=False)
        self.main_effect_output_bias = self.add_weight(name="main_effect_output_bias",
                                           shape=[1],
                                           initializer=tf.zeros_initializer(),
                                           trainable=True)
        self.interaction_output_bias = self.add_weight(name="interaction_output_bias",
                                           shape=[1],
                                           initializer=tf.zeros_initializer(),
                                           trainable=True)

    def call(self, inputs):
        self.input_main_effect = inputs[:,:self.input_num]
        if self.interact_num > 0:
            self.input_interaction = inputs[:,self.input_num:]
            output = (tf.matmul(self.input_main_effect, self.main_effect_switcher * self.main_effect_weights) 
                   + tf.matmul(self.input_interaction, self.interaction_switcher * self.interaction_weights) 
                   + self.main_effect_output_bias + self.interaction_output_bias)
        else:
            output = (tf.matmul(self.input_main_effect, self.main_effect_switcher * self.main_effect_weights) 
                   + self.main_effect_output_bias)

        return output
