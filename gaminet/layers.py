import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

    
class CategNet(tf.keras.layers.Layer):

    def __init__(self, depth, bn_flag=True, cagetnet_id=0):
        super(CategNet, self).__init__()
        self.bn_flag = bn_flag
        self.depth = depth
        self.cagetnet_id = cagetnet_id
        
        self.categ_bias = self.add_weight(name="cate_bias_" + str(self.cagetnet_id),
                                                 shape=[self.depth, 1],
                                                 initializer=tf.zeros_initializer(),
                                                 trainable=True)
        self.moving_mean = self.add_weight(name="mean"+str(self.cagetnet_id), shape=[1], 
                                           initializer=tf.zeros_initializer(),trainable=False)
        self.moving_norm = self.add_weight(name="norm"+str(self.cagetnet_id), shape=[1], 
                                           initializer=tf.ones_initializer(),trainable=False)
        self.built = True

    def call(self, inputs, training=False):

        dummy = tf.one_hot(indices=tf.cast(inputs[:,0], tf.int32), depth=self.depth)
        self.output_original = tf.matmul(dummy, self.categ_bias)

        if training:
            self.subnet_mean = tf.reduce_mean(self.output_original, 0)
            self.subnet_norm = tf.maximum(tf.math.reduce_std(self.output_original, 0), 1e-10) 
            self.moving_mean.assign(self.subnet_mean)
            self.moving_norm.assign(self.subnet_norm)
        else:
            self.subnet_mean = self.moving_mean
            self.subnet_norm = self.moving_norm

        if self.bn_flag:
            output = (self.output_original - self.subnet_mean) / (self.subnet_norm)
        else:
            output = self.output_original
        return output

    
class CategNetBlock(tf.keras.layers.Layer):

    def __init__(self, meta_info, categ_variable_list=[], categ_index_list=[], bn_flag=True):
        super(CategNetBlock, self).__init__()
        self.meta_info = meta_info
        self.categ_variable_list = categ_variable_list
        self.categ_index_list = categ_index_list
        self.bn_flag = bn_flag
        
        self.categnets = []
        for i, key in enumerate(self.categ_variable_list):
            self.categnets.append(CategNet(depth=len(self.meta_info[key]['values']), bn_flag=self.bn_flag, cagetnet_id=i))
        self.built = True

    def call(self, inputs, training=False):
        output = 0
        if len(self.categ_variable_list) > 0:
            self.categ_output = []
            for i, key in enumerate(self.categ_variable_list):
                self.categ_output.append(self.categnets[i](tf.gather(inputs, [self.categ_index_list[i]], axis=1), training=training))  
            output = tf.reshape(tf.squeeze(tf.stack(self.categ_output, 1)), [-1, len(self.categ_variable_list)])
        return output


class Subnetwork(tf.keras.layers.Layer):

    def __init__(self, subnet_arch=[10, 6], activation_func=tf.tanh, bn_flag=True, subnet_id=0):
        super(Subnetwork, self).__init__()
        self.layers = []
        self.subnet_arch = subnet_arch
        self.activation_func = activation_func
        self.bn_flag = bn_flag
        self.subnet_id = subnet_id

        for nodes in self.subnet_arch:
            self.layers.append(layers.Dense(nodes, activation=self.activation_func, kernel_initializer=tf.keras.initializers.GlorotNormal()))
        self.output_layer = layers.Dense(1, activation=tf.identity, kernel_initializer=tf.keras.initializers.GlorotNormal())
        self.moving_mean = self.add_weight(name="mean"+str(self.subnet_id), shape=[1], initializer=tf.zeros_initializer(), trainable=False)
        self.moving_norm = self.add_weight(name="norm"+str(self.subnet_id), shape=[1], initializer=tf.ones_initializer(), trainable=False)

    def call(self, inputs, training=False):
        
        x = inputs
        for dense_layer in self.layers:
            x = dense_layer(x)
        self.output_original = self.output_layer(x)
        
        if training:
            
            x = tf.expand_dims(np.array(np.linspace(-1, 1, 101), dtype=np.float32), 1)
            for dense_layer in self.layers:
                x = dense_layer(x)
            output_grid = self.output_layer(x)
            self.subnet_mean = tf.reduce_mean(output_grid, 0)
            self.subnet_norm = tf.maximum(tf.math.reduce_std(output_grid, 0), 1e-10) 
            self.moving_mean.assign(self.subnet_mean)
            self.moving_norm.assign(self.subnet_norm)
        else:
            self.subnet_mean = self.moving_mean
            self.subnet_norm = self.moving_norm

        if self.bn_flag:
            output = (self.output_original - self.subnet_mean) / (self.subnet_norm)
        else:
            output = self.output_original
        return output


class SubnetworkBlock(tf.keras.layers.Layer):

    def __init__(self, subnet_num, numerical_index_list, subnet_arch=[10, 6], activation_func=tf.tanh, bn_flag=True):
        super(SubnetworkBlock, self).__init__()
        
        self.bn_flag = bn_flag
        self.subnet_num = subnet_num
        self.subnet_arch = subnet_arch
        self.activation_func = activation_func
        self.numerical_index_list = numerical_index_list

        self.subnets = []
        for i in range(self.subnet_num):
            self.subnets.append(Subnetwork(self.subnet_arch,
                               self.activation_func,
                               self.bn_flag, 
                               subnet_id=i))
        self.built = True

    def call(self, inputs, training=False):
        
        self.subnet_outputs = []
        self.subnet_inputs = tf.split(tf.gather(inputs, self.numerical_index_list, axis=1), self.subnet_num, 1)
        for i in range(self.subnet_num):
            subnet = self.subnets[i]
            subnet_output = subnet(self.subnet_inputs[i], training=training)
            self.subnet_outputs.append(subnet_output)

        output = tf.reshape(tf.squeeze(tf.stack(self.subnet_outputs, 1)), [-1, self.subnet_num])
        return output


class Interactnetwork(tf.keras.layers.Layer):

    def __init__(self, meta_info, categ_index_list, interact_arch=[100, 60], activation_func=tf.tanh, bn_flag=True, interact_id=0):
        super(Interactnetwork, self).__init__()
        
        self.layers = []
        self.categ_index_list = categ_index_list
        self.meta_info = meta_info
        self.interact_arch = interact_arch
        self.activation_func = activation_func
        self.bn_flag = bn_flag
        self.interact_id = interact_id

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

        input_list = []
        self.interaction = interaction
        self.length1 = len(self.meta_info[list(self.meta_info.keys())[self.interaction[0]]]['values']) \
                    if self.interaction[0] in self.categ_index_list else 101
        self.length2 = len(self.meta_info[list(self.meta_info.keys())[self.interaction[1]]]['values']) \
                    if self.interaction[1] in self.categ_index_list else 101

        if self.interaction[0] in self.categ_index_list:
            interact_input1 = np.array(np.arange(self.length1), dtype=np.float32)
            input_list.append(tf.unstack(interact_input1, axis=-1))
        else:
            input_list.append(tf.expand_dims(np.array(np.linspace(-1, 1, self.length1), dtype=np.float32), 1))
        if self.interaction[1] in self.categ_index_list:
            interact_input2 = np.array(np.arange(self.length2), dtype=np.float32)
            input_list.append(tf.unstack(interact_input2, axis=-1))
        else:
            input_list.append(tf.expand_dims(np.array(np.linspace(-1, 1, self.length2), dtype=np.float32), 1))
        x1, x2 = tf.meshgrid(input_list[0], input_list[1])
        input_grid = tf.concat([tf.reshape(x1, [-1, 1]), tf.reshape(x2, [-1, 1])], 1)
        self.input_grid_onehot = tf.stack(self.onehot_encoding(input_grid), axis=1)

        self.moving_mean1 = self.add_weight(name="mean1_"+str(self.interact_id), 
                    shape=[self.length1], initializer=tf.zeros_initializer(), trainable=False)
        self.moving_mean2 = self.add_weight(name="mean2_"+str(self.interact_id), 
                    shape=[self.length2], initializer=tf.zeros_initializer(), trainable=False)
        self.moving_mean = self.add_weight(name="mean_"+str(self.interact_id), 
                                shape=[1], initializer=tf.zeros_initializer(), trainable=False)
        self.moving_norm = self.add_weight(name="norm_"+str(self.interact_id),
                                shape=[1], initializer=tf.ones_initializer(), trainable=False)

    def call(self, inputs, training=False):
        
        x = tf.stack(self.onehot_encoding(inputs), 1)
        for dense_layer in self.layers:
            x = dense_layer(x)
        output = self.output_layer(x)
        
        if self.bn_flag & training:
            x_grid = self.input_grid_onehot
            for dense_layer in self.layers:
                x_grid = dense_layer(x_grid)
            output_grid = self.output_layer(x_grid)

            self.output_grid = tf.reshape(output_grid, [self.length2, self.length1])
            self.subnet_mean1 = tf.reduce_mean(self.output_grid, 0)
            self.subnet_mean2 = tf.reduce_mean(self.output_grid, 1)
            self.subnet_mean = tf.reshape(tf.reduce_mean(self.output_grid), [1])

            self.output_grid_normalized = self.output_grid - self.subnet_mean1 - tf.reshape(self.subnet_mean2, [-1, 1]) + self.subnet_mean
            self.subnet_norm = tf.reshape(tf.maximum(tf.math.reduce_std(self.output_grid_normalized), 1e-10), [1]) 

            self.moving_mean1.assign(self.subnet_mean1)
            self.moving_mean2.assign(self.subnet_mean2)
            self.moving_mean.assign(self.subnet_mean)
            self.moving_norm.assign(self.subnet_norm)
        else:
            self.subnet_mean1 = self.moving_mean1
            self.subnet_mean2 = self.moving_mean2
            self.subnet_mean = self.moving_mean
            self.subnet_norm = self.moving_norm

        if self.bn_flag:
            if self.interaction[0] in self.categ_index_list:
                value1 = tf.gather(self.subnet_mean1, tf.cast(inputs[:, 0], tf.int32), axis=0)
            else:
                ratio1 = tf.cast(tf.math.mod((inputs[:, 0] + 1) / (2 / (self.length1 - 1)), 1), tf.float32)
                loclow1 = tf.cast((tf.math.floor(((inputs[:, 0] + 1) / (2 / (self.length1 - 1))))), tf.int32)
                lochigh1 = tf.minimum(tf.cast((tf.math.ceil(((inputs[:, 0] + 1) / (2 / (self.length1 - 1))))), tf.int32), self.length1 - 1)
                value1 = (1 - ratio1) * tf.gather(self.subnet_mean1, loclow1, axis=0) + ratio1 * tf.gather(self.subnet_mean1, lochigh1, axis=0)
            if self.interaction[1] in self.categ_index_list:
                value2 = tf.gather(self.subnet_mean2, tf.cast(inputs[:, 1], tf.int32), axis=0)
            else:
                ratio2 = tf.cast(tf.math.mod((inputs[:, 1] + 1) / (2 / (self.length2 - 1)), 1), tf.float32)
                loclow2 = tf.cast((tf.math.floor(((inputs[:, 1] + 1) / (2 / (self.length2 - 1))))), tf.int32)
                lochigh2 = tf.minimum(tf.cast((tf.math.ceil(((inputs[:, 1] + 1) / (2 / (self.length2 - 1))))), tf.int32), self.length2 - 1)
                value2 = (1 - ratio2) * tf.gather(self.subnet_mean2, loclow2, axis=0) + ratio2 * tf.gather(self.subnet_mean2, lochigh2, axis=0)
            output = (output - tf.reshape(value1, [-1, 1]) - tf.reshape(value2, [-1, 1]) + self.subnet_mean) / (self.subnet_norm)           
            
        return output


class InteractionBlock(tf.keras.layers.Layer):

    def __init__(self, interact_num, meta_info, interact_arch=[10, 6], activation_func=tf.tanh, bn_flag=True):

        super(InteractionBlock, self).__init__()
        self.interact_num = interact_num
        self.interact_arch = interact_arch
        self.activation_func = activation_func
        self.bn_flag = bn_flag
        
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
                                      self.bn_flag,
                                      interact_id=i))

    def set_interaction_list(self, interaction_list):
        
        self.interaction_list = interaction_list
        for i in range(self.interact_num):
            self.interacts[i].set_interaction(interaction_list[i])

    def call(self, inputs, training=False):

        self.interact_outputs = []
        for i in range(self.interact_num):
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

    def __init__(self, input_num, interact_num, l1_subnet=0.001, l1_inter=0.001):

        super(OutputLayer, self).__init__()
        self.l1_subnet = l1_subnet
        self.l1_inter = l1_inter
        self.input_num = input_num
        self.interact_num = interact_num

        self.subnet_weights = self.add_weight(name="subnet_weights",
                                              shape=[self.input_num, 1],
                                              initializer=tf.keras.initializers.GlorotNormal(),
                                              regularizer=tf.keras.regularizers.l1(self.l1_subnet),
                                              trainable=True)
        self.subnet_switcher = self.add_weight(name="subnet_switcher",
                                              shape=[self.input_num, 1],
                                              initializer=tf.ones_initializer(),
                                              trainable=False)
        
        self.interaction_weights = self.add_weight(name="interaction_weights",
                                              shape=[self.interact_num, 1],
                                              initializer=tf.keras.initializers.GlorotNormal(),
                                              regularizer=tf.keras.regularizers.l1(self.l1_inter),
                                              trainable=True)
        self.interaction_switcher = self.add_weight(name="interaction_switcher",
                                              shape=[self.interact_num, 1],
                                              initializer=tf.ones_initializer(),
                                              trainable=False)
        self.output_bias = self.add_weight(name="output_bias",
                                           shape=[1],
                                           initializer=tf.zeros_initializer(),
                                           trainable=True)

    def call(self, inputs, training=False):
        input_subnets = inputs[:,:self.input_num]
        if self.interact_num > 0:
            input_interactions = inputs[:,-self.interact_num:]
            output = (tf.matmul(input_subnets, self.subnet_switcher * self.subnet_weights) 
                   + tf.matmul(input_interactions, self.interaction_switcher * self.interaction_weights) 
                   + self.output_bias)
        else:
            output = (tf.matmul(input_subnets, self.subnet_switcher * self.subnet_weights) 
                   + self.output_bias)

        return output
