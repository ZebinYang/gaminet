import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

    
class CategNet(tf.keras.layers.Layer):

    def __init__(self, depth, bn_flag=True, cagetnet_id=0):
        super(CategNet, self).__init__()
        self.bn_flag = bn_flag
        self.depth = depth
        self.cagetnet_id = cagetnet_id
        
    def build(self, input_shape=None):
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
        
    def build(self, input_shape=None):
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

    def __init__(self, subnet_arch=[10, 6], activation_func=tf.tanh, bn_flag=False, subnet_id=0):
        super(Subnetwork, self).__init__()
        self.dense = []
        self.subnet_arch = subnet_arch
        self.activation_func = activation_func
        self.bn_flag = bn_flag
        self.subnet_id = subnet_id

    def build(self, input_shape=None):
        for nodes in self.subnet_arch:
            self.dense.append(layers.Dense(nodes, activation=self.activation_func, kernel_initializer=tf.keras.initializers.GlorotNormal()))
        self.output_layer = layers.Dense(1, activation=tf.identity, kernel_initializer=tf.keras.initializers.GlorotNormal())
        self.moving_mean = self.add_weight(name="mean"+str(self.subnet_id), shape=[1], initializer=tf.zeros_initializer(), trainable=False)
        self.moving_norm = self.add_weight(name="norm"+str(self.subnet_id), shape=[1], initializer=tf.ones_initializer(), trainable=False)

    def call(self, inputs, training=False):
        
        x = inputs
        for dense_layer in self.dense:
            x = dense_layer(x)
        self.output_original = self.output_layer(x)

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


class SubnetworkBlock(tf.keras.layers.Layer):

    def __init__(self, subnet_num, numerical_index_list, subnet_arch=[10, 6], activation_func=tf.tanh, bn_flag=True):
        super(SubnetworkBlock, self).__init__()
        
        self.bn_flag = bn_flag
        self.subnet_num = subnet_num
        self.subnet_arch = subnet_arch
        self.activation_func = activation_func
        self.numerical_index_list = numerical_index_list

    def build(self, input_shape=None):
        self.subnets = []
        for i in range(self.subnet_num):
            self.subnets.append(Subnetwork(self.subnet_arch,
                               self.activation_func,
                               self.bn_flag))
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


class InteractionBlock(tf.keras.layers.Layer):

    def __init__(self, inter_num, meta_info, inter_subnet_arch=[10, 6], activation_func=tf.tanh, bn_flag=True):

        super(InteractionBlock, self).__init__()
        self.inter_num = inter_num
        self.inter_subnet_arch = inter_subnet_arch
        self.activation_func = activation_func
        self.bn_flag = bn_flag
        
        self.meta_info = meta_info
        self.categ_variable_num = 0
        self.categ_variable_list = []
        self.categ_index_list = []
        for i, (key, item) in enumerate(self.meta_info.items()):
            if key == "target":
                continue
            if item['type'] == "categorical":
                self.categ_variable_num += 1
                self.categ_variable_list.append(key)
                self.categ_index_list.append(i)

    def build(self, input_shape=None):
        self.sub_interacts = []
        for i in range(self.inter_num):
            self.sub_interacts.append(Subnetwork(self.inter_subnet_arch,
                                   self.activation_func,
                                   self.bn_flag))
        self.built = True

    def set_inter_list(self, interaction_list):
        self.interaction_list = interaction_list

    def call(self, inputs, training=False):
        
        if self.inter_num > 0:
            self.sub_interact_inputs = []
            self.sub_interact_outputs = []
            for i in range(self.inter_num):
                sub_interact = self.sub_interacts[i]
                sub_interact_input = tf.gather(inputs, self.interaction_list[i], axis=1)

                sub_interact_input1 = sub_interact_input[:, 0]
                sub_interact_input2 = sub_interact_input[:, 1]
                sub_interact_input_list = []

                if self.interaction_list[i][0] in self.categ_index_list:
                    depth = len(self.meta_info[list(self.meta_info.keys())[self.interaction_list[i][0]]]['values'])
                    sub_interact_input1 = tf.one_hot(indices=tf.cast(sub_interact_input[:,0], tf.int32), depth=depth)
                    sub_interact_input_list.extend(tf.unstack(sub_interact_input1, axis=-1))
                else:
                    sub_interact_input_list.append(sub_interact_input1)
                if self.interaction_list[i][1] in self.categ_index_list:
                    depth = len(self.meta_info[list(self.meta_info.keys())[self.interaction_list[i][1]]]['values'])
                    sub_interact_input2 = tf.one_hot(indices=tf.cast(sub_interact_input[:,1], tf.int32), depth=depth)
                    sub_interact_input_list.extend(tf.unstack(sub_interact_input2, axis=-1))
                else:
                    sub_interact_input_list.append(sub_interact_input2)
                sub_interact_input = tf.stack(sub_interact_input_list, 1)
                sub_interact_output = sub_interact(sub_interact_input, training=training)
                self.sub_interact_inputs.append(sub_interact_input) 
                self.sub_interact_outputs.append(sub_interact_output)
            output = tf.reshape(tf.squeeze(tf.stack(self.sub_interact_outputs, 1)), [-1, self.inter_num])
        else:
            output = 0
        return output
    

class OutputLayer(tf.keras.layers.Layer):

    def __init__(self, input_num, inter_num, l1_subnet=0.001, l1_inter=0.001):

        super(OutputLayer, self).__init__()
        self.l1_subnet = l1_subnet
        self.l1_inter = l1_inter
        self.input_num = input_num
        self.inter_num = inter_num

    def build(self, input_shape=None):
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
                                              shape=[self.inter_num, 1],
                                              initializer=tf.keras.initializers.GlorotNormal(),
                                              regularizer=tf.keras.regularizers.l1(self.l1_inter),
                                              trainable=True)
        self.interaction_switcher = self.add_weight(name="interaction_switcher",
                                              shape=[self.inter_num, 1],
                                              initializer=tf.ones_initializer(),
                                              trainable=False)
        self.output_bias = self.add_weight(name="output_bias",
                                           shape=[1],
                                           initializer=tf.zeros_initializer(),
                                           trainable=True)

    def call(self, inputs, training=False):
        input_subnets = inputs[:,:self.input_num]
        if self.inter_num > 0:
            input_interactions = inputs[:,-self.inter_num:]
            output = (tf.matmul(input_subnets, self.subnet_switcher * self.subnet_weights) 
                   + tf.matmul(input_interactions, self.interaction_switcher * self.interaction_weights) 
                   + self.output_bias)
        else:
            output = (tf.matmul(input_subnets, self.subnet_switcher * self.subnet_weights) 
                   + self.output_bias)

        return output
