import os
import numpy as np
import pandas as pd 
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt

from .layers import *
from .utils import get_interaction_list


class GAMINet(tf.keras.Model):

    def __init__(self, input_num,
                 meta_info=None, 
                 subnet_arch=[10, 6],
                 inter_num=10,
                 inter_subnet_arch=[100, 60],
                 task_type="Regression",
                 activation_func=tf.tanh,
                 bn_flag=True,
                 lr_bp=0.001,
                 l1_subnet=0.001,
                 l1_inter=0.001,
                 batch_size=1000,
                 init_training_epochs=10000,
                 interact_training_epochs=1000, 
                 tuning_epochs=500,
                 beta_threshold=0.01,
                 verbose=False,
                 val_ratio=0.2,
                 early_stop_thres=100):

        super(GAMINet, self).__init__()
        # Parameter initiation
        self.input_num = input_num
        self.meta_info = meta_info
        
        self.bn_flag = bn_flag
        self.task_type = task_type
        self.subnet_arch = subnet_arch
        self.activation_func = activation_func
        self.inter_subnet_arch = inter_subnet_arch
        self.inter_num = min(inter_num, int(round(input_num * (input_num - 1) / 2)))

        self.lr_bp = lr_bp
        self.l1_inter = l1_inter
        self.l1_subnet = l1_subnet
        self.batch_size = batch_size
        self.tuning_epochs = tuning_epochs
        self.init_training_epochs = init_training_epochs
        self.interact_training_epochs = interact_training_epochs
        self.beta_threshold = beta_threshold

        self.verbose = verbose
        self.val_ratio = val_ratio
        self.early_stop_thres = early_stop_thres

        self.categ_variable_num = 0
        self.numerical_input_num = 0
        self.categ_variable_list = []
        self.categ_index_list = []
        self.noncateg_index_list = []
        self.noncateg_variable_list = []
        self.variables_names = []
        for i, (key, item) in enumerate(self.meta_info.items()):
            if item['type'] == "target":
                continue
            if item['type'] == "categorical":
                self.categ_variable_num += 1
                self.categ_variable_list.append(key)
                self.categ_index_list.append(i)
                self.variables_names.append(key)
            else:
                self.numerical_input_num +=1
                self.noncateg_index_list.append(i)
                self.noncateg_variable_list.append(key)
                self.variables_names.append(key)
        # build
        self.categ_blocks = CategNetBlock(meta_info=self.meta_info, 
                                 categ_variable_list=self.categ_variable_list, 
                                 categ_index_list=self.categ_index_list,
                                 bn_flag=self.bn_flag)
        self.subnet_blocks = SubnetworkBlock(subnet_num=self.numerical_input_num,
                                 numerical_index_list=list(self.noncateg_index_list),
                                 subnet_arch=self.subnet_arch,
                                 activation_func=self.activation_func,
                                 bn_flag=self.bn_flag)
        self.interact_blocks = InteractionBlock(inter_num=self.inter_num,
                                meta_info=self.meta_info, 
                                inter_subnet_arch=self.inter_subnet_arch, 
                                activation_func=self.activation_func,
                                bn_flag=self.bn_flag)
        self.output_layer = OutputLayer(input_num=self.input_num,
                                inter_num=self.inter_num, 
                                l1_subnet=self.l1_subnet,
                                l1_inter=self.l1_inter)

        self.fit_interaction = False
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_bp)
        if self.task_type == "Regression":
            self.loss_fn = tf.keras.losses.MeanSquaredError()
        elif self.task_type == "Classification":
            self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        else:
            raise ValueError('The task type is not supported')

    def call(self, inputs, training=False):

        self.categ_outputs = self.categ_blocks(inputs, training=training)
        self.subnet_outputs = self.subnet_blocks(inputs, training=training)

        if self.fit_interaction:
            self.interact_outputs = self.interact_blocks(inputs, training=training)
        else:
            self.interact_outputs = tf.zeros([self.subnet_outputs.shape[0], self.inter_num])

        concat_list = []
        if self.numerical_input_num > 0:
            concat_list.append(self.subnet_outputs)
        if self.categ_variable_num > 0:
            concat_list.append(self.categ_outputs)
        if self.inter_num > 0:
            concat_list.append(self.interact_outputs)

        if self.task_type == "Regression":
            output = self.output_layer(tf.concat(concat_list, 1), training=training)
        elif self.task_type == "Classification":
            output = tf.nn.sigmoid(self.output_layer(tf.concat(concat_list, 1), training=training))
        else:
            raise ValueError('The task type is not supported')
        
        return output
    
    @tf.function
    def predict_graph(self, x):
        return self.apply(tf.cast(x, tf.float32), training=False)

    def predict(self, x):
        return self.predict_graph(x).numpy()
    
    @tf.function
    def evaluate_graph_init(self, x, y, training=False):
        return self.loss_fn(y, self.apply(tf.cast(x, tf.float32), training=training))

    @tf.function
    def evaluate_graph_inter(self, x, y, training=False):
        return self.loss_fn(y, self.apply(tf.cast(x, tf.float32), training=training))

    def evaluate(self, x, y, training=False):
        if self.fit_interaction:
            return self.evaluate_graph_inter(x, y, training=training).numpy()
        else:
            return self.evaluate_graph_init(x, y, training=training).numpy()

    @tf.function
    def train_step_init(self, inputs, labels):

        with tf.GradientTape() as tape:
            pred = self.apply(inputs, training=True)
            total_loss = self.loss_fn(labels, pred)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

    @tf.function
    def train_step_interact(self, inputs, labels):

        with tf.GradientTape() as tape:
            pred = self.apply(inputs, training=True)
            pred_loss = self.loss_fn(labels, pred)
            regularization_loss = tf.math.add_n(self.output_layer.losses)
            total_loss = pred_loss + regularization_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

    @tf.function
    def train_step_finetune(self, inputs, labels):

        with tf.GradientTape() as tape:
            pred = self.apply(inputs, training=True)
            total_loss = self.loss_fn(labels, pred)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

    def get_active_subnets(self):
        if self.bn_flag:
            beta = self.output_layer.subnet_weights.numpy()
        else:
            subnet_norm = [self.subnet_blocks.subnets[i].moving_norm.numpy()[0] for i in range(self.numerical_input_num)]
            categ_norm = [self.categ_blocks.categnets[i].moving_norm.numpy()[0]for i in range(self.categ_variable_num)]
            beta = self.output_layer.subnet_weights.numpy() * np.array([subnet_norm]).reshape([-1, 1])
        beta = beta * self.output_layer.subnet_switcher.numpy()
        if self.bn_flag:
            gamma = self.output_layer.interaction_weights.numpy() * self.output_layer.interaction_switcher.numpy()
        else:
            interaction_norm = [self.interact_blocks.subnets[i].moving_variance.numpy()[0] ** 0.5 for i in range(self.input_num)]
            gamma = (self.output_layer.interaction_weights.numpy() 
                  * np.array([interaction_norm]).reshape([-1, 1])
                  * self.output_layer.interaction_switcher.numpy())

        componment_coefs = np.vstack([beta, gamma])
        componment_scales = (np.abs(componment_coefs) / np.sum(np.abs(componment_coefs))).reshape([-1])
        sorted_index = np.argsort(componment_scales)
        active_index = sorted_index[componment_scales[sorted_index].cumsum()>self.beta_threshold][::-1]
        active_univariate_index = active_index[active_index<beta.shape[0]]
        active_interaction_index = active_index[active_index>=beta.shape[0]] - beta.shape[0]
        return active_univariate_index, active_interaction_index, beta, gamma, componment_scales

    def fit(self, train_x, train_y):

        self.err_val = []
        self.err_train = []
        tr_x = train_x[:int(round((1 - self.val_ratio) * train_x.shape[0])), :]
        tr_y = train_y[:int(round((1 - self.val_ratio) * train_x.shape[0])), :]
        val_x = train_x[int(round((1 - self.val_ratio) * train_x.shape[0])):, :]
        val_y = train_y[int(round((1 - self.val_ratio) * train_x.shape[0])):, :]

        # 1. Training
        last_improvement = 0
        best_validation = np.inf
        train_size = tr_x.shape[0]
        if self.verbose:
            print("Initial training.")
        for epoch in range(self.init_training_epochs):
            shuffle_index = np.arange(tr_x.shape[0])
            np.random.shuffle(shuffle_index)
            tr_x = tr_x[shuffle_index]
            tr_y = tr_y[shuffle_index]

            for iterations in range(train_size // self.batch_size):
                offset = (iterations * self.batch_size) % train_size
                batch_xx = tr_x[offset:(offset + self.batch_size), :]
                batch_yy = tr_y[offset:(offset + self.batch_size)]
                self.train_step_init(tf.cast(batch_xx, tf.float32), batch_yy)

            self.err_train.append(self.evaluate(tr_x, tr_y, training=True))
            self.err_val.append(self.evaluate(val_x, val_y, training=True))
            if self.verbose & (epoch % 1 == 0):
                print("Training epoch: %d, train loss: %0.5f, val loss: %0.5f" %
                      (epoch + 1, self.err_train[-1], self.err_val[-1]))

            if self.err_val[-1] < best_validation:
                best_validation = self.err_val[-1]
                last_improvement = epoch
            if epoch - last_improvement > self.early_stop_thres:
                if self.verbose:
                    print("Early stop at epoch %d, With Testing Error: %0.5f" % (epoch + 1, self.err_val[-1]))
                break

        # 2. interaction detection
        if self.verbose:
            print("Interaction training.")

        last_improvement = 0
        best_validation = np.inf
        train_pred = self.apply(tf.cast(train_x, tf.float32), training=False).numpy()
        residual = train_pred - train_y
        self.interaction_list = get_interaction_list(train_x,
                                      residual.ravel(),
                                      interactions=self.inter_num,
                                      meta_info=self.meta_info,
                                      task_type=self.task_type)
        self.interact_blocks.set_inter_list(self.interaction_list)
        self.fit_interaction = True 
        for epoch in range(self.interact_training_epochs):
            shuffle_index = np.arange(tr_x.shape[0])
            np.random.shuffle(shuffle_index)
            tr_x = tr_x[shuffle_index]
            tr_y = tr_y[shuffle_index]

            for iterations in range(train_size // self.batch_size):
                offset = (iterations * self.batch_size) % train_size
                batch_xx = tr_x[offset:(offset + self.batch_size), :]
                batch_yy = tr_y[offset:(offset + self.batch_size)]
                self.train_step_interact(tf.cast(batch_xx, tf.float32), batch_yy)

            self.err_train.append(self.evaluate(tr_x, tr_y, training=True))
            self.err_val.append(self.evaluate(val_x, val_y, training=True))
            if self.verbose & (epoch % 1 == 0):
                print("Training epoch: %d, train loss: %0.5f, val loss: %0.5f" %
                      (epoch + 1, self.err_train[-1], self.err_val[-1]))

            if self.err_val[-1] < best_validation:
                best_validation = self.err_val[-1]
                last_improvement = epoch
            if epoch - last_improvement > self.early_stop_thres:
                if self.verbose:
                    print("Early stop at epoch %d, With Testing Error: %0.5f" % (epoch + 1, self.err_val[-1]))
                break

        # 3. pruning & fine tune
        if self.verbose:
            print("Subnetwork pruning & Fine tuning.")
        last_improvement = 0
        best_validation = np.inf
        subnet_scal_factor = np.zeros((self.input_num, 1))
        interaction_scal_factor = np.zeros((self.inter_num, 1))
        active_univariate_index, active_interaction_index, beta, gamma, componment_scales = self.get_active_subnets()
        subnet_scal_factor[active_univariate_index] = 1
        interaction_scal_factor[active_interaction_index] = 1
        self.output_layer.subnet_switcher.assign(tf.constant(subnet_scal_factor, dtype=tf.float32))
        self.output_layer.interaction_switcher.assign(tf.constant(interaction_scal_factor, dtype=tf.float32))
        
        for epoch in range(self.tuning_epochs):
            shuffle_index = np.arange(tr_x.shape[0])
            np.random.shuffle(shuffle_index)
            tr_x = tr_x[shuffle_index]
            tr_y = tr_y[shuffle_index]

            for iterations in range(train_size // self.batch_size):
                offset = (iterations * self.batch_size) % train_size
                batch_xx = tr_x[offset:(offset + self.batch_size), :]
                batch_yy = tr_y[offset:(offset + self.batch_size)]
                self.train_step_finetune(tf.cast(batch_xx, tf.float32), batch_yy)

            self.err_train.append(self.evaluate(tr_x, tr_y, training=True))
            self.err_val.append(self.evaluate(val_x, val_y, training=True))
            if self.verbose & (epoch % 1 == 0):
                print("Tuning epoch: %d, train loss: %0.5f, val loss: %0.5f" %
                      (epoch + 1, self.err_train[-1], self.err_val[-1]))
            if self.err_val[-1] < best_validation:
                best_validation = self.err_val[-1]
                last_improvement = epoch
            if epoch - last_improvement > self.early_stop_thres:
                if self.verbose:
                    print("Early stop at epoch %d, With Testing Error: %0.5f" % (epoch + 1, self.err_val[-1]))
                break

        self.tr_x = tr_x
        self.tr_y = tr_y
        self.evaluate(tr_x, tr_y, training=True)

    def local_explain(self, x, y=None, folder="./results", name="demo", save_png=False, save_eps=False):
        
        if not os.path.exists(folder):
            os.makedirs(folder)
        save_path = folder + name

        f = plt.figure(figsize=(6, round((len(self.variables_names)+1) * 0.45)))
        predicted = self.predict(x)
        intercept = self.output_layer.output_bias.numpy()

        if self.numerical_input_num > 0:
            subnet_output = self.subnet_blocks.apply(x).numpy()
        else:
            subnet_output = np.array([])
        if self.categ_variable_num > 0:
            categ_output = self.categ_blocks.apply(x).numpy()
        else:
            categ_output = np.array([])
        if self.inter_num > 0:
            interact_output = self.interact_blocks.apply(x).numpy()
        else:
            interact_output = np.array([])
        active_univariate_index, active_interaction_index, beta, gamma, componment_scales = self.get_active_subnets()
        scores = np.hstack([intercept[0], (np.hstack([subnet_output.ravel(), categ_output.ravel(), 
                                       interact_output.ravel()]) * np.hstack([beta.ravel(), gamma.ravel()]).ravel()).ravel()])
        active_indice = 1 + np.hstack([-1, active_univariate_index, self.numerical_input_num + self.categ_variable_num + active_interaction_index])
        effect_names = np.hstack(["Intercept", np.array(self.variables_names)[self.noncateg_index_list],
                   np.array(self.variables_names)[self.categ_index_list],
                   [self.variables_names[self.interaction_list[i][0]] + " x " 
                    + self.variables_names[self.interaction_list[i][1]] for i in range(self.inter_num)]])
        
        plt.barh(np.arange(len(active_indice)), scores[active_indice][::-1])
        plt.yticks(np.arange(len(active_indice)), effect_names[active_indice][::-1])
        title = "Predicted: %0.4f | Actual: %0.4f"%(predicted, y) if y is not None else "Predicted: %0.4f"%(predicted)
        plt.title(title, fontsize=12)
        if save_eps:
            f.savefig("%s.png" % save_path, bbox_inches='tight', dpi=100)
        if save_png:
            f.savefig("%s.eps" % save_path, bbox_inches='tight', dpi=100)
    

    def global_explain(self, folder="./results", name="demo", cols_per_row=3, main_density=3, save_png=False, save_eps=False):

        if not os.path.exists(folder):
            os.makedirs(folder)
        save_path = folder + name

        idx = 0
        tick_step = 20
        input_grid_num = 101
        numerical_tick_loc = np.arange(10, input_grid_num - 10, tick_step).reshape([-1, 1])
        active_univariate_index, active_interaction_index, beta, gamma, componment_scales = self.get_active_subnets()
        max_ids = len(active_univariate_index) + len(active_interaction_index)
        f = plt.figure(figsize=(6 * cols_per_row, round(max_ids * 6 / cols_per_row)))
        for indice in active_univariate_index:

            if indice < self.numerical_input_num:
                
                ax1 = plt.subplot2grid((main_density * int(np.ceil(max_ids/cols_per_row)), cols_per_row),
                                (main_density * int(idx/cols_per_row), idx%cols_per_row), 
                                rowspan=main_density - 1)

                subnet = self.subnet_blocks.subnets[indice]
                feature_name = list(self.variables_names)[self.noncateg_index_list[indice]]

                sx = self.meta_info[feature_name]["scaler"]
                data_min, data_max = sx.feature_range[0], sx.feature_range[1]
                input_grid_step = (data_max - data_min) / 100
                subnets_inputs = np.arange(data_min, data_max + 0.01, input_grid_step).reshape([-1, 1])

                subnets_outputs = np.sign(beta[indice]) * subnet.apply(tf.cast(tf.constant(subnets_inputs), tf.float32)).numpy()
                x_tick_loc = numerical_tick_loc.ravel()
                x_tick_values = sx.inverse_transform(subnets_inputs).ravel()[x_tick_loc].ravel()

                if (np.max(x_tick_values) - np.min(x_tick_values)) < 0.01:
                    x_tick_values = np.array([np.format_float_scientific(x_tick_values[i], 
                                     precision=1) for i in range(x_tick_values.shape[0])])
                elif (np.max(x_tick_values) - np.min(x_tick_values)) < 10:
                    x_tick_values = np.round(x_tick_values, 2)
                elif (np.max(x_tick_values) - np.min(x_tick_values)) < 1000:
                    x_tick_values = np.round(x_tick_values).astype(int)
                else:
                    x_tick_values = np.array([np.format_float_scientific(x_tick_values[i],
                                     precision=1) for i in range(x_tick_values.shape[0])])

                ax1.plot(subnets_inputs, subnets_outputs)
                plt.xticks(subnets_inputs[x_tick_loc].ravel(), x_tick_values.ravel(), fontsize=10)
                ax1.set_ylabel("Score", fontsize=12)

                ax2 = plt.subplot2grid((main_density * int(np.ceil(max_ids/cols_per_row)), cols_per_row),
                                (main_density * int(idx/cols_per_row) + main_density - 1, idx%cols_per_row))

                ax2.hist(sx.inverse_transform(self.tr_x[:,[self.noncateg_index_list[indice]]]), density=True, bins=30)
                ax2.set_ylabel("Density", fontsize=12)
                
            else:

                ax1 = plt.subplot2grid((main_density * int(np.ceil(max_ids/cols_per_row)), cols_per_row),
                                (main_density * int(idx/cols_per_row), idx%cols_per_row), 
                                rowspan=main_density - 1)
                
                feature_name = self.categ_variable_list[indice - self.numerical_input_num]
                dummy_gamma = self.categ_blocks.categnets[indice - self.numerical_input_num].categ_bias.numpy()
                norm = self.categ_blocks.categnets[indice - self.numerical_input_num].moving_norm.numpy()
                ax1.bar(np.arange(len(self.meta_info[feature_name]['values'])), np.sign(beta[indice]) * dummy_gamma[:, 0] / norm)
                plt.xticks(np.arange(len(self.meta_info[feature_name]['values'])), 
                          self.meta_info[self.categ_variable_list[indice - self.numerical_input_num]]['values'])
                ax1.set_ylabel("Score", fontsize=12)
                
                ax2 = plt.subplot2grid((main_density * int(np.ceil(max_ids/cols_per_row)), cols_per_row),
                                (main_density * int(idx/cols_per_row) + main_density - 1, idx%cols_per_row))
                
                unique, counts = np.unique(self.tr_x[:, 
                                  self.categ_index_list[indice - self.numerical_input_num]], return_counts=True)
                ax2.bar(np.arange(len(self.meta_info[feature_name]['values'])), counts)
                plt.xticks(np.arange(len(self.meta_info[feature_name]['values'])), 
                        self.meta_info[self.categ_variable_list[indice - self.numerical_input_num]]['values'])
                ax2.set_ylabel("Density", fontsize=12)
                
            ax1.set_title(feature_name + " (" + str(np.round(100 * componment_scales[indice], 1)) + "%)", fontsize=12)
            idx = idx + 1

        for indice in active_interaction_index:

            ax1 = plt.subplot2grid((main_density * int(np.ceil(max_ids/cols_per_row)), cols_per_row),
                            (main_density * int(idx/cols_per_row), idx%cols_per_row), 
                            rowspan=main_density)

            response = []
            inter_net = self.interact_blocks.sub_interacts[indice]
            feature_name1 = self.variables_names[self.interaction_list[indice][0]]
            feature_name2 = self.variables_names[self.interaction_list[indice][1]]
            if ((feature_name1 in self.categ_variable_list) & (feature_name2 not in self.categ_variable_list)):

                sx2 = self.meta_info[feature_name2]["scaler"]
                data_min, data_max = sx2.feature_range[0], sx2.feature_range[1]
                input_grid_step = (data_max - data_min) / 100
                numerical_tick_value = np.arange(data_min, data_max + 0.01, input_grid_step).reshape([-1, 1])

                depth1 = len(self.meta_info[feature_name1]['values'])
                for i in range(depth1):
                    temp = np.zeros([input_grid_num, depth1])
                    temp[:, i] = 1   
                    grid_input = tf.cast(np.hstack([temp, numerical_tick_value[::-1]]), tf.float32)
                    grid_output = np.sign(gamma[indice]) * inter_net.apply(grid_input).numpy().reshape([-1, 1])
                    response.append(grid_output)

                response = np.hstack(response)
                x1_tick_loc = np.arange(depth1)
                x2_tick_loc = numerical_tick_loc.ravel()
                x1_real_values = self.meta_info[feature_name1]['values']
                x2_real_values = sx2.inverse_transform(numerical_tick_value).ravel()[x2_tick_loc][::-1]
                
            elif ((feature_name1 not in self.categ_variable_list) & (feature_name2 in self.categ_variable_list)):

                sx1 = self.meta_info[feature_name1]["scaler"] 
                data_min, data_max = sx1.feature_range[0], sx1.feature_range[1]
                input_grid_step = (data_max - data_min) / 100
                numerical_tick_value = np.arange(data_min, data_max + 0.01, input_grid_step).reshape([-1, 1])

                depth2 = len(self.meta_info[feature_name2]['values'])
                for i in range(depth2):
                    temp = np.zeros([input_grid_num, depth2])
                    temp[:, i] = 1   
                    grid_input = tf.cast(np.hstack([numerical_tick_value, temp]), tf.float32)
                    grid_output = np.sign(gamma[indice]) * inter_net.apply(grid_input).numpy().reshape([-1, 1])
                    response.append(grid_output)

                response = np.hstack(response).T
                x1_tick_loc = numerical_tick_loc.ravel()
                x2_tick_loc = np.arange(depth2)
                x1_real_values = sx1.inverse_transform(numerical_tick_value).ravel()[x1_tick_loc]
                x2_real_values = self.meta_info[feature_name2]['values'][::-1]

            elif ((feature_name1 in self.categ_variable_list) & (feature_name2 in self.categ_variable_list)):

                depth1 = len(self.meta_info[feature_name1]['values'])
                depth2 = len(self.meta_info[feature_name2]['values'])
                for i in range(depth2):
                    temp2 = np.zeros([1, depth2])
                    temp2[:, i] = 1
                    response_temp = []
                    for j in range(depth1):
                        temp1 = np.zeros([1, depth1])
                        temp1[:, j] = 1
                        grid_input = tf.cast(np.hstack([temp1, temp2]), tf.float32)
                        grid_output = np.sign(gamma[indice])[0] * inter_net.apply(grid_input).numpy()[0][0]
                        response_temp.append(grid_output)
                    response.append(response_temp)

                response = np.array(response)
                x1_tick_loc = np.arange(depth1)
                x2_tick_loc = np.arange(depth2)
                x1_real_values = self.meta_info[feature_name1]['values']
                x2_real_values = self.meta_info[feature_name2]['values'][::-1]

            else:
                sx1 = self.meta_info[feature_name1]["scaler"]
                sx2 = self.meta_info[feature_name2]["scaler"]
                
                data_min, data_max = sx1.feature_range[0], sx1.feature_range[1]
                input_grid_step = (data_max - data_min) / 100
                numerical_tick_value1 = np.arange(data_min, data_max + 0.01, input_grid_step).reshape([-1, 1])
                data_min, data_max = sx2.feature_range[0], sx2.feature_range[1]
                input_grid_step = (data_max - data_min) / 100
                numerical_tick_value2 = np.arange(data_min, data_max + 0.01, input_grid_step).reshape([-1, 1])

                x1, x2 = np.meshgrid(numerical_tick_value1, numerical_tick_value2[::-1])
                response = np.sign(gamma[indice]) * inter_net.apply(tf.cast(np.vstack([x1.ravel(), 
                                        x2.ravel()]).T, tf.float32)).numpy().reshape([input_grid_num, input_grid_num]) 

                x1_tick_loc = numerical_tick_loc.ravel() 
                x2_tick_loc = numerical_tick_loc.ravel()
                x1_real_values = sx1.inverse_transform(numerical_tick_value1).ravel()[x1_tick_loc]
                x2_real_values = sx2.inverse_transform(numerical_tick_value2).ravel()[x2_tick_loc][::-1]

            if feature_name1 not in self.categ_variable_list:
                if (np.max(x1_real_values) - np.min(x1_real_values)) < 0.01:
                    x1_real_values = np.array([np.format_float_scientific(x1_real_values[i], precision=1) for i in range(x1_real_values.shape[0])])
                elif (np.max(x1_real_values) - np.min(x1_real_values)) < 10:
                    x1_real_values = np.round(x1_real_values, 2)
                elif (np.max(x1_real_values) - np.min(x1_real_values)) < 1000:
                    x1_real_values = np.round(x1_real_values).astype(int)
                else:
                    x1_real_values = np.array([np.format_float_scientific(x1_real_values[i], precision=1) for i in range(x1_real_values.shape[0])])

            if feature_name2 not in self.categ_variable_list:

                if (np.max(x2_real_values) - np.min(x2_real_values)) < 0.01:
                    x2_real_values = np.array([np.format_float_scientific(x2_real_values[i], precision=1) for i in range(x2_real_values.shape[0])])
                elif (np.max(x2_real_values) - np.min(x2_real_values)) < 10:
                    x2_real_values = np.round(x2_real_values, 2)
                elif (np.max(x2_real_values) - np.min(x2_real_values)) < 1000:
                    x2_real_values = np.round(x2_real_values).astype(int)
                else:
                    x2_real_values = np.array([np.format_float_scientific(x2_real_values[i], precision=1) for i in range(x2_real_values.shape[0])])

            cf = ax1.imshow(response, interpolation='nearest', aspect='auto')
            plt.xticks(x1_tick_loc, x1_real_values, fontsize=10)
            plt.yticks(x2_tick_loc, x2_real_values, fontsize=10)
            ax1.set_title(feature_name1 + " X " + feature_name2 + " (" + 
                          str(np.round(100 * componment_scales[beta.shape[0] + indice], 1)) + "%)", fontsize=12)
            f.colorbar(cf, ax=ax1, format='%0.1f', orientation='horizontal')
            idx = idx + 1
        f.tight_layout()
        if max_ids > 0:
            if save_eps:
                f.savefig("%s.png" % save_path, bbox_inches='tight', dpi=100)
            if save_png:
                f.savefig("%s.eps" % save_path, bbox_inches='tight', dpi=100)
