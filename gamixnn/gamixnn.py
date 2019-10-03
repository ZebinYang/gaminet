import os
import numpy as np
import pandas as pd 
import tensorflow as tf
from scipy import stats
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split

from .layers import *
from .utils import get_interaction_list


class GAMIxNN(tf.keras.Model):

    def __init__(self, meta_info, 
                 subnet_arch=[10, 6],
                 interact_num=10,
                 interact_arch=[100, 60],
                 task_type="Regression",
                 activation_func=tf.tanh,
                 grid_size=21,
                 lr_bp=0.001,
                 batch_size=1000,
                 init_training_epochs=10000,
                 interact_training_epochs=1000, 
                 tuning_epochs=500,
                 main_threshold=0.01,
                 total_threshold=0.01,
                 verbose=False,
                 val_ratio=0.2,
                 early_stop_thres=100,
                 random_state=0):

        super(GAMIxNN, self).__init__()
        # Parameter initiation
        self.meta_info = meta_info
        self.input_num = len(meta_info) - 1
        
        self.task_type = task_type
        self.subnet_arch = subnet_arch
        self.grid_size = grid_size
        self.activation_func = activation_func
        self.interact_arch = interact_arch
        self.max_interact_num = int(round(self.input_num * (self.input_num - 1) / 2))
        self.interact_num = min(interact_num, self.max_interact_num)

        self.lr_bp = lr_bp
        self.batch_size = batch_size
        self.tuning_epochs = tuning_epochs
        self.init_training_epochs = init_training_epochs
        self.interact_training_epochs = interact_training_epochs
        self.main_threshold = main_threshold
        self.total_threshold = total_threshold

        self.verbose = verbose
        self.val_ratio = val_ratio
        self.early_stop_thres = early_stop_thres
        self.random_state = random_state
        
        np.random.seed(random_state)
        tf.random.set_seed(random_state)

        self.categ_variable_num = 0
        self.numerical_input_num = 0
        self.categ_variable_list = []
        self.categ_index_list = []
        self.numerical_index_list = []
        self.numerical_variable_list = []
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
                self.numerical_index_list.append(i)
                self.numerical_variable_list.append(key)
                self.variables_names.append(key)
        # build
        self.maineffect_blocks = MainEffectBlock(meta_info=self.meta_info,
                                 numerical_index_list=list(self.numerical_index_list),
                                 categ_index_list=self.categ_index_list,
                                 subnet_arch=self.subnet_arch,
                                 activation_func=self.activation_func,
                                 grid_size=self.grid_size)
        self.interact_blocks = InteractionBlock(interact_num=self.interact_num,
                                meta_info=self.meta_info,
                                interact_arch=self.interact_arch,
                                activation_func=self.activation_func,
                                grid_size=self.grid_size)
        self.output_layer = OutputLayer(input_num=self.input_num,
                                interact_num=self.interact_num)

        self.fit_interaction = False
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_bp)
        if self.task_type == "Regression":
            self.loss_fn = tf.keras.losses.MeanSquaredError()
        elif self.task_type == "Classification":
            self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        else:
            raise ValueError('The task type is not supported')

    def call(self, inputs, training=False):

        self.maineffect_outputs = self.maineffect_blocks(inputs, training=training)
        if self.fit_interaction:
            self.interact_outputs = self.interact_blocks(inputs, training=training)
        else:
            self.interact_outputs = tf.zeros([self.maineffect_outputs.shape[0], self.interact_num])

        concat_list = [self.maineffect_outputs]
        if self.interact_num > 0:
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
    def train_step_main(self, inputs, labels):

        with tf.GradientTape() as tape:
            pred = self.apply(inputs, training=True)
            total_loss = self.loss_fn(labels, pred)

        train_weights = self.maineffect_blocks.weights
        train_weights.append(self.output_layer.subnet_weights)
        train_weights.append(self.output_layer.output_bias)
        train_weights_list = []
        trainable_weights_names = [self.trainable_weights[j].name for j in range(len(self.trainable_weights))]
        for i in range(len(train_weights)):
            if train_weights[i].name in trainable_weights_names:
                train_weights_list.append(train_weights[i])
        grads = tape.gradient(total_loss, train_weights_list)
        self.optimizer.apply_gradients(zip(grads, train_weights_list))

    @tf.function
    def train_step_interact(self, inputs, labels):

        with tf.GradientTape() as tape:
            pred = self.apply(inputs, training=True)
            total_loss = self.loss_fn(labels, pred)

        train_weights = self.interact_blocks.weights
        train_weights.append(self.output_layer.interaction_weights)
        train_weights.append(self.output_layer.output_bias)
        train_weights_list = []
        trainable_weights_names = [self.trainable_weights[j].name for j in range(len(self.trainable_weights))]
        for i in range(len(train_weights)):
            if train_weights[i].name in trainable_weights_names:
                train_weights_list.append(train_weights[i])
        grads = tape.gradient(total_loss, train_weights_list)
        self.optimizer.apply_gradients(zip(grads, train_weights_list))

    def get_active_main_effects(self):

        subnet_norm = [self.maineffect_blocks.subnets[i].moving_norm.numpy()[0] for i in range(self.input_num)]
        beta = (self.output_layer.subnet_weights.numpy() * np.array([subnet_norm]).reshape([-1, 1])
             * self.output_layer.subnet_switcher.numpy())

        componment_coefs = beta
        componment_scales = (np.abs(componment_coefs) / np.sum(np.abs(componment_coefs))).reshape([-1])
        sorted_index = np.argsort(componment_scales)
        active_univariate_index = sorted_index[componment_scales[sorted_index].cumsum()>self.main_threshold][::-1]
        return active_univariate_index, beta, componment_scales
    
    def get_active_interactions(self):

        subnet_norm = [self.maineffect_blocks.subnets[i].moving_norm.numpy()[0] for i in range(self.input_num)]
        beta = (self.output_layer.subnet_weights.numpy() * np.array([subnet_norm]).reshape([-1, 1]) 
             * self.output_layer.subnet_switcher.numpy())

        interaction_norm = [self.interact_blocks.interacts[i].moving_norm.numpy()[0] for i in range(self.interact_num)]
        gamma = (self.output_layer.interaction_weights.numpy() 
              * np.array([interaction_norm]).reshape([-1, 1])
              * self.output_layer.interaction_switcher.numpy())

        componment_coefs = np.vstack([beta, gamma])
        componment_scales = (np.abs(componment_coefs) / np.sum(np.abs(componment_coefs))).reshape([-1])
        componment_scales_main = componment_scales[:self.input_num]
        componment_scales_interact = componment_scales[self.input_num:]

        sorted_index = np.argsort(componment_scales_interact)
        active_interaction_index = sorted_index[(componment_scales_interact[sorted_index].cumsum())>self.total_threshold][::-1]
        return active_interaction_index, gamma, componment_scales

    def get_active_effects(self):

        subnet_norm = [self.maineffect_blocks.subnets[i].moving_norm.numpy()[0] for i in range(self.input_num)]
        beta = (self.output_layer.subnet_weights.numpy() * np.array([subnet_norm]).reshape([-1, 1]) 
             * self.output_layer.subnet_switcher.numpy())

        interaction_norm = [self.interact_blocks.interacts[i].moving_norm.numpy()[0] for i in range(self.interact_num)]
        gamma = (self.output_layer.interaction_weights.numpy() 
              * np.array([interaction_norm]).reshape([-1, 1])
              * self.output_layer.interaction_switcher.numpy())

        componment_coefs = np.vstack([beta, gamma])
        componment_scales = (np.abs(componment_coefs) / np.sum(np.abs(componment_coefs))).reshape([-1])
        sorted_index = np.argsort(componment_scales)
        active_index = sorted_index[componment_scales[sorted_index].cumsum()>0][::-1]
        active_univariate_index = active_index[active_index<beta.shape[0]]
        active_interaction_index = active_index[active_index>=beta.shape[0]] - beta.shape[0]
        return active_univariate_index, active_interaction_index, beta, gamma, componment_scales

    def fit(self, train_x, train_y):

        self.err_val = []
        self.err_train = []
        if self.task_type == "Regression":
            tr_x, val_x, tr_y, val_y = train_test_split(train_x, train_y, test_size=self.val_ratio, 
                                          random_state=self.random_state)
        elif self.task_type == "Classification":
            tr_x, val_x, tr_y, val_y = train_test_split(train_x, train_y, test_size=self.val_ratio, 
                                      stratify=train_y, random_state=self.random_state)

        for i in range(self.input_num):

            values = train_x[:,[i]]
            if i in self.numerical_index_list:
                input_grid = np.linspace(0, 1, self.grid_size)
                kde = stats.gaussian_kde(values.T)
                pdf_grid = kde(input_grid)
                pdf_grid = np.array(pdf_grid / np.sum(pdf_grid), dtype=np.float32) 
            elif i in self.categ_index_list:
                key = self.variables_names[i]
                input_grid = np.arange(len(self.meta_info[key]['values']))
                pdf_grid, _ = np.histogram(values, bins=np.arange(len(self.meta_info[key]['values']) + 1), density=True)

            self.maineffect_blocks.subnets[i].set_pdf(np.array(input_grid, dtype=np.float32).reshape([-1, 1]),
                                        np.array(pdf_grid, dtype=np.float32).reshape([1, -1]))
        #### 1. Main Effects Training
        if self.verbose:
            print("Main Effects Training.")

        last_improvement = 0
        best_validation = np.inf
        train_size = tr_x.shape[0]
        for epoch in range(self.init_training_epochs):
            shuffle_index = np.arange(tr_x.shape[0])
            np.random.shuffle(shuffle_index)
            tr_x = tr_x[shuffle_index]
            tr_y = tr_y[shuffle_index]

            for iterations in range(train_size // self.batch_size):
                offset = (iterations * self.batch_size) % train_size
                batch_xx = tr_x[offset:(offset + self.batch_size), :]
                batch_yy = tr_y[offset:(offset + self.batch_size)]
                self.train_step_main(tf.cast(batch_xx, tf.float32), batch_yy)

            self.err_train.append(self.evaluate(tr_x, tr_y, training=False))
            self.err_val.append(self.evaluate(val_x, val_y, training=False))
            if self.verbose & (epoch % 1 == 0):
                print("Main effects training epoch: %d, train loss: %0.5f, val loss: %0.5f" %
                      (epoch + 1, self.err_train[-1], self.err_val[-1]))

            if self.err_val[-1] < best_validation:
                best_validation = self.err_val[-1]
                last_improvement = epoch
            if epoch - last_improvement > self.early_stop_thres:
                if self.verbose:
                    print("Early stop at epoch %d, With Testing Error: %0.5f" % (epoch + 1, self.err_val[-1]))
                break

        subnet_scal_factor = np.zeros((self.input_num, 1))
        active_univariate_index, beta, componment_scales = self.get_active_main_effects()
        subnet_scal_factor[active_univariate_index] = 1
        self.output_layer.subnet_switcher.assign(tf.constant(subnet_scal_factor, dtype=tf.float32))
        for epoch in range(self.tuning_epochs):
            shuffle_index = np.arange(tr_x.shape[0])
            np.random.shuffle(shuffle_index)
            tr_x = tr_x[shuffle_index]
            tr_y = tr_y[shuffle_index]

            for iterations in range(train_size // self.batch_size):
                offset = (iterations * self.batch_size) % train_size
                batch_xx = tr_x[offset:(offset + self.batch_size), :]
                batch_yy = tr_y[offset:(offset + self.batch_size)]
                self.train_step_main(tf.cast(batch_xx, tf.float32), batch_yy)

            self.err_train.append(self.evaluate(tr_x, tr_y, training=False))
            self.err_val.append(self.evaluate(val_x, val_y, training=False))
            if self.verbose & (epoch % 1 == 0):
                print("Main effects tunning epoch: %d, train loss: %0.5f, val loss: %0.5f" %
                      (epoch + 1, self.err_train[-1], self.err_val[-1]))

        # 2. interaction Training
        if self.interact_num>0:
            if self.verbose:
                print("Interaction Training.")

            last_improvement = 0
            best_validation = np.inf
            tr_pred = self.apply(tf.cast(tr_x, tf.float32), training=False).numpy().astype(np.float64)
            val_pred = self.apply(tf.cast(val_x, tf.float32), training=False).numpy().astype(np.float64)
            interaction_list_all = get_interaction_list(tr_x, val_x, tr_y.ravel(), val_y.ravel(),
                                          tr_pred.ravel(), val_pred.ravel(),
                                          interactions=int(round(self.input_num * (self.input_num - 1) / 2)),
                                          meta_info=self.meta_info,
                                          task_type=self.task_type)

            self.interaction_list = [interaction_list_all[i] for i in range(self.max_interact_num) 
                                     if (interaction_list_all[i][0] in active_univariate_index)
                                     or (interaction_list_all[i][1] in active_univariate_index)][:self.interact_num]
            self.interact_blocks.set_interaction_list(self.interaction_list)

            for interact_id, (idx1, idx2) in enumerate(self.interaction_list):

                interact_input_list = []
                values = train_x[:,[idx1, idx2]]
                feature_name1 = self.variables_names[idx1]
                feature_name2 = self.variables_names[idx2]
                if (feature_name1 in self.categ_variable_list) & (feature_name2 in self.categ_variable_list):
                    pdf_grid = np.zeros((len(self.meta_info[feature_name1]['values']),
                                  len(self.meta_info[feature_name2]['values'])))
                    for i in np.arange(len(self.meta_info[feature_name1]['values'])):
                        for j in np.arange(len(self.meta_info[feature_name2]['values'])):
                            pdf_grid[i, j] = np.sum((values[:, 0] == i)&(values[:, 1] == j))

                    pdf_grid = pdf_grid / np.sum(pdf_grid)
                    x1, x2 = np.meshgrid(np.arange(len(self.meta_info[feature_name1]['values'])), 
                                  np.arange(len(self.meta_info[feature_name2]['values'])))
                    input_grid = np.hstack([np.reshape(x1, [-1, 1]), np.reshape(x2, [-1, 1])])
                    pdf_grid = np.ones([len(self.meta_info[feature_name1]['values']), len(self.meta_info[feature_name2]['values'])]) / (len(self.meta_info[feature_name1]['values']) * len(self.meta_info[feature_name2]['values']))

                if (feature_name1 in self.categ_variable_list) & (feature_name2 not in self.categ_variable_list):

                    pdf_grid = np.zeros((len(self.meta_info[feature_name1]['values']), 
                                  self.grid_size))
                    x1, x2 = np.meshgrid(np.arange(len(self.meta_info[feature_name1]['values'])), 
                                  np.linspace(0, 1, self.grid_size))
                    input_grid = np.hstack([np.reshape(x1, [-1, 1]), np.reshape(x2, [-1, 1])])
                    for i in np.arange(len(self.meta_info[feature_name1]['values'])):
                        kde = stats.gaussian_kde(values[values[:, 0] == i][:, 1].T)
                        pdf_grid_temp = kde(np.linspace(0, 1, self.grid_size))
                        pdf_grid[i, :] = (np.sum(values[:, 0] == i) / values.shape[0]) * pdf_grid_temp / np.sum(pdf_grid_temp)
                    pdf_grid = np.ones([len(self.meta_info[feature_name1]['values']), self.grid_size]) / (self.grid_size * len(self.meta_info[feature_name1]['values']))

                if (feature_name1 not in self.categ_variable_list) & (feature_name2 in self.categ_variable_list):

                    pdf_grid = np.zeros((self.grid_size,
                                  len(self.meta_info[feature_name2]['values'])))
                    x1, x2 = np.meshgrid(np.linspace(0, 1, self.grid_size), 
                                  np.arange(len(self.meta_info[feature_name2]['values'])))
                    input_grid = np.hstack([np.reshape(x1, [-1, 1]), np.reshape(x2, [-1, 1])])
                    for j in np.arange(len(self.meta_info[feature_name2]['values'])):
                        kde = stats.gaussian_kde(values[values[:, 1] == j][:, 0].T)
                        pdf_grid_temp = kde(np.linspace(0, 1, self.grid_size))
                        pdf_grid[:, j] = (np.sum(values[:, 1] == j) / values.shape[0]) * pdf_grid_temp / np.sum(pdf_grid_temp)
                    pdf_grid = np.ones([self.grid_size, len(self.meta_info[feature_name2]['values'])]) / (self.grid_size * len(self.meta_info[feature_name2]['values']))
                    
                if (feature_name1 not in self.categ_variable_list) & (feature_name2 not in self.categ_variable_list):

                    x1, x2 = np.meshgrid(np.linspace(0, 1, self.grid_size), 
                                  np.linspace(0, 1, self.grid_size))
                    input_grid = np.hstack([np.reshape(x1, [-1, 1]), np.reshape(x2, [-1, 1])])
                    kde = stats.gaussian_kde(values.T)
                    pdf_grid = kde(np.vstack([x1.ravel(), x2.ravel()]))
                    pdf_grid = np.reshape(pdf_grid / np.sum(pdf_grid), [self.grid_size, self.grid_size])
                    pdf_grid = np.ones([self.grid_size, self.grid_size]) / (self.grid_size * self.grid_size)

                self.interact_blocks.interacts[interact_id].set_pdf(np.array(input_grid, dtype=np.float32),
                                                   np.array(pdf_grid, dtype=np.float32).T)

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

                self.err_train.append(self.evaluate(tr_x, tr_y, training=False))
                self.err_val.append(self.evaluate(val_x, val_y, training=False))
                if self.verbose & (epoch % 1 == 0):
                    print("Interaction training epoch: %d, train loss: %0.5f, val loss: %0.5f" %
                          (epoch + 1, self.err_train[-1], self.err_val[-1]))

                if self.err_val[-1] < best_validation:
                    best_validation = self.err_val[-1]
                    last_improvement = epoch
                if epoch - last_improvement > self.early_stop_thres:
                    if self.verbose:
                        print("Early stop at epoch %d, With Testing Error: %0.5f" % (epoch + 1, self.err_val[-1]))
                    break

            interaction_scal_factor = np.zeros((self.interact_num, 1))
            active_interaction_index, gamma, componment_scales = self.get_active_interactions()
            interaction_scal_factor[active_interaction_index] = 1
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
                    self.train_step_interact(tf.cast(batch_xx, tf.float32), batch_yy)

                self.err_train.append(self.evaluate(tr_x, tr_y, training=False))
                self.err_val.append(self.evaluate(val_x, val_y, training=False))
                if self.verbose & (epoch % 1 == 0):
                    print("Interaction tunning epoch: %d, train loss: %0.5f, val loss: %0.5f" %
                          (epoch + 1, self.err_train[-1], self.err_val[-1]))

        self.tr_x = tr_x
        self.tr_y = tr_y
        self.val_x = val_x
        self.val_y = val_y

    def local_explain(self, x, y=None, folder="./results", name="demo", save_png=False, save_eps=False):
        
        if not os.path.exists(folder):
            os.makedirs(folder)
        save_path = folder + name

        f = plt.figure(figsize=(6, round((len(self.variables_names)+1) * 0.45)))
        predicted = self.predict(x)
        intercept = self.output_layer.output_bias.numpy()

        subnet_output = self.maineffect_blocks.apply(x).numpy()
        if self.interact_num > 0:
            interact_output = self.interact_blocks.apply(x).numpy()
        else:
            interact_output = np.array([])
        active_univariate_index, active_interaction_index, beta, gamma, componment_scales = self.get_active_effects()
        scores = np.hstack([intercept[0], (np.hstack([subnet_output.ravel(), interact_output.ravel()]) 
                                           * np.hstack([beta.ravel(), gamma.ravel()]).ravel()).ravel()])
        active_indice = 1 + np.hstack([-1, active_univariate_index, self.numerical_input_num + self.categ_variable_num + active_interaction_index])
        effect_names = np.hstack(["Intercept", np.array(self.variables_names)[self.numerical_index_list],
                   np.array(self.variables_names)[self.categ_index_list],
                   [self.variables_names[self.interaction_list[i][0]] + " x " 
                    + self.variables_names[self.interaction_list[i][1]] for i in range(self.interact_num)]])
        
        plt.barh(np.arange(len(active_indice)), scores[active_indice][::-1])
        plt.yticks(np.arange(len(active_indice)), effect_names[active_indice][::-1])
        title = "Predicted: %0.4f | Actual: %0.4f"%(predicted, y) if y is not None else "Predicted: %0.4f"%(predicted)
        plt.title(title, fontsize=12)
        if save_eps:
            f.savefig("%s.png" % save_path, bbox_inches='tight', dpi=100)
        if save_png:
            f.savefig("%s.eps" % save_path, bbox_inches='tight', dpi=100)
    

    def global_explain(self, folder="./results", name="demo", cols_per_row=4, save_png=False, save_eps=False):

        if not os.path.exists(folder):
            os.makedirs(folder)
        save_path = folder + name

        idx = 0
        grid_length = 101
        active_univariate_index, active_interaction_index, beta, gamma, componment_scales = self.get_active_effects()
        max_ids = len(active_univariate_index) + len(active_interaction_index)
        
        fig = plt.figure(figsize=(6 * cols_per_row, 
                         4.6 * int(np.ceil(max_ids / cols_per_row))))
        outer = gridspec.GridSpec(int(np.ceil(max_ids/cols_per_row)), cols_per_row, wspace=0.25, hspace=0.25)
        for indice in active_univariate_index:

            feature_name = list(self.variables_names)[indice]
            subnet = self.maineffect_blocks.subnets[indice]
            if indice in self.numerical_index_list:
                            
                sx = self.meta_info[feature_name]['scaler']
                subnets_inputs = np.linspace(0, 1, grid_length).reshape([-1, 1])
                subnets_outputs = np.sign(beta[indice]) * subnet.apply(tf.cast(tf.constant(subnets_inputs), tf.float32)).numpy()

                inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[idx], wspace=0.1, hspace=0.1, height_ratios=[4, 1])
                ax1 = plt.Subplot(fig, inner[0]) 
                ax1.plot(subnets_inputs, subnets_outputs)
                ax1.set_ylabel("Score", fontsize=12)
                ax1.get_yaxis().set_label_coords(-0.15, 0.5)
                ax1.set_title(feature_name, fontsize=12)
                fig.add_subplot(ax1)

                ax2 = plt.Subplot(fig, inner[1]) 
                ax2.hist(sx.inverse_transform(self.tr_x[:,[indice]]), bins=30)
                ax1.get_shared_x_axes().join(ax1, ax2)
                ax1.set_xticklabels([])
                ax2.set_ylabel("Histogram", fontsize=12)
                ax2.get_yaxis().set_label_coords(-0.15, 0.5)
                if np.sum([len(ax1.get_yticklabels()[i].get_text()) for i in range(len(ax1.get_yticklabels()))]) > 20:
                    ax1.yaxis.set_tick_params(rotation=15)
                if np.sum([len(ax2.get_xticklabels()[i].get_text()) for i in range(len(ax2.get_xticklabels()))]) > 20:
                    ax2.xaxis.set_tick_params(rotation=15)
                if np.sum([len(ax2.get_yticklabels()[i].get_text()) for i in range(len(ax2.get_yticklabels()))]) > 20:
                    ax2.yaxis.set_tick_params(rotation=15)
                fig.add_subplot(ax2)

            elif indice in self.categ_index_list:

                dummy_gamma = subnet.categ_bias.numpy()
                norm = subnet.moving_norm.numpy()

                inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[idx], wspace=0.1, hspace=0.1, height_ratios=[4, 1])
                ax1 = plt.Subplot(fig, inner[0])
                ax1.bar(np.arange(len(self.meta_info[feature_name]['values'])), np.sign(beta[indice]) * dummy_gamma[:, 0] / norm)
                ax1.set_ylabel("Score", fontsize=12)
                ax1.get_yaxis().set_label_coords(-0.15, 0.5)
                ax1.set_title(feature_name, fontsize=12)
                fig.add_subplot(ax1)

                ax2 = plt.Subplot(fig, inner[1])
                unique, counts = np.unique(self.tr_x[:, indice], return_counts=True)
                ybar_ticks = np.zeros((len(self.meta_info[feature_name]['values'])))
                ybar_ticks[unique.astype(int)] = counts
                ax2.bar(np.arange(len(self.meta_info[feature_name]['values'])), ybar_ticks)
                ax1.get_shared_x_axes().join(ax1, ax2)
                ax1.set_xticklabels([])
                
                xtick_loc = (np.arange(len(self.meta_info[feature_name]['values'])) if len(self.meta_info[feature_name]['values']) <= 12 else 
                         np.arange(0, len(self.meta_info[feature_name]['values']) - 1, 
                               int(len(self.meta_info[feature_name]['values']) / 6)).astype(int))
                xtick_label = [self.meta_info[feature_name]["values"][i] for i in xtick_loc]
                ax2.set_xticks(xtick_loc)
                ax2.set_xticklabels(xtick_label)
                ax2.set_ylabel("Histogram", fontsize=12)
                ax2.get_yaxis().set_label_coords(-0.15, 0.5)
                if np.sum([len(ax1.get_yticklabels()[i].get_text()) for i in range(len(ax1.get_yticklabels()))]) > 20:
                    ax1.yaxis.set_tick_params(rotation=15)
                if np.sum([len(ax2.get_xticklabels()[i].get_text()) for i in range(len(ax2.get_xticklabels()))]) > 20:
                    ax2.xaxis.set_tick_params(rotation=15)
                if np.sum([len(ax2.get_yticklabels()[i].get_text()) for i in range(len(ax2.get_yticklabels()))]) > 20:
                    ax2.yaxis.set_tick_params(rotation=15)
                fig.add_subplot(ax2)

            idx = idx + 1
            ax1.set_title(feature_name + " (" + str(np.round(100 * componment_scales[indice], 1)) + "%)", fontsize=12)

        for indice in active_interaction_index:
            
            response = []
            inter_net = self.interact_blocks.interacts[indice]
            feature_name1 = self.variables_names[self.interaction_list[indice][0]]
            feature_name2 = self.variables_names[self.interaction_list[indice][1]]

            axis_extent = []
            interact_input_list = []
            if feature_name1 in self.categ_variable_list:
                interact_label1 = self.meta_info[feature_name1]['values']
                interact_input1 = np.array(np.arange(inter_net.length1), dtype=np.float32)
                interact_input_list.append(interact_input1)
                axis_extent.extend([-0.5, inter_net.length1 - 0.5])
            else:
                sx1 = self.meta_info[feature_name1]['scaler']
                interact_input_list.append(np.array(np.linspace(0, 1, grid_length), dtype=np.float32))
                interact_label1 = sx1.inverse_transform(np.array([0, 1], dtype=np.float32).reshape([-1, 1])).ravel()
                axis_extent.extend([interact_label1.min(), interact_label1.max()])
            if feature_name2 in self.categ_variable_list:
                interact_label2 = self.meta_info[feature_name2]['values']
                interact_input2 = np.array(np.arange(inter_net.length2), dtype=np.float32)
                interact_input_list.append(interact_input2)
                axis_extent.extend([-0.5, inter_net.length2 - 0.5])
            else:
                sx2 = self.meta_info[feature_name2]['scaler']
                interact_input_list.append(np.array(np.linspace(0, 1, grid_length), dtype=np.float32))
                interact_label2 = sx2.inverse_transform(np.array([0, 1], dtype=np.float32).reshape([-1, 1])).ravel()
                axis_extent.extend([interact_label2.min(), interact_label2.max()])

            x1, x2 = np.meshgrid(interact_input_list[0], interact_input_list[1][::-1])
            input_grid = np.hstack([np.reshape(x1, [-1, 1]), np.reshape(x2, [-1, 1])])
            response = np.sign(gamma[indice]) * inter_net.apply(input_grid, training=False).numpy().reshape([x1.shape[0], x1.shape[1]])

            ax = plt.Subplot(fig, outer[idx]) 
            cf = ax.imshow(response, interpolation='nearest', aspect='auto', extent=axis_extent)

            if feature_name1 in self.categ_variable_list:
                xtick_loc = (np.arange(inter_net.length1) if inter_net.length1 <= 12 else
                            np.arange(0, inter_net.length1 - 1, int(inter_net.length1 / 6)).astype(int))
                xtick_label = [interact_label1[i] for i in xtick_loc]
                ax.set_xticks(xtick_loc)
                ax.set_xticklabels(xtick_label)
            elif np.sum([len(str(interact_label1[i])) for i in range(len(interact_label1))]) > 20:
                ax.xaxis.set_tick_params(rotation=15)
            if feature_name2 in self.categ_variable_list:
                ytick_loc = (np.arange(inter_net.length2) if inter_net.length2 <= 12 else
                            np.arange(0, inter_net.length2 - 1, int(inter_net.length2 / 6)).astype(int))
                ytick_label = [interact_label2[i] for i in ytick_loc]
                ax.set_yticks(ytick_loc)
                ax.set_yticklabels(ytick_label)
            elif np.sum([len(str(interact_label2[i])) for i in range(len(interact_label2))]) > 20:
                ax.yaxis.set_tick_params(rotation=15)

            response_precision = max(int(- np.log10(np.max(response) - np.min(response))) + 2, 0)
            fig.colorbar(cf, ax=ax, format='%0.' + str(response_precision) + 'f')
            ax.set_title(feature_name1 + " vs. " + feature_name2 + " (" + 
                          str(np.round(100 * componment_scales[beta.shape[0] + indice], 1)) + "%)", fontsize=12)
            fig.add_subplot(ax)
            idx = idx + 1

        if max_ids > 0:
            if save_eps:
                fig.savefig("%s.png" % save_path, bbox_inches='tight', dpi=100)
            if save_png:
                fig.savefig("%s.eps" % save_path, bbox_inches='tight', dpi=100)
