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
        self.interact_num_heredity = self.interact_num

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
        self.data_dict = {}
        for indice, (feature_name, feature_info) in enumerate(self.meta_info.items()):
            if feature_info['type'] == "target":
                continue
            if feature_info['type'] == "categorical":
                self.categ_variable_num += 1
                self.categ_index_list.append(indice)
                self.categ_variable_list.append(feature_name)
            else:
                self.numerical_input_num +=1
                self.numerical_index_list.append(indice)
                self.numerical_variable_list.append(feature_name)
            self.variables_names.append(feature_name)
            self.data_dict.update({feature_name:{}})
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
        train_weights.append(self.output_layer.main_effect_weights)
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

        main_effect_norm = [self.maineffect_blocks.subnets[i].moving_norm.numpy()[0] for i in range(self.input_num)]
        beta = (self.output_layer.main_effect_weights.numpy() * np.array([main_effect_norm]).reshape([-1, 1])
             * self.output_layer.main_effect_switcher.numpy())

        componment_coefs = beta
        componment_scales = (np.abs(componment_coefs) / np.sum(np.abs(componment_coefs))).reshape([-1])
        sorted_index = np.argsort(componment_scales)
        active_univariate_index = sorted_index[componment_scales[sorted_index].cumsum()>self.main_threshold][::-1]
        return active_univariate_index
    
    def get_active_interactions(self):

        main_effect_norm = [self.maineffect_blocks.subnets[i].moving_norm.numpy()[0] for i in range(self.input_num)]
        beta = (self.output_layer.main_effect_weights.numpy() * np.array([main_effect_norm]).reshape([-1, 1]) 
             * self.output_layer.main_effect_switcher.numpy())

        interaction_norm = [self.interact_blocks.interacts[i].moving_norm.numpy()[0] for i in range(self.interact_num_heredity)]
        gamma = (self.output_layer.interaction_weights.numpy()[:self.interact_num_heredity] 
              * np.array([interaction_norm]).reshape([-1, 1])
              * self.output_layer.interaction_switcher.numpy()[:self.interact_num_heredity])

        componment_coefs = np.vstack([beta, gamma])
        componment_scales = (np.abs(componment_coefs) / np.sum(np.abs(componment_coefs))).reshape([-1])
        componment_scales_main = componment_scales[:self.input_num]
        componment_scales_interact = componment_scales[self.input_num:]

        sorted_index = np.argsort(componment_scales_interact)
        active_interaction_index = sorted_index[(componment_scales_interact[sorted_index].cumsum())>self.total_threshold][::-1]
        return active_interaction_index

    def get_active_effects(self):

        main_effect_norm = [self.maineffect_blocks.subnets[i].moving_norm.numpy()[0] for i in range(self.input_num)]
        beta = (self.output_layer.main_effect_weights.numpy() * np.array([main_effect_norm]).reshape([-1, 1]) 
             * self.output_layer.main_effect_switcher.numpy())

        interaction_norm = [self.interact_blocks.interacts[i].moving_norm.numpy()[0] for i in range(self.interact_num_heredity)]
        gamma = (self.output_layer.interaction_weights.numpy()[:self.interact_num_heredity] 
              * np.array([interaction_norm]).reshape([-1, 1])
              * self.output_layer.interaction_switcher.numpy()[:self.interact_num_heredity])

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
        
        ### data splits ###
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

        ### density ###
        self.density = []
        for indice in range(self.input_num):
            feature_name = list(self.variables_names)[indice]
            if indice in self.numerical_index_list:
                sx = self.meta_info[feature_name]['scaler']
                hist, bin_edges = np.histogram(sx.inverse_transform(train_x[:,[indice]]), bins=10, density=True)
                self.data_dict[feature_name].update({'density':{"names":bin_edges,"scores":hist}})
            elif indice in self.categ_index_list:
                unique, counts = np.unique(train_x[:, indice], return_counts=True)
                density = np.zeros((len(self.meta_info[feature_name]['values'])))
                density[unique.astype(int)] = counts / n_samples
                self.data_dict[feature_name].update({'density':{"names":unique,"scores":ybar_ticks}})

        #### 1. Main Effects Training
        if self.verbose:
            print("Main Effects Training.")

        last_improvement = 0
        best_validation = np.inf
        train_size = tr_x.shape[0]
        
        for i in range(self.input_num):
            if i in self.categ_index_list:
                length = len(self.meta_info[self.variables_names[i]]['values'])
                input_grid = np.arange(len(self.meta_info[self.variables_names[i]]['values']))
            else:
                length = self.grid_size
                input_grid = np.linspace(0, 1, length)
            pdf_grid = np.ones([length]) / length            
            self.maineffect_blocks.subnets[i].set_pdf(np.array(input_grid, dtype=np.float32).reshape([-1, 1]),
                                        np.array(pdf_grid, dtype=np.float32).reshape([1, -1]))

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

            self.err_train.append(self.evaluate(tr_x, tr_y, training=True))
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

        main_effect_switcher = np.zeros((self.input_num, 1))
        active_univariate_index = self.get_active_main_effects()
        main_effect_switcher[active_univariate_index] = 1
        self.output_layer.main_effect_switcher.assign(tf.constant(main_effect_switcher, dtype=tf.float32))
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

            self.err_train.append(self.evaluate(tr_x, tr_y, training=True))
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
            
            self.interact_num_heredity = len(self.interaction_list)
            interaction_switcher = np.zeros((self.interact_num, 1))
            interaction_switcher[:self.interact_num_heredity] = 1
            self.output_layer.interaction_switcher.assign(tf.constant(interaction_switcher, dtype=tf.float32))
            self.interact_blocks.set_interaction_list(self.interaction_list)

            for interact_id, (idx1, idx2) in enumerate(self.interaction_list):

                feature_name1 = self.variables_names[idx1]
                feature_name2 = self.variables_names[idx2]
                if feature_name1 in self.categ_variable_list:
                    length1 = len(self.meta_info[feature_name1]['values']) 
                    length1_grid = np.arange(length1)
                else:
                    length1 = self.grid_size
                    length1_grid = np.linspace(0, 1, self.grid_size)
                if feature_name2 in self.categ_variable_list:
                    length2 = len(self.meta_info[feature_name2]['values']) 
                    length2_grid = np.arange(length2)
                else:
                    length2 = self.grid_size
                    length2_grid = np.linspace(0, 1, self.grid_size)

                x1, x2 = np.meshgrid(length1_grid, length2_grid)
                input_grid = np.hstack([np.reshape(x1, [-1, 1]), np.reshape(x2, [-1, 1])])
                pdf_grid = np.ones([length1, length2]) / (length1 * length2)
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

                self.err_train.append(self.evaluate(tr_x, tr_y, training=True))
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

            interaction_switcher = np.zeros((self.interact_num, 1))
            active_interaction_index = self.get_active_interactions()
            interaction_switcher[active_interaction_index] = 1
            self.output_layer.interaction_switcher.assign(tf.constant(interaction_switcher, dtype=tf.float32))

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

                self.err_train.append(self.evaluate(tr_x, tr_y, training=True))
                self.err_val.append(self.evaluate(val_x, val_y, training=False))
                if self.verbose & (epoch % 1 == 0):
                    print("Interaction tunning epoch: %d, train loss: %0.5f, val loss: %0.5f" %
                          (epoch + 1, self.err_train[-1], self.err_val[-1]))

    def local_visualize(self, x, y=None, folder="./results", name="demo", save_png=False, save_eps=False):
        
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


    def global_visualize(self, folder="./results", name="demo", cols_per_row=4, save_png=False, save_eps=False, save_dict=False):
        
        self.global_explain(save_dict)
        if not os.path.exists(folder):
            os.makedirs(folder)
        save_path = folder + name

        active_univariate_index, active_interaction_index, beta, gamma, componment_scales = self.get_active_effects()
        max_ids = len(active_univariate_index) + len(active_interaction_index)
        
        idx = 0
        fig = plt.figure(figsize=(6 * cols_per_row, 
                         4.6 * int(np.ceil(max_ids / cols_per_row))))
        outer = gridspec.GridSpec(int(np.ceil(max_ids/cols_per_row)), cols_per_row, wspace=0.25, hspace=0.25)
        for indice in active_univariate_index:

            feature_name = list(self.variables_names)[indice]
            subnet = self.maineffect_blocks.subnets[indice]
            if indice in self.numerical_index_list:

                inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[idx], wspace=0.1, hspace=0.1, height_ratios=[4, 1])
                ax1 = plt.Subplot(fig, inner[0]) 
                ax1.plot(self.data_dict[feature_name]["inputs"], self.data_dict[feature_name]["outputs"])
                ax1.set_ylabel("Score", fontsize=12)
                ax1.get_yaxis().set_label_coords(-0.15, 0.5)
                ax1.set_title(feature_name, fontsize=12)
                fig.add_subplot(ax1)

                ax2 = plt.Subplot(fig, inner[1]) 
                xint = ((np.array(self.data_dict[feature_name]['density']['names'][1:]) 
                                + np.array(self.data_dict[feature_name]['density']['names'][:-1]))/2).reshape([-1, 1]).reshape([-1])
                ax2.bar(xint, self.data_dict[feature_name]['density']['scores'], width=xint[1]-xint[0])
                ax1.get_shared_x_axes().join(ax1, ax2)
                ax1.set_xticklabels([])
                ax2.set_ylabel('Density', fontsize=12)
                ax2.get_yaxis().set_label_coords(-0.15, 0.5)
                if np.sum([len(ax2.get_xticklabels()[i].get_text()) for i in range(len(ax2.get_xticklabels()))]) > 20:
                    ax2.xaxis.set_tick_params(rotation=20)
                fig.add_subplot(ax2)

            elif indice in self.categ_index_list:

                inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[idx],
                                            wspace=0.1, hspace=0.1, height_ratios=[4, 1])
                ax1 = plt.Subplot(fig, inner[0])
                ax1.bar(np.arrange(self.data_dict[feature_name]["inputs"]),
                             self.data_dict[feature_name]["outputs"])
                ax1.set_ylabel("Score", fontsize=12)
                ax1.get_yaxis().set_label_coords(-0.15, 0.5)
                ax1.set_title(feature_name, fontsize=12)
                fig.add_subplot(ax1)

                ax2 = plt.Subplot(fig, inner[1])
                ax2.bar(np.arrange(len(self.data_dict[feature_name]["density"]["names"])),
                        self.data_dict[feature_name]["density"]["scores"])
                ax1.get_shared_x_axes().join(ax1, ax2)
                ax1.set_xticklabels([])
                
                xtick_loc = (subnets_inputs if len(subnets_inputs) <= 12 else np.arange(0, len(subnets_inputs) - 1,
                                                                 int(subnets_inputs / 6)).astype(int))
                xtick_label = [self.meta_info[feature_name]["values"][i] for i in xtick_loc]
                if len("".join(list(map(str, xtick_label)))) > 30:
                    xtick_label = [self.meta_info[feature_name]["values"][i][:4] for i in xtick_loc]

                ax2.set_xticks(xtick_loc)
                ax2.set_xticklabels(xtick_label)
                ax2.set_ylabel("Density", fontsize=12)
                ax2.get_yaxis().set_label_coords(-0.15, 0.5)
                if np.sum([len(ax2.get_xticklabels()[i].get_text()) for i in range(len(ax2.get_xticklabels()))]) > 20:
                    ax2.xaxis.set_tick_params(rotation=20)
                fig.add_subplot(ax2)

            idx = idx + 1
            ax1.set_title(feature_name + " (" + str(np.round(100 * self.data_dict[feature_name]["importance"], 1)) + "%)", fontsize=12)

        for indice in active_interaction_index:
            
            feature_name1 = self.variables_names[self.interaction_list[indice][0]]
            feature_name2 = self.variables_names[self.interaction_list[indice][1]]
            name = feature_name1 + " vs. " + feature_name2

            ax = plt.Subplot(fig, outer[idx]) 
            cf = ax.imshow(self.data_dict[name]["outputs"], interpolation='nearest', aspect='auto',
                      extent=self.data_dict[name]["axis_extent"])

            if feature_name1 in self.categ_variable_list:
                ax.set_xticks(self.data_dict[name]["interact_input1"])
                ax.set_xticklabels(self.data_dict[name]["interact_label1"])
                if np.sum([len(ax.get_xticklabels()[i].get_text()) for i in range(len(ax.get_xticklabels()))]) > 20:
                    ax.xaxis.set_tick_params(rotation=20)
            elif (max(self.data_dict[name]["input1"]) - min(self.data_dict[name]["input1"])) > 10000:
                    ax.xaxis.set_tick_params(rotation=20)
            if feature_name2 in self.categ_variable_list:
                ax.set_xticks(self.data_dict[name]["interact_input2"])
                ax.set_xticklabels(self.data_dict[name]["interact_label2"])
                if np.sum([len(ax.get_yticklabels()[i].get_text()) for i in range(len(ax.get_yticklabels()))]) > 20:
                    ax.yaxis.set_tick_params(rotation=20)
            elif (max(self.data_dict[name]["input2"]) - min(self.data_dict[name]["input2"])) > 10000:
                    ax.yaxis.set_tick_params(rotation=20)

            response_precision = max(int(- np.log10(np.max(self.data_dict[name]["outputs"]) 
                                            - np.min(self.data_dict[name]["outputs"]))) + 2, 0)
            fig.colorbar(cf, ax=ax, format='%0.' + str(response_precision) + 'f')
            ax.set_title(name + " (" + str(np.round(100 * self.data_dict[feature_name]["importance"], 1)) + "%)", fontsize=12)
            fig.add_subplot(ax)
            idx = idx + 1

        if max_ids > 0:
            if save_eps:
                fig.savefig("%s.png" % save_path, bbox_inches='tight', dpi=100)
            if save_png:
                fig.savefig("%s.eps" % save_path, bbox_inches='tight', dpi=100)

    
    def global_explain(self, grid_length=None, save_dict=False):

        ## By default, we use the same grid_length as that of the zero mean constraint
        ## Alternatively, we can also specify it manually, e.g., when we want to have the same grid size as EBM (256).
        if grid_length is None:
            grid_length = self.grid_size
        else:
            grid_length = 256
        
        _, _, _, _, componment_scales = self.get_active_effects()
        for indice in range(self.input_num):
            feature_name = list(self.variables_names)[indice]
            subnet = self.maineffect_blocks.subnets[indice]
            if indice in self.numerical_index_list:
                sx = self.meta_info[feature_name]['scaler']
                subnets_inputs = np.linspace(0, 1, grid_length).reshape([-1, 1])
                subnets_inputs_original = sx.inverse_transform(subnets_inputs)
                subnets_outputs = (self.output_layer.main_effect_weights.numpy()[indice]
                            * self.output_layer.main_effect_switcher.numpy()[indice]
                            * subnet.apply(tf.cast(tf.constant(subnets_inputs), tf.float32)).numpy())
                self.data_dict[feature_name].update({"inputs":subnets_inputs_original.ravel(), "outputs":subnets_outputs.ravel(),
                              "importance":componment_scales[indice]})
                
            elif indice in self.categ_index_list:
                subnets_inputs = np.arange(len(self.meta_info[feature_name]['values']))
                subnets_inputs_original = self.meta_info[feature_name]['values']
                subnets_outputs = (self.output_layer.main_effect_weights.numpy()[indice]
                            * self.output_layer.main_effect_switcher.numpy()[indice]
                            * subnet.apply(tf.cast(tf.constant(subnets_inputs), tf.float32)).numpy())
                self.data_dict[feature_name].upadte({"inputs":subnets_inputs_original.ravel(), "outputs":subnets_outputs.ravel(),
                                      "importance":componment_scales[indice]})

        for indice in range(self.interact_num_heredity):
            
            response = []
            inter_net = self.interact_blocks.interacts[indice]
            feature_name1 = self.variables_names[self.interaction_list[indice][0]]
            feature_name2 = self.variables_names[self.interaction_list[indice][1]]

            axis_extent = []
            interact_input_list = []
            if feature_name1 in self.categ_variable_list:
                interact_input1 = np.arange(len(self.meta_info[feature_name1]["values"]), dtype=np.float32)
                interact_input1_original = self.meta_info[feature_name1]["values"]
                interact_input1_ticks = (interact_input1 if len(interact_input1) <= 12 else 
                             np.arange(0, len(interact_input1) - 1, int(len(interact_input1) / 6)).astype(int))
                interact_input1_labels = [self.meta_info[feature_name1]["values"][i] for i in interact_input1_ticks]
                if len("".join(list(map(str, interact_input1_labels)))) > 30:
                    interact_input1_labels = [self.meta_info[feature_name1]["values"][i][:4] for i in interact_input1_ticks]
                interact_input_list.append(interact_input1)
                axis_extent.extend([-0.5, len(self.meta_info[feature_name1]["values"]) - 0.5])
            else:
                sx1 = self.meta_info[feature_name1]['scaler']
                interact_input1 = np.array(np.linspace(0, 1, grid_length), dtype=np.float32).reshape([-1, 1])
                interact_input1_original = sx1.inverse_transform(interact_input1)
                interact_input1_lables = []
                interact_input_list.append(interact_input1)
                axis_extent.extend([interact_input1_original.min(), interact_input1_original.max()])
            if feature_name2 in self.categ_variable_list:
                interact_input2 = np.arange(len(self.meta_info[feature_name2]["values"]), dtype=np.float32)
                interact_input2_original = self.meta_info[feature_name2]["values"]
                interact_input2_ticks = (interact_input2 if len(interact_input2) <= 12 else 
                             np.arange(0, len(interact_input2) - 1, int(len(interact_input2) / 6)).astype(int))
                interact_input2_lables = [self.meta_info[feature_name2]["values"][i] for i in interact_input2_ticks]
                if len("".join(list(map(str, interact_input1_labels)))) > 30:
                    interact_input1_labels = [self.meta_info[feature_name2]["values"][i][:4] for i in interact_input2_ticks]
                interact_input_list.append(interact_input2)
                axis_extent.extend([-0.5, len(self.meta_info[feature_name2]["values"]) - 0.5])
            else:
                sx2 = self.meta_info[feature_name1]['scaler']
                interact_input2 = np.array(np.linspace(0, 1, grid_length), dtype=np.float32).reshape([-1, 1])
                interact_input2_original = sx2.inverse_transform(interact_input2)
                interact_input2_lables = []
                interact_input_list.append(interact_input2)
                axis_extent.extend([interact_input2_original.min(), interact_input2_original.max()])

            x1, x2 = np.meshgrid(interact_input_list[0], interact_input_list[1][::-1])
            input_grid = np.hstack([np.reshape(x1, [-1, 1]), np.reshape(x2, [-1, 1])])
            
            interact_outputs = (self.output_layer.interaction_weights.numpy()[indice]
                        * self.output_layer.interaction_switcher.numpy()[indice]
                        * inter_net.apply(input_grid, training=False).numpy().reshape(x1.shape))
            self.data_dict.update({feature_name1 + " vs. " + feature_name2:{"input1":interact_input1_original.ravel(),
                                                        "input2":interact_input2_original.ravel(),
                                                        "outputs":interact_outputs,
                                                        "axis_extent":axis_extent,
                                                        "interact_input1":interact_input1.ravel(),
                                                        "interact_input2":interact_input2.ravel(),
                                                        "interact_label1":interact_input1_lables,
                                                        "interact_label2":interact_input2_lables,
                                                        "importance":componment_scales[self.input_num + indice]}})
        if save_dict:
            np.save("%s.npy" % save_path, self.data_dict)
