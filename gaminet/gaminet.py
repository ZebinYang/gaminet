import os
import numpy as np
import pandas as pd 
import tensorflow as tf
from scipy import stats
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from .layers import *
from .utils import get_interaction_list
from .utils import global_visualize_density
from .utils import global_visualize_wo_density


class GAMINet(tf.keras.Model):

    def __init__(self, meta_info,
                 subnet_arch=[10, 6],
                 interact_num=10,
                 interact_arch=[100, 60],
                 task_type='Regression',
                 activation_func=tf.tanh,
                 main_grid_size=101,
                 interact_grid_size=21,
                 lr_bp=0.001,
                 batch_size=1000,
                 init_training_epochs=10000,
                 interact_training_epochs=1000,
                 tuning_epochs=500,
                 verbose=False,
                 val_ratio=0.2,
                 early_stop_thres=100,
                 random_state=0):

        super(GAMINet, self).__init__()
        # Parameter initiation
        self.meta_info = meta_info
        self.input_num = len(meta_info) - 1
        
        self.task_type = task_type
        self.subnet_arch = subnet_arch
        self.main_grid_size = main_grid_size
        self.interact_grid_size = interact_grid_size
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
            if feature_info['type'] == 'target':
                continue
            if feature_info['type'] == 'categorical':
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
                                 grid_size=self.main_grid_size)
        self.interact_blocks = InteractionBlock(interact_num=self.interact_num,
                                meta_info=self.meta_info,
                                interact_arch=self.interact_arch,
                                activation_func=self.activation_func,
                                grid_size=self.interact_grid_size)
        self.output_layer = OutputLayer(input_num=self.input_num,
                                interact_num=self.interact_num)

        self.fit_interaction = False
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_bp)
        if self.task_type == 'Regression':
            self.loss_fn = tf.keras.losses.MeanSquaredError()
        elif self.task_type == 'Classification':
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

        if self.task_type == 'Regression':
            output = self.output_layer(tf.concat(concat_list, 1), training=training)
        elif self.task_type == 'Classification':
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
        return sorted_index
    
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
        return sorted_index

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
        if self.task_type == 'Regression':
            tr_x, val_x, tr_y, val_y, tr_idx, val_idx = train_test_split(train_x, train_y, indices, test_size=self.val_ratio, 
                                          random_state=self.random_state)
        elif self.task_type == 'Classification':
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
                density, bins = np.histogram(sx.inverse_transform(train_x[:,[indice]]), bins=10, density=True)
                self.data_dict[feature_name].update({'density':{'names':bins,'scores':density}})
            elif indice in self.categ_index_list:
                unique, counts = np.unique(train_x[:, indice], return_counts=True)
                density = np.zeros((len(self.meta_info[feature_name]['values'])))
                density[unique.astype(int)] = counts / n_samples
                self.data_dict[feature_name].update({'density':{'names':unique,'scores':density}})

        #### 1. Main Effects Training
        if self.verbose:
            print('Main Effects Training.')

        last_improvement = 0
        best_validation = np.inf
        train_size = tr_x.shape[0]
        
        for i in range(self.input_num):
            if i in self.categ_index_list:
                length = len(self.meta_info[self.variables_names[i]]['values'])
                input_grid = np.arange(len(self.meta_info[self.variables_names[i]]['values']))
            else:
                length = self.main_grid_size
                input_grid = np.linspace(0, 1, length)
            pdf_grid = np.ones([length]) / length    
            # H, bin_edges = np.histogram(train_x[:, i], bins=length)
            # pdf_grid = H / np.sum(H)
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
                print('Main effects training epoch: %d, train loss: %0.5f, val loss: %0.5f' %
                      (epoch + 1, self.err_train[-1], self.err_val[-1]))

            if self.err_val[-1] < best_validation:
                best_validation = self.err_val[-1]
                last_improvement = epoch
            if epoch - last_improvement > self.early_stop_thres:
                if self.verbose:
                    print('Early stop at epoch %d, with validation loss: %0.5f' % (epoch + 1, self.err_val[-1]))
                break

        main_effect_switcher = np.zeros((self.input_num, 1))
        sortted_index = self.get_active_main_effects()
        val_loss = []
        for idx, _ in enumerate(sortted_index):
            self.output_layer.main_effect_switcher.assign(tf.constant(np.ones((self.input_num, 1)), dtype=tf.float32))
            main_effect_switcher = np.zeros((self.input_num, 1))
            main_effect_switcher[sortted_index[:(idx + 1)]] = 1
            self.output_layer.main_effect_switcher.assign(tf.constant(main_effect_switcher, dtype=tf.float32))
            val_loss.append(self.evaluate(val_x, val_y, training=False))

        best_main_effect_num = np.argmin(val_loss)
        main_effect_switcher[sortted_index[:(best_main_effect_num + 1)] = 1
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
                print('Main effects tunning epoch: %d, train loss: %0.5f, val loss: %0.5f' %
                      (epoch + 1, self.err_train[-1], self.err_val[-1]))

        # 2. interaction training
        if self.interact_num>0:
            if self.verbose:
                print('Interaction Training.')

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
                    length1 = self.interact_grid_size
                    length1_grid = np.linspace(0, 1, length1)
                if feature_name2 in self.categ_variable_list:
                    length2 = len(self.meta_info[feature_name2]['values']) 
                    length2_grid = np.arange(length2)
                else:
                    length2 = self.interact_grid_size
                    length2_grid = np.linspace(0, 1, length2)

                x1, x2 = np.meshgrid(length1_grid, length2_grid)
                input_grid = np.hstack([np.reshape(x1, [-1, 1]), np.reshape(x2, [-1, 1])])
                pdf_grid = np.ones([length1, length2]) / (length1 * length2)
                # H, xedges, yedges = np.histogram2d(train_x[:,idx1], train_x[:,idx2], bins=[length1, length2])
                # pdf_grid = H / np.sum(H)
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
                    print('Interaction training epoch: %d, train loss: %0.5f, val loss: %0.5f' %
                          (epoch + 1, self.err_train[-1], self.err_val[-1]))

                if self.err_val[-1] < best_validation:
                    best_validation = self.err_val[-1]
                    last_improvement = epoch
                if epoch - last_improvement > self.early_stop_thres:
                    if self.verbose:
                        print('Early stop at epoch %d, with validation loss: %0.5f' % (epoch + 1, self.err_val[-1]))
                    break

            interaction_switcher = np.zeros((self.interact_num, 1))
            sortted_index = self.get_active_interactions()
            val_loss = []
            for idx, _ in enumerate(sortted_index):
                self.output_layer.interaction_switcher.assign(tf.constant(np.ones((self.interact_num, 1)), dtype=tf.float32))
                interaction_switcher = np.zeros((self.interact_num, 1))
                interaction_switcher[sortted_index[:(idx + 1)]] = 1
                self.output_layer.interaction_switcher.assign(tf.constant(interaction_switcher, dtype=tf.float32))
                val_loss.append(self.evaluate(val_x, val_y, training=False))

            best_interact_num = np.argmin(val_loss)
            interaction_switcher[sortted_index[:(best_interact_num + 1)] = 1
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
                    print('Interaction tunning epoch: %d, train loss: %0.5f, val loss: %0.5f' %
                          (epoch + 1, self.err_train[-1], self.err_val[-1]))
    
    def local_explain(self, x, y=None, save_dict=False, folder='./', name='local_explain'):
        
        predicted = self.predict(x)
        intercept = self.output_layer.output_bias.numpy()

        subnet_output = self.maineffect_blocks.apply(tf.cast(tf.constant(x), tf.float32)).numpy()
        if self.interact_num > 0:
            interact_output = self.interact_blocks.apply(tf.cast(tf.constant(x), tf.float32)).numpy()
        else:
            interact_output = np.array([])
        active_univariate_index, active_interaction_index, beta, gamma, componment_scales = self.get_active_effects()
        scores = np.hstack([intercept[0], (np.hstack([subnet_output.ravel(), interact_output.ravel()]) 
                                           * np.hstack([beta.ravel(), gamma.ravel()]).ravel()).ravel()])
        active_indice = 1 + np.hstack([-1, active_univariate_index, self.numerical_input_num + self.categ_variable_num + active_interaction_index])
        effect_names = np.hstack(['Intercept', np.array(self.variables_names)[self.numerical_index_list],
                   np.array(self.variables_names)[self.categ_index_list],
                   [self.variables_names[self.interaction_list[i][0]] + ' x ' 
                    + self.variables_names[self.interaction_list[i][1]] for i in range(self.interact_num)]])
        
        data_dict = {'active_indice': active_indice, 
                 'scores': scores,
                 'effect_names': effect_names,
                 'predicted': predicted, 
                 'actual': y}
        
        if save_dict:
            if not os.path.exists(folder):
                os.makedirs(folder)
            save_path = folder + name
            np.save('%s.npy' % save_path, data_dict)

        return data_dict

    
    def global_explain(self, main_grid_size=None, interact_grid_size=None, save_dict=False, folder='./', name='global_explain'):

        ## By default, we use the same main_grid_size and interact_grid_size as that of the zero mean constraint
        ## Alternatively, we can also specify it manually, e.g., when we want to have the same grid size as EBM (256).        
        if main_grid_size is None:
            main_grid_size = self.main_grid_size
        if interact_grid_size is None:
            interact_grid_size = self.interact_grid_size      

        data_dict = self.data_dict
        _, _, _, _, componment_scales = self.get_active_effects()
        for indice in range(self.input_num):
            feature_name = list(self.variables_names)[indice]
            subnet = self.maineffect_blocks.subnets[indice]
            if indice in self.numerical_index_list:
                sx = self.meta_info[feature_name]['scaler']
                subnets_inputs = np.linspace(0, 1, main_grid_size).reshape([-1, 1])
                subnets_inputs_original = sx.inverse_transform(subnets_inputs)
                subnets_outputs = (self.output_layer.main_effect_weights.numpy()[indice]
                            * self.output_layer.main_effect_switcher.numpy()[indice]
                            * subnet.apply(tf.cast(tf.constant(subnets_inputs), tf.float32)).numpy())
                data_dict[feature_name].update({'type':'continuous',
                                         'importance':componment_scales[indice],
                                         'inputs':subnets_inputs_original.ravel(),
                                         'outputs':subnets_outputs.ravel()})
                
            elif indice in self.categ_index_list:
                subnets_inputs = np.arange(len(self.meta_info[feature_name]['values'])).reshape([-1, 1])
                subnets_inputs_original = self.meta_info[feature_name]['values']
                subnets_outputs = (self.output_layer.main_effect_weights.numpy()[indice]
                            * self.output_layer.main_effect_switcher.numpy()[indice]
                            * subnet.apply(tf.cast(subnets_inputs, tf.float32)).numpy())
                data_dict[feature_name].update({'type':'categorical',
                                         'importance':componment_scales[indice],
                                         'inputs':subnets_inputs_original,
                                         'outputs':subnets_outputs.ravel()})

        for indice in range(self.interact_num_heredity):
            
            response = []
            inter_net = self.interact_blocks.interacts[indice]
            feature_name1 = self.variables_names[self.interaction_list[indice][0]]
            feature_name2 = self.variables_names[self.interaction_list[indice][1]]
            feature_type1 = 'categorical' if feature_name1 in self.categ_variable_list else 'continuous'
            feature_type2 = 'categorical' if feature_name2 in self.categ_variable_list else 'continuous'
            
            axis_extent = []
            interact_input_list = []
            if feature_name1 in self.categ_variable_list:
                interact_input1 = np.arange(len(self.meta_info[feature_name1]['values']), dtype=np.float32)
                interact_input1_original = self.meta_info[feature_name1]['values']
                interact_input1_ticks = (interact_input1.astype(int) if len(interact_input1) <= 12 else 
                             np.arange(0, len(interact_input1) - 1, int(len(interact_input1) / 6)).astype(int))
                interact_input1_labels = [self.meta_info[feature_name1]['values'][i] for i in interact_input1_ticks]
                if len(''.join(list(map(str, interact_input1_labels)))) > 30:
                    interact_input1_labels = [self.meta_info[feature_name1]['values'][i][:4] for i in interact_input1_ticks]
                interact_input_list.append(interact_input1)
                axis_extent.extend([-0.5, len(interact_input1_original) - 0.5])
            else:
                sx1 = self.meta_info[feature_name1]['scaler']
                interact_input1 = np.array(np.linspace(0, 1, interact_grid_size), dtype=np.float32)
                interact_input1_original = sx1.inverse_transform(interact_input1.reshape([-1, 1])).ravel()
                interact_input1_ticks = []
                interact_input1_labels = []
                interact_input_list.append(interact_input1)
                axis_extent.extend([interact_input1_original.min(), interact_input1_original.max()])
            if feature_name2 in self.categ_variable_list:
                interact_input2 = np.arange(len(self.meta_info[feature_name2]['values']), dtype=np.float32)
                interact_input2_original = self.meta_info[feature_name2]['values']
                interact_input2_ticks = (interact_input2.astype(int) if len(interact_input2) <= 12 else 
                             np.arange(0, len(interact_input2) - 1, int(len(interact_input2) / 6)).astype(int))
                interact_input2_labels = [self.meta_info[feature_name2]['values'][i] for i in interact_input2_ticks]
                if len(''.join(list(map(str, interact_input2_labels)))) > 30:
                    interact_input2_labels = [self.meta_info[feature_name2]['values'][i][:4] for i in interact_input2_ticks]
                interact_input_list.append(interact_input2)
                axis_extent.extend([-0.5, len(interact_input2_original) - 0.5])
            else:
                sx2 = self.meta_info[feature_name2]['scaler']
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
                        * inter_net.apply(input_grid, training=False).numpy().reshape(x1.shape))
            data_dict.update({feature_name1 + ' vs. ' + feature_name2:{'type':'pairwise',
                                                       'xtype':feature_type1,
                                                       'ytype':feature_type2,
                                                       'importance':componment_scales[self.input_num + indice],
                                                       'input1':interact_input1_original,
                                                       'input2':interact_input2_original,
                                                       'outputs':interact_outputs,
                                                       'input1_ticks': interact_input1_ticks,
                                                       'input2_ticks': interact_input2_ticks,
                                                       'input1_labels': interact_input1_labels,
                                                       'input2_labels': interact_input2_labels,
                                                       'axis_extent':axis_extent}})

        if save_dict:
            if not os.path.exists(folder):
                os.makedirs(folder)
            save_path = folder + name
            np.save('%s.npy' % save_path, data_dict)
            
        return data_dict
