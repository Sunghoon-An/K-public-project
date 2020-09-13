import tensorflow as tf
######################### keras import #########################
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise, BatchNormalization, Activation, Embedding, ZeroPadding2D, LeakyReLU, UpSampling2D, Conv2D, add, Lambda, LSTM, TimeDistributed, GaussianDropout, Bidirectional, Concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Nadam, Adagrad, Adadelta
from tensorflow.keras import losses
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler, LambdaCallback
from tensorflow.keras.initializers import glorot_uniform

######################### tensorflow addons import #########################
import tensorflow_addons as tfa
from tensorflow_addons.activations import *
from tensorflow_addons.optimizers import RectifiedAdam, LazyAdam, ConditionalGradient, Lookahead
from tensorflow_addons.layers import GroupNormalization, InstanceNormalization, WeightNormalization
from tensorflow_addons.losses import ContrastiveLoss, NpairsMultilabelLoss, SparsemaxLoss, TripletSemiHardLoss
from tensorflow_addons.metrics import MultiLabelConfusionMatrix

######################### scikit learn #########################
from sklearn.metrics import classification_report, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.utils import class_weight

from keras.utils.generic_utils import get_custom_objects
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import *
from imblearn.under_sampling import *
from collections import Counter

from config import *
from utils import *

import warnings

import networkx as nx


import argparse

tf.random.set_seed(RANDOM_STATE)
warnings.filterwarnings("ignore")
def swish(x):
        return (K.sigmoid(x) * x)
get_custom_objects().update({'swish': Activation(swish)})
##########################################
# Model Argument
##########################################

def args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--savepath', type = str, default = None
                        , help = 'save history training')
    
    parser.add_argument('--fold_num', type = int, default = 0
                        , choices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                        , help = 'K fold type. max is 9, and min is 0')
    
    parser.add_argument('--data_path', type = str, default = None
                        , help = 'train data')
    
    parser.add_argument('--val_path', type = str, default = None
                        , help = 'validation data')
    
    parser.add_argument('--model', type = str, default = 'dnn_1'
                        , choices = ['dnn_1', 'bilstm', 'rwn', 'semi_gan']
                        , help = 'Choose model what you training')
    
    parser.add_argument('--sample_type', type = str, default = 'both'
                        , choices = ['over', 'under', 'both', 'None']
                        , required = False, help = 'Sampling type - over and under, both')
    parser.add_argument('--over_method', type = str, default = 'ros'
                        , required = False, choices = ['ros', 'adasyn'
                                                       , 'border_smote', 'svmsmote', 'cate_smote']
                        , help = 'oversampling method - 5 type')
    parser.add_argument('--under_method', type = str, default = 'rus'
                        , required = False, choices = ['rus', 'tomek', 'centroid', 'one_sided']
                        , help = 'oversampling method - 4 type')
    parser.add_argument('--scaler', type = str, default = 'minmax'
                        , choices = ['robust', 'standard', 'minmax']
                        , help = 'scaler before training')
    
    
    parser.add_argument('--nb_class', type = int, default = 16
                        , choices = range(1,16), help = 'Binary or Multiclass')
    parser.add_argument('--opts', type = str, default = 'adam'
                        , choices = ['adam', 'rmsprop', 'adagrad', 'sgd', 'amsgrad'
                                     , 'adagrad', 'nadam', 'adadelta'
                                     , 'radam', 'lazyadam', 'conditionalgradient', 'lookahead']
                        , help = 'Optimizer Choose')
    parser.add_argument('--lossfunc', type = str, default = 'sparse_categorical_crossentropy'
                        , choices = ['binary_crossentropy', 'hinge', 'squared_hinge'
                                     , 'categorical_crossentropy', 'sparse_categorical_crossentropy'
                                     , 'kulback_leibler_divergence', 'focal_loss']
                        , help = 'Loss function Choose')
    parser.add_argument('--batch_size', type = int, default = 64
                        , help = 'Training Batch Size')
    parser.add_argument('--learning_rate', type = float, default = 0.0001
                        , help = 'Training learning rate')
    parser.add_argument('--callback_type', type = str, default = ['checkpoint', 'earlystopping']
                        , nargs = '+', choices = ['checkpoint', 'earlystopping', 'tensorboard'
                                                  , 'rateschedule', 'interval_check']
                        , help = 'Custom Callback Choose')
    parser.add_argument('--restore', action = 'store_true'
                        , help = 'if restore, use it')
    parser.add_argument('--epoch', type = int, default = 100
                        , help = 'Training interation')
    parser.add_argument('--jobs', type = int, default = 4
                        , help = 'sampling using cpu')
    parser.add_argument('--random', action = 'store_true'
                        , help = 'if random, use it')
    
    args = parser.parse_args()
    return args


# def swish(x):
#     return (K.sigmoid(x) * x)

# get_custom_objects().update({'swish': Activation(swish)})

##########################################
# RWN
##########################################
class RWN(object):
    def __init__(self, input_shape, num_stages, hidden_size, num_class, graph_model, graph_param):
        """
        Randomly Wired Neural Nets
        Args
            input_shape : input shape
            num_stages : number of stages
            hidden_size : the size of node'output in stage
            graph_model : random graph model,  choose one from ['ws', 'ba', 'er']
            graph_param : parameters when create random graph. In paper, best parameter is [32, 4, 0.75]
            num_class : the number of class to classify
        Return
            RWN model
        """
        self.input_shape = input_shape
        self.num_stages = num_stages
        self.hidden_size = hidden_size
        self.num_class = num_class
        self.graph_model = graph_model
        self.graph_param = graph_param
        

    def build_model(self):
        x = Input(shape = (self.input_shape,))
        outputs = self.regime(x, self.num_stages, self.hidden_size, self.num_class, self.graph_model, self.graph_param)
        model = Model(inputs = [x], outputs = [outputs])
        model.build(input_shape=[self.input_shape])

        return model

    def dense_block(self, x, hidden_size):
        """
        operation in Node
        Args
            x : input data
            hidden_size : Node's output size
        Return
            Node operation
        """
        x = ReLU()(x)
        x = Dense(hidden_size, kernel_initializer="he_uniform")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)

        return x

    def build_stage(self, x, hidden_size, graph_data):
        """
        Create Neural Net using random graph model. this is a stage
        Args
        x : input data
            hidden_size : Node's output size
            graph_data : Returned values after create random graph
        Return
            A neural net, stage
        """

        graph, graph_order, start_node, end_node = graph_data

        interms = {}
        for node in graph_order:
            if node in start_node:
                interm = self.dense_block(x, hidden_size)
                nterms[node] = interm
            else:
                in_node = list(nx.ancestors(graph, node))
                if len(in_node) > 1:
                    weight = tf.Variable(
                        initial_value = tf.keras.initializers.GlorotNormal()(shape=[len(in_node)])
                        , name = 'sum_weight'
                        , dtype = tf.float32
                        , constraint = lambda x : tf.clip_by_value(x, 0, np.infty))
                    weight = tf.nn.sigmoid(weight)
                    interm = 0
                    for idx in range(len(in_node)):
                        interm += weight[idx] * interms[in_node[idx]]
                    interm = self.dense_block(interm, hidden_size)
                    interms[node] = interm
                elif len(in_node) == 1:
                    interm = self.dense_block(interms[in_node[0]], hidden_size)
                    interms[node] = interm
        output = 0
        for idx in range(len(end_node)):
            output += interms[end_node[idx]]

        return output

    def regime(self, x, num_stages, hidden_size, num_class, graph_model, graph_param):
        """
        Create total Network. 
        
        """
        #### stage 1 ######
        x = Dense(256, kernel_initializer = "he_uniform", activation= "swish")(x)
        # x = ReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        #### stage 2 ######
        x = self.dense_block(x, hidden_size)
        
        #### stage 3 ~ 4 ####
        for stage in range(3, num_stages+1):
            graph_data = self.graph_generator(graph_model
                , graph_param, "graph_model", 'dense'+str(stage)+'_'+graph_model)
            x = self.build_stage(x, hidden_size, graph_data)

        #### stage 5 ######
        x = self.dense_block(x, hidden_size)
        x = Dense(64, kernel_initializer = "he_uniform", activation= "swish")(x)
        x = BatchNormalization()(x)
        # x = ReLU()(x)
        x = Dense(num_class, activation="softmax", kernel_initializer="glorot_uniform")(x)

        return x

    def graph_generator(self, graph_model, graph_param, save_path, file_name):
        """
        Create random graph

        """

        graph_param[0] = int(graph_param[0])
        if graph_model == 'ws':
            graph_param[1] = int(graph_param[1])
            graph = nx.random_graphs.connected_watts_strogatz_graph(*graph_param)
        elif graph_model == 'er':
            graph = nx.random_graphs.erdos_renyi_graph(*graph_param)
        elif graph_model == 'ba':
            graph_param[1] = int(graph_param[1])
            graph = nx.random_graphs.barabasi_albert_graph(*graph_param)

        if os.path.isfile(save_path + '/' + file_name + '.yaml') is True:
            print('graph loaded')
            dgraph = nx.read_yaml(save_path + '/' + file_name + '.yaml')

        else:
            dgraph = nx.DiGraph()
            dgraph.add_nodes_from(graph.nodes)
            dgraph.add_edges_from(graph.edges)

        dgraph = nx.DiGraph()
        dgraph.add_nodes_from(graph.nodes)
        dgraph.add_edges_from(graph.edges)

        in_node = []
        out_node = []
        for indeg, outdeg in zip(dgraph.in_degree, dgraph.out_degree):
            if indeg[1] == 0:
                in_node.append(indeg[0])
            elif outdeg[1] == 0:
                out_node.append(outdeg[0])
        _sorted = list(nx.topological_sort(dgraph))

        if os.path.isdir(save_path) is False:
            os.makedirs(save_path)

        if os.path.isfile(save_path + '/' + file_name + '.yaml') is False:
            print('graph_saved')
            nx.write_yaml(dgraph, save_path + '/' + file_name + '.yaml')

        return dgraph, _sorted, in_node, out_node
    
##########################################
# Semi-Gan
##########################################
class SGAN(object):

    def __init__(self, random_noise, n_samples, num_classes, input_shape):
        """
        Semi Supervised Generative Adversarial Networks
        Args
            random_noise : generator input
            n_samples : sampled real data
            num_classes : the number of labels to classify
            input shape : input data size
        """

        self.random_noise = random_noise
        self.n_samples = n_samples
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.c_model, self.d_model = self.create_discriminator()
        self.g_model = self.create_generator()
        self.gan_model = self.create_gan()


    def select_supervised_sampled(self, x_train, y_train):  
        """
        Extract sampled data from real data to balace class
        """

        x_train_list = []
        y_train_list = []
        n_per_class = int(self.n_samples / self.num_classes)
        for i in range(0, self.num_classes):
            x_with_class = x_train[y_train == i]
            ix = list(np.random.choice(x_with_class.index, size=n_per_class))
            [x_train_list.append(x_with_class.loc[j]) for j in ix]
            [y_train_list.append(i) for _ in ix]
        
        return x_train, y_train, np.asarray(x_train_list), np.asarray(y_train_list)

    def generate_real_samples(self, df, n_samples):
        """
        Choice as number of n_samples in sampled and balanced data
        """
        data, labels = df
        ix = np.random.randint(0, data.shape[0], n_samples)
        X, labels = data[ix], labels[ix]
        y = np.ones((n_samples, 1))
        
        return [X, labels], y     # y = Fake label

    def generate_fake_samples(self, n_samples):
        """
        Generate sampled data by generator
        """
        z_input = np.random.randn(self.random_noise * n_samples)
        z_input = z_input.reshape(n_samples, self.random_noise)
        gen_data = self.g_model.predict(z_input)
        y = np.zeros((n_samples, 1))
        
        return gen_data, y

    def create_generator(self):
        in_lat = Input(shape=self.random_noise,)
        generator = Dense(100, activation="selu")(in_lat)
        generator = BatchNormalization(momentum=0.8)(generator)
        generator = Dense(50, activation="selu")(generator)
        generator = BatchNormalization(momentum=0.8)(generator)
        generator = Dense(25, activation="selu")(generator)
        generator = BatchNormalization(momentum=0.8)(generator)
        out_layer = Dense(281, activation="tanh")(generator)
        generator = Model(in_lat, out_layer)
        
        return generator

    def create_discriminator(self):
        in_data = Input(shape=self.input_shape)
        discriminator = Dense(281, activation="selu", input_dim=self.input_shape)(in_data)
        discriminator = BatchNormalization(momentum=0.8)(discriminator)
        discriminator = Dense(64, activation="selu")(discriminator)
        discriminator = BatchNormalization(momentum=0.8)(discriminator)
        discriminator = Dense(32, activation="selu")(discriminator)
        discriminator = BatchNormalization(momentum=0.8)(discriminator)
        discriminator = Dense(28, activation="selu")(discriminator)
        discriminator = BatchNormalization(momentum=0.8)(discriminator)
        c_out_layer = Dense(self.num_classes, activation="softmax")(discriminator)
        c_model = Model(in_data, c_out_layer)
        # c_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=["accuracy"])
        d_out_layer = Dense(1, activation="sigmoid")(discriminator)
        d_model = Model(in_data, d_out_layer)
        # d_model.compile(loss=['binary_crossentropy'], optimizer=Adam(0.0002, 0.5))
        
        return c_model, d_model

    def create_gan(self):
        """
        Combine g_model and d_model
        """
        
        gan_output = self.d_model(self.g_model.output)
        gan_model = Model(self.g_model.input, gan_output)
        # gan_model.compile(loss=['binary_crossentropy'], optimizer = Adam(0.0002, 0.5))
        
        return gan_model

    def model_compile(self, loss_func1, loss_func2, opts, metrics):
        
        self.c_model.compile(loss = loss_func1, optimizer = opts, metrics = metrics)
        # generater 다음 mse 사용할지 검토
        self.d_model.compile(loss = loss_func2, optimizer = opts)
        self.d_model.trainable = False
        self.gan_model.compile(loss = loss_func2, optimizer = opts)

    def train(self, x_train, y_train, batch_size, epochs):

        # Select sampled real dataset (label O, balanced class)
        x_train, y_train, x_super, y_super = self.select_supervised_sampled(x_train, y_train)
        x_train, y_train = x_train.values, y_train.values
        # Calculate the number of batches per training epoch
        bat_per_epo = int(x_train.shape[0] / batch_size)

        # Calulate the number of trainging iterations
        n_steps = bat_per_epo * epochs
        half_batch = batch_size // 2

        tensorboard = TensorBoard(log_dir = "./logs", histogram_freq = 3, 
            write_graph = True, write_images = True)
        tensorboard.set_model(self.gan_model)
        train_log_dir = "./logs"
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        metrics = {}

        print('epochs=%d, batch_size=%d, 1/2=%d, b/e=%d, steps=%d' % (epochs, batch_size, half_batch, bat_per_epo, n_steps))
        for step in range(n_steps):
            # update supervised discriminator (c)
            [Xsup_real, labels], _ = self.generate_real_samples([x_super, y_super], half_batch)
            metrics["c_loss"], metrics["c_acc"], metrics["c_recall"], metrics["c_precision"] = self.c_model.train_on_batch(Xsup_real, labels)

            with train_summary_writer.as_default():
                tf.summary.scalar("c_loss", metrics["c_loss"], step = step)
                tf.summary.scalar("c_acc", metrics["c_acc"], step = step)
                tf.summary.scalar("c_recall", metrics["c_recall"], step = step)
                tf.summary.scalar("c_precision", metrics["c_precision"], step = step)

            # update unsupervised discriminator (d)
            [X_real, _], y_real = self.generate_real_samples([x_train, y_train], half_batch)
            d_loss1 = self.d_model.train_on_batch(X_real, y_real)
            X_fake, y_fake = self.generate_fake_samples(half_batch)
            d_loss2 = self.d_model.train_on_batch(X_fake, y_fake)
            
            # update generator (g)
            z_input = np.random.randn(self.random_noise * half_batch)
            z_input = z_input.reshape(half_batch, self.random_noise)
            X_gan = z_input
            y_gan = np.ones((X_gan.shape[0], 1))
            g_loss = self.gan_model.train_on_batch(X_gan, y_gan)

            if step % 10 == 0:

                print('>%d, c[loss: %.3f, acc: %.3f, recall : %.3f, precision : %.3f],  d[loss1: %.3f, loss2: %.3f],  g_loss[%.3f]'
                    % (step+1, metrics["c_loss"], metrics["c_acc"]*100, metrics["c_recall"]*100, metrics["c_precision"]*100, d_loss1, d_loss2, g_loss))
            
            
            
            
##########################################
# Trainer
##########################################


#################################
## Preprocess
#################################



class ModelTrain(object):
    def __init__(self, data = (), nb_class = 16, opts = 'adam'
                 , lossfunc = 'sparse_categorical_crossentropy'
                 , model = 'dnn_1'
                 , batch_size = 64, val_data = (), learning_rate = 0.001
                 , savepath = None , callback_type = None
                 , epoch = 100, restore = False
                 , logger = None
                ):
       
        self.model = model
        self.x_train, self.y_train = data
        self.nb_class = nb_class
        self.opts = opts
        self.lossfunc = lossfunc
        self.batch_size = batch_size
        self.x_val, self.y_val = val_data
        self.learning_rate = learning_rate
        self.savepath = savepath
        self.callback_type = callback_type
        self.epoch = epoch
        self.restore = restore

        input_shape = self.x_train.shape[1]
        nb_class = self.nb_class

        # model choose
        if self.model == 'dnn_1':
            model = dnn_1(input_shape, nb_class)
        elif self.model == 'bilstm':
            model = bilstm(input_shape, nb_class)
        elif self.model == 'advanced_dnn':
            model = sample_dnn(input_shape, nb_class)
        elif self.model == 'rwn':
            model = RWN(self, input_shape)
        elif self.model == 'semi_gan':
            model = SemiGan(self, input_shape)
        else:
            pass

        # loss funtion choose
        if self.lossfunc == 'focal_loss':
            lossfunc = focal_loss(alpha = 1)
        elif self.lossfunc == 'binary_crossentropy':
            lossfunc = 'binary_crossentropy'
        elif self.lossfunc == 'hinge':
            lossfunc = 'hinge'
        elif self.lossfunc == 'squared_hinge':
            lossfunc = 'squared_hinge'
        elif self.lossfunc == 'categorical_crossentropy':
            lossfunc = 'categorical_crossentropy'
        elif self.lossfunc == 'sparse_categorical_crossentropy':
            lossfunc = 'sparse_categorical_crossentropy'
        elif self.lossfunc == 'kulback_leibler_divergence':
            lossfunc = 'kulback_leibler_divergence'
        else:
            pass

        # optimizer choose
        if self.opts == 'adam':
            opts = Adam(lr = self.learning_rate, decay = 1e-6)
        elif self.opts == 'rmsprop':
            opts = RMSprop(lr = self.learning_rate, epsilon = 1.0)
        elif self.opts == 'adagrad':
            opts = Adam(lr = self.learning_rate, decay = 1e-6, adagrad = True)
        elif self.opts == 'sgd':
            opts = SGD(lr = self.learning_rate, momentum = 0.9, decay = 1e-6, nestrov = True)
        elif self.opts == 'amsgrad':
            opts = Adam(lr = self.learning_rate, decay = 1e-6, amsgrad = True)
        elif self.opts == 'adagrad':
            opts = Adagrad(lr = self.learning_rate, decay = 1e-6)
        elif self.opts == 'nadam':
            opts = Nadam(lr = self.learning_rate, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, schedule_decay = 0.004)
        elif self.opts == 'adadelta':
            opts = Adadelta(lr = 1.0, rho = 0.95, epsilon = None, decay = 0.0)
        elif self.opts == 'radam':
            opts = RectifiedAdam(lr = 1e-3, total_steps = self.epoch, warmup_proportion = 0.1, min_lr = 1e-6)
        elif self.opts == 'lookahead':
            radam = RectifiedAdam()
            opts = Lookahead(radam, sync_period = 6, slow_step_size = 0.5)
        elif self.opts == 'lazyadam':
            opts = LazyAdam(lr = self.learning_rate)
        elif self.opts == 'conditionalgradient':
            opts = ConditionalGradient(lr = 0.99949, lambda_ = 203)
        else:
            pass
        print("loss func is {}".format(lossfunc))
        auc = tf.keras.metrics.AUC()
        recall = tf.keras.metrics.Recall()
        precision = tf.keras.metrics.Precision()

        model.compile(optimizer = opts, loss = lossfunc, metrics = ['acc', auc, recall, precision])
        
#         x_train = self.x_train
#         x_val = self.x_val
#         y_train = self.y_train
#         y_val = self.y_val
        x_train = np.asarray(self.x_train)
        x_val = np.asarray(self.x_val)
        y_train = np.asarray(self.y_train)
        y_val = np.asarray(self.y_val)
        
        MODEL_SAVE_FOLDER_PATH = os.path.join(self.savepath, self.model)
        if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
            os.mkdir(MODEL_SAVE_FOLDER_PATH)

        def lrdropping(self):
            initial_lrate = self.learning_rate
            drop = 0.9
            epochs_drop = 3.0
            lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
            return lrate
        
        callbacks = []
        if 'checkpoint' in self.callback_type:
            checkpoint = ModelCheckpoint(os.path.join(MODEL_SAVE_FOLDER_PATH, 'checkpoint-{epoch:02d}.h5')
            , monitor = 'val_auc', save_best_only = False, mode = 'max')
            callbacks.append(checkpoint)
        else:
            pass
        if 'elarystopping' in self.callback_type:
            earlystopping = EarlyStopping(monitor = 'val_auc', patience = 5, verbose = 1, mode = 'max')
            callbacks.append(earlystopping)
        else:
            pass
        if 'tensorboard' in self.callback_type:
            logger.info("tensorboard path : {}".format(MODEL_SAVE_FOLDER_PATH))
            tensorboard = TensorBoard(log_dir = os.path.join("tensorboard", MODEL_SAVE_FOLDER_PATH), histogram_freq = 0
                , write_graph = True, write_images = True)
            tensorboard.set_model(model)
            train_summary_writer = tf.summary.create_file_writer(os.path.join("tensorboard", MODEL_SAVE_FOLDER_PATH))
            callbacks.append(tensorboard)
        else:
            pass
        if 'rateschedule' in self.callback_type:
            lrd = LearningRateScheduler(lrdropping)
            callbacks.append(lrd)
        else:
            pass
        if 'interval_check' in self.callback_type:
            inter = IntervalEvaluation(validation_data = (x_val, y_val), interval = 1, savedir = os.path.join(self.savepath, self.model), file = 'Evaluation_{}.csv'.format(self.model), logger = logger)
            callbacks.append(inter)
        else:
            pass
        if self.opts == 'conditionalgradient':
            def frobenius_norm(m):
                total_reduce_sum = 0
                for i in range(len(m)):
                    total_reduce_sum = total_reduce_sum + tf.math.reduce_sum(m[i]**2)
                norm = total_reduce_sum ** 0.5
                return norm
            CG_frobenius_norm_of_weight = []
            CG_get_weight_norm = LambdaCallback(
                on_epoch_end = lambda batch, logs : 
                CG_frobenius_norm_of_weight.append(frobenius_norm(model.trainable_weights).np()))
            cgnorm = CG_frobenius_norm_of_weight
            callbacks.append(cgnorm)
        else:
            pass

        model_json = model.to_json()
        
        with open(os.path.join(MODEL_SAVE_FOLDER_PATH, 'model.json'), 'w')as json_file:
            json_file.write(model_json)

        if self.restore:
            chkp = last_checkpoint(MODEL_SAVE_FOLDER_PATH)
            model.load_weight(chkp)
            init_epoch = int(os.path.basename(chkp).split('-')[1])
            print("================== restore checkpoint ==================")
        else:
            init_epoch = 0
            print("================== restore failed ==================")

        cw = class_weight.compute_class_weight(class_weight = 'balanced'
                                               , classes = np.unique(y_train)
                                               , y = y_train)
        logger.info("callback is {}".format(callbacks))

        hist = model.fit(x = x_train, y = y_train
            , epochs = self.epoch
            , verbose = 1
            , validation_data = (x_val, y_val)
            , shuffle = True
            , callbacks = callbacks
            , class_weight = cw)
        
def dnn_1(input_dim, nb_class):

    model = Sequential()
    model.add(Dense(128, input_dim = input_dim, kernel_initializer = 'he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('selu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, kernel_initializer = 'he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('selu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, kernel_initializer = 'he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('selu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, kernel_initializer = 'he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('selu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation = 'sigmoid', kernel_initializer = 'glorot_uniform'))
    return model

def dnn_2(input_dim, nb_class):
    inputs = Input(shape = input_dim)
    x = Dense(256, kernel_initializer = 'he_uniform')(x)
    x = InstanceNormalization()(x)
    x = Activation(swish)(x)
    x = GaussianDropout(0.3)(x)
    x = Dense(128, kernel_initializer = 'he_uniform')(x)
    x = InstanceNormalization()(x)
    x = Activation(swish)(x)
    x = GaussianDropout(0.3)(x)
    x = Dense(64, kernel_initializer = 'he_uniform')(x)
    x = InstanceNormalization()(x)
    x = Activation(swish)(x)
    x = GaussianDropout(0.3)(x)
    x = Dense(32, kernel_initializer = 'he_uniform')(x)
    x = InstanceNormalization()(x)
    x = Activation(swish)(x)
    x = GaussianDropout(0.3)(x)

    output = Dense(1, activation = 'sigmoid', kernel_initializer = 'glorot_uniform')(model)
    model = Model(inputs = inputs, outputs = outputs)
    
    return model



def sample_dnn(input_dim, nb_class):
    def residual(x, nodes):
        shortcut = Dense(nodes, kernel_initializer = 'he_uniform', kernel_constraint = max_norm(3))(x)
        shortcut = BatchNormalization()(shortcut)
        shortcut = Activation(swish)(shortcut)

        x = BatchNormalization()(x)
        x = Dense(nodes, kernel_initializer = 'he_uniform', activation = swish)(x)
        x = Dropout(0.3)(x)

        x = BatchNormalization()(x)
        x = Dense(nodes, kernel_initializer = 'he_uniform', activation = swish)(x)
        x = Dropout(0.3)(x)

        x = BatchNormalization()(x)
        x = Dense(nodes, kernel_initializer = 'he_uniform', activation = swish)(x)
        x = Dropout(0.3)(x)

        added = keras.layers.add([x, shortcut])

        return added

    inputs = Input(shape = (input_dim,))
    x = residual(inputs, 64)
    x = residual(x, 64)
    x = residual(x, 128)
    x = residual(x, 128)
    x = residual(x, 256)
    x = residual(x, 256)

    outputs = Dense(nb_class, activation = 'sigmoid', kernel_initializer = 'glorot_uniform')(x)
    model = Model(inputs = inputs, outputs = outputs)

    return model

def bilstm(input_dim, nb_class):
    sequence_input = Input(shape = (input_dim[0], input_dim[1]))
    lstm, forward_h, forward_c, backward_h, backward_c = Bidirectional(LSTM(
        128
        , dropout=0.3
        , return_sequences = True
        , return_state = True
        , recurrent_activation = 'selu'
        , recurrent_initializer='he_uniform'))(sequence_input)
    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    attention = BahdanauAttention(128)
    context_vector, attention_weights = attention(lstm, state_h)
    hidden = BatchNormalization()(context_vector)
    lstm_1, forward_h_1, forward_c_1, backward_h_1, backward_c_1 = Bidirectional(LSTM(
        64
        , dropout=0.3
        , return_sequences = True
        , return_state = True
        , recurrent_activation = 'selu'
        , recurrent_initializer='he_uniform'))(hidden)
    state_h_1 = Concatenate()([forward_h, backward_h])
    state_c_1 = Concatenate()([forward_c, backward_c])
    attention_1 = BahdanauAttention(64)
    context_vector_1, attention_weights_1 = attention_1(lstm_1, state_h_1)
    hidden_1 = BatchNormalization()(context_vector_1)
    lstm_2, forward_h_2, forward_c_2, backward_h_2, backward_c_2 = Bidirectional(LSTM(
        32
        , dropout=0.3
        , return_sequences = True
        , return_state = True
        , recurrent_activation = 'selu'
        , recurrent_initializer='he_uniform'))(hidden_1)
    state_h_2 = Concatenate()([forward_h, backward_h])
    state_c_2 = Concatenate()([forward_c, backward_c])
    attention_2 = BahdanauAttention(32)
    context_vector_2, attention_weights_2 = attention_2(lstm_2, state_h_2)
    hidden_2 = BatchNormalization()(context_vector_2)
    output = TimeDistributed(Dense(nb_class, activation = 'sigmoid', kernel_initializer = 'glorot_uniform'))(hidden_2)
    model = Model(inputs = sequence_input, outputs = output)

    return model

## LSTM
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, values, query):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
    
