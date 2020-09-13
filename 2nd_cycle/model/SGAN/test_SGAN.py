from SGAN import SGAN
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD, Adam, Adagrad, Adadelta
import time
import argparse
import tensorflow.keras.backend as K

        # self.random_noise = 100
        # self.n_samples =100
        # self.num_classes = 16
        # self.input_shape = 281
        # self.c_model, self.d_model = self.create_discriminator()
        # self.g_model = self.create_generator()
        # self.gan_model = self.create_gan()
def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opts', type = str, default = 'adam'
                        , choices = ['adam', 'rmsprop', 'agagrad', 'sgd'])
    parser.add_argument('--lossfunc1', type = str, default = 'sparse_categorical_crossentropy'
                        , choices = ['binary_crossentropy', 'hinge', 'sparse_categorical_crossentropy'])
    parser.add_argument('--lossfunc2', type = str, default = 'binary_crossentropy'
                        , choices = ['binary_crossentropy', 'hinge', 'sparse_categorical_crossentropy'])
    parser.add_argument('--learning_rate', type = float, default = 0.0001
        , help = 'Training learning rate')
    parser.add_argument('--epoch', type = int, default = 100
        , help = 'Training interation')
 

    args = parser.parse_args()
    return args


def data_load(data_dir):
    df = pd.read_csv(data_dir)
    x_train = df.iloc[:,:-1]
    y_train = df.iloc[:,-1]
    return [x_train, y_train]


def recall(y_test, y_pred):
    y_target_yn = K.round(K.clip(y_test, 0, 1))
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))

    count_true_positive = K.sum(y_target_yn * y_pred_yn)
    count_true_positive_false_negative = K.sum(y_target_yn)

    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon()) 

    return recall 


def precision(y_test, y_pred):
    y_target_yn = K.round(K.clip(y_test, 0, 1))
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))

    count_true_positive = K.sum(y_target_yn * y_pred_yn)
    count_true_positive_false_negative = K.sum(y_pred_yn)

    precision = count_true_positive / (count_true_positive_false_negative + K.epsilon()) 

    return precision


def main():
    arg = args()

    df = data_load("SGAN_test_dataset.csv")
    model = SGAN(100, 100, 16, 281)

    # loss function 1  choose
    if arg.lossfunc1 == 'binary_crossentropy':
        loss_func1 = 'binary_crossentropy'
    elif arg.lossfunc1 == 'hinge':
        loss_func1 = 'hinge'
    elif arg.lossfunc1 == 'sparse_categorical_crossentropy':
        loss_func1 = 'sparse_categorical_crossentropy'
    else:
        raise ValueError('loss function is not support yet.')

    

    # loss function 2  choose

    if arg.lossfunc2 == 'binary_crossentropy':
        loss_func2 = 'binary_crossentropy'
    elif arg.lossfunc2 == 'hinge':
        loss_func2 = 'hinge'
    elif arg.lossfunc2 == 'sparse_categorical_crossentropy':
        loss_func2 = 'sparse_categorical_crossentropy'
    else:
        raise ValueError('loss function is not support yet.')

    # optimizer choose

    if arg.opts == 'adam':
        opts = Adam(learning_rate = arg.learning_rate, decay = 1e-6)
    elif arg.opts == 'rmsprop':
        opts = RMSprop(learning_rate = arg.learning_rate, epsilon = 1.0)
    elif arg.opts == 'adagrad':
        opts = Adam(learning_rate = arg.learning_rate, decay = 1e-6, adagrad = True)
    elif arg.opts == 'sgd':
        opts = SGD(learning_rate = arg.learning_rate, momentum = 0.9, decay = 1e-6, nestrov = True)
    else:
        raise ValueError('loss function is not support yet.')

    model.model_compile(loss_func1, loss_func2, opts, ["accuracy", recall, precision])
    model.train(df, 80, 100)

if __name__ == "__main__":
    main()