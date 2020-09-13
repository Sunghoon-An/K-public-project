from __future__ import print_function, division
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from tensorflow.keras.metrics import Recall
from tensorflow.keras.metrics import Precision

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm


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

    def select_supervised_sampled(self, df):
        """
        Extract sampled data from real data to balace class
        """
        x_train, y_train = df.iloc[:,:-1], df.iloc[:,-1]
        x_train_list = []
        y_train_list = []
        n_per_class = int(self.n_samples / self.num_classes)
        for i in range(0, self.num_classes):
            x_with_class = x_train[y_train == i]
            ix = list(np.random.choice(x_with_class.index, size=n_per_class))
            [x_train_list.append(x_with_class.loc[j]) for j in ix]
            [y_train_list.append(i) for _ in ix]
        
        return np.asarray(x_train_list), np.asarray(y_train_list)

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
        generator = Dense(100, activation="relu")(in_lat)
        generator = BatchNormalization(momentum=0.8)(generator)
        generator = Dense(56, activation="relu")(generator)
        generator = BatchNormalization(momentum=0.8)(generator)
        generator = Dense(32, activation="relu")(generator)
        generator = BatchNormalization(momentum=0.8)(generator)
        out_layer = Dense(281, activation="tanh")(generator)
        generator = Model(in_lat, out_layer)
        
        return generator

    def create_discriminator(self):
        in_data = Input(shape=self.input_shape)
        discriminator = Dense(281, activation="relu", input_dim=self.input_shape)(in_data)
        discriminator = BatchNormalization(momentum=0.8)(discriminator)
        discriminator = Dense(56, activation="relu")(discriminator)
        discriminator = BatchNormalization(momentum=0.8)(discriminator)
        discriminator = Dense(32, activation="relu")(discriminator)
        discriminator = BatchNormalization(momentum=0.8)(discriminator)
        discriminator = Dropout(0.4)(discriminator)
        discriminator = Dense(28, activation="relu")(discriminator)
        discriminator = BatchNormalization(momentum=0.8)(discriminator)
        c_out_layer = Dense(self.num_classes, activation="softmax")(discriminator)
        c_model = Model(in_data, c_out_layer)
        c_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=["accuracy"])
        d_out_layer = Dense(1, activation="sigmoid")(discriminator)
        d_model = Model(in_data, d_out_layer)
        d_model.compile(loss=['binary_crossentropy'], optimizer=Adam(0.0002, 0.5))
        
        return c_model, d_model

    def create_gan(self):
        """
        Combine g_model and d_model
        """
        self.d_model.trainable = False
        gan_output = self.d_model(self.g_model.output)
        gan_model = Model(self.g_model.input, gan_output)
        gan_model.compile(loss=['binary_crossentropy'], optimizer = Adam(0.0002, 0.5))
        
        return gan_model

    def train(self, df, batch_size, epochs):
        # split df
        x_train, y_train = df.iloc[:,:-1].values, df.iloc[:,-1].values

        # Select sampled real dataset (label O, balanced class)
        x_super, y_super = self.select_supervised_sampled(df)

        # Calculate the number of batches per training epoch
        bat_per_epo = int(x_train.shape[0] / batch_size)

        # Calulate the number of trainging iterations
        n_steps = bat_per_epo * epochs
        half_batch = batch_size // 2

        print('epochs=%d, batch_size=%d, 1/2=%d, b/e=%d, steps=%d' % (epochs, batch_size, half_batch, bat_per_epo, n_steps))
        for step in range(n_steps):
            # update supervised discriminator (c)
            [Xsup_real, labels], _ = self.generate_real_samples([x_super, y_super], half_batch)
            c_loss, c_acc = self.c_model.train_on_batch(Xsup_real, labels)

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
            print('>%d, c[loss: %.3f, acc: %.3f],  d[loss1: %.3f, loss2: %.3f],  g_loss[%.3f]'
                  % (step+1, c_loss, c_acc*100, d_loss1, d_loss2, g_loss))

        return  step+1, c_loss, c_acc*100, d_loss1, d_loss2, g_loss