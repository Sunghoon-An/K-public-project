from __future__ import print_function, division

from tf.keras.datasets import mnist
from tf.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from tf.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from tf.keras.layers import LeakyReLU
from tf.keras.models import Sequential, Model
from tf.keras.optimizers import Adam
from tf.keras import losses
from tf.keras.utils import to_categorical
import tf.keras.backend as K

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

def data_load_keit(data_dir,test_dir):
    df_train=pd.read_csv(data_dir)
    df_test=pd.read_csv(test_dir)
    x_train,y_train=df_train.iloc[:,:-1],df_train.iloc[:,-1]
    x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.2,random_state=42)
    x_test,y_test=df_test.iloc[:,:-1],df_test.iloc[:,-1]

    return [x_train,y_train,x_val,y_val,x_test,y_test]

def make_args():
    args={}
    args["input shape"]=281
    args["num class"]=2


class SGAN:
    def __init__(self,args):
        self.input_shape = 28
        self.num_classes = 2
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss=['binary_crossentropy', 'categorical_crossentropy'],
            loss_weights=[0.5, 0.5],
            optimizer=optimizer,
            metrics=['accuracy']
        )

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        noise = Input(shape=(64,))
        img = self.generator(noise)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid, _ = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model(noise, valid)
        self.combined.compile(loss=['binary_crossentropy'], optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(78, activation="relu", input_dim=self.latent_dim))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(56, activation="relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(32, activation="relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(28, activation="tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(78, activation="relu", input_dim=self.input_shape))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(56, activation="relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(32, activation="relu"))
        model.add(Dropout(rate=0.3))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(28, activation="relu"))
        model.add(Dropuout(rate=0.3))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(10, activation="relu"))

        model.summary()

        img = Input(shape=self.img_shape)

        features = model(img)
        valid = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes+1, activation="softmax")(features)

        return Model(img, [valid, label])

    def train(self, epochs, batch_size=100, sample_interval=50):

        # Load the dataset
        data=data_load("credit_fraud_sampled.csv")
        x_train,y_train=data[:2]
        x_val,y_val=data[2:4]
        x_test,y_test=data[4:]

        # Class weights:
        # To balance the difference in occurences of digit class labels.
        # 50% of labels that the discriminator trains on are 'fake'.
        # Weight = 1 / frequency
        half_batch = batch_size // 2
        cw1 = {0: 1, 1: 1}
        cw2 = {i: self.num_classes / half_batch for i in range(self.num_classes)}
        cw2[self.num_classes] = 1 / half_batch

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # One-hot encoding of labels
            labels = to_categorical(y_train[idx], num_classes=self.num_classes+1)
            fake_labels = to_categorical(np.full((batch_size, 1), self.num_classes), num_classes=self.num_classes+1)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, labels], class_weight=[cw1, cw2])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, fake_labels], class_weight=[cw1, cw2])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid, class_weight=[cw1, cw2])

            # Plot the progress
            if epoch % 100 == 0:
                print ("%d [D loss: %f, acc: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "mnist_sgan_generator")
        save(self.discriminator, "mnist_sgan_discriminator")
        save(self.combined, "mnist_sgan_adversarial")


if __name__ == '__main__':
    sgan = SGAN()
    sgan.train(epochs=20000, batch_size=32, sample_interval=50)