import torch
import tensorflow as tf
from utils.Functions import *


extractor = Extractor('CPD-R.pth')


class Attention_net(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.block1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(filters=8, kernel_size=1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
        ])
        self.block2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=8, kernel_size=1),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(filters=4, kernel_size=1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
        ])
        self.block3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='same'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(filters=4, kernel_size=2, strides=2)
        ])
        self.block4 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=4, kernel_size=3, padding='same'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(filters=4, kernel_size=1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=2, strides=2)
        ])
        self.block5 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=3, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU()
        ])

    def call(self, X):
        Y_c1 = self.block1(X)
        Y = pool_and_concat(Y_c1)
        Y_c2 = self.block2(Y)
        Y = pool_and_concat(Y_c2)
        Y = self.block3(Y)
        Y += Y_c2
        Y = self.block4(Y)
        Y += Y_c1
        return self.block5(Y)


class SFT_layer(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.block1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
        ])
        self.block2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
        ])

    def call(self, X, F):
        return X * self.block1(F) + self.block2(F)


class SFT_block(tf.keras.Model):
    def __init__(self, extractor):
        super().__init__()
        self.extractor = extractor
        self.preprossNet = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
        ])
        self.layer1 = SFT_layer()
        self.layer2 = SFT_layer()
        # self.layer3 = SFT_layer()
        # self.layer4 = SFT_layer()
        # self.layer5 = SFT_layer()
        self.featureList = []

    def call(self, X):
        self.featureList = self.extractor.extract_features(X)
        X = self.layer1(self.preprossNet(X), self.featureList[0])
        X = self.layer2(X, self.featureList[1])
        # X = self.layer3(X, self.featureList[0])
        # X = self.layer4(X, self.featureList[1])
        # Y = self.layer5(X, self.featureList[0])
        return X


class resBlock(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.net = Attention_net()
        self.preprossNet = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
        ])

    def call(self, X):
        Y = self.net(self.preprossNet(X))
        Y *= X
        Y += X
        return Y


class resBlockV2(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.block = SFT_block(extractor)
        self.net = Attention_net()

    def call(self, X):
        Y = self.block(X)
        Y = self.net(Y)
        Y *= X
        Y += X
        return Y

