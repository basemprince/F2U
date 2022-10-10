import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


class Generator_tf():

    def create_model():
        model = tf.keras.Sequential()
        model.add(layers.Conv2DTranspose(512, (4,4), strides=(1, 1),input_shape=(1, 1, 128), padding='valid', use_bias=False))
        model.add(layers.BatchNormalization(momentum=0.1))
        model.add(layers.ReLU())
        model.add(layers.Conv2DTranspose(256, (4,4), strides=(2, 2),input_shape=(4, 4, 512), padding='same', use_bias=False))
        model.add(layers.BatchNormalization(momentum=0.1))
        model.add(layers.ReLU()) 
        model.add(layers.Conv2DTranspose(128, (4,4), strides=(2, 2),input_shape=(8, 8, 256), padding='same', use_bias=False))
        model.add(layers.BatchNormalization(momentum=0.1))
        model.add(layers.ReLU())
        model.add(layers.Conv2DTranspose(64, (4,4), strides=(2, 2),input_shape=(16, 16, 128), padding='same', use_bias=False))
        model.add(layers.BatchNormalization(momentum=0.1))
        model.add(layers.ReLU())
        model.add(layers.Conv2DTranspose(3, (3, 3), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
        return model
