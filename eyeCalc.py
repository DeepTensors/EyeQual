from keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, AveragePooling2D, Activation, Input, Flatten, concatenate, GlobalAveragePooling2D, LeakyReLU
from keras.models import Model
from keras import backend as K
from keras.layers import Layer, InputSpec
from keras import initializers
import numpy as np
import tensorflow as tf
'''

    1. __init__ method to initialize class variable and super class variables
    2. build method to define weights.
    3. call method where you will perform all your operations.
    4. compute_output_shape method to define output shape of this custom layer
'''


class WeightedAveragePooling(Layer):
    def __init__(self, output_shape, **kwargs):
        self.shape = output_shape
        super(WeightedAveragePooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(name='W1', shape=self.shape, initializer='uniform') # creating W

        super(WeightedAveragePooling, self).build(input_shape)

    def call(self, input_):
        w_absolute = K.abs(self.w)  # making w values positive
        numerator = input_*w_absolute
        numerator_sum = K.expand_dims(K.sum(numerator, axis=(1, 2, 3)))
        denominator = K.sum(w_absolute, axis=(1, 2, 3))
        denominator_sum = K.expand_dims(K.sum(w_absolute, axis=(1, 2, 3)))
        return numerator_sum / (denominator_sum + 1e-7)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)

    def get_config(self):
        config = {
            'shape': self.shape,
        }
        base_config = super(WeightedAveragePooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SWAP(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(SWAP, self).__init__(**kwargs)

    def build(self, input_shape):

        # print(input_shape[-1], self.output_shape_)
        input_dim = input_shape[-1]
        print(type(input_dim))

        self.w = self.add_weight(name='w', shape=(input_dim, self.output_dim), initializer='uniform')
        super(SWAP, self).build(input_shape)

    def call(self, inputs):
        w = np.abs(self.w)
        w = w/(np.sum(w) + 1e-7)
        x = K.dot(inputs, K.abs(w))          # weights need to be non negative
        bias_ = -0.5*np.ones(1,)
        output = x + bias_
        output = Activation('sigmoid')(output)

        return output

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        shape[-1] = self.output_dim
        return tuple(shape)

    def get_config(self):
        config = {'output_dim': self.output_dim
                 }
        base_config = super(SWAP, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class EyeCalc:
    def __init__(self, input_, pooling_type):
        self.input = input_
        self.pooling_layer = pooling_type

    def convolution(self, input_, kernel_size, filters, strides=1, activation='relu', max_pool="True", batch_norm="True"):

        x = Conv2D(kernel_size=kernel_size, filters=filters, strides=strides, padding='same')(input_)

        if activation == 'sigmoid':
            x = Activation('sigmoid')(x)
        else:
            x = Activation('relu')(x)
#             x = LeakyReLU(0.02)(x)

        if batch_norm:
            x = BatchNormalization()(x)

        if max_pool:
            x = MaxPooling2D((2, 2))(x)

        return x

    def conv2d(self):
        num_filters = 64
        x = self.convolution(self.input, 3, num_filters, strides=1)
        for i in range(3):
            num_filters *= 2
            x = self.convolution(x, 3, num_filters, strides=1)

        x = self.convolution(x, kernel_size=1, filters=1, strides=1, activation='sigmoid', max_pool=False, batch_norm=False)

        return x

    def pooling(self, input_, type='wap'):
        if type == 'SWAP':
            x = Flatten()(input_)
            x = SWAP(1)(x)

        else:
            y = int(self.input.shape[2]//16)
            x = WeightedAveragePooling((1, y, y, 1))(input_)

        return x

    def forward(self):
        x = self.conv2d()
        x = self.pooling(x, type=self.pooling_layer)

        return x

    def build_model(self):

        output = self.forward()
        model = Model(self.input, output)

        return model

