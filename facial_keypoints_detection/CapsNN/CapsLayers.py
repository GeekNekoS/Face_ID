from keras import backend as K
import tensorflow as tf
import numpy as np
from keras import layers, initializers, regularizers, constraints
from keras.utils import conv_utils
from keras.layers import InputSpec
from keras.utils.conv_utils import conv_output_length
from batchdot import own_batch_dot


class Conv2DCaps(layers.Layer):

    def __init__(self, ch_j, n_j,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 r_num=1,
                 b_alphas=[8, 8, 8],
                 padding='same',
                 data_format='channels_last',
                 dilation_rate=(1, 1),
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        super(Conv2DCaps, self).__init__(**kwargs)
        rank = 2
        self.ch_j = ch_j  # Number of capsules in layer J
        self.n_j = n_j  # Number of neurons in a capsule in J
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.r_num = r_num
        self.b_alphas = b_alphas
        self.padding = conv_utils.normalize_padding(padding)
        #self.data_format = conv_utils.normalize_data_format(data_format)
        self.data_format = K.normalize_data_format(data_format)
        self.dilation_rate = (1, 1)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.input_spec = InputSpec(ndim=rank + 3)

    def build(self, input_shape):

        self.h_i, self.w_i, self.ch_i, self.n_i = input_shape[1:5]

        self.h_j, self.w_j = [conv_utils.conv_output_length(input_shape[i + 1],
                                                            self.kernel_size[i],
                                                            padding=self.padding,
                                                            stride=self.strides[i],
                                                            dilation=self.dilation_rate[i]) for i in (0, 1)]

        self.ah_j, self.aw_j = [conv_utils.conv_output_length(input_shape[i + 1],
                                                              self.kernel_size[i],
                                                              padding=self.padding,
                                                              stride=1,
                                                              dilation=self.dilation_rate[i]) for i in (0, 1)]

        self.w_shape = self.kernel_size + (self.ch_i, self.n_i,
                                           self.ch_j, self.n_j)

        self.w = self.add_weight(shape=self.w_shape,
                                 initializer=self.kernel_initializer,
                                 name='kernel',
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)

        self.built = True

    def call(self, inputs):
        if self.r_num == 1:
            # if there is no routing (and this is so when r_num is 1 and all c are equal)
            # then this is a common convolution
            outputs = K.conv2d(K.reshape(inputs, (-1, self.h_i, self.w_i,
                                                  self.ch_i * self.n_i)),
                               K.reshape(self.w, self.kernel_size +
                                         (self.ch_i * self.n_i, self.ch_j * self.n_j)),
                               data_format='channels_last',
                               strides=self.strides,
                               padding=self.padding,
                               dilation_rate=self.dilation_rate)

            outputs = squeeze(K.reshape(outputs, ((-1, self.h_j, self.w_j,
                                                   self.ch_j, self.n_j))))

        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.h_j, self.w_j, self.ch_j, self.n_j)

    def get_config(self):
        config = {
            'ch_j': self.ch_j,
            'n_j': self.n_j,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'b_alphas': self.b_alphas,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint)
        }
        base_config = super(Conv2DCaps, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))