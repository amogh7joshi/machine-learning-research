#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, AveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten, Activation, concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import get_file

# A custom build of the Inception v4 model as described in
# https://arxiv.org/pdf/1602.07261.pdf

class ChannelError(Exception):
   # Error if image_data_format != 'channels_last'
   def __init__(self):
      self.message = "You should be using a TensorFlow backend, which uses channel_last."
      super().__init__(self, self.message)

def conv_block(x, filters, kernel_size = (), padding = 'same', strides = (1, 1), l2_reg = 0.01, use_bias = False):
   # A convolution block.
   if K.image_data_format() == 'channels_last':
      c_axis = -1
   else: raise ChannelError

   with K.name_scope("conv_block"):
      x = Conv2D(filters, kernel_size = kernel_size, strides = strides, padding = padding,
                 kernel_regularizer = l2(l2_reg), kernel_initializer = 'he_normal', use_bias = use_bias)(x)
      x = BatchNormalization(axis = c_axis)(x)
      x = Activation('relu')(x)

   return x

def inception_a_block(input):
   # The first Inception (A) Block.
   if K.image_data_format() == 'channels_last':
      c_axis = -1
   else: raise ChannelError

   with tf.name_scope("inception-a_block"):
      b_1 = conv_block(input, 64, kernel_size = (1, 1))
      b_1 = conv_block(b_1, 96, kernel_size = (3, 3))
      b_1 = conv_block(b_1, 96, kernel_size = (3, 3))

      b_2 = conv_block(input, 64, kernel_size = (1, 1))
      b_2 = conv_block(b_2, 93, kernel_size = (3, 3))

      b_3 = conv_block(input, 96, kernel_size = (1, 1))

      b_4 = AveragePooling2D(pool_size = (3, 3), strides = (1, 1), padding = 'same')(input)
      b_4 = conv_block(b_4, 96, kernel_size = (1, 1))

      x = concatenate([b_1, b_2, b_3, b_4], axis = c_axis)

   return x

def inception_b_block(input):
   # The second Inception (B) Block.
   if K.image_data_format() == 'channels_last':
      c_axis = -1
   else: raise ChannelError

   with tf.name_scope("inception-b_block"):
      b_1 = conv_block(input, 192, kernel_size = (1, 1))
      b_1 = conv_block(b_1, 192, kernel_size = (1, 7))
      b_1 = conv_block(b_1, 224, kernel_size = (7, 1))
      b_1 = conv_block(b_1, 224, kernel_size = (1, 7))
      b_1 = conv_block(b_1, 256, kernel_size = (7, 1))

      b_2 = conv_block(input, 192, kernel_size = (1, 1))
      b_2 = conv_block(b_2, 224, kernel_size = (1, 7))
      b_2 = conv_block(b_2, 256, kernel_size = (7, 1))

      b_3 = conv_block(input, 384, kernel_size = (1, 1))

      b_4 = AveragePooling2D(pool_size = (3, 3), strides = (1, 1), padding = 'same')(input)
      b_4 = conv_block(b_4, 128, kernel_size = (1, 1))

      x = concatenate([b_1, b_2, b_3, b_4], axis = c_axis)

   return x

def inception_c_block(input):
   # The third Inception (C) Block.
   if K.image_data_format() == 'channels_last':
      c_axis = -1
   else: raise ChannelError

   with tf.name_scope("inception-c_block"):
      b_1 = conv_block(input, 384, kernel_size = (1, 1))
      b_1 = conv_block(b_1, 448, kernel_size = (1, 3))
      b_1 = conv_block(b_1, 512, kernel_size = (3, 1))
      b_1_1 = conv_block(b_1, 256, kernel_size = (1, 3))
      b_1_2 = conv_block(b_1, 256, kernel_size = (3, 1))

      b_2 = conv_block(input, 384, kernel_size = (1, 1))
      b_2_1 = conv_block(b_2, 256, kernel_size = (1, 3))
      b_2_2 = conv_block(b_2, 256, kernel_size = (3, 1))

      b_3 = conv_block(input, 256, kernel_size = (1, 1))

      b_4 = AveragePooling2D(pool_size = (3, 3), strides = (1, 1), padding = 'same')(input)
      b_4 = conv_block(b_4, 256, kernel_size = (1, 1))

      x = concatenate([b_1_1, b_1_2, b_2_1, b_2_2, b_3, b_4], axis = c_axis)

   return x

def reduction_a_block(input):
   # The first Reduction (A) Block.
   if K.image_data_format() == 'channels_last':
      c_axis = -1
   else: raise ChannelError

   with tf.name_scope("reduction-a_block"):
      b_1 = conv_block(input, 192, kernel_size = (1, 1))
      b_1 = conv_block(b_1, 224, kernel_size = (3, 3))
      b_1 = conv_block(b_1, 256, kernel_size = (3, 3), strides = (2, 2), padding = 'valid')

      b_2 = conv_block(input, 384, kernel_size = (3, 3), strides = (2, 2), padding = 'valid')

      b_3 = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'valid')(input)

      x = concatenate([b_1, b_2, b_3], axis = c_axis)

   return x

def reduction_b_block(input):
   # The second Reduction (B) Block.
   if K.image_data_format() == 'channels_last':
      c_axis = -1
   else: raise ChannelError

   with tf.name_scope("reduction-b_block"):
      b_1 = conv_block(input, 256, kernel_size = (1, 1))
      b_1 = conv_block(b_1, 256, kernel_size = (1, 7))
      b_1 = conv_block(b_1, 320, kernel_size = (7, 1))
      b_1 = conv_block(b_1, 320, kernel_size = (3, 3), strides = (2, 2), padding = 'valid')

      b_2 = conv_block(input, 192, kernel_size = (1, 1))
      b_2 = conv_block(b_2, 192, kernel_size = (3, 3), strides = (2, 2), padding = 'valid')

      b_3 = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'valid')(input)

      x = concatenate([b_1, b_2, b_3], axis = c_axis)

   return x

def stem(input):
   if K.image_data_format() == 'channels_last':
      c_axis = -1
   else: raise ChannelError

   with tf.name_scope("stem"):
      x = conv_block(input, 32, kernel_size = (3, 3), strides = (2, 2), padding = 'valid')
      x = conv_block(x, 32, kernel_size = (3, 3), padding = 'valid')
      x = conv_block(x, 64, kernel_size = (3, 3))

      b_1 = conv_block(x, 96, kernel_size = (3, 3), strides = (2, 2), padding = 'valid')

      b_2 = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'valid')(x)

      x = concatenate([b_1, b_2], axis = c_axis)

      b_3 = conv_block(x, 64, kernel_size = (1, 1))
      b_3 = conv_block(b_3, 64, kernel_size = (1, 7))
      b_3 = conv_block(b_3, 64, kernel_size = (7, 1))
      b_3 = conv_block(b_3, 96, kernel_size = (3, 3), padding = 'valid')

      b_4 = conv_block(x, 64, kernel_size = (1, 1))
      b_4 = conv_block(b_4, 96, kernel_size = (3, 3), padding = 'valid')

      x = concatenate([b_3, b_4], axis = c_axis)

      b_5 = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'valid')(x)

      b_6 = conv_block(x, 192, kernel_size = (3, 3), strides = (2, 2), padding = 'valid')

      x = concatenate([b_5, b_6], axis = c_axis)

   return x

def inception_v4(classes, input_shape = (299, 299, 3), weights = None):
   if K.image_data_format() == 'channels_last':
      input = Input(input_shape)
   else: raise ChannelError

   # Construct the Stem first.
   x = stem(input)

   for i in range(4):
      x = inception_a_block(x)

   x = reduction_a_block(x)

   for i in range(7):
      x = inception_b_block(x)

   x = reduction_b_block(x)

   for i in range(3):
      x = inception_c_block(x)

   x = AveragePooling2D(pool_size = (8, 8), padding = 'valid')(x)
   x = Dropout(0.2)(x)
   x = Flatten()(x)
   x = Dense(classes, activation = 'softmax')(x)

   model = Model(input, x, name = 'Inception-v4')

   if weights is not None:
      if weights == 'imagenet':
         path = get_file(
            'inception-v4_weights_tf_dim_ordering_tf_kernels.h5',
            'https://github.com/kentsommer/keras-inceptionV4/releases/download/2.1/inception-v4_weights_tf_dim_ordering_tf_kernels.h5',
            cache_subdir = 'models',
            md5_hash = '9fe79d77f793fe874470d84ca6ba4a3b')
         model.load_weights(path, by_name = True)
      elif os.path.isfile(weights):
         model.load_weights(weights)
      else: raise ValueError("That is not a valid weights file")

   return model






