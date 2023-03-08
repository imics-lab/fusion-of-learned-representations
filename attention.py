"""
Regular Simple Attention Layer

Authors: Alexander Katrompas
Organization: Texas State University

self.em is is the attention capture
    this variable is public and can be accessed programmatically
    with a predict callback and written to a file
"""

from tensorflow.keras.layers import *
from tensorflow.keras import backend as K

class Attention(Layer):
    def __init__(self, return_sequences=True):
        self.__return_sequences = return_sequences
        self.em = None  # this is the attention capture variable
        self.__W = None
        self.__b = None
        self.__out = None
        super(Attention,self).__init__()

    def build(self, input_shape):
        self.__W =self.add_weight(name="att_weight", shape=(input_shape[-1],1))
        self.__b =self.add_weight(name="att_bias", shape=(input_shape[1],1))
        super(Attention,self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x,self.__W)+self.__b)
        alpha = K.softmax(e, axis=1)
        self.em = alpha + 0 # updated for capture
        self.out = x*alpha
        if not self.__return_sequences:
            self.out = K.sum(self.out, axis=1)
        return self.out
