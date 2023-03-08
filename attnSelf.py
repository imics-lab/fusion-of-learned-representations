"""
Self-Attention Layer

Authors: Alexander Katrompas
Organization: Texas State University

Stand alone self-attention layer class for use with LSTM layer
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer, Flatten, Activation, Permute
from tensorflow.keras.layers import Permute
from tensorflow.keras import backend as K

class AttnSelf(Layer):
    """
    Stand alone self-attenion layer class for use with LSTM layer

    @param (int) alength: attention length
    @param (bool) return_sequences: return sequences true (default) or false
    """
    def __init__(self, alength, return_sequences = True):
        self.alength = alength
        self.return_sequences = return_sequences
        super(AttnSelf, self).__init__()

    def build(self, input_shape):
        self.W1 = self.add_weight(name='W1',
                                  shape=(self.alength, input_shape[2]),
                                  initializer='random_uniform',
                                  trainable=True)
        self.W2 = self.add_weight(name='W2',
                                  shape=(input_shape[1], self.alength),
                                  initializer='random_uniform',
                                  trainable=True)
        super(AttnSelf, self).build(input_shape)

    def call(self, inputs):
        W1, W2 = self.W1[None, :, :], self.W2[None, :, :]

        #key matrix
        hidden_states_transposed = Permute(dims=(2, 1))(inputs)
        e = tf.matmul(W1, hidden_states_transposed)
        e = Activation('tanh')(e)

        #value matrix
        attention_weights = tf.matmul(W2, e)
        attention_weights = Activation('softmax')(attention_weights)

        embedding_matrix = tf.matmul(attention_weights, inputs) # will be size seq x lstm
        if not self.return_sequences:
            embedding_matrix = K.sum(embedding_matrix, axis=1)
        return embedding_matrix

# example with 16 features, size 5 sequence, 256 LSTMs
# (16, 5, 256)
# data is from last point only, but represents the 5 in the sequence
# you'll always get all 5 in the sequence at each time-step, representing the
# whole sequence at that point
# 256, one for each lstm
# 5 time steps for the sequence
# 16 for each feature
