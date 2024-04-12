import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import LayerNormalization, Dropout, Layer, MultiHeadAttention
from tensorflow.keras.layers import Embedding, Input, GlobalAveragePooling1D, Dense
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential, Model


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()

        # Multi-head attention
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)

        # Feed forward network
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), 
             Dense(embed_dim),]
        )

        # Normalization layers
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):

        # Multi-head attention
        attn_output = self.att(inputs, inputs)

        # Dropout and residual connection
        attn_output = self.dropout1(attn_output, training=training)

        # Add and normalize
        out1 = self.layernorm1(inputs + attn_output)

        # Feed forward network
        ffn_output = self.ffn(out1)

        # Dropout and residual connection
        ffn_output = self.dropout2(ffn_output, training=training)

        # Add and normalize
        out = self.layernorm2(out1 + ffn_output)
        
        return out

class PositionEmbedding(Layer):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(PositionEmbedding, self).__init__()
        self.pos_emb = Embedding(input_dim=input_dim, output_dim=output_dim)

    def call(self, x):
        seq_len = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions = self.pos_emb(positions)
        return positions

# Encoder architecture. This can be changed to any other architecture.
def encoder(input_shape=None, output_width=32, name="encoder"):
    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer
    k_size = 5

    return keras.Sequential(
        [
            layers.Input(shape=input_shape),

            layers.Conv1D(filters=50, kernel_size=k_size, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            # layers.Dropout(0.2),

            layers.Conv1D(filters=50, kernel_size=k_size, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            # layers.Dropout(0.2),

			layers.Conv1D(filters=embed_dim, kernel_size=k_size, activation='relu'),
			layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),

			# layers.Flatten(),
            TransformerBlock(embed_dim, num_heads, ff_dim),
            layers.GlobalAveragePooling1D(),
            # layers.Dropout(0.1),
            layers.Flatten(),
			layers.Dense(output_width, activation='relu'),
        ],
        name=name,
    )