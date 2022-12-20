from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

class Jittering(layers.Layer):
    '''
    Jittering the signal. The signal is jittered by adding a random value to each point.
    Args:
        sigma: Standard deviation of the Gaussian distribution.
        
    Returns:
        Jittered signal.
    '''

    def __init__(self, sigma: float = 0.03):
        super(Jittering, self).__init__()
        self.sigma = sigma

    def call(self, signal):
        return signal + tf.random.normal(tf.shape(signal), stddev=self.sigma)
    
        
class Scaling(layers.Layer):
    '''
    Scaling the signal. The signal is scaled by multiplying a random value to each point.
    Args:
        sigma: Standard deviation of the Gaussian distribution.

    Returns:
        Scaled signal.
    '''
    def __init__(self, sigma: float = 0.1):
        super(Scaling, self).__init__()
        self.sigma = sigma

    def call(self, signal):
        return signal * tf.random.normal(tf.shape(signal), stddev=self.sigma)


class TimeWarping(layers.Layer):
    '''
    Time warping the signal. The signal is warped by adding a random value to a random point.
    Args:
        sigma: Standard deviation of the Gaussian distribution.
        knot: Number of knots.

    Returns:
        Time warped signal.
    '''

    def __init__(self, sigma=0.2, knot=4):
        super(TimeWarping, self).__init__()
        self.sigma = sigma
        self.knot = knot

    def call(self, signal):
        # Time warping the signal
        signal_length = tf.shape(signal)[1]
        signal_length_float = tf.cast(signal_length, dtype=tf.float32)
        time_warping_path = tf.linspace(0.0, signal_length_float - 1, signal_length)
        for i in range(self.knot):
            random_point = tf.random.uniform(
                shape=(1,), minval=0, maxval=signal_length - 1, dtype=tf.int32
            )
            random_warping = tf.random.normal(
                shape=(1,), mean=0.0, stddev=self.sigma, dtype=tf.float32
            )
            time_warping_path = tf.concat(
                [
                    time_warping_path[:random_point[0]],
                    time_warping_path[random_point[0]] + random_warping,
                    time_warping_path[random_point[0] + 1 :],
                ],
                axis=0,
            )

        time_warping_path = tf.clip_by_value(time_warping_path, 0.0, signal_length_float - 1)
        time_warping_path = tf.cast(time_warping_path, dtype=tf.int32)
        warped_signal = tf.gather(signal, time_warping_path, axis=1)
        return warped_signal


def augmenter(name="augmenter", input_shape=None):
    return keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            Jittering(sigma=0.03),
            Scaling(sigma=0.1),
            TimeWarping(sigma=0.2, knot=4),
        ],
        name=name,
    )

