import tensorflow as tf


class Linear(tf.keras.layers.Layer):
    """y = w.x + b"""

    def __init__(self, w, b):
        super(Linear, self).__init__()
        self.w = w
        self.b = b

    def build(self, input_shape):
        pass

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b