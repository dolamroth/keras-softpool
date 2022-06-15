import tensorflow.keras.backend as K
from tensorflow.python.keras.layers.pooling import GlobalPooling2D


class GlobalLogAvgExpPooling2D(GlobalPooling2D):
    """
    A drop-in replacement for GlobalAveragePooling2D or GlobalMaxPooling2D.
    Soft approximation to maximum via Log-Avg-Exp.
    # https://arxiv.org/abs/1909.03469

    :param r: Free parameter, controls the sharpness of pooling.
    """
    def __init__(self, r=20, data_format=None, keepdims=False, **kwargs):
        self.r = r
        super().__init__(data_format=data_format, keepdims=keepdims, **kwargs)

    def call(self, inputs):
        r = self.r
        axis = [1, 2] if self.data_format == 'channels_last' else [2, 3]
        return (
            K.logsumexp(r * inputs, axis=axis) * (1.0 / r)
            - K.logsumexp(r * K.zeros_like(inputs), axis=axis) * (1.0 / r)
        )
