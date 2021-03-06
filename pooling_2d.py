import tensorflow.keras.backend as K
from tensorflow.keras import layers


def LogAvgExpPooling2D(
    pool_size=(2, 2),
    data_format=None,
    r=1.0,
):
    # TODO: enable passing "strides" and "padding"
    # TODO: enable passing layers, that have WxH not evenly divided by pool_size/strides
    """
    A drop-in replacement for AveragePooling2D or MaxPooling2D.
    Soft approximation to maximum via Log-Avg-Exp.
    # https://arxiv.org/abs/1909.03469

    :param pool_size: same, as in MaxPooling2D
    :param data_format: same, as in MaxPooling2D
    :param r: Free parameter, controls the sharpness of pooling.
    """
    def get_layer(input_layer):
        mp = layers.MaxPooling2D(pool_size, data_format=data_format, padding='same')(input_layer)
        zp = layers.UpSampling2D(pool_size, interpolation='nearest')(mp)

        # Subtract maximum within pool, in order to make computation stable
        x = layers.Lambda(lambda y: y[0] - y[1])([input_layer, zp])

        x = layers.Lambda(lambda y: K.exp(r * y))(x)
        x = layers.AveragePooling2D(pool_size, data_format=data_format, padding='same')(x)
        x = layers.Lambda(lambda y: K.log(y) / r)(x)

        x = layers.Lambda(lambda y: y[0] + y[1])([x, mp])
        return x

    return get_layer
