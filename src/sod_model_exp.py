from typing import Tuple
import tensorflow as tf
from keras import layers, models


def build_unet_experiment(
    input_shape: Tuple[int, int, int] = (128, 128, 3)
) -> tf.keras.Model:

    inputs = layers.Input(shape=input_shape)

    # Encoder
    def enc_block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x, layers.MaxPooling2D((2, 2))(x)

    c1, p1 = enc_block(inputs, 32)
    c2, p2 = enc_block(p1, 64)
    c3, p3 = enc_block(p2, 128)

    # Bottleneck with Dropout
    b = layers.Conv2D(256, 3, padding="same")(p3)
    b = layers.BatchNormalization()(b)
    b = layers.ReLU()(b)
    b = layers.Dropout(0.4)(b)   # added
    b = layers.Conv2D(256, 3, padding="same")(b)
    b = layers.BatchNormalization()(b)
    b = layers.ReLU()(b)

    # Decoder
    def dec_block(x, skip, filters):
        x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(x)
        x = layers.concatenate([x, skip])
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    u3 = dec_block(b, c3, 128)
    u2 = dec_block(u3, c2, 64)
    u1 = dec_block(u2, c1, 32)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(u1)

    model = models.Model(inputs=[inputs], outputs=[outputs], name="SOD_UNet_Experiment")
    return model


# Same loss function
def bce_iou_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection + 1e-7
    iou = intersection / union

    return bce + 0.5 * (1 - iou)
