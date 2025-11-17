from typing import Tuple

import tensorflow as tf
from keras import layers, models


def build_unet_like_model(
    input_shape: Tuple[int, int, int] = (128, 128, 3)
) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)

    # Encodr
    c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, 3, activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(64, 3, activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(128, 3, activation="relu", padding="same")(p2)
    c3 = layers.Conv2D(128, 3, activation="relu", padding="same")(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck
    b = layers.Conv2D(256, 3, activation="relu", padding="same")(p3)
    b = layers.Conv2D(256, 3, activation="relu", padding="same")(b)

    # Decoder
    u3 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(b)
    u3 = layers.concatenate([u3, c3])
    c4 = layers.Conv2D(128, 3, activation="relu", padding="same")(u3)
    c4 = layers.Conv2D(128, 3, activation="relu", padding="same")(c4)

    u2 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(c4)
    u2 = layers.concatenate([u2, c2])
    c5 = layers.Conv2D(64, 3, activation="relu", padding="same")(u2)
    c5 = layers.Conv2D(64, 3, activation="relu", padding="same")(c5)

    u1 = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(c5)
    u1 = layers.concatenate([u1, c1])
    c6 = layers.Conv2D(32, 3, activation="relu", padding="same")(u1)
    c6 = layers.Conv2D(32, 3, activation="relu", padding="same")(c6)

    outputs = layers.Conv2D(1, 1, activation="sigmoid", padding="same")(c6)

    model = models.Model(inputs=[inputs], outputs=[outputs], name="SOD_UNet")
    return model


def bce_iou_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection + 1e-7
    iou = intersection / union

    iou_loss = 1.0 - iou
    return bce + 0.5 * iou_loss
