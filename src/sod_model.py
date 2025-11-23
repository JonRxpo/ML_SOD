from typing import Tuple
import tensorflow as tf
from keras import layers, models, losses


def build_unet_baseline(
    input_shape: Tuple[int, int, int] = (128, 128, 3)
) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)

    c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, 3, activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(64, 3, activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(128, 3, activation="relu", padding="same")(p2)
    c3 = layers.Conv2D(128, 3, activation="relu", padding="same")(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    b = layers.Conv2D(256, 3, activation="relu", padding="same")(p3)
    b = layers.Conv2D(256, 3, activation="relu", padding="same")(b)

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
    return models.Model(inputs=inputs, outputs=outputs, name="SOD_UNet_Baseline")



def build_unet_improved(
    input_shape: Tuple[int, int, int] = (128, 128, 3)
) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)


    c1 = layers.Conv2D(32, 3, padding="same")(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.ReLU()(c1)
    c1 = layers.Dropout(0.2)(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, 3, padding="same")(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.ReLU()(c2)
    c2 = layers.Dropout(0.3)(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(128, 3, padding="same")(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.ReLU()(c3)
    c3 = layers.Dropout(0.4)(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    b = layers.Conv2D(256, 3, padding="same")(p3)
    b = layers.BatchNormalization()(b)
    b = layers.ReLU()(b)


    u3 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(b)
    u3 = layers.concatenate([u3, c3])
    c4 = layers.Conv2D(128, 3, padding="same")(u3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.ReLU()(c4)

    u2 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(c4)
    u2 = layers.concatenate([u2, c2])
    c5 = layers.Conv2D(64, 3, padding="same")(u2)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.ReLU()(c5)

    u1 = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(c5)
    u1 = layers.concatenate([u1, c1])
    c6 = layers.Conv2D(32, 3, padding="same")(u1)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.ReLU()(c6)

    outputs = layers.Conv2D(1, 1, activation="sigmoid", padding="same")(c6)
    return models.Model(inputs=inputs, outputs=outputs, name="SOD_UNet_Improved")



def bce_iou_loss(y_true, y_pred):
    bce = losses.binary_crossentropy(y_true, y_pred)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection + 1e-7
    iou_loss = 1.0 - (intersection / union)
    return bce + 0.5 * iou_loss
