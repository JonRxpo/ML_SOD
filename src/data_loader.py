import os
from typing import Tuple

import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def _load_image(path: tf.Tensor, img_size: Tuple[int, int]) -> tf.Tensor:
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def _load_mask(path: tf.Tensor, img_size: Tuple[int, int]) -> tf.Tensor:
    mask = tf.io.read_file(path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, img_size)
    mask = tf.cast(mask > 127, tf.float32)
    return mask


def _augment(image: tf.Tensor, mask: tf.Tensor):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    image = tf.image.random_brightness(image, max_delta=0.1)
    return image, mask


def build_dataset(
    images_dir: str,
    masks_dir: str,
    img_size: Tuple[int, int] = (128, 128),
    batch_size: int = 8,
    shuffle: bool = True,
    augment: bool = True,
) -> tf.data.Dataset:
    image_paths = sorted(
        [
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )
    mask_paths = sorted(
        [
            os.path.join(masks_dir, f)
            for f in os.listdir(masks_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )

    if len(image_paths) != len(mask_paths):
        print(
            f"[WARNING] Number of images ({len(image_paths)}) "
            f"and masks ({len(mask_paths)}) do not match."
        )

    img_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    msk_ds = tf.data.Dataset.from_tensor_slices(mask_paths)

    def _load_pair(ip, mp):
        img = _load_image(ip, img_size)
        msk = _load_mask(mp, img_size)
        return img, msk

    ds = tf.data.Dataset.zip((img_ds, msk_ds))
    ds = ds.map(_load_pair, num_parallel_calls=AUTOTUNE)

    if augment:
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_paths))

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds
