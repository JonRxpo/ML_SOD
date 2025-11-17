import argparse
import os

import tensorflow as tf
from keras.optimizers import Adam

from src.data_loader import build_dataset
from src.sod_model import build_unet_like_model, bce_iou_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train SOD model (TensorFlow/Keras).")
    parser.add_argument("--train_images", type=str, required=True)
    parser.add_argument("--train_masks", type=str, required=True)
    parser.add_argument("--val_images", type=str, required=True)
    parser.add_argument("--val_masks", type=str, required=True)
    parser.add_argument("--img_size", type=int, nargs=2, default=[128, 128])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_ds = build_dataset(
        args.train_images,
        args.train_masks,
        img_size=tuple(args.img_size),
        batch_size=args.batch_size,
        shuffle=True,
        augment=True,
    )

    val_ds = build_dataset(
        args.val_images,
        args.val_masks,
        img_size=tuple(args.img_size),
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
    )

    model = build_unet_like_model(
        input_shape=(args.img_size[0], args.img_size[1], 3)
    )
    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss=bce_iou_loss,
        metrics=["accuracy"], 
    )

    ckpt_path = os.path.join(args.checkpoint_dir, "best_model.keras")
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[checkpoint_cb, early_stop_cb],
    )

    model.save(os.path.join(args.checkpoint_dir, "final_model.keras"))
    print("Training completed. Models saved in:", args.checkpoint_dir)


if __name__ == "__main__":
    main()
