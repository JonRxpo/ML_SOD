import sys
import os
import argparse
import tensorflow as tf
from keras.optimizers import Adam

sys.path.append(".")

from src.data_loader import build_dataset
from src.sod_model_exp import build_unet_experiment, bce_iou_loss 


def parse_args():

    parser = argparse.ArgumentParser(description="Train SOD model on ECSSD dataset")
    parser.add_argument("--train_images", required=True, help="Path to training images")
    parser.add_argument("--train_masks", required=True, help="Path to training masks")
    parser.add_argument("--val_images", required=True, help="Path to validation images")
    parser.add_argument("--val_masks", required=True, help="Path to validation masks")
    parser.add_argument("--img_size", nargs=2, type=int, default=[128, 128])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/ecssd")
    return parser.parse_args()


def main():
    args = parse_args()

  
    train_ds = build_dataset(
        args.train_images, args.train_masks,
        img_size=tuple(args.img_size),
        batch_size=args.batch_size,
        shuffle=True, augment=True
    )

    val_ds = build_dataset(
        args.val_images, args.val_masks,
        img_size=tuple(args.img_size),
        batch_size=args.batch_size,
        shuffle=False, augment=False
    )

  
    model = build_unet_experiment(input_shape=(args.img_size[0], args.img_size[1], 3))
    model.compile(
        optimizer=Adam(args.lr),
        loss=bce_iou_loss,
        metrics=["accuracy"]
    )

    model.summary()

  
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{args.checkpoint_dir}/best_model.keras",
        save_best_only=True,
        monitor="val_loss",
        verbose=1
    )

    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[checkpoint_cb, early_stop_cb]
    )

    model.save(f"{args.checkpoint_dir}/final_model.keras")
    print(f"Training complete. Models saved in: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
