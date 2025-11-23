import os
import shutil
import random

RAW_DIR = "data/ecssd_raw"
OUT_DIR = "data/ecssd"

SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}


def ensure_dirs():
    """Create train/val/test folders if not existing."""
    for split in SPLIT_RATIOS.keys():
        os.makedirs(os.path.join(OUT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, split, "masks"), exist_ok=True)


def main():
    ensure_dirs()

    images_dir = os.path.join(RAW_DIR, "images")
    masks_dir = os.path.join(RAW_DIR, "masks")

    images = sorted([f for f in os.listdir(images_dir) if f.endswith(".jpg")])
    masks = sorted([f for f in os.listdir(masks_dir) if f.endswith(".png")])

    print(f"Total images found: {len(images)}")
    print(f"Total masks found: {len(masks)}")

    # Verify matching filenames
    missing = [img for img in images if img.replace(".jpg", ".png") not in masks]
    if missing:
        print(f"Warning: {len(missing)} images do not have matching masks.")
        print("Example:", missing[:5])
    else:
        print("All image and mask filenames match correctly.")

    # Shuffle and split
    combined = list(zip(images, masks))
    random.shuffle(combined)

    total = len(combined)
    n_train = int(total * SPLIT_RATIOS["train"])
    n_val = int(total * SPLIT_RATIOS["val"])

    splits = {
        "train": combined[:n_train],
        "val": combined[n_train:n_train + n_val],
        "test": combined[n_train + n_val:]
    }

    for split, items in splits.items():
        for img_file, mask_file in items:
            shutil.copy(
                os.path.join(images_dir, img_file),
                os.path.join(OUT_DIR, split, "images", img_file)
            )
            shutil.copy(
                os.path.join(masks_dir, mask_file),
                os.path.join(OUT_DIR, split, "masks", mask_file)
            )
        print(f"{split}: {len(items)} samples copied.")

    print(f"\nDataset split complete. Output saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
