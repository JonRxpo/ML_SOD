import sys, os, time
sys.path.append(".")

import numpy as np
import tensorflow as tf
import cv2
import random
import matplotlib.pyplot as plt

from src.sod_model import bce_iou_loss


def load_model(model_path: str):
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"bce_iou_loss": bce_iou_loss}
    )


def predict_and_visualize(model, image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: could not read image at: {image_path}")
        return None, None

    img_resized = cv2.resize(img, (128, 128))
    inp = img_resized / 255.0
    inp = np.expand_dims(inp, axis=0)

    start = time.time()
    pred = model.predict(inp)[0]
    inference_time = time.time() - start

    mask_bin = (pred > 0.5).astype(np.uint8)

    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(img_resized[..., ::-1])
    plt.title("Input")

    plt.subplot(1, 3, 2)
    plt.imshow(pred.squeeze(), cmap="gray")
    plt.title("Prediction")

    overlay = img_resized.copy()
    overlay[:, :, 0] = overlay[:, :, 0] * (1 - mask_bin.squeeze()) + 255 * mask_bin.squeeze()

    plt.subplot(1, 3, 3)
    plt.imshow(overlay[..., ::-1])
    plt.title("Overlay")

    plt.tight_layout()
    plt.show()

    return pred.squeeze(), inference_time


def evaluate_mask(pred, true):
    # Resize mask to 128x128 to match prediction size
    true = cv2.resize(true, (128, 128))

    pred_bin = (pred > 0.5).astype(np.uint8).flatten()
    true_bin = (true / 255).astype(np.uint8).flatten()

    tp = np.sum((pred_bin == 1) & (true_bin == 1))
    fp = np.sum((pred_bin == 1) & (true_bin == 0))
    fn = np.sum((pred_bin == 0) & (true_bin == 1))
    union = np.sum(pred_bin | true_bin)

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    iou = tp / (union + 1e-7)

    return precision, recall, f1, iou


if __name__ == "__main__":
    model = load_model("checkpoints/ecssd/best_model.keras")

    test_dir = "data/ecssd/test/images"
    test_files = os.listdir(test_dir)
    test_image_path = os.path.join(test_dir, random.choice(test_files))

    print("Using test image:", test_image_path)

    pred, inference_time = predict_and_visualize(model, test_image_path)
    if pred is None:
        exit()

    mask_path = test_image_path.replace("images", "masks").replace(".jpg", ".png")
    true_mask = cv2.imread(mask_path, 0)
    if true_mask is None:
        print("Mask not found:", mask_path)
        exit()

    precision, recall, f1, iou = evaluate_mask(pred, true_mask)

    print("\nEvaluation Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"IoU:       {iou:.4f}")
    print(f"Inference Time: {inference_time:.4f} seconds")
