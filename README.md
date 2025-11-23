Salient Object Detection (SOD) with UNet – TensorFlow

This project implements a Salient Object Detection (SOD) model using a UNet-style CNN, built from scratch in TensorFlow. The goal is to detect the most important object in an image and generate a binary saliency mask.

Project Structure – What Was Done

Dataset preprocessing & splitting (ECSSD)
Data augmentation & normalization
UNet-style CNN (Encoder–Decoder)
Custom loss: Binary Cross Entropy + IoU
Training with checkpointing & early stopping
Evaluation: Precision, Recall, F1, IoU
Final demo using Jupyter Notebook

1. Dataset (ECSSD)

All images are resized to 128×128, normalized, and split:

Split	Ratio
Train	70%
Val	15%
Test	15%

Download Dataset
https://www.cuhk.edu.cn/en/saliency/dataset/ecssd/

Place files like this:

data/ecssd_raw/ – This is where you place the original ECSSD dataset after downloading it.

images/ → contains all input images

masks/ → contains the ground-truth saliency masks

data/ecssd/ – This folder is created automatically after running split_ecssd.py.
It contains the processed dataset split into:

train/ → used for model training

val/ → used for validation

test/ → used for final testing and evaluation


Then split the dataset:

python src/split_ecssd.py

2. Setup & Virtual Environment

Recommended to use a virtual environment:

Create venv
python -m venv venv

(Windows)
venv\Scripts\activate

(Mac/Linux)
source venv/bin/activate

Install dependencies
pip install -r requirements.txt

3. Model – UNet Architecture

Encoder → Bottleneck → Decoder (skip connections)
Output: 1-channel mask (sigmoid)
Loss: BCE + 0.5 × (1 − IoU)
Optimizer: Adam (lr = 1e-3)

4. Training
python src/train.py \
 --train_images data/ecssd/train/images \
 --train_masks data/ecssd/train/masks \
 --val_images data/ecssd/val/images \
 --val_masks data/ecssd/val/masks \
 --epochs 7 --lr 0.001 --batch_size 8 \
 --checkpoint_dir checkpoints/ecssd_exp


Best model is saved to:

checkpoints/ecssd_exp/best_model.keras

5. Evaluation

After training, metrics were tested on random images:

Metric	Score (approx.)
Precision	~0.78
Recall	~0.76
F1-score	~0.74
IoU	~0.65
Demo – Jupyter Notebook

Open the demo:

notebooks/demo_notebook.ipynb


It:
Loads trained model
Picks random test image
Predicts mask
Shows Input + Prediction + Overlay

7. Requirements
TensorFlow
NumPy
OpenCV
Matplotlib
scikit-learn


Install dependencies:

pip install -r requirements.txt

8. Notes

This project helped understand:

Image Segmentation & CNN design

Training pipelines & augmentation

Evaluation metrics (Precision, Recall, F1, IoU)

Visualizations using OpenCV & Matplotlib

Future Improvements

More data augmentations

Larger image size

Deeper UNet or ResNet backbone
