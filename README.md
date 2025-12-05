# 🧠 DenseNet121-MRI-Classification

A deep learning project for classifying Brain Tumor types from MRI images using Transfer Learning with the DenseNet121 architecture.

## ✨ Features

* **Multi-Class Classification:** Distinguishes between four tumor types: Glioma, Meningioma, Pituitary, and No Tumor.
* **Three-Phase Fine-Tuning:** Implements a robust training strategy to optimize transfer learning performance:
    1.  Training only the **Classifier** layer (frozen backbone).
    2.  Unfreezing and training the **last dense block** (`denseblock4`).
    3.  **Full Network Fine-Tuning** (unfreezing all layers) with a very low learning rate.
* **Early Stopping & Checkpointing:** Uses validation loss for early stopping and saves the best model weights across all phases.
* **Explainability (Grad-CAM):** Includes a function to generate **Grad-CAM heatmaps** to visualize which parts of the MRI image the model focuses on for its predictions.

## ⚙️ Setup and Installation

### 1. Requirements

This project requires Python 3.8+ and the following libraries:

``bash

pip install torch torchvision pandas numpy seaborn matplotlib scikit-learn opencv-python-headless

2. Data
The model is trained on the Brain Tumor MRI Dataset available on Kaggle.

Download the data: Obtain your Kaggle API key (kaggle.json).

Use the Kaggle CLI:

Bash

kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
unzip brain-tumor-mri-dataset.zip -d ./
Ensure the unzipped folders (train/, Testing/, val/) are in the root directory.

3. Clone the Repository
Bash

git clone [https://github.com/AmaarDevelops/DenseNet121-MRI-Classification.git]
cd DenseNet121-MRI-Classification

🚀 How to Run
The entire pipeline, including data loading, model definition, multi-phase training, evaluation, and visualization, is contained in a single script.

Bash

python model.py

📊 Evaluation Metrics
The script outputs comprehensive performance metrics on the test set, including:

Accuracy

Weighted F1 Score

Precision

Recall

Confusion Matrix (with visual heatmap)

ROC AUC Score (OVR)

📦 Saved Files
The final trained model and essential metadata are saved upon completion:

brain_tumor_mri_classifier.pth: The Pytorch state dictionary containing the final model weights.

class_to_idx.pth: A dictionary mapping class names to their corresponding integer labels.
