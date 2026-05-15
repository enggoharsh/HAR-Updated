# Human Activity Recognition Using Deep Learning and Machine Learning Hybrid Models

This project focuses on **Human Activity Recognition (HAR)** using different combinations of Deep Learning and Machine Learning models.

The system classifies different human activities from images using:

- AlexNet
- GoogleNet
- SVM
- KNN
- Softmax Classification

---

# Human Activity Classes

The dataset contains the following activity classes:

| Class ID | Activity |
|---|---|
| A1 | Walking |
| A2 | Sitting-down |
| A3 | StandUp |
| A4 | PickObject |
| A5 | DrinkWater |
| A6 | Fall |

---

# Dataset Split

The dataset is divided into:

- 70% Training Data
- 15% Validation Data
- 15% Testing Data

---

# Dataset Folder Structure

```text
dataset/
│
├── train/
│    ├── A1-Walking/
│    ├── A2-Sitting-down/
│    ├── A3-StandUp/
│    ├── A4-PickObject/
│    ├── A5-DrinkWater/
│    ├── A6-Fall/
│
├── validation/
│    ├── A1-Walking/
│    ├── A2-Sitting-down/
│    ├── A3-StandUp/
│    ├── A4-PickObject/
│    ├── A5-DrinkWater/
│    ├── A6-Fall/
│
└── test/
     ├── A1-Walking/
     ├── A2-Sitting-down/
     ├── A3-StandUp/
     ├── A4-PickObject/
     ├── A5-DrinkWater/
     └── A6-Fall/
```

---

# Project Models Included

| Notebook File | Description |
|---|---|
| `alexnet_svm.ipynb` | AlexNet feature extractor + SVM classifier |
| `alxnet_knn.ipynb` | AlexNet feature extractor + KNN classifier |
| `google_net_knn.ipynb` | GoogleNet feature extractor + KNN classifier |
| `googlenet_googlenet.ipynb` | End-to-end GoogleNet classification using Softmax |
| `googlnet_svm.ipynb` | GoogleNet feature extractor + SVM classifier |

---

# Project Folder Structure

```text
Project Folder/
│
├── alexnet_svm.ipynb
├── alxnet_knn.ipynb
├── google_net_knn.ipynb
├── googlenet_googlenet.ipynb
├── googlnet_svm.ipynb
│
├── *.pkl files
│    ├── SVM models
│    ├── KNN models
│    ├── scaler files
│    ├── PCA files
│
├── *.h5 files
│    ├── AlexNet feature extractor
│    ├── GoogleNet feature extractor
│    ├── trained CNN models
│
├── dataset/
│
└── README.md
```

---

# Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- NumPy
- OpenCV
- Matplotlib
- Pandas
- Joblib

---

# Deep Learning Concepts Used

- Convolutional Neural Networks (CNN)
- Feature Extraction
- Transfer Learning
- PCA (Principal Component Analysis)
- Feature Scaling
- Hybrid Deep Learning Models

---

# Machine Learning Classifiers Used

## 1. SVM (Support Vector Machine)

Used for:
- classification using extracted CNN features

Advantages:
- strong classifier for medium-sized datasets
- good generalization capability

---

## 2. KNN (K-Nearest Neighbors)

Used for:
- nearest-neighbor based classification

Advantages:
- simple implementation
- effective for feature-based classification

---

## 3. Softmax Classification

Used in:
- end-to-end GoogleNet model

Advantages:
- direct multiclass probability prediction
- fully deep learning-based pipeline

---

# CNN Models Used

## AlexNet

- Input size: `227 × 227`
- Used as feature extractor
- Simpler and faster CNN architecture

---

## GoogleNet

- Deep CNN architecture
- Uses Inception modules
- Better feature extraction capability

---

# Workflow of the Project

```text
Input Image
     ↓
Preprocessing
     ↓
CNN Feature Extraction
     ↓
Feature Scaling
     ↓
PCA Dimensionality Reduction
     ↓
Classifier (SVM / KNN / Softmax)
     ↓
Predicted Human Activity
```

---

# Installation

Install required libraries before running the notebooks.

## Step 1 — Create Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

## Step 2 — Install Dependencies

```bash
pip install tensorflow scikit-learn numpy matplotlib opencv-python pandas joblib notebook
```

---

# How to Run the Project

## 1. Download Project Files

Download all files from the shared Google Drive link.

---

## 2. Extract Files

Extract ZIP folder if compressed.

---

## 3. Open Jupyter Notebook

```bash
jupyter notebook
```

---

## 4. Open Required Notebook

| Notebook | Purpose |
|---|---|
| `alexnet_svm.ipynb` | AlexNet + SVM |
| `alxnet_knn.ipynb` | AlexNet + KNN |
| `google_net_knn.ipynb` | GoogleNet + KNN |
| `googlenet_googlenet.ipynb` | GoogleNet + Softmax |
| `googlnet_svm.ipynb` | GoogleNet + SVM |

---

## 5. Run Cells Sequentially

Run all notebook cells one by one from top to bottom.

---

# Saved Models

## `.h5` Files

Contain:
- trained CNN models
- feature extractors

Examples:

```text
alexnet_feature_extractor.h5
googlenet_model.h5
```

---

## `.pkl` Files

Contain:
- SVM models
- KNN models
- scalers
- PCA transformers

Examples:

```text
alexnet_svm_model.pkl
alexnet_svm_scaler.pkl
alexnet_svm_pca.pkl
```

---

# Prediction Pipeline Example

```text
Image
 ↓
AlexNet Feature Extractor
 ↓
Feature Vector
 ↓
Scaling
 ↓
PCA
 ↓
SVM Classification
 ↓
Predicted Activity
```

---

# Evaluation Metrics Used

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

# Example Output

```text
Predicted Class: Sitting-down
Confidence: 95%
```

---

# Research Purpose

This project demonstrates:

- Hybrid deep learning architectures
- CNN feature extraction
- Machine learning classification
- Transfer learning concepts
- Human activity recognition systems
- Comparison between classifiers

---

# Future Improvements

Possible future enhancements:

- Real-time activity detection
- Video-based HAR
- Mobile deployment
- Advanced CNN architectures
- Attention mechanisms
- Larger datasets

---

# Notes

- Ensure dataset paths are correct before running.
- Keep `.pkl` and `.h5` files in the same directory as notebooks.
- GPU is recommended for faster training.
- Training time depends on dataset size and system specifications.

---

# Author
Harsh Mishra, Om Kumar, Anupam Sarashwat.

Project developed for educational and research purposes.
