# Vegetable Image Classification Project
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12.0-orange)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📌 Project Overview
A deep learning model achieving **96.97% test accuracy** in classifying 15 vegetable types, developed for Dicoding's "Belajar Fundamental Deep Learning" certification. Implements a CNN with optimization techniques and multi-platform deployment.

## 🥦 Dataset
**Kaggle Vegetable Image Dataset**  
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue)](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset)

| Category       | Count |
|----------------|-------|
| Training       | 15,000 images |
| Validation     | 3,000 images |
| Test           | 3,000 images |
| **Classes**    | 15 (Brinjal, Broccoli, etc.) |

## 🧠 Model Architecture
```
# Membangun model CNN
model = Sequential([
    # Conv block 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    # Conv block 2
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    # Conv block 3
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    # Fully connected
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(15, activation='softmax')  # 15 kelas sayuran
])
```

## 🏆 Performance Metrics
| Metric              | Score  |
| ------------------- | ------ |
| Training Accuracy   | 97.99% |
| Validation Accuracy | 98.33% |
| Test Accuracy       | 96.97% |
| Test Loss           | 0.1913 |


## 📊 Training Visualization
<p align="center">
  <img src="https://github.com/user-attachments/assets/1ac31ce8-3e99-4a11-ae22-4c7eb29fe3f2" alt="Training Metrics">
</p>

<p align="center">
  <em>Figure 1. Accuracy and loss curves showing no overfitting</em>
</p>

## 🔍 Inference Examples
The project includes code to perform inference on random test images, showing both the predicted class and confidence score.

<div align="center">

### Direct Model Inference
<img src="https://github.com/user-attachments/assets/39db667c-188d-45f1-ba22-2b95abb1cd42" alt="Direct Model Inference Results" width="80%">

<p align="center">
  <em>Figure 2. Direct model inference</em>
</p>

### SavedModel Inference  
<img src="https://github.com/user-attachments/assets/537484ab-81ba-4d32-803e-bb036f22f483" alt="SavedModel Inference Results" width="80%">

<p align="center">
  <em>Figure 3. SavedModel inference</em>
</p>
</div>

## 🛠️ Requirements
To run this project, you'll need:
- Python 3.7+
- TensorFlow 2.x
- Other dependencies listed in `requirements.txt`

Install requirements with:
```
pip install -r requirements.txt
```

## ✅ Dicoding Submission Checklist
1. >1000 images dataset ✔️
2. Original dataset ✔️
3. Proper train/val/test split ✔️
4. Sequential + Conv2D model ✔️
5. >85% accuracy (Achieved 96.97%) ✔️
6. Accuracy/loss plots ✔️
7. SavedModel/TFLite/TFJS exports ✔️

## Future Improvements
1. Implement data augmentation to improve generalization
2. Experiment with transfer learning using pre-trained models
3. Add more vegetable classes
4. Develop a web/mobile interface for easy classification

## 👩‍💻 Author  
**Rebecca Olivia Javenka Br. Manurung**  
📧 Email: [rebeccaolivia1601@gmail.com](mailto:rebeccaolivia1601@gmail.com)  
👤 Dicoding ID: [rebeccaolivia](https://www.dicoding.com/users/rebeccaolivia)
