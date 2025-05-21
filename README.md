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
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    BatchNormalization(),
    # ... (your full architecture)
    Dense(15, activation='softmax')
])

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    BatchNormalization(),
    # ... (your full architecture)
    Dense(15, activation='softmax')
])



# Vegetable Image Classification Project

## Overview
This project implements a Convolutional Neural Network (CNN) to classify images of 15 different types of vegetables. The model achieves high accuracy in distinguishing between various vegetables, making it potentially useful for applications in agriculture, food recognition, or inventory management systems.

## Dataset
The project uses the [Vegetable Image Dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset?select=Vegetable+Images) from Kaggle, which contains:
- 15 vegetable classes
- Pre-split train, validation, and test sets
- Images of various vegetables at different angles and lighting conditions

## Training
The model was trained with:
- Optimizer: Adam
- Loss function: Categorical Crossentropy
- Metrics: Accuracy
- Callbacks:
  - EarlyStopping (patience=5)
  - ReduceLROnPlateau (patience=3, factor=0.5)
- Epochs: 30 (early stopping triggered)

## Performance
- Training Accuracy: 97.99%
- Validation Accuracy: 98.33%
- Test Accuracy: 96.97%
- Test Loss: 0.1913

## Results Visualization
<p align="center">
  <img src="https://github.com/user-attachments/assets/1ac31ce8-3e99-4a11-ae22-4c7eb29fe3f2" alt="Training Metrics">
</p>

<p align="center">
  <em>Pict 1. Model accuracy and loss graph during training</em>
</p>

## Requirements
To run this project, you'll need:
- Python 3.7+
- TensorFlow 2.x
- Other dependencies listed in `requirements.txt`

## Inference Examples
The project includes code to perform inference on random test images, showing both the predicted class and confidence score.

<div align="center">

### Direct Model Inference
<img src="https://github.com/user-attachments/assets/39db667c-188d-45f1-ba22-2b95abb1cd42" alt="Direct Model Inference Results" width="80%">

<p align="center">
<em>Pict 2. Direct inference results from CNN models</em><br>
Prediction of vegetable class with confidence score > 95%
</p>

### SavedModel Inference  
<img src="https://github.com/user-attachments/assets/537484ab-81ba-4d32-803e-bb036f22f483" alt="SavedModel Inference Results" width="80%">

<p align="center">
<em>Pict 3. Inference using SavedModel format</em><br>
Validation of prediction consistency across model formats
</p>

</div>

## Future Improvements
1. Implement data augmentation to improve generalization
2. Experiment with transfer learning using pre-trained models
3. Add more vegetable classes
4. Develop a web/mobile interface for easy classification

## Author
Rebecca Olivia Javenka Br. Manurung
Email: [rebeccaolivia1601@gmail.com]
Dicoding ID: [rebeccaolivia]
