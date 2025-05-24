# Vegetable Image Classification Project
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12.0-orange)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-96.00%25-brightgreen)]()


## 📌 Project Overview
A deep learning model achieving **96.00% test accuracy** in classifying 15 vegetable types, developed for Dicoding's "Belajar Fundamental Deep Learning" certification. Implements a CNN with optimization techniques and multi-platform deployment.

## 🥦 Dataset
**Kaggle Vegetable Image Dataset**  
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue)](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset)

| Split        | Images | Classes |
|--------------|--------|---------|
| **Training** | 14700 | 15      |
| **Validation** | 3150  | 15      |
| **Test**     | 3150  | 15      |

**Classes:** 'Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato'

## 🧠 Model Architecture
```
model = tf.keras.Sequential([
    # Blok Conv 1
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.BatchNormalization(),

    # Blok Conv 2
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.BatchNormalization(),

    # Blok Conv 3
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.BatchNormalization(),

    # Fully Connected
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(len(class_names), activation='softmax')  # 15 kelas
])
```

## 🏆 Performance Metrics
| Metric              | Score  |
|---------------------|--------|
| Training Accuracy   | 98.52% |
| Validation Accuracy | 96.83% |
| Test Accuracy       | 96.00% |
| Test Loss           | 0.2951 |

## 📊 Training Visualization
<p align="center">
  <img src="https://github.com/user-attachments/assets/e8003f3d-af99-4d34-b02d-d214ee6588a4" alt="Training Metrics">
</p>

<p align="center">
  <em>Figure 1. Accuracy and loss curves showing no overfitting</em>
</p>

## 🔍 Inference Examples
<p align="center">
  <img src="https://github.com/user-attachments/assets/938fbbe6-aaa8-424b-94e6-3eba4071bf01" alt="Training Metrics">
</p>

<p align="center">
  <em>Figure 2. SavedModel inference</em>
</p>

## 🛠️ Requirements
To run this project, you'll need:
- Python 3.7+
- TensorFlow 2.x
- Other dependencies listed in [`requirements.txt`](https://github.com/rebeccaaolivia/vegetable-vision-classifier/blob/main/requirements.txt)

Install requirements with:
```
pip install -r requirements.txt
```

## ✅ Dicoding Submission Checklist
1. \>1000 images dataset ✔️
2. Original dataset ✔️
3. Proper train/val/test split ✔️
4. Sequential + Conv2D model ✔️
5. \>85% accuracy (Achieved 96.97%) ✔️
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
