# 😴 Facial State Classification using LSTM

## 📌 Overview

This project presents a functional system for automatic classification of human facial states based on real-time video analysis.

The system distinguishes between three states:

- **Sleep** – closed eyes  
- **Tired** – yawning and fatigued eyes  
- **Normal** – neutral facial state  

The solution focuses on temporal facial dynamics rather than static image classification.

---

## 🧠 Model Architecture

Instead of processing individual frames independently, the system uses a custom **LSTM (Long Short-Term Memory)** neural network to capture sequential facial changes over time.

This allows the model to recognize gradual transitions, such as slow eye closing.

### Model Structure

The architecture includes three primary learning components:

1. **LSTM Layer**  
   - Analyzes timing and sequence of facial movements  
   - Captures temporal dependencies between frames  

2. **Dropout Layer**  
   - Reduces overfitting  
   - Improves generalization on unseen data  

3. **Dense Layers**  
   - Fully connected layers  
   - Final output layer uses **Softmax activation** for multi-class classification  

---

## 🔍 Features & Methodology

To reduce computational complexity, the system does not process raw pixel data.

Instead, it uses **MediaPipe Face Mesh** to extract geometric facial features.

For each frame, the following metrics are computed:

- **EAR (Eye Aspect Ratio)**  
  Measures the degree of eye openness and helps detect closed or fatigued eyes.

- **MAR (Mouth Aspect Ratio)**  
  Measures mouth openness and is used to detect yawning.

### Detection Stabilization

The system requires a short observation period before confirming a state.

It uses:

- A **sequence buffer**
- A **stabilization mechanism**

A prediction is confirmed only after several consecutive consistent outputs.  
This reduces accidental misclassifications and increases reliability.

---

## 📊 Performance

Despite using a limited and specialized dataset, the model achieves approximately:

**~75% classification accuracy**

This demonstrates that meaningful results can be achieved even with relatively small datasets when feature engineering and temporal modeling are applied correctly.

---

## 🔒 Data Privacy & Availability

The dataset consists of recordings from **17 individuals**.

Most recordings were collected manually from volunteers who were personally assured that their data would remain private and secure.

For this reason:

- ❌ Raw dataset is not included  
- ❌ Training videos are not included  

The repository provides the full methodology and custom model architecture instead.

---

## 🚀 Key Highlights

- Real-time facial state classification  
- Temporal modeling using LSTM  
- Lightweight geometric feature extraction (EAR & MAR)  
- Stabilized prediction mechanism  
- Privacy-conscious dataset handling  

---

## 👩‍💻 Author

**Yauheniya Drozd**
