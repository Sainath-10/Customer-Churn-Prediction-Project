# Customer-Churn-Prediction-Project
Neural Network based Machine Learning project to predict bank customer churn

# 💳 Bank Customer Churn Prediction

A simple project that predicts whether a bank customer will continue using the service or leave (churn), based on their profile and account information.

---

## 📘 Overview

This project takes customer data and tries to understand patterns of those who exit the bank vs those who stay.
A basic Artificial Neural Network (ANN) model is trained on the dataset and a small prediction app is made to test new inputs.

The aim is to identify customers who are likely to leave so actions can be taken earlier.

---

## 📂 Dataset Details

* **Input**: Bank customer information (age, balance, credit score, geography, etc.)
* **Target**: Churn (Yes / No)
* **Goal**: Predict if a customer will leave the bank

---

## ⚙️ Tech Stack

* **Language**: Python
* **Libraries**: Pandas, NumPy, TensorFlow/Keras, Scikit-learn, Streamlit

---

## 🧩 Workflow

### 1. Importing Libraries

Basic dependencies for data handling and model building.

### 2. Data Preprocessing

* Encoding categorical fields
* Scaling numeric values
* Splitting data into train/test

### 3. Model

A simple ANN model trained to classify customers into churn or not churn.

### 4. Evaluation

Model performance checked on test data.

---

## 📊 Results

**Accuracy : 96.4%**