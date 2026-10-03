# 🏆 Sports Performance Analytics & Athlete Segmentation

### Machine Learning Based Athlete Analysis, Performance Prediction & Recommendations

#  https://athleteapp-gavdjwqeenuahmvvbweraz.streamlit.app/

## 📌 Project Overview

**Sports Performance Analytics & Athlete Segmentation** is a machine learning project designed to analyze athlete performance data, group athletes based on similar characteristics, predict their overall performance, estimate medal chances, and provide personalized training recommendations.

The system combines **K-Means Clustering, Random Forest Regression, Linear Regression, and a Recommendation System** into a single analytical pipeline.

> 🎯 **Goal:** Transform athlete performance data into meaningful groups, predictions, and actionable insights.

---

## 🚀 Key Features

* 🧑‍🤝‍🧑 **Athlete Segmentation** using K-Means Clustering
* 📊 **Overall Performance Prediction** using Random Forest Regression
* 🏅 **Medal Chance Prediction** using Linear Regression
* 💡 **Personalized Recommendations** for athletes below 75% predicted performance
* 📈 Interactive performance analytics
* 🔎 Athlete data exploration
* 📊 Model performance and feature importance visualization
* 🖥️ User-friendly Streamlit interface

---

## 🔄 Machine Learning Workflow

```text
Athlete Dataset
      ↓
Data Loading & Preprocessing
      ↓
Feature Scaling
      ↓
K-Means Clustering
      ↓
Athlete Segmentation
      ↓
Random Forest Regression
      ↓
Overall Performance Prediction
      ↓
Linear Regression
      ↓
Medal Chance Prediction
      ↓
Recommendation System
      ↓
Final Athlete Insights
```

---

## 🧠 Methodologies Used

| Method                       | Purpose                                                      |
| ---------------------------- | ------------------------------------------------------------ |
| **K-Means Clustering**       | Groups athletes with similar performance characteristics     |
| **Random Forest Regression** | Predicts overall athlete performance                         |
| **Linear Regression**        | Predicts medal chance                                        |
| **Recommendation System**    | Provides suggestions when predicted performance is below 75% |

---

## 📊 Athlete Parameters

The system analyzes multiple athlete-related features:

* Age
* Weight
* Training Experience
* Training Hours
* VO₂ Max
* Sprint Speed
* Reaction Time
* Strength Test Score
* Resting Heart Rate
* Sleep Hours
* Training Days
* Recovery Heart Rate

These parameters help the system understand different aspects of an athlete's performance.

---

## 🧩 Athlete Segmentation

K-Means Clustering is used to identify athletes with similar characteristics.

The system can classify athletes into groups such as:

```text
🏆 High Performer
⚡ Intermediate
🌱 Developing
```

This allows athletes with similar performance patterns to be analyzed together.

---

## 🤖 Performance Prediction

### Random Forest Regression

Random Forest Regression is used to predict:

**Overall Performance Score (%)**

The model uses athlete characteristics and the assigned cluster to generate the predicted performance.

> ℹ️ Feature scaling is used for K-Means clustering. Random Forest does not require feature scaling.

---

## 🏅 Medal Chance Prediction

### Linear Regression

Linear Regression estimates:

**Medal Chance (%)**

The prediction uses selected athlete information along with the predicted overall performance and athlete cluster.

---

## 💡 Recommendation System

The project includes a simple recommendation system based on the predicted overall performance.

### If performance is below 75%

The system analyzes the athlete's parameters and provides suggestions related to areas such as:

* 🏃 Training consistency
* ❤️ Aerobic fitness
* 💪 Strength
* ⚡ Speed
* 🧠 Reaction training
* 😴 Sleep and recovery
* 🫀 Cardiovascular fitness
* 🏋️ Conditioning

### If performance is 75% or above

```text
Good performance - maintain current training.
```

---

## 🖥️ Application Interface

The Streamlit application contains:

### 🏠 Dashboard

Provides an overview of the athlete dataset, clusters, model performance, and key statistics.

### 🧑 Athlete Prediction

Enter athlete details and receive:

* Performance Group
* Overall Performance %
* Medal Chance %
* Cluster
* Recommendation

### 🔎 Athlete Explorer

Explore athlete records and filter athletes based on their performance groups.

### 📈 Model Insights

View:

* K-Means clustering analysis
* Feature importance
* Model evaluation metrics
* Methodology overview

---

## 🛠️ Technologies Used

### Programming Language

🐍 **Python**

### Libraries

* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Streamlit

### Machine Learning

* K-Means Clustering
* Random Forest Regression
* Linear Regression

---

## 📁 Project Structure

```text
Sports_Performance_Analytics_Athlete_Segmentation/
│
├── app.py
├── athletes_analytics.csv
├── requirements.txt
├── README.md
│
└── models/
    └── README.txt
```

---

## ⚙️ Installation & Setup

### 1️⃣ Open Command Prompt

Navigate to the project folder:

```bash
cd Sports_Performance_Analytics_Athlete_Segmentation
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
```

### 3️⃣ Activate the Environment

```bash
venv\Scripts\activate
```

### 4️⃣ Install Required Libraries

```bash
pip install -r requirements.txt
```

### 5️⃣ Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📋 Example Prediction Flow

```text
Enter Athlete Details
        ↓
K-Means identifies athlete group
        ↓
Random Forest predicts performance
        ↓
Linear Regression predicts medal chance
        ↓
System checks performance threshold
        ↓
Personalized recommendation
```

---

## 🎯 Project Objective

The main objective of this project is to demonstrate how machine learning can be applied to sports performance data to:

**Analyze → Segment → Predict → Recommend**

This provides a simple data-driven approach for understanding athlete performance.

---

## 📌 Dataset Note

The dataset used in this academic project is a **synthetic/demo dataset** created for demonstrating the machine learning workflow.

It is intended for **educational and project demonstration purposes**, not for making real-world athlete selection or medical decisions.

---

## 🔮 Future Enhancements

Possible future improvements include:

* 📱 Mobile-friendly application
* 📊 Larger real-world athlete datasets
* 🧠 Advanced machine learning models
* 📈 Historical performance tracking
* 👤 Athlete profile management
* 📅 Training-plan generation
* ☁️ Cloud deployment
* 🔐 Secure user authentication

---

## 👨‍💻 Project

### **Sports Performance Analytics & Athlete Segmentation**

**Built using Python & Machine Learning**

> *Analyze athlete data. Discover patterns. Predict performance. Generate insights.*
