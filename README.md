# Sports Performance Analytics & Athlete Segmentation

## Project Overview
This project analyzes athlete performance and segments athletes using machine learning.

## ML Pipeline
1. K-Means Clustering -> Athlete Segmentation
2. Random Forest Regression -> Overall Performance Prediction
3. Linear Regression -> Medal Chance Prediction
4. Recommendation System -> Training Suggestions

## Main Features
- Athlete performance dashboard
- K-Means athlete segmentation
- New athlete prediction
- Overall performance prediction
- Medal chance prediction
- Athlete explorer and filtering
- Random Forest feature importance
- Silhouette-score analysis
- Model evaluation using MAE, RMSE and R²
- Recommendation system for athletes below 75% predicted performance

## Project Structure

Sports_Performance_Analytics_Athlete_Segmentation/
│
├── app.py
├── athletes_analytics.csv
├── requirements.txt
├── README.md
└── models/
    └── README.txt

## Requirements
- Python 3.9 or newer
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Run in Windows CMD

Open CMD inside this project folder and run:

    python -m venv venv

Then activate it:

    venv\Scripts\activate

Install the libraries:

    pip install -r requirements.txt

Run the application:

    streamlit run app.py

The Streamlit application will open in your browser.

## Dataset
The project uses `athletes_analytics.csv`.

The dataset contains athlete-related measurements such as:
- Age
- Weight
- Training experience
- Training hours
- VO2 Max
- Sprint speed
- Reaction time
- Strength
- Resting heart rate
- Sleep
- Training days
- Recovery heart rate

It also contains performance and medal-chance target columns.

## Important Implementation Note
Scaling is used for K-Means because distance-based clustering is affected by feature scales.

Random Forest does not require feature scaling, so the Random Forest prediction section uses the original feature values.

## Academic Project Title
Sports Performance Analytics & Athlete Segmentation

Dataset type: Synthetic/Demo
