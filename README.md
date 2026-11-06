# Food Delivery Time Prediction using Machine Learning

## Project Overview

Food Delivery Time Prediction is a machine learning project that estimates the time required to deliver a food order based on factors such as delivery distance, weather conditions, traffic level, time of day, vehicle type, food preparation time, and courier experience.

The project uses regression algorithms and a machine learning pipeline to predict delivery time. An interactive web application built with Streamlit allows users to enter delivery details and receive predictions.

## Project Objectives

- Predict food delivery time using machine learning.
- Analyze the impact of traffic, weather, and delivery distance.
- Compare multiple regression algorithms.
- Apply preprocessing techniques using Scikit-learn pipelines.
- Build an interactive prediction application using Streamlit.

## Dataset Information

- **Dataset:** Food_Delivery_Times.csv
- **Total Records:** 1,000
- **Total Columns:** 9
- **Target Variable:** `Delivery_Time_min`

### Features

| Feature | Description |
|---|---|
| Order_ID | Unique order identifier |
| Distance_km | Delivery distance in kilometers |
| Weather | Weather conditions |
| Traffic_Level | Traffic congestion level |
| Time_of_Day | Time when the order is placed |
| Vehicle_Type | Delivery vehicle |
| Preparation_Time_min | Food preparation time |
| Courier_Experience_yrs | Courier experience in years |
| Delivery_Time_min | Actual delivery time in minutes |

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Streamlit
- Joblib
- Jupyter Notebook

## Machine Learning Workflow

1. Data collection and loading
2. Exploratory Data Analysis (EDA)
3. Data cleaning and missing-value treatment
4. Feature selection
5. Categorical feature encoding
6. Numerical feature scaling
7. Train-test splitting
8. Model training and comparison
9. Cross-validation
10. Hyperparameter tuning
11. Model evaluation
12. Model serialization
13. Streamlit application development
14. Automated model testing

## Data Preprocessing

A Scikit-learn preprocessing pipeline was developed.

**Numerical features:**
- Missing values handled using median imputation.
- Features scaled using StandardScaler.

**Categorical features:**
- Missing values handled using most-frequent imputation.
- Features encoded using OneHotEncoder.

The preprocessing pipeline was combined with machine learning models to ensure consistent transformations during training and prediction.

## Models Implemented

The following regression algorithms were evaluated:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Random Forest Regressor
5. Gradient Boosting Regressor
6. XGBoost Regressor

## Model Performance

The following results were obtained during model evaluation.

| Model | CV R² | Test R² | MAE |
|---|---:|---:|---:|
| Linear Regression | 0.7512 | 0.8212 | 6.0634 |
| Ridge Regression | 0.7513 | 0.8207 | 6.0745 |
| Lasso Regression | 0.7513 | 0.8206 | 6.0766 |
| Gradient Boosting | 0.7266 | 0.7848 | 6.8437 |
| XGBoost | 0.7221 | 0.7860 | 6.8298 |
| Random Forest | 0.6871 | 0.7674 | 7.0836 |

**Selected Model:** Linear Regression

**Test R² Score:** 0.8212

**Test MAE:** 6.0634 minutes

The selected model achieved an R² score of approximately 82.12% on the test dataset.

## Project Structure

```text
Food Time Prediction/
│
├── Data/
│   └── Food_Delivery_Times.csv
│
├── models/
│   └── delivery_time_model.pkl
│
├── tests/
│   └── test_model.py
│
├── app.py
├── train.py
├── project.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/food-delivery-time-prediction.git
```

### 2. Navigate to the Project Directory

```bash
cd food-delivery-time-prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

## Run Model Training

To train the machine learning models and generate the model artifact, run:

```bash
python train.py
```

## Run Automated Tests

To execute the model tests:

```bash
python -m pytest tests/ -v
```

## Streamlit Application Features

- Interactive dashboard
- Food delivery time prediction
- Input-based prediction system
- Prediction history
- Analytics and visualizations
- Downloadable prediction history

## Future Enhancements

- Integrate real-time weather and traffic data.
- Train models using larger real-world delivery datasets.
- Explore advanced ensemble and boosting algorithms.
- Deploy the application with cloud infrastructure.
- Develop an API for integration with external applications.

## Author

**Jangili Ch M Srinivasa vara Prasad**

B.Tech – Computer Science and Engineering

GitHub: https://github.com/chiru2710

LinkedIn: https://www.linkedin.com/in/jangili-ch-m-srinivasa-vara-prasad-213097315/

---

*This project was developed as an end-to-end machine learning application demonstrating data preprocessing, regression modeling, model evaluation, deployment, and automated testing.*