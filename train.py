
# ============================================================
# FOOD DELIVERY TIME PREDICTION - MODEL TRAINING
# ============================================================

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# 1. LOAD DATASET
# ------------------------------------------------------------

DATA_PATH = "Data/Food_Delivery_Times.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())


# 2. SEPARATE FEATURES AND TARGET
# ------------------------------------------------------------

X = df.drop(columns=["Order_ID", "Delivery_Time_min"])
y = df["Delivery_Time_min"]


# 3. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ------------------------------------------------------------

numerical_features = [
    "Distance_km",
    "Preparation_Time_min",
    "Courier_Experience_yrs"
]

categorical_features = [
    "Weather",
    "Traffic_Level",
    "Time_of_Day",
    "Vehicle_Type"
]


# 4. PREPROCESSING
# ------------------------------------------------------------

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        drop="first",
        handle_unknown="ignore"
    ))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])


# 5. TRAIN-TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)


# 6. DEFINE MODELS AND HYPERPARAMETERS
# ------------------------------------------------------------

models = {
    "Linear Regression": (
        LinearRegression(),
        {}
    ),

    "Ridge": (
        Ridge(),
        {"model__alpha": [0.1, 1, 10, 100]}
    ),

    "Lasso": (
        Lasso(max_iter=10000),
        {"model__alpha": [0.001, 0.01, 0.1, 1]}
    ),

    "Random Forest": (
        RandomForestRegressor(random_state=42),
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 10, 20]
        }
    ),

    "Gradient Boosting": (
        GradientBoostingRegressor(random_state=42),
        {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3]
        }
    ),

    "XGBoost": (
        XGBRegressor(
            random_state=42,
            objective="reg:squarederror",
            verbosity=0
        ),
        {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3, 5]
        }
    )
}


# 7. MODEL TRAINING AND HYPERPARAMETER TUNING
# ------------------------------------------------------------

results = []
best_models = {}

for name, (estimator, params) in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Test predictions
    y_pred = best_model.predict(X_test)

    # Evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "Best Parameters": grid_search.best_params_,
        "CV R2": grid_search.best_score_,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

    best_models[name] = best_model

    print("Best Parameters:", grid_search.best_params_)
    print("CV R2:", round(grid_search.best_score_, 4))
    print("Test R2:", round(r2, 4))
    print("MAE:", round(mae, 4))


# 8. MODEL COMPARISON
# ------------------------------------------------------------

comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    by="CV R2",
    ascending=False
).reset_index(drop=True)

print("\n========== MODEL COMPARISON ==========")
print(comparison.round(4).to_string(index=False))


# 9. IDENTIFY BEST MODELS
# ------------------------------------------------------------

# Best model based on cross-validation
cv_best_name = comparison.loc[0, "Model"]

# Best model based on test R2
test_best_name = comparison.loc[
    comparison["R2 Score"].idxmax(), "Model"
]

print("\n========== MODEL SELECTION ==========")

print("Best Model based on CV R2:", cv_best_name)
print("Best Model based on Test R2:", test_best_name)


# 10. SELECT MODEL FOR DEPLOYMENT
# ------------------------------------------------------------

# Use Linear Regression based on the reported
# test R2 and MAE results.
deployment_model_name = "Linear Regression"

best_model = best_models[deployment_model_name]

print("\nSelected Deployment Model:", deployment_model_name)

selected_result = comparison[
    comparison["Model"] == deployment_model_name
].iloc[0]

print("CV R2:", round(selected_result["CV R2"], 4))
print("Test R2:", round(selected_result["R2 Score"], 4))
print("MAE:", round(selected_result["MAE"], 4))


# 11. SAVE SELECTED MODEL
# ------------------------------------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(
    best_model,
    "models/delivery_time_model.pkl"
)

print("\nModel saved successfully!")
print("Model:", deployment_model_name)
print("Path: models/delivery_time_model.pkl")
