
import os
import joblib
import pandas as pd
import numpy as np

# Model path
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "delivery_time_model.pkl"
)

# Sample input data
sample_data = pd.DataFrame([{
    "Distance_km": 5.0,
    "Weather": "Clear",
    "Traffic_Level": "Medium",
    "Time_of_Day": "Afternoon",
    "Vehicle_Type": "Bike",
    "Preparation_Time_min": 15,
    "Courier_Experience_yrs": 2.0
}])


def test_model_file_exists():
    assert os.path.exists(MODEL_PATH)


def test_model_loading():
    model = joblib.load(MODEL_PATH)
    assert model is not None


def test_model_prediction():
    model = joblib.load(MODEL_PATH)

    prediction = model.predict(sample_data)

    assert len(prediction) == 1
    assert np.isfinite(prediction[0])


def test_prediction_is_positive():
    model = joblib.load(MODEL_PATH)

    prediction = model.predict(sample_data)

    assert prediction[0] > 0
