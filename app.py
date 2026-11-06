
import streamlit as st
import pandas as pd
import joblib
import numpy as np

from pathlib import Path
from datetime import datetime


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="FoodTime AI | Delivery Intelligence",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# 2. LOAD TRAINED MODEL
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "delivery_time_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("Trained model not found. Please run train.py first.")
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()


# ==========================================================
# 3. SESSION STATE
# ==========================================================

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "history" not in st.session_state:
    st.session_state.history = []

if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None


def go_to_prediction():
    st.session_state.page = "Prediction"


def clear_history():
    st.session_state.history = []
    st.session_state.latest_prediction = None


# ==========================================================
# 4. CUSTOM CSS
# ==========================================================

st.html("""
<style>

/* MAIN APPLICATION */

.stApp {
    background-color: #F4F7FC;
    color: #172033;
    font-family: 'Inter', sans-serif;
}

/* TOP NAVBAR */

[data-testid="stHeader"] {
    background: rgba(244, 247, 252, 0.98) !important;
    border-bottom: 1px solid #E5EAF2;
}

[data-testid="stToolbar"] {
    background: transparent !important;
}

/* PAGE SPACING */

.block-container {
    padding-top: 5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1450px;
}

/* HIDE DEFAULT BRANDING */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #111827 0%,
        #1E293B 100%
    );
}

[data-testid="stSidebar"] * {
    color: #F8FAFC;
}

[data-testid="stSidebar"] hr {
    border-color: #475569;
}

/* MAIN HEADINGS */

.main-title {
    width: 100%;
    text-align: center;
    font-size: 36px;
    font-weight: 800;
    color: #172033;
    letter-spacing: -1px;
    margin: 10px 0 12px 0;
}

.subtitle {
    width: 100%;
    text-align: center;
    color: #64748B;
    font-size: 16px;
    margin-bottom: 32px;
}

.section-heading {
    width: 100%;
    text-align: center;
    font-size: 22px;
    font-weight: 750;
    color: #172033;
    margin-top: 30px;
    margin-bottom: 20px;
}

/* HERO BANNER */

.hero {
    text-align: center;
    background: linear-gradient(
        120deg,
        #1D4ED8,
        #6D28D9
    );
    padding: 38px;
    border-radius: 18px;
    color: white;
    margin-bottom: 28px;
    box-shadow: 0 8px 25px rgba(37, 99, 235, 0.16);
}

.hero h2 {
    color: white;
    font-size: 30px;
    font-weight: 800;
    margin-bottom: 12px;
}

.hero p {
    color: #E0E7FF;
    font-size: 16px;
    line-height: 1.8;
}

/* METRIC CARDS */

.metric-card {
    background: white;
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #E5EAF2;
    box-shadow: 0 4px 15px rgba(15, 23, 42, 0.04);
    transition: all 0.25s ease;
    min-height: 115px;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
}

.metric-label {
    color: #64748B;
    font-size: 13px;
    font-weight: 600;
}

.metric-value {
    color: #2563EB;
    font-size: 27px;
    font-weight: 800;
    margin-top: 8px;
}

/* PREDICTION RESULT */

.result-card {
    background: linear-gradient(
        120deg,
        #ECFDF5,
        #EFF6FF
    );
    border: 1px solid #BBF7D0;
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    margin-top: 25px;
    margin-bottom: 20px;
}

.result-label {
    color: #475569;
    font-size: 15px;
    font-weight: 600;
    letter-spacing: 1px;
}

.result-value {
    color: #047857;
    font-size: 46px;
    font-weight: 850;
    margin: 12px 0;
}

.result-note {
    color: #64748B;
    font-size: 13px;
}

/* INPUT FIELDS */

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    border-radius: 10px;
}

/* BUTTONS */

.stButton > button,
.stFormSubmitButton > button,
.stDownloadButton > button {
    width: 100%;
    min-height: 45px;
    border: none;
    border-radius: 10px;
    background: linear-gradient(
        110deg,
        #2563EB,
        #7C3AED
    );
    color: white;
    font-weight: 700;
    transition: all 0.25s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 7px 18px rgba(37, 99, 235, 0.25);
    color: white;
}

/* FORM */

[data-testid="stForm"] {
    background: white;
    padding: 25px;
    border: 1px solid #E5EAF2;
    border-radius: 16px;
}

/* DATA TABLE */

[data-testid="stDataFrame"] {
    border-radius: 12px;
}

/* RESPONSIVE DESIGN */

@media (max-width: 768px) {

    .main-title {
        font-size: 26px;
    }

    .section-heading {
        font-size: 19px;
    }

    .hero {
        padding: 24px;
    }

    .hero h2 {
        font-size: 22px;
    }

    .result-value {
        font-size: 35px;
    }

}

</style>
""")


# ==========================================================
# 5. SIDEBAR NAVIGATION
# ==========================================================

with st.sidebar:

    st.markdown("# 🚴 FoodTime AI")
    st.caption("DELIVERY INTELLIGENCE PLATFORM")

    st.divider()

    st.markdown("### MENU")

    st.radio(
        "Navigation",
        ["Dashboard", "Prediction", "Analytics"],
        key="page",
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### 🤖 MODEL INFORMATION")

    st.write("**Algorithm:** Linear Regression")
    st.write("**CV R²:** 75.12%")
    st.write("**Test R²:** 82.12%")
    st.write("**MAE:** 6.06 minutes")

    st.divider()

    st.markdown("### 📊 PROJECT STATUS")
    st.success("Model Loaded")

    st.caption("FoodTime AI")
    st.caption("Machine Learning Project")


# ==========================================================
# 6. DASHBOARD PAGE
# ==========================================================

if st.session_state.page == "Dashboard":

    st.html("""
    <div class="main-title">
        Delivery Intelligence Dashboard
    </div>

    <div class="subtitle">
        Monitor and predict food delivery performance using Machine Learning.
    </div>
    """)

    # HERO SECTION

    st.html("""
    <div class="hero">
        <h2>Welcome to FoodTime AI 👋</h2>

        <p>
            Predict estimated food delivery duration using distance,
            weather, traffic, vehicle and courier information.
        </p>

        <p>
            <b>Smarter Predictions. Better Delivery Planning.</b>
        </p>
    </div>
    """)

    # METRICS

    total_predictions = len(st.session_state.history)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.html(f"""
        <div class="metric-card">
            <div class="metric-label">TOTAL PREDICTIONS</div>
            <div class="metric-value">{total_predictions}</div>
        </div>
        """)

    with col2:
        st.html("""
        <div class="metric-card">
            <div class="metric-label">MODEL TEST R²</div>
            <div class="metric-value">82.12%</div>
        </div>
        """)

    with col3:
        st.html("""
        <div class="metric-card">
            <div class="metric-label">MODEL MAE</div>
            <div class="metric-value">6.06 min</div>
        </div>
        """)

    # HOW IT WORKS

    st.html("""
    <div class="section-heading">
        How FoodTime AI Works
    </div>
    """)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.info(
            "**01 | Enter Details**\n\n"
            "Provide delivery and courier information."
        )

    with c2:
        st.info(
            "**02 | ML Prediction**\n\n"
            "The trained model processes your inputs."
        )

    with c3:
        st.info(
            "**03 | Get Estimate**\n\n"
            "View your estimated delivery duration."
        )

    st.divider()

    st.html("""
    <div class="section-heading">
        Get Started
    </div>
    """)

    st.button(
        "🚀 Start New Prediction",
        on_click=go_to_prediction,
        use_container_width=True
    )

    # RECENT PREDICTIONS

    if st.session_state.history:

        st.html("""
        <div class="section-heading">
            Recent Predictions
        </div>
        """)

        recent_df = pd.DataFrame(
            st.session_state.history[-5:]
        )

        st.dataframe(
            recent_df,
            use_container_width=True,
            hide_index=True
        )


# ==========================================================
# 7. PREDICTION PAGE
# ==========================================================

elif st.session_state.page == "Prediction":

    st.html("""
    <div class="main-title">
        Delivery Time Prediction
    </div>

    <div class="subtitle">
        Enter delivery details to estimate delivery duration.
    </div>

    <div class="section-heading">
        📦 Delivery Information
    </div>
    """)

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            distance = st.number_input(
                "Distance (km)",
                min_value=0.1,
                max_value=100.0,
                value=5.0,
                step=0.5
            )

            weather = st.selectbox(
                "Weather",
                ["Clear", "Windy", "Foggy", "Rainy", "Snowy"]
            )

            traffic = st.selectbox(
                "Traffic Level",
                ["Low", "Medium", "High"]
            )

            time_of_day = st.selectbox(
                "Time of Day",
                ["Morning", "Afternoon", "Evening", "Night"]
            )

        with col2:

            vehicle = st.selectbox(
                "Vehicle Type",
                ["Scooter", "Bike", "Car"]
            )

            preparation = st.number_input(
                "Preparation Time (minutes)",
                min_value=0,
                max_value=300,
                value=15
            )

            experience = st.number_input(
                "Courier Experience (years)",
                min_value=0.0,
                max_value=50.0,
                value=2.0,
                step=0.5
            )

        submitted = st.form_submit_button(
            "🚀 Predict Delivery Time",
            use_container_width=True
        )

    # PREDICTION EXECUTION

    if submitted:

        input_data = pd.DataFrame([{
            "Distance_km": distance,
            "Weather": weather,
            "Traffic_Level": traffic,
            "Time_of_Day": time_of_day,
            "Vehicle_Type": vehicle,
            "Preparation_Time_min": preparation,
            "Courier_Experience_yrs": experience
        }])

        try:

            prediction = float(
                model.predict(input_data)[0]
            )

            if not np.isfinite(prediction):
                raise ValueError(
                    "Model returned an invalid prediction."
                )

            if prediction <= 0:

                st.warning(
                    "The model returned a non-positive estimate. "
                    "Please review the input values."
                )

            else:

                st.session_state.latest_prediction = {
                    "prediction": round(prediction, 2),
                    "distance": distance,
                    "weather": weather,
                    "traffic": traffic,
                    "vehicle": vehicle
                }

                st.session_state.history.append({

                    "Date & Time": datetime.now().strftime(
                        "%d-%m-%Y %H:%M:%S"
                    ),

                    "Distance (km)": distance,
                    "Weather": weather,
                    "Traffic": traffic,
                    "Vehicle": vehicle,

                    "Predicted Time (min)": round(
                        prediction, 2
                    )
                })

                st.success(
                    "Prediction generated successfully!"
                )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # DISPLAY LATEST RESULT

    if st.session_state.latest_prediction is not None:

        result = st.session_state.latest_prediction

        st.html(f"""
        <div class="result-card">

            <div class="result-label">
                ESTIMATED DELIVERY TIME
            </div>

            <div class="result-value">
                {result['prediction']:.2f} min
            </div>

            <div class="result-note">
                This is a machine learning estimate,
                not a guaranteed delivery time.
            </div>

        </div>
        """)

        st.markdown("### Prediction Summary")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Distance",
            f"{result['distance']} km"
        )

        c2.metric(
            "Weather",
            result["weather"]
        )

        c3.metric(
            "Traffic",
            result["traffic"]
        )

    else:

        st.info(
            "Fill in the delivery details and click "
            "'Predict Delivery Time' to get an estimate."
        )


# ==========================================================
# 8. ANALYTICS PAGE
# ==========================================================

elif st.session_state.page == "Analytics":

    st.html("""
    <div class="main-title">
        Prediction Analytics
    </div>

    <div class="subtitle">
        Analyze predictions generated during your current session.
    </div>
    """)

    if len(st.session_state.history) == 0:

        st.info(
            "No prediction history available. "
            "Generate your first prediction to view analytics."
        )

        st.button(
            "🚀 Go to Prediction",
            on_click=go_to_prediction
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        # ANALYTICS CALCULATIONS

        avg_time = history_df[
            "Predicted Time (min)"
        ].mean()

        max_time = history_df[
            "Predicted Time (min)"
        ].max()

        min_time = history_df[
            "Predicted Time (min)"
        ].min()

        # KEY METRICS

        st.html("""
        <div class="section-heading">
            Key Metrics
        </div>
        """)

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Average Delivery Time",
            f"{avg_time:.2f} min"
        )

        col2.metric(
            "Maximum Estimate",
            f"{max_time:.2f} min"
        )

        col3.metric(
            "Minimum Estimate",
            f"{min_time:.2f} min"
        )

        # PREDICTION TREND

        st.html("""
        <div class="section-heading">
            Prediction Trend
        </div>
        """)

        chart_data = history_df[
            ["Predicted Time (min)"]
        ].copy()

        chart_data.index = range(
            1, len(chart_data) + 1
        )

        chart_data.index.name = "Prediction Number"

        st.line_chart(
            chart_data,
            use_container_width=True
        )

        # VEHICLE ANALYSIS

        st.html("""
        <div class="section-heading">
            Vehicle Analysis
        </div>
        """)

        vehicle_analysis = (
            history_df.groupby("Vehicle")[
                "Predicted Time (min)"
            ]
            .mean()
            .reset_index()
        )

        vehicle_analysis.columns = [
            "Vehicle",
            "Average Predicted Time"
        ]

        st.bar_chart(
            vehicle_analysis,
            x="Vehicle",
            y="Average Predicted Time"
        )

        # HISTORY TABLE

        st.html("""
        <div class="section-heading">
            Prediction History
        </div>
        """)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        # DOWNLOAD CSV

        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download Prediction History",
            data=csv_data,
            file_name="food_delivery_prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.divider()

        # CLEAR HISTORY

        st.button(
            "🗑️ Clear Prediction History",
            on_click=clear_history,
            use_container_width=True
        )


# ==========================================================
# 9. FOOTER
# ==========================================================

st.divider()

st.html("""
<div style="
    text-align: center;
    color: #94A3B8;
    padding: 20px;
    font-size: 13px;
">
    <b>FoodTime AI</b> | Food Delivery Time Prediction
    <br>
    Built with Python, Scikit-learn and Streamlit
</div>
""")
