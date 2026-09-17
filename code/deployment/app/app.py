"""
Streamlit Web Application for Wine Quality Prediction.

Interacts with the FastAPI model service running in a separate container.
Provides:
- Interactive sliders and numeric inputs for physicochemical wine features
- Quick sample presets (High Quality vs Standard Quality)
- Real-time API connectivity and model health check
- Prediction button and detailed probability visualizer
"""

import os
import requests
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Wine Quality AI Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API URL from environment variable or default
API_URL = os.getenv("API_URL", "http://api:8000").rstrip("/")

# Sample presets
PRESETS = {
    "Standard Quality Sample (Rating ~5)": {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.70,
        "citric_acid": 0.00,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "ph": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    },
    "High Quality Sample (Rating ~7-8)": {
        "fixed_acidity": 8.5,
        "volatile_acidity": 0.28,
        "citric_acid": 0.45,
        "residual_sugar": 2.3,
        "chlorides": 0.055,
        "free_sulfur_dioxide": 18.0,
        "total_sulfur_dioxide": 45.0,
        "density": 0.9942,
        "ph": 3.32,
        "sulphates": 0.82,
        "alcohol": 12.8
    }
}


def check_api_health(url: str):
    """Check if the FastAPI backend is reachable and ready."""
    try:
        r = requests.get(f"{url}/health", timeout=3)
        if r.status_code == 200:
            return True, r.json()
        return False, {"error": f"Status {r.status_code}"}
    except Exception as e:
        return False, {"error": str(e)}


def fetch_api_metadata(url: str):
    """Retrieve model training metrics from FastAPI service."""
    try:
        r = requests.get(f"{url}/metadata", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}


# ---------------- Sidebar ----------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?w=500&auto=format&fit=crop&q=60", use_container_width=True)
    st.title("⚙️ Service Status")

    api_endpoint = st.text_input("FastAPI Endpoint", value=API_URL)
    is_healthy, health_data = check_api_health(api_endpoint)

    if is_healthy:
        st.success("🟢 API Connected & Ready")
        st.caption(f"**Run ID:** `{health_data.get('run_id', 'N/A')[:8]}...`")
        st.caption(f"**Loaded At:** {health_data.get('model_loaded_at', 'N/A')}")
    else:
        st.error("🔴 API Offline")
        st.caption(f"Error: {health_data.get('error', 'Cannot connect')}")
        st.info("Ensure the FastAPI container is running on " + api_endpoint)

    st.markdown("---")
    st.subheader("📊 Model Metrics")
    meta = fetch_api_metadata(api_endpoint)
    if meta and "metrics" in meta:
        metrics = meta["metrics"]
        st.metric("Test Accuracy", f"{metrics.get('accuracy', 0):.1%}")
        st.metric("F1 Score", f"{metrics.get('f1_score', 0):.3f}")
        st.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.3f}")
    else:
        st.caption("Connect to API to view live model metrics.")

    st.markdown("---")
    st.caption("PMLDL Assignment 1 • MLOps Automated Deployment")


# ---------------- Main Page ----------------
st.title("🍷 Wine Quality Predictor")
st.markdown(
    "Predict whether a red wine is **Good Quality (Rating ≥ 6)** or **Standard Quality (Rating < 6)** "
    "based on its chemical and physical attributes."
)

# Preset Selector
preset_col1, preset_col2 = st.columns([3, 1])
with preset_col1:
    selected_preset = st.selectbox("Quick Presets (select to auto-fill inputs):", list(PRESETS.keys()))
default_vals = PRESETS[selected_preset]

st.markdown("### 🧪 Physicochemical Features")

# 3-column input layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### 🍋 Acidity & pH")
    fixed_acidity = st.slider("Fixed Acidity (g/dm³)", 4.0, 16.0, float(default_vals["fixed_acidity"]), 0.1)
    volatile_acidity = st.slider("Volatile Acidity (g/dm³)", 0.1, 1.6, float(default_vals["volatile_acidity"]), 0.01)
    citric_acid = st.slider("Citric Acid (g/dm³)", 0.0, 1.0, float(default_vals["citric_acid"]), 0.01)
    ph = st.slider("pH Value", 2.5, 4.2, float(default_vals["ph"]), 0.01)

with col2:
    st.markdown("##### 🍬 Sugar, Salt & Density")
    residual_sugar = st.slider("Residual Sugar (g/dm³)", 0.5, 16.0, float(default_vals["residual_sugar"]), 0.1)
    chlorides = st.slider("Chlorides (g/dm³)", 0.01, 0.4, float(default_vals["chlorides"]), 0.001)
    density = st.slider("Density (g/cm³)", 0.9900, 1.0040, float(default_vals["density"]), 0.0002, format="%.4f")

with col3:
    st.markdown("##### 💨 Sulfur & Alcohol")
    free_so2 = st.slider("Free Sulfur Dioxide (mg/dm³)", 1.0, 72.0, float(default_vals["free_sulfur_dioxide"]), 1.0)
    total_so2 = st.slider("Total Sulfur Dioxide (mg/dm³)", 6.0, 280.0, float(default_vals["total_sulfur_dioxide"]), 1.0)
    sulphates = st.slider("Sulphates (g/dm³)", 0.3, 2.0, float(default_vals["sulphates"]), 0.02)
    alcohol = st.slider("Alcohol (% vol)", 8.0, 15.0, float(default_vals["alcohol"]), 0.1)

st.markdown("---")

# Predict Button
predict_btn = st.button("🔮 Make Prediction", type="primary", use_container_width=True)

if predict_btn:
    payload = {
        "fixed_acidity": fixed_acidity,
        "volatile_acidity": volatile_acidity,
        "citric_acid": citric_acid,
        "residual_sugar": residual_sugar,
        "chlorides": chlorides,
        "free_sulfur_dioxide": free_so2,
        "total_sulfur_dioxide": total_so2,
        "density": density,
        "ph": ph,
        "sulphates": sulphates,
        "alcohol": alcohol
    }

    with st.spinner("Requesting inference from FastAPI backend..."):
        try:
            response = requests.post(f"{api_endpoint}/predict", json=payload, timeout=5)
            if response.status_code == 200:
                result = response.json()
                is_good = result.get("prediction") == 1
                prob_good = result.get("probability_good", 0.0)
                prob_standard = result.get("probability_standard", 0.0)

                st.subheader("🎯 Prediction Result")
                res_col1, res_col2 = st.columns([2, 1])

                with res_col1:
                    if is_good:
                        st.success(f"### 🌟 **{result.get('prediction_label', 'Good Quality')}**")
                        st.write(f"The model predicts with **{prob_good:.1%} confidence** that this wine is of superior quality.")
                    else:
                        st.warning(f"### 🍷 **{result.get('prediction_label', 'Standard Quality')}**")
                        st.write(f"The model predicts with **{prob_standard:.1%} confidence** that this is a standard table wine.")

                    st.markdown("**Good Quality Confidence:**")
                    st.progress(prob_good)

                with res_col2:
                    st.metric("Good Quality Probability", f"{prob_good:.1%}")
                    st.metric("Standard Quality Probability", f"{prob_standard:.1%}")

                with st.expander("🔍 View Raw API Response"):
                    st.json(result)

            else:
                st.error(f"Prediction failed with HTTP {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as req_err:
            st.error(f"Cannot connect to API at `{api_endpoint}`: {req_err}")
            st.info("Tip: If running locally without Docker Compose, set API Endpoint to `http://localhost:8000` in the sidebar.")
