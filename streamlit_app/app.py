"""Machine Guard web page: move the sliders, get a failure risk from the live API.

This page is prepared for you (Day 3). It does not load the model itself: it sends the readings
to the Machine Guard API and shows the answer, exactly like any other client of the API would.

Settings (never in Git):
    API_URL   the API's base URL, for example https://machine-guard-xxxx.onrender.com
    API_KEY   the same key the API expects in X-API-Key

On Streamlit Community Cloud: App settings -> Secrets, in TOML:
    API_URL = "https://machine-guard-xxxx.onrender.com"
    API_KEY = "paste-the-key-here"

Locally, in the codespace (no secrets file needed):
    API_URL=http://localhost:8000 API_KEY=$API_KEY streamlit run streamlit_app/app.py
"""
import os

import requests
import streamlit as st

# These mirror machine_guard.features.VALID_RANGES. They are copied, not imported, because
# Streamlit Cloud installs only streamlit_app/requirements.txt (no scikit-learn, no machine_guard).
# If a value is out of range anyway, the API answers 422 and the page shows why.
SLIDERS = {
    # column: (label, min, max, default, step)
    "air_temp_k": ("Air temperature (K)", 290.0, 310.0, 298.1, 0.1),
    "process_temp_k": ("Process temperature (K)", 300.0, 320.0, 308.6, 0.1),
    "rotational_speed_rpm": ("Rotational speed (rpm)", 1000, 3000, 1551, 10),
    "torque_nm": ("Torque (Nm)", 0.0, 80.0, 42.8, 0.1),
    "tool_wear_min": ("Tool wear (min)", 0, 260, 0, 1),
}
TIMEOUT_S = 90  # a sleeping free Render service can take about a minute to wake up


def setting(name: str) -> str:
    """Read a setting from an environment variable, else from Streamlit secrets."""
    if os.environ.get(name):
        return os.environ[name]
    # load_if_toml_exists() is False when there is no secrets file (normal when running locally).
    if st.secrets.load_if_toml_exists() and name in st.secrets:
        return str(st.secrets[name])
    return ""


st.set_page_config(page_title="Machine Guard", page_icon="🛠️")
st.title("Machine Guard")
st.write("Will this machine run fail? Set the sensor readings and ask the model.")

api_url = setting("API_URL").rstrip("/")
api_key = setting("API_KEY")
if not api_url or not api_key:
    st.error("API_URL and API_KEY are not set. Add them under App settings -> Secrets "
             "(or as environment variables when running locally).")
    st.stop()

with st.form("reading"):
    product_type = st.selectbox("Product quality type", ["L", "M", "H"])
    values = {}
    for column, (label, low, high, default, step) in SLIDERS.items():
        values[column] = st.slider(label, min_value=low, max_value=high, value=default, step=step)
    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {"type": product_type, **values}
    with st.spinner("Asking the model (the first call can take a minute while the API wakes up)..."):
        try:
            response = requests.post(f"{api_url}/predict", json=payload,
                                     headers={"X-API-Key": api_key}, timeout=TIMEOUT_S)
        except requests.RequestException as err:
            st.error(f"Could not reach the API at {api_url}: {err}")
            st.stop()

    if response.status_code == 401:
        st.error("The API rejected the key (401). Check API_KEY in the app's secrets.")
    elif response.status_code == 422:
        st.error("The API rejected the readings as out of range (422).")
        st.json(response.json())
    elif not response.ok:
        st.error(f"The API returned an error: {response.status_code}")
        st.text(response.text[:500])
    else:
        result = response.json()
        probability = result["failure_probability"]
        st.metric("Failure probability", f"{probability:.1%}")
        st.progress(min(max(probability, 0.0), 1.0))
        if result["prediction"] == 1:
            st.warning(f"Likely failure (probability is at or above the {result['threshold']} threshold). "
                       "Schedule an inspection before the next run.")
        else:
            st.success("No failure expected for this run.")
        st.caption(f"Model: {result['model_version']} · API: {api_url}")

with st.expander("Try these"):
    st.write("- **Worn tool under load:** tool wear 220 min, torque 65 Nm, speed 1,300 rpm.\n"
             "- **Poor heat dissipation:** air 302 K, process 310.5 K (a small difference), speed 1,300 rpm.\n"
             "- Then move one slider at a time and watch the probability change.")
