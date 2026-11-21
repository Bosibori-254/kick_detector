"""Streamlit interface for the experimental drilling kick detection model.

Run from the project root with:
    streamlit run app/streamlit_app.py

The interface is a research decision-support prototype, NOT a well-control system.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_config
from src.predict import load_bundle, predict_one, required_raw_features

st.set_page_config(page_title="Drilling Kick Detection", page_icon="🛢️", layout="wide")


@st.cache_resource(show_spinner=True)
def get_bundle_and_config():
    """Load the trained bundle once per Streamlit session."""
    config = load_config()
    bundle = load_bundle(config=config)
    return bundle, config


st.title("Drilling Kick Detection and Early Warning System")
st.caption(
    "Experimental research prototype based on the DataDRILL simulator dataset. "
    "Output is a model prediction, not a confirmed kick and not a well-control decision."
)

try:
    bundle, config = get_bundle_and_config()
except Exception as exc:  # pragma: no cover - Streamlit runtime path
    st.error(f"Cannot load the trained model: {exc}")
    st.info("Run `python -m src.cli train` from the project root, then refresh this page.")
    st.stop()

required = required_raw_features(bundle)
bounds = bundle.get("feature_bounds", {})
model_names = list(bundle["models"].keys())
best = bundle.get("best_model", model_names[0])

with st.sidebar:
    st.header("Model settings")
    selected_model = st.selectbox("Model", model_names, index=model_names.index(best))
    st.markdown("---")
    st.subheader("About this interface")
    st.write(
        "Enter instantaneous drilling measurements. Required inputs are validated "
        "against the training-data envelope. Flow imbalance is computed internally "
        "from FOut - FIn."
    )
    st.warning(
        "Do not use this output to replace approved well-control procedures, "
        "pit-volume checks, flow monitoring, or qualified drilling personnel."
    )

st.subheader("Drilling parameter inputs")
st.markdown(
    "Values outside a generous tolerance around the training range are rejected. "
    "Ranges shown below are observed in the training split of the simulated dataset."
)
inputs: dict[str, float] = {}
columns = st.columns(3)
for index, feature in enumerate(required):
    limit = bounds.get(feature, {})
    lower = float(limit.get("min", 0.0))
    upper = float(limit.get("max", 1.0))
    if upper <= lower:
        upper = lower + 1.0
    default = (lower + upper) / 2.0
    with columns[index % 3]:
        inputs[feature] = st.number_input(
            label=feature,
            value=default,
            min_value=float(-1e12),
            max_value=float(1e12),
            format="%.6g",
            help=f"Training range: {lower:.6g} to {upper:.6g}",
        )

col1, col2 = st.columns([1, 3])
with col1:
    submitted = st.button("Evaluate condition", type="primary")
with col2:
    st.caption("Predictions are experimental and should be compared with standard kick indicators.")

if submitted:
    try:
        result = predict_one(inputs, bundle, config, model_name=selected_model)
    except ValueError as exc:
        st.error(f"Invalid input: {exc}")
    else:
        positive = result["predicted_class"] == bundle["target"]["positive_label"]
        st.markdown("---")
        st.subheader("Prediction result")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Predicted condition", result["predicted_label"])
        metric_col2.metric("Model score", f"{result['score']:.4f}")
        metric_col3.metric("Decision threshold", f"{result['decision_threshold']:.4f}")

        if result.get("uncertain"):
            st.warning(result["warning"])
        elif positive:
            st.error(result["warning"])
        else:
            st.success(result["warning"])

        st.write(f"**Model:** `{result['model']}` ({result['model_family']})")
        st.caption(
            "The score is a probability for supervised classifiers and a calibrated "
            "anomaly/rule percentile for the anomaly detector and conventional baseline."
        )

        st.subheader("Most influential parameters (global importance)")
        for item in result["top_global_contributors"]:
            st.write(f"- `{item['feature']}`: {item['importance']:.4f}")
        st.caption("Global importance describes the model overall, not this individual prediction.")

        with st.expander("Input values sent to the model"):
            st.code(json.dumps(inputs, indent=2))

st.markdown("---")
st.caption(bundle["target"].get("proxy_disclaimer", "") or
           "WARNING: labels used for training were experimental proxies derived from pit gain/loss. "
           "Confirm the target definition with a drilling supervisor before interpreting results.")
st.caption(
    "Research limitation: the source dataset has no timestamps or event IDs, so this interface "
    "performs kick-condition classification, not a validated early-warning lead-time estimate."
)
