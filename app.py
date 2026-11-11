"""TIPS Outcome Navigator | 365-day three-outcome MT-GBM calculator.
Run: streamlit run app.py
This is research decision support, NOT a competing-risk survival calculator.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from feature_schema import FEATURES, OUTCOME_LABELS
from model_runtime import PortableMTGBM

MODEL_FILE = Path(__file__).resolve().parent / "models" / "mtgbm_365.json.gz"

FEATURE_EN = {
    "Age": {
        "label": "Age",
        "unit_label": "years",
        "help": "Patient age in years.",
        "fallback": 50,
    },
    "PLT": {
        "label": "Platelet count",
        "unit_label": "×10⁹/L",
        "help": "Platelet count before TIPS.",
        "fallback": 57,
    },
    "Hb": {
        "label": "Hemoglobin",
        "unit_label": "g/L",
        "help": "Hemoglobin before TIPS.",
        "fallback": 82,
    },
    "Alb": {
        "label": "Albumin",
        "unit_label": "g/L",
        "help": "Serum albumin before TIPS.",
        "fallback": 34,
    },
    "TBIL": {
        "label": "Total bilirubin",
        "unit_label": "μmol/L",
        "help": "Total bilirubin before TIPS.",
        "fallback": 22,
    },
    "Scr": {
        "label": "Serum creatinine",
        "unit_label": "μmol/L",
        "help": "Serum creatinine before TIPS.",
        "fallback": 77,
    },
    "INR": {
        "label": "International normalized ratio",
        "unit_label": "ratio",
        "help": "INR before TIPS.",
        "fallback": 1.3,
    },
    "PT_prolongation": {
        "label": "PT prolongation",
        "unit_label": "s",
        "help": "Prothrombin time prolongation before TIPS.",
        "fallback": 1.9,
    },
}

OUTCOME_EN = {
    "Rebleeding_365": ("Rebleeding", "Rebleeding · 365 days"),
    "OHE_365": ("Overt hepatic encephalopathy", "Overt hepatic encephalopathy · 365 days"),
    "Mortality_365": ("All-cause mortality", "All-cause mortality · 365 days"),
}

st.set_page_config(
    page_title="TIPS Outcome Navigator | MT-GBM",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .stApp { background: linear-gradient(160deg,#f1f7fa 0%, #f9fbfd 49%,#f5f8fb 100%); }
    .block-container { max-width: 1390px; padding-top: 2rem; padding-bottom: 4rem; }
    #MainMenu, footer { visibility: hidden; }
    .app-eyebrow {font-size:.75rem;font-weight:780;letter-spacing:.13em;color:#0e7783;text-transform:uppercase;margin-bottom:.6rem;}
    .hero {border:1px solid #d9e8ed;background:linear-gradient(111deg,#0d3c51 0%,#07566a 58%,#087881 100%);
           border-radius:22px; padding:31px 38px 28px;color:white;position:relative;overflow:hidden;
           box-shadow:0 22px 55px rgba(24,68,82,.13);margin-bottom:1.4rem;}
    .hero:after {content:"";position:absolute;right:-25px;top:-95px;width:340px;height:340px;border:55px solid rgba(255,255,255,.06);border-radius:50%;}
    .hero h1 {color:white!important;font-size:2.25rem!important;letter-spacing:-.035em;font-weight:800!important;line-height:1.12!important;margin:.28rem 0 .7rem!important;}
    .hero p {color:#d4e6ec;font-size:1rem;max-width:750px;margin:0;line-height:1.65;}
    .hero-pills {display:flex;gap:9px;flex-wrap:wrap;margin-top:18px;}
    .hero-pills span {border:1px solid rgba(233,251,255,.27);border-radius:100px;padding:6px 11px;color:#e4f7fa;font-size:.77rem;background:rgba(255,255,255,.07);}
    .section-head {display:flex;align-items:center;gap:9px;font-weight:780;color:#17354a;font-size:1.07rem;margin:1.1rem 0 .35rem;}
    .section-num {height:28px;min-width:28px;display:grid;place-items:center;border-radius:9px;background:#e4f2f4;color:#0b7180;font-weight:750;font-size:.79rem;}
    .fineprint {color:#688196;font-size:.83rem;line-height:1.6;}
    .risk-card {border:1px solid #e4eaf0;border-radius:18px;background:white;padding:22px 22px 19px;
                box-shadow:0 8px 26px rgba(31,64,88,.055);min-height:171px;}
    .risk-name {font-size:.94rem;color:#27465b;font-weight:720;margin:0;line-height:1.4;}
    .risk-en {font-size:.77rem;color:#8294a1;line-height:1.3;margin-top:2px;}
    .risk-number {font-size:2.7rem;color:#132e43;font-weight:790;letter-spacing:-.06em;margin-top:11px;line-height:1.2;}
    .risk-unit {font-size:1.1rem;color:#7a909d;font-weight:500;margin-left:4px;}
    .risk-track {height:7px;border-radius:12px;background:#ecf1f5;margin-top:13px;overflow:hidden;}
    .risk-fill {height:100%;border-radius:12px;}
    .info-panel {background:white;border:1px solid #e5edf1;border-radius:16px;padding:18px 22px;margin-top:11px;
                 color:#496578;font-size:.86rem;line-height:1.68;}
    .tag {display:inline-block;background:#e8f4f4;color:#15747d;border:1px solid #d5e9e9;border-radius:99px;padding:4px 9px;font-size:.74rem;font-weight:700;}
    div[data-testid="stForm"] {background:white;border:1px solid #dde9ec;border-radius:18px;padding:22px 24px 17px;
                    box-shadow:0 8px 26px rgba(31,64,88,.045);}
    div[data-testid="stNumberInput"] label {color:#284558;font-weight:650!important;}
    div[data-testid="stFormSubmitButton"] button {border-radius:11px;background:#0a6876;color:white;border:none;
                  height:3rem;font-weight:720;box-shadow:0 7px 17px rgba(9,105,116,.2);}
    div[data-testid="stFormSubmitButton"] button:hover {background:#0a5562;color:white;border:none;}
    [data-testid="stMetric"] {border:1px solid #e6edf0;padding:11px 13px;border-radius:12px;background:white;}
    [data-testid="stDownloadButton"] button {border-radius:10px;}
    @media(max-width:800px){.hero{padding:23px 21px;}.hero h1{font-size:1.78rem!important;}}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the MT-GBM model…")
def load_model(path: str, modified_ns: int):
    # mtime changes invalidate Streamlit cache when a new model is exported.
    return PortableMTGBM(path)


def get_outcome_color(outcome: str, idx: int) -> str:
    if outcome in OUTCOME_LABELS and len(OUTCOME_LABELS[outcome]) >= 3:
        return OUTCOME_LABELS[outcome][2]
    fallback = ["#2563eb", "#8b5cf6", "#ea580c"]
    return fallback[idx % len(fallback)]


def risk_cards(risks):
    cols = st.columns(3, gap="medium")
    for idx, outcome in enumerate(OUTCOME_EN):
        name, subtitle = OUTCOME_EN[outcome]
        color = get_outcome_color(outcome, idx)
        p = float(risks[idx]) * 100.0
        with cols[idx]:
            st.markdown(
                f'<div class="risk-card">'
                f'<div class="risk-name">{name}</div><div class="risk-en">{subtitle}</div>'
                f'<div class="risk-number">{p:.1f}<span class="risk-unit">%</span></div>'
                f'<div class="risk-track"><div class="risk-fill" style="width:{p:.3f}%;background:{color};"></div></div>'
                '</div>', unsafe_allow_html=True,
            )


def risk_figure(risks):
    outcomes = list(OUTCOME_EN)
    names = [OUTCOME_EN[o][0] for o in outcomes]
    colors = [get_outcome_color(o, i) for i, o in enumerate(outcomes)]
    vals = np.asarray(risks) * 100
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names[::-1], x=vals[::-1], orientation="h", marker_color=colors[::-1],
        text=[f"{v:.1f}%" for v in vals[::-1]], textposition="outside",
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        cliponaxis=False, width=0.48,
    ))
    fig.update_layout(
        title=None, paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        height=264, margin=dict(l=10, r=58, t=14, b=30),
        xaxis=dict(range=[0, 107], tickvals=[0, 25, 50, 75, 100],
                   ticksuffix="%", gridcolor="#e9eef2", zeroline=False,
                   title="Predicted risk (%)"),
        yaxis=dict(showgrid=False, tickfont=dict(size=13, color="#314d60")),
        font=dict(family="Arial, sans-serif", color="#536b7c"),
        showlegend=False,
    )
    return fig


def in_development_range(model, name, value):
    lim = model.reference.get("features", {}).get(name, {})
    low, high = lim.get("p01"), lim.get("p99")
    return (low is None or float(value) >= float(low)) and (high is None or float(value) <= float(high))


st.markdown("""
<div class="hero">
  <div class="app-eyebrow" style="color:#a1e9ed">Clinical prediction · Translational research</div>
  <h1>TIPS Outcome Navigator</h1>
  <p>A 365-day multi-outcome risk prediction platform after transjugular intrahepatic portosystemic shunt (TIPS),
     based on a shared-tree multi-task gradient boosting model (MT-GBM).<br>
     Enter eight routine clinical variables once to estimate the independent risks of rebleeding,
     overt hepatic encephalopathy, and all-cause mortality.</p>
  <div class="hero-pills"><span>MT-GBM</span><span>365-day horizon</span><span>3 clinical outcomes</span><span>8 predictors</span><span>Research prototype</span></div>
</div>
""", unsafe_allow_html=True)

model = None
if MODEL_FILE.exists():
    try:
        model = load_model(str(MODEL_FILE), MODEL_FILE.stat().st_mtime_ns)
    except Exception as exc:
        st.error(f"Failed to load the model file: {exc}")
else:
    st.warning("**Model not connected yet.** The web interface is ready, but you must first export the trained model to `models/mtgbm_365.json.gz`. No demo or fabricated risk values are shown before a real model is loaded.")

left, right = st.columns([0.92, 1.42], gap="large")
with left:
    st.markdown('<div class="section-head"><span class="section-num">01</span>Patient data input</div>', unsafe_allow_html=True)
    st.caption("Patient-specific clinical variables · Pre-TIPS routine measures")
    with st.form("input_form", clear_on_submit=False):
        user_inputs = {}
        input_cols = st.columns(2, gap="medium")
        for idx, (name, cfg) in enumerate(FEATURES.items()):
            reference_median = (model.reference.get("features", {}).get(name, {}).get("median")
                                if model is not None else None)
            fallback = FEATURE_EN.get(name, {}).get("fallback", float(cfg.get("minimum", 0)))
            value = float(reference_median) if reference_median is not None else float(fallback)
            low = float(min(cfg["minimum"], value))
            high = float(max(cfg["maximum"], value))
            label = FEATURE_EN.get(name, {}).get("label", name)
            unit_label = FEATURE_EN.get(name, {}).get("unit_label", cfg.get("unit", ""))
            help_text = FEATURE_EN.get(name, {}).get("help", cfg.get("help", ""))
            with input_cols[idx % 2]:
                user_inputs[name] = st.number_input(
                    f'{label} · {unit_label}', min_value=low, max_value=high,
                    value=value, step=float(cfg["step"]), help=help_text,
                    key=f"field_{name}", format="%.1f",
                )
        st.markdown('<div class="fineprint">All 8 inputs are processed strictly in the same order used for model training. Missing values are not imputed automatically, and no hidden unit conversion is applied.</div>', unsafe_allow_html=True)
        can_calculate = model is not None
        submitted = st.form_submit_button(
            "Estimate 365-day risk for all 3 outcomes →", use_container_width=True,
            disabled=not can_calculate,
        )
    if model is not None:
        st.markdown(
            f'<div class="info-panel"><span class="tag">MODEL READY</span><br>'
            f'<b>Model version:</b> {model.model_version}<br>'
            f'<b>Number of shared trees:</b> {len(model.trees)}<br>'
            f'<b>Training sample size:</b> {model.reference.get("n", "Not provided")}<br>'
            f'<b>Algorithm:</b> Shared-tree MT-GBM · binary 365-day outcomes</div>',
            unsafe_allow_html=True,
        )

    if submitted:
        x = np.asarray([[float(user_inputs[n]) for n in FEATURES]], dtype=np.float64)
        try:
            predicted = model.predict_proba(x)[0]
            st.session_state["risk_result"] = {
                "inputs": dict(user_inputs),
                "risks": predicted.tolist(),
                "version": model.model_version,
                "fingerprint": model.fingerprint,
            }
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

with right:
    st.markdown('<div class="section-head"><span class="section-num">02</span>Individualized risk assessment</div>', unsafe_allow_html=True)
    st.caption("Individualized risk estimates · Three independent binary outcomes")
    result = st.session_state.get("risk_result")
    if result and model is not None and result.get("fingerprint") == model.fingerprint:
        risks = np.asarray(result["risks"], dtype=float)
        risk_cards(risks)
        st.markdown("#### Outcome risk comparison")
        st.plotly_chart(risk_figure(risks), use_container_width=True, config={"displayModeBar": False})
        st.caption("The three outcomes may occur in the same patient. The probabilities shown are **not mutually exclusive** and should **not** be summed to 100%. These predictions are based on fixed 365-day binary labels and are not cumulative incidence functions (CIFs) from a competing-risk model.")
        entered = result["inputs"]
        out_of_range = [FEATURE_EN.get(n, {}).get("label", n) for n, v in entered.items() if not in_development_range(model, n, v)]
        if out_of_range:
            st.warning("Some inputs are outside the 1st–99th percentile reference range of the training cohort: " + ", ".join(out_of_range) + ". Interpretation should therefore be cautious.")
        report = {"model_version": model.model_version, "horizon_days": 365,
                  "created_at_UTC": datetime.now(timezone.utc).isoformat(), **entered}
        for idx, outcome in enumerate(OUTCOME_EN):
            report[f"pred_{outcome}"] = float(risks[idx])
        csv_text = pd.DataFrame([report]).to_csv(index=False).encode("utf-8-sig")
        st.download_button("↓ Download anonymous prediction results (CSV)", data=csv_text,
                           file_name="tips_mtgbm_risk_report.csv", mime="text/csv")
    else:
        st.markdown('''
        <div class="info-panel" style="min-height:245px;display:flex;align-items:center;">
          <div><span class="tag">NO PREDICTION</span><h3 style="color:#193a50;margin-top:14px;">Risk results will appear here</h3>
          <p>After a trained model is loaded and all 8 variables are entered, this panel will display the individualized predicted probabilities for all three outcomes, a graphical comparison, and a downloadable anonymous report.</p>
          <p style="color:#8c9eaa;font-size:.85rem">No fabricated or demo predictions are shown without a real exported model.</p></div>
        </div>''', unsafe_allow_html=True)

if model is not None and st.session_state.get("risk_result"):
    result = st.session_state["risk_result"]
    if result.get("fingerprint") == model.fingerprint:
        st.markdown('<div class="section-head"><span class="section-num">03</span>One-variable scenario explorer</div>', unsafe_allow_html=True)
        st.caption("Explore how the model output changes when a single input is modified. This does not imply that changing that variable will causally change clinical risk to the same degree.")
        with st.expander("Open scenario simulation (not treatment advice)"):
            col1, col2, col3 = st.columns([1.0, 1.0, 1.15])
            names = list(FEATURES)
            with col1:
                selected = st.selectbox("Variable to modify", options=names, format_func=lambda x: FEATURE_EN.get(x, {}).get("label", x))
            with col2:
                config = FEATURES[selected]
                scenario_value = st.number_input(
                    f'Scenario value ({FEATURE_EN.get(selected, {}).get("unit_label", config.get("unit", ""))})',
                    value=float(result["inputs"][selected]),
                    min_value=float(config["minimum"]), max_value=float(config["maximum"]),
                    step=float(config["step"]), key=f"scenario_{selected}", format="%.1f",
                )
            with col3:
                st.markdown("<div style='height:1.78rem'></div>", unsafe_allow_html=True)
                compare_clicked = st.button("Compare updated prediction", use_container_width=True)
            if compare_clicked:
                alternative = dict(result["inputs"])
                alternative[selected] = float(scenario_value)
                x_new = np.asarray([[alternative[n] for n in names]], dtype=float)
                alt_pred = model.predict_proba(x_new)[0]
                deltas = (alt_pred - np.asarray(result["risks"])) * 100
                output_cols = st.columns(3)
                for i, outcome in enumerate(OUTCOME_EN):
                    with output_cols[i]:
                        st.metric(OUTCOME_EN[outcome][0], f"{alt_pred[i]*100:.1f}%", delta=f"{deltas[i]:+.2f} pp", delta_color="off")
                if not in_development_range(model, selected, scenario_value):
                    st.warning("The scenario value is outside the 1st–99th percentile training reference range. The result may therefore be unstable.")
                st.info("Scenario simulation only demonstrates the numerical response of the machine-learning model to a changed input. It does not imply treatment effect or causality.")

st.markdown("---")
st.markdown("#### Scope of use and important notes")
a, b, c = st.columns(3, gap="medium")
with a:
    st.markdown("**Prediction time horizon**")
    st.caption("365 days after TIPS. The output is a fixed-time binary risk estimate rather than a full survival curve at arbitrary time points.")
with b:
    st.markdown("**Model inputs**")
    st.caption("The calculator strictly uses the 8 predictors from the development dataset. Please verify laboratory units, preprocessing steps, and compatibility with the target population.")
with c:
    st.markdown("**Clinical interpretation**")
    st.caption("For research use only. This tool does not replace professional clinical judgment. Outcome handling, external calibration, and privacy safeguards should be verified before public deployment.")

if model is not None and model.shared_feature_importance is not None:
    with st.expander("Model transparency: view global shared-tree feature importance"):
        importance = np.maximum(model.shared_feature_importance, 0)
        if importance.sum() > 0:
            ranking = np.argsort(importance)
            chart = go.Figure(go.Bar(
                x=100 * importance[ranking] / importance.sum(),
                y=[FEATURE_EN.get(model.features[i], {}).get("label", model.features[i]) for i in ranking],
                orientation="h", marker_color="#177987",
            ))
            chart.update_layout(height=300, margin=dict(l=5, r=15, t=10, b=20),
                                xaxis_title="Relative split gain (%)", plot_bgcolor="white",
                                paper_bgcolor="white")
            st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})
            st.caption("This figure shows the global split gain of the shared-tree structure across all three tasks. It is **not** a patient-level explanation and **not** a SHAP plot for a single outcome.")

st.caption("TIPS Outcome Navigator · MT-GBM research web application · Prediction results are not validated treatment recommendations. No patient name or patient ID is required; data protection depends on your deployment environment.")
