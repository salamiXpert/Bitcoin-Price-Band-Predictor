"""
Bitcoin Price-Band Predictor (Streamlit)
=========================================
Artifacts expected next to this file:
    Bitcoin_model.pkl    -> RandomForestClassifier (18 input features, classes 0/1/2)
    Bitcoin_scaler.pkl   -> StandardScaler fitted on all 18 features
    Bitcoin_features.pkl -> pandas.Index with the exact training column order
    BTC.png              -> optional banner image

Run locally:
    pip install -r requirements.txt
    python -m streamlit run app.py

Feature layout the model was trained on (verified from the .pkl files):
    Open, High, Low, Close, Volume   -> numeric prices / volume
    Year                             -> numeric (e.g. 2021), NOT one-hot
    Day                              -> DAY OF THE MONTH (1-31), NOT weekday
    Month_names_<Month>              -> one-hot, April is the dropped baseline
    (all 18 columns, including the month dummies, go through the scaler)
"""

from __future__ import annotations

import calendar
import datetime as dt
import os

import joblib
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# PAGE CONFIG (must be the first Streamlit call)
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "BTC.png")
HAS_LOGO = os.path.exists(LOGO_PATH)

st.set_page_config(
    page_title="Bitcoin Price-Band Predictor",
    page_icon=LOGO_PATH if HAS_LOGO else "₿",
    layout="centered",
)

MODEL_PATH = os.path.join(BASE_DIR, "Bitcoin_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "Bitcoin_scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "Bitcoin_features.pkl")

# ----------------------------------------------------------------------------
# CLASS LABELS
# ----------------------------------------------------------------------------
# Probing the model shows: class 0 = very low prices (below ~$850),
# class 2 = mid prices (~$850 - ~$22,800), class 1 = high prices (above ~$22,800).
# So the numeric codes are NOT ordered Low/Medium/High. Check this against the
# label-encoding step in your training notebook and edit here if needed.
LABEL_MAPS = {
    "Inferred from model behaviour (0=Low, 1=High, 2=Medium)": {0: "Low", 1: "High", 2: "Medium"},
    "Original app mapping (0=Low, 1=Medium, 2=High)": {0: "Low", 1: "Medium", 2: "High"},
}
LABEL_ICONS = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}

MONTHS = list(calendar.month_name)[1:]  # January ... December
MIN_DATE = dt.date(2014, 1, 1)
MAX_DATE = dt.date(2040, 12, 31)
LAST_TRAIN_YEAR_HINT = 2023  # approximate; the scaler mean year is ~2017.5


# ----------------------------------------------------------------------------
# LOAD ARTIFACTS
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_order = list(joblib.load(FEATURES_PATH))
    scaler_cols = list(getattr(scaler, "feature_names_in_", feature_order))
    return model, scaler, feature_order, scaler_cols


try:
    model, scaler, feature_order, scaler_cols = load_artifacts()
except FileNotFoundError as e:
    st.error(
        f"A required .pkl file is missing: {e}\n\n"
        "Put Bitcoin_model.pkl, Bitcoin_scaler.pkl and Bitcoin_features.pkl "
        "in the same folder as app.py."
    )
    st.stop()
except ImportError as e:
    st.error(f"A required library is missing: {e}\n\nRun: pip install -r requirements.txt")
    st.stop()
except Exception as e:  # noqa: BLE001 - show any unpickling / version problem clearly
    st.error(f"Could not load the model files: {e}")
    st.stop()


# ----------------------------------------------------------------------------
# FEATURE ENGINEERING + PREDICTION
# ----------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df needs: Open, High, Low, Close, Volume, Date. Returns RAW (unscaled) model features."""
    dates = pd.to_datetime(df["Date"])
    out = pd.DataFrame(
        {
            "Open": df["Open"].astype(float),
            "High": df["High"].astype(float),
            "Low": df["Low"].astype(float),
            "Close": df["Close"].astype(float),
            "Volume": df["Volume"].astype(float),
            "Year": dates.dt.year,
            "Day": dates.dt.day,  # day of month, as in training
        }
    )
    month_names = dates.dt.month_name()
    for col in feature_order:
        if col.startswith("Month_names_"):
            out[col] = (month_names == col.replace("Month_names_", "")).astype(int)
    # Any column the model expects that we did not build -> 0 (e.g. baseline month handled implicitly)
    return out.reindex(columns=feature_order, fill_value=0)


def predict(raw: pd.DataFrame):
    """Scale in the scaler's column order, then predict. Returns (class_codes, probabilities)."""
    scaled = pd.DataFrame(scaler.transform(raw[scaler_cols]), columns=scaler_cols, index=raw.index)
    scaled = scaled.reindex(columns=feature_order)
    X = scaled.values
    return model.predict(X), model.predict_proba(X)


def validate_ohlc(o, h, l, c, v) -> list[str]:
    problems = []
    if min(o, h, l, c) <= 0:
        problems.append("Open, High, Low and Close must all be greater than 0.")
    if h < max(o, l, c):
        problems.append("High must be the highest of Open, Low and Close.")
    if l > min(o, h, c):
        problems.append("Low must be the lowest of Open, High and Close.")
    if v < 0:
        problems.append("Volume cannot be negative.")
    return problems


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    map_name = st.selectbox("Class label mapping", list(LABEL_MAPS.keys()))
    label_map = LABEL_MAPS[map_name]
    st.caption(
        "The model outputs codes 0, 1 and 2. If the labels look wrong to you, "
        "switch the mapping here and compare with your training notebook."
    )
    st.divider()
    st.subheader("About the model")
    st.write(f"**Type:** {type(model).__name__}")
    st.write(f"**Trees:** {getattr(model, 'n_estimators', 'n/a')}")
    st.write(f"**Features:** {len(feature_order)}")
    st.write(f"**Classes:** {list(model.classes_)}")

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
if HAS_LOGO:
    st.image(LOGO_PATH, width=140)
st.title("Bitcoin Price-Band Predictor")
st.caption(
    "Enter one day's Bitcoin market data and the model classifies the price into a band "
    "(Low / Medium / High). It does not forecast tomorrow's price and is not financial advice."
)

tab_single, tab_batch, tab_info = st.tabs(["Single prediction", "Batch (CSV)", "Model info"])

# ----------------------------------------------------------------------------
# TAB 1: SINGLE PREDICTION
# ----------------------------------------------------------------------------
with tab_single:
    with st.form("single_form"):
        st.subheader("Market data (USD)")
        c1, c2 = st.columns(2)
        with c1:
            open_p = st.number_input("Open ($)", min_value=0.0, value=30000.0, step=100.0, format="%.2f")
            high_p = st.number_input("High ($)", min_value=0.0, value=31000.0, step=100.0, format="%.2f")
            low_p = st.number_input("Low ($)", min_value=0.0, value=29500.0, step=100.0, format="%.2f")
        with c2:
            close_p = st.number_input("Close ($)", min_value=0.0, value=30500.0, step=100.0, format="%.2f")
            volume = st.number_input("Volume", min_value=0.0, value=2.0e10, step=1.0e8, format="%.0f")

        st.subheader("Date")
        date_val = st.date_input(
            "Date of the candle",
            value=dt.date.today(),
            min_value=MIN_DATE,
            max_value=MAX_DATE,
        )
        submitted = st.form_submit_button("Predict price band", type="primary", width="stretch")

    if submitted:
        problems = validate_ohlc(open_p, high_p, low_p, close_p, volume)
        if problems:
            for p in problems:
                st.error(p)
        else:
            row = pd.DataFrame(
                [{"Open": open_p, "High": high_p, "Low": low_p, "Close": close_p,
                  "Volume": volume, "Date": pd.Timestamp(date_val)}]
            )
            raw = build_features(row)
            codes, probas = predict(raw)
            code = int(codes[0])
            label = label_map.get(code, f"Class {code}")
            probs = pd.Series(
                {label_map.get(int(k), f"Class {k}"): float(p) for k, p in zip(model.classes_, probas[0])}
            ).sort_values(ascending=False)

            st.divider()
            st.subheader("Result")
            m1, m2 = st.columns(2)
            m1.metric("Predicted price band", f"{LABEL_ICONS.get(label, '')} {label}")
            m2.metric("Confidence", f"{probs.max():.1%}")

            st.write("**Probability by band**")
            st.bar_chart(probs.rename("Probability"), horizontal=True)

            if date_val.year > LAST_TRAIN_YEAR_HINT:
                st.warning(
                    f"The model was trained on data up to roughly {LAST_TRAIN_YEAR_HINT}. "
                    "Random forests cannot extrapolate, so for later years the result is driven mostly "
                    "by the price and volume values you entered."
                )
            if probs.max() < 0.6:
                st.info("Low confidence: the input sits close to a boundary between two bands.")

            with st.expander("Show the exact feature vector sent to the model"):
                st.dataframe(raw.T.rename(columns={0: "raw value"}), width="stretch")

# ----------------------------------------------------------------------------
# TAB 2: BATCH PREDICTION
# ----------------------------------------------------------------------------
with tab_batch:
    st.write("Upload a CSV with the columns: **Date, Open, High, Low, Close, Volume**.")
    template = pd.DataFrame(
        {"Date": ["2021-03-15", "2022-08-01"], "Open": [55000, 23000], "High": [56500, 24000],
         "Low": [54000, 22500], "Close": [55800, 23400], "Volume": [4.5e10, 2.8e10]}
    )
    st.download_button("Download CSV template", template.to_csv(index=False), "bitcoin_template.csv", "text/csv")

    up = st.file_uploader("Upload CSV", type="csv")
    if up is not None:
        try:
            data = pd.read_csv(up)
            required = ["Date", "Open", "High", "Low", "Close", "Volume"]
            missing = [c for c in required if c not in data.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
                for col in required[1:]:
                    data[col] = pd.to_numeric(data[col], errors="coerce")
                bad = data[required].isna().any(axis=1)
                if bad.any():
                    st.warning(f"Skipped {int(bad.sum())} row(s) with missing or invalid values.")
                data = data.loc[~bad].copy()
                if data.empty:
                    st.error("No valid rows to predict.")
                else:
                    raw = build_features(data)
                    codes, probas = predict(raw)
                    result = data.copy()
                    result["Predicted band"] = [label_map.get(int(c), f"Class {c}") for c in codes]
                    result["Confidence"] = probas.max(axis=1)
                    st.dataframe(
                        result.style.format({"Confidence": "{:.1%}"}),
                        width="stretch",
                    )
                    st.bar_chart(result["Predicted band"].value_counts())
                    st.download_button(
                        "Download predictions",
                        result.to_csv(index=False),
                        "bitcoin_predictions.csv",
                        "text/csv",
                    )
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not process the file: {e}")

# ----------------------------------------------------------------------------
# TAB 3: MODEL INFO
# ----------------------------------------------------------------------------
with tab_info:
    st.subheader("Feature importance")
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_order).sort_values(ascending=False)
        st.bar_chart(imp.head(10), horizontal=True)
        st.caption(
            "Price columns (Open/High/Low/Close) and Year dominate. Day and the month columns "
            "contribute almost nothing."
        )
    st.subheader("Approximate band boundaries")
    st.write(
        "Found by sweeping the price with other inputs fixed: roughly **Low below ~$850**, "
        "**Medium from ~$850 to ~$22,800**, **High above ~$22,800**. "
        "Treat these as indicative and verify against your training data."
    )
    st.subheader("Training feature order")
    st.code(", ".join(feature_order))

st.divider()
st.caption("For education only. Not financial advice.")