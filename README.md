# ₿ Bitcoin Price-Band Predictor

A Streamlit web app that uses a trained **Random Forest classifier** to classify a day's Bitcoin market data (Open, High, Low, Close, Volume and date) into a price band: **Low**, **Medium** or **High**.

> **Disclaimer:** This project is for learning and demonstration only. It classifies the price level of the data you enter. It does not forecast future prices and is **not financial advice**.

---

## Features

- **Single prediction:** enter OHLC prices, volume and a date, then get the predicted band, a confidence score and a probability chart.
- **Input validation:** catches impossible candles (for example, a High lower than the Close, or zero and negative prices).
- **Batch mode:** upload a CSV, get predictions for every row and download the results.
- **Model info tab:** feature importances, approximate band boundaries and the training feature order.
- **Transparent:** an expander shows the exact feature vector sent to the model.
- **Switchable label mapping:** a sidebar option lets you swap the class-label mapping (see [Class labels](#class-labels)).

## Project structure

```
.
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── Bitcoin_model.pkl       # Trained RandomForestClassifier
├── Bitcoin_scaler.pkl      # StandardScaler fitted on all 18 features
├── Bitcoin_features.pkl    # pandas.Index with the training column order
├── BTC.png                 # Optional banner/logo image
└── README.md
```

## Getting started

### 1. Clone the repository

```bash
https://github.com/salamiXpert/Bitcoin-Price-Band-Predictor.git
cd <salamiXpert>
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python -m streamlit run app.py
```

The app opens at <https://bitcoin-price-band-predictor.streamlit.app/>.

> The `.pkl` files were created with **scikit-learn 1.8.0**, which is pinned in `requirements.txt`. Using a different version may produce warnings or load errors. `pyarrow` is required to load `Bitcoin_features.pkl`.

## How to use

### Single prediction
1. Open the **Single prediction** tab.
2. Enter Open, High, Low, Close and Volume, and pick a date.
3. Click **Predict price band**.

### Batch prediction
1. Open the **Batch (CSV)** tab and download the template.
2. Fill in a CSV with these columns: `Date, Open, High, Low, Close, Volume`.
3. Upload it, review the table and download the predictions.

Example:

```csv
Date,Open,High,Low,Close,Volume
2021-03-15,55000,56500,54000,55800,45000000000
2022-08-01,23000,24000,22500,23400,28000000000
```

## Model details

| Item | Value |
|---|---|
| Algorithm | `RandomForestClassifier` (100 trees, entropy criterion, `max_depth=100`, `random_state=42`) |
| Input features | 18 |
| Target | 3 price-band classes (codes 0, 1, 2) |
| Scaling | `StandardScaler` applied to all 18 features |

**Feature layout (in training order):**

`Open, High, Low, Close, Volume, Year, Day` followed by one-hot month columns `Month_names_<Month>`.

- `Year` is a numeric value (for example, 2021).
- `Day` is the **day of the month** (1–31), not the day of the week.
- April is the dropped baseline month, so it has no column (all month columns are 0).

**What drives the model:** feature importance is dominated by Open, High, Low, Close and Year. Day, Month and Volume contribute very little.

### Class labels

The model outputs the codes `0`, `1` and `2`. Probing the model shows the following approximate boundaries:

| Code | Band | Approximate price range |
|---|---|---|
| 0 | Low | below ~$850 |
| 2 | Medium | ~$850 to ~$22,800 |
| 1 | High | above ~$22,800 |

These boundaries were found by sweeping the price with other inputs held fixed, so treat them as indicative and confirm them against your training notebook. If your label encoding differs, change the mapping in the sidebar or in `LABEL_MAPS` inside `app.py`.

## Limitations

- Random forests **cannot extrapolate**. For dates after the training period (roughly up to 2023), predictions are driven almost entirely by the prices and volume you enter.
- The model classifies the **level** of the price. It does not predict direction or future movement.
- Results depend on the quality and time range of the original training data.

## Tech stack

- [Python](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/)
- [joblib](https://joblib.readthedocs.io/) and [PyArrow](https://arrow.apache.org/docs/python/)

## Future improvements

- Add technical indicators (moving averages, RSI) as features.
- Retrain on more recent data and report validation metrics in the app.
- Deploy to Streamlit Community Cloud.

## Author

**Salami Emmanuel**

## License

Add a license of your choice (for example [MIT](https://choosealicense.com/licenses/mit/)) by creating a `LICENSE` file.
