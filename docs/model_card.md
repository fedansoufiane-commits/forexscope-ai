# Model Card - WealthScope AI 1.0

## Intended use

The classifier demonstrates a complete QUA³CK machine-learning workflow on
historical US equity and ETF data. It is an educational artefact, not a trading
system and not financial advice.

## Data and target

- Source: Kaggle US Stocks & ETFs (CC0), 26 selected tickers
- Period: 1962-2017
- Target: price is higher after 20 trading days (`target_20d`)
- Inputs: returns, moving-average distances, volatility and drawdown

## Validation

Rows are ordered by date. The newest 20% of trading dates form the untouched
test period. Twenty trading days immediately before the test are purged because
the label itself looks 20 days ahead. Four earlier expanding walk-forward
windows estimate stability through time.

This is stricter and more realistic than a random split: no later market regime
may train a model that is evaluated on the past.

## Compared models

The same data windows and metrics are used for a Dummy baseline, Logistic
Regression, Decision Tree, Linear SVM and Random Forest. The comparison exposes
the historical development from transparent statistical and rule-based models
to margin methods and ensemble learning.

## Metrics

Accuracy is shown but never used alone. Balanced Accuracy, ROC-AUC, Average
Precision, Precision, Recall, F1, fit time and inference time provide a fuller
view. The exact generated values live in `models/diagnostics.json`.

## Limitations

- Historical prices alone contain little durable predictive signal.
- Tickers have different histories and survival/selection effects may remain.
- Transaction costs, slippage, taxes and portfolio constraints are excluded.
- Live observations after 2017 are a distribution shift.
- Feature importance and SHAP explain associations, not causality.
- Results must not be interpreted as expected investment returns.

## Reproduction

```bash
python scripts/train_and_diagnose.py
python -m pytest
python scripts/validate_app.py
```
