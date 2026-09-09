# 🐉🐯 Dragon Tiger Analyzer

A transparent statistical-analysis and backtesting starter app for Dragon Tiger.

## Important

This project **does not claim to predict casino outcomes with certainty**. It separates:

- historical frequency
- conditional frequency
- descriptive patterns
- theoretical baseline
- model signals

A signal is not the same thing as a true probability of the next hand.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Input format

Enter results oldest → newest:

```text
D T T D D X T D T
```

Where:

- `D` = Dragon
- `T` = Tiger
- `X` = Tie

The app also accepts `Dragon`, `Tiger`, and `Tie`.

## Current MVP features

- Dragon/Tiger/Tie frequencies
- current streak
- longest Dragon/Tiger streak
- switching/alternation rate
- recent-window analysis
- conditional pattern frequency
- simple baseline comparison
- transparent Dragon/Tiger/LEAVE descriptive signal
- CSV export

## Planned versions

### V2 — Card-level data
Store actual ranks and suits.

### V3 — Road/chart engine
Reconstruct and analyze road layouts.

### V4 — Backtesting
Evaluate candidate rules on historical data.

### V5 — Statistical validation
Confidence intervals, hypothesis tests, multiple-testing controls.

### V6 — Model comparison
Compare transparent statistical models and calibrated probabilities.

### V7 — Database + dashboard
Persist tables, sessions, providers and variants.

### V8 — Screenshot assistance
Optional OCR/data-entry assistance for chart screenshots.

## Suggested repository structure

```text
dragon-tiger-analyzer/
├── app.py
├── requirements.txt
├── README.md
├── data/
├── src/
│   ├── probability.py
│   ├── patterns.py
│   ├── statistics.py
│   ├── backtest.py
│   └── models.py
└── tests/
```
