# Commodity Diversification & the Dollar Factor

**Empirical analysis of commodity diversification benefits using DCC-GARCH, Markov Regime-Switching, Hidden Markov Models, and CVaR Portfolio Optimization on 20 years of weekly data (2005–2025).**

*SF Club USI — Andrea Botta & Gianluca Ladelfa — June 2026*

---

## Abstract

This project investigates whether commodities provide genuine diversification benefits for equity investors, and how the US Dollar factor mediates these relationships. Using a unified quantitative framework built on conditional volatility models, regime-switching techniques, and tail-risk optimization, we analyze five commodity markets (Copper, Silver, Gold, Wheat, Natural Gas) against the S&P 500 and DXY over a 20-year period spanning multiple macroeconomic regimes.

---

## Methodology

| Layer | Method | Purpose |
|---|---|---|
| Volatility modeling | GARCH(1,1) | Conditional variance estimation per asset |
| Regime detection | Markov Regime-Switching (2-state) | Identify S&P 500 bull/bear regimes |
| Correlation dynamics | DCC-GARCH (Engle, 2002) | Time-varying conditional correlations |
| Regime prediction | Hidden Markov Model (HMM) | Unsupervised correlation regime detection |
| Forecasting | Random Forest + Stacked Ensemble | Predict next-period DCC correlation |
| Feature attribution | SHAP values | Interpretability of RF predictions |
| Portfolio construction | Min-CVaR vs Min-Volatility | Tail-risk-aware allocation |
| Sentiment signal | COT Report (CFTC) | Managed money positioning z-scores |
| Causality testing | Granger Causality | VIX → DCC transmission |

---

## Data Sources

| Source | Series | Frequency |
|---|---|---|
| Yahoo Finance (`yfinance`) | S&P 500, DXY, Copper, Silver, Gold, Wheat, Nat. Gas, 10Y Treasury | Weekly |
| FRED (St. Louis Fed) | Industrial Production, Manufacturing Employment, VIX, Fed Funds Rate, CPI | Monthly → resampled weekly |
| CFTC (cot.gov) | COT Disaggregated Report — Managed Money positions | Weekly (2010–2025) |

**Sample period:** 2005-01-01 → 2025-12-31 (main dataset, ~1,040 weeks)  
**ML/COT dataset:** 2010-01-01 → 2025-12-31 (~800 weeks)

---

## Key Results

- Gold and Copper exhibit **time-varying, regime-dependent** correlations with S&P 500: near-zero in bull markets, negative during crises
- The **DXY factor** amplifies commodity sell-offs — strong dollar periods coincide with depressed commodity-equity correlations
- HMM correctly identifies **high-integration regimes** (GFC 2008, COVID 2020, Ukraine 2022) without supervision
- Min-CVaR portfolios allocate significantly more to **Gold** relative to Min-Volatility, reflecting tail-risk hedging demand
- Random Forest achieves **R² > 0.85** in predicting next-week DCC correlations; COT z-scores are the most important feature (SHAP)

---

## Notebook Structure

```
SFClubUSI_Commodity_Clean.ipynb
│
├── 1.  Imports & Configuration
├── 2.  Data Acquisition
│   ├── 2.1  Price Data (yfinance)
│   ├── 2.2  Macro Data (FRED)
│   └── 2.3  COT Report (CFTC)
├── 3.  Data Merging
├── 4.  Feature Engineering
│   ├── 4.1  Analytical Variables
│   ├── 4.2  GARCH(1,1) — Conditional Volatility
│   ├── 4.3  Markov Regime-Switching
│   └── 4.4  DCC-GARCH — Dynamic Conditional Correlations
├── 5.  Statistical Analysis
├── 6.  Visualization  (Figures 1–7 + GARCH figure)
├── 7.  Machine Learning
│   ├── 7.2  HMM — Unsupervised Regime Detection
│   ├── 7.3  Feature Engineering & Train/Test Split
│   ├── 7.4  AR(1) Benchmark
│   ├── 7.5  Random Forest — GridSearchCV Optimization
│   └── 7.6  Figure 8 — HMM Regimes & SHAP
├── 8.  Portfolio Optimization
│   ├── 8.1  Min-CVaR vs Min-Volatility
│   └── 8.2  Transaction Cost Optimization
├── 9.  Performance Evaluation & Dynamic Strategy
│   ├── 9.1  RF Test Set Metrics
│   ├── 9.2  Dynamic Strategy Backtest
│   └── 9.3  Stacked Ensemble (RF + XGBoost + SVR)
├── 10. Granger Causality — VIX → DCC
├── 11. Summary Tables
└── 12. Export
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/commodity-dollar-factor.git
cd commodity-dollar-factor
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. FRED API Key

The notebook requires a free FRED API key. Get one at [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html) and replace the placeholder in **Section 1**:

```python
FRED_API_KEY = "your_key_here"
```

### 5. Run the notebook

```bash
jupyter notebook SFClubUSI_Commodity_Clean.ipynb
```

---

## Output Files

Running the notebook end-to-end produces:

| File | Description |
|---|---|
| `fig1_overview.png` | Normalized price overview + Markov regime |
| `fig_garch_regime.png` | GARCH(1,1) conditional volatility by asset |
| `fig2_dcc_sp500.png` | DCC correlations vs S&P 500 (2005–2025) |
| `fig3_dcc_dxy.png` | DCC correlations vs DXY |
| `fig4_dcc_heatmap.png` | Sub-period DCC heatmap |
| `fig5_cot_signals.png` | COT signal → next-week return |
| `fig6_portfolio_metrics.png` | Risk/return, beta, Sharpe |
| `fig7_efficient_frontier.png` | Efficient frontier + optimal portfolios |
| `fig8_ml_final.png` | HMM regimes + SHAP feature importance |
| `fig9_cvar_allocation.png` | Min-CVaR vs Min-Vol allocation |
| `commodity_dataset_final.csv` | Full processed panel dataset |
| `commodity_analysis_results.zip` | All outputs bundled |

---

## Requirements

See `requirements.txt`. Main dependencies:

```
arch · fredapi · hmmlearn · matplotlib · numpy · pandas
pyportfolioopt · scikit-learn · scipy · shap · statsmodels
xgboost · yfinance
```

---

## Authors

**Andrea Botta** — BSc Economics, Università della Svizzera italiana (USI)  
**Gianluca Ladelfa** — BSc Economics, Università della Svizzera italiana (USI)  

*SF Club USI — Student Finance Club*

---

## References

- Engle, R. (2002). Dynamic Conditional Correlation. *Journal of Business & Economic Statistics*, 20(3), 339–350.
- Hamilton, J.D. (1989). A New Approach to the Economic Analysis of Nonstationary Time Series. *Econometrica*, 57(2), 357–384.
- Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327.
- Rockafeller, R.T. & Uryasev, S. (2000). Optimization of Conditional Value-at-Risk. *Journal of Risk*, 2(3), 21–41.

---

## License

This project is released under the MIT License. See `LICENSE` for details.
