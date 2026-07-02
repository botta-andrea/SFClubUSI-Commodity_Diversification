# Commodity Diversification & the Dollar Factor

**Empirical analysis of commodity diversification benefits using DCC-GARCH, Markov Regime-Switching, Hidden Markov Models, and CVaR Portfolio Optimization on 20 years of weekly data (2005–2025).**

*Starting Finance Club USI — Andrea Botta & Gianluca Ladelfa — June 2026*

---

## Abstract

This project investigates whether commodities provide genuine diversification benefits for equity investors, and how the US Dollar factor mediates these relationships. Using a unified quantitative framework built on conditional volatility models, regime-switching techniques, and tail-risk optimization, we analyze five commodity markets (Copper, Silver, Gold, Wheat, Natural Gas) against the S&P 500 and DXY over a 20-year period spanning multiple macroeconomic regimes.

---

## Authors

**Andrea Botta** — BSc Economics, Università della Svizzera italiana (USI)  
**Gianluca Ladelfa** — BSc Economics, Università della Svizzera italiana (USI)  

*SF Club USI — Student Finance Club*

---

## Methodology

| Layer | Method | Purpose |
|---|---|---|
| Volatility modeling | GARCH(1,1) | Conditional variance estimation per asset (Bollerslev, 1986) |
| Regime detection | Markov Regime-Switching (2-state & 3-state) | S&P 500 bull/bear identification & Gold conditional states |
| Correlation dynamics | DCC-GARCH (Engle, 2002) | Time-varying conditional commodity-equity & commodity-fiat correlations |
| Regime prediction | Hidden Markov Model (HMM) | Unsupervised correlation regime detection (high vs. low integration) |
| Forecasting | Random Forest + Stacked Ensemble | Predict next-period DCC correlation trajectories |
| Feature attribution | SHAP values | Interpretability and economic attribution of Random Forest predictions |
| Portfolio construction | Min-CVaR vs Min-Volatility | Tail-risk-aware asset allocation (Rockafellar & Uryasev, 2000) |
| Sentiment signal | COT Report (CFTC) | Managed money net positioning rolling Z-scores |
| Causality testing | Granger Causality | Financial transmission testing (VIX $\to$ DCC) |

---

## Data Sources

| Source | Series | Frequency |
|---|---|---|
| Yahoo Finance (`yfinance`) | S&P 500, DXY, Copper, Silver, Gold, Wheat, Nat. Gas, 10Y Treasury | Weekly |
| FRED (St. Louis Fed) | Industrial Production, Manufacturing Employment, VIX, Fed Funds Rate, CPI | Monthly $\to$ resampled weekly |
| CFTC (cot.gov) | COT Disaggregated Report — Managed Money positions | Weekly |

*   **Main Dataset (Core Analysis):** 1,057 weeks spanning from **2005-01-06 to 2025-12-25** [1].
*   **ML/COT Dataset (COT Focus):** 597 weeks spanning from **2013-01-17 to 2024-11-07** [2].

---

## Key Results

*   **Regime-Dependent Decoupling:** Gold maintains a persistently near-zero conditional correlation with equities across both Markov regimes ($\bar{\rho}_{\text{bull}} = +0.065$, $\bar{\rho}_{\text{bear}} = +0.068$), strictly validating its role as a structural safe haven [3]. Conversely, copper exhibits a procyclical co-movement ($\bar{\rho} = +0.370$) that dynamically intensifies to $+0.385$ during bear markets, illustrating the diversification paradox [3].
*   **The Dollar Anchor:** Commodity-DXY conditional correlations are structurally negative. Heatmap and sub-period analyses reveal that precious and industrial metals act as powerful fiat hedges, with gold and copper averaging DCC correlations of $-0.478$ and $-0.351$ respectively, remaining stable even during periods of pronounced dollar strength [4, 5].
*   **Unsupervised Regime Detection:** The Hidden Markov Model (HMM) successfully identifies systemic **high-integration regimes** corresponding to major macro-financial shocks (GFC 2008, COVID-19 2020, Ukraine War 2022) without requiring prior label supervision.
*   **Tail-Risk Hedging:** Incorporating gold significantly enhances portfolio metrics. While a classic minimum-volatility portfolio allocates roughly $41.2\%$ to gold, the tail-risk-aware **minimum-CVaR** optimization shifts weights away from equities (reducing S&P 500 from $48\%$ to $38\%$) to increase gold exposure to **$46\%$**, providing superior downside protection.
*   **ML Correlation Predictability:** Predicting time-varying DCC correlations proves highly challenging due to extreme autoregressive persistence. The Naive AR(1) baseline slightly outperforms the tuned Random Forest in overall variance explained ($R^2 = 0.783$ vs. $0.763$) [6]. However, SHAP feature attributions show that macro-financial features—specifically HMM high-integration probabilities and CFTC COT speculative positioning—are critical for successfully timing non-linear structural breakouts that a univariate baseline is blind to.


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

## References

- Engle, R. (2002). Dynamic Conditional Correlation. *Journal of Business & Economic Statistics*, 20(3), 339–350.
- Hamilton, J.D. (1989). A New Approach to the Economic Analysis of Nonstationary Time Series. *Econometrica*, 57(2), 357–384.
- Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327.
- Rockafeller, R.T. & Uryasev, S. (2000). Optimization of Conditional Value-at-Risk. *Journal of Risk*, 2(3), 21–41.

---

## License

This project is released under the MIT License. See `LICENSE` for details.
