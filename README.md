# S&P 500 Volatility Prediction Pipeline

**5-day realized volatility prediction for S&P 500 using machine learning**

> End-to-end pipeline that downloads historical data from 467 S&P 500 companies, engineers 29 technical indicators, and trains an XGBoost model with temporal validation to predict future volatility. The model explains **32.65%** of realized volatility variance, is especially useful during crisis periods (R²=0.527 in 2008), and generalizes across tickers without ticker-specific encoding.

---

## 📊 Executive Summary of Results

### Dataset
- **2.67 million samples** of daily prices and volume
- **467 S&P 500 tickers**
- **22 years of history** (2000-03-16 → 2026-02-12)
- **29 technical indicators** derived (returns, EMAs, RSI, MACD, volatility, volume, etc.)
- **Target**: 5-day realized volatility (continuous)

### Final Model
| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **Global R²** | 0.3265 ± 0.1197 | Explains 32.65% of variance (solid for volatility) |
| **MAE** | 0.006941 | Average absolute error (~38.6% relative) |
| **RMSE** | 0.010952 | Squared error (penalizes outliers) |
| **Pearson r** | 0.6406 | Linear correlation with actual |
| **Total Time** | 137.3s | 5 models, 2.67M samples |

### Architecture
- **5 independent XGBoost models** (1 per temporal fold)
- **Pure Temporal Validation**: TimeSeriesSplit with expanding windows
- **5-day gap** between train and test (anti-leakage)
- **29 features** (no ticker one-hot)
- **Train time**: 135.6s | **Inference**: 0.72 μs/sample

---

## 🎯 Key Results per Fold

The model shows a critical pattern: **it predicts better during crises than during calm periods**.

| Fold | Period | Avg Volatility | **R²** | MAE | RMSE | Interpretation |
|------|--------|----------------|--------|------|------|-----------------|
| **0** | 2005–2010 | 0.0216 (🔴 High) | **0.527** | 0.0081 | 0.0135 | ✅ 2008 Crisis — Best R² |
| **1** | 2010–2014 | 0.0154 (Medium) | 0.267 | 0.0064 | 0.0093 | Post-crisis recovery |
| **2** | 2014–2018 | 0.0133 (🟢 Low) | 0.216 | 0.0056 | 0.0086 | Extremely calm period |
| **3** | 2018–2022 | 0.0185 (🔴 High) | **0.398** | 0.0072 | 0.0121 | ✅ COVID — Second best R² |
| **4** | 2022–2026 | 0.0175 (Medium) | 0.224 | 0.0073 | 0.0113 | ⚙️ Current production |

**Pattern**: R² in crisis (0.527, 0.398) >> R² in calm (0.216, 0.224). This is **correct for risk**: the model is most useful when it matters most.

---

## 📈 Temporal Validation (TimeSeriesSplit 5-Fold)

The pipeline ensures **zero temporal leakage** using expanding windows:

```
Fold 0: [Train: 2000-2005]        → [Test: 2005-2010]   ~445K/445K
Fold 1: [Train: 2000-2010]        → [Test: 2010-2014]   ~889K/445K
Fold 2: [Train: 2000-2014]        → [Test: 2014-2018]   ~1.33M/445K
Fold 3: [Train: 2000-2018]        → [Test: 2018-2022]   ~1.78M/445K
Fold 4: [Train: 2000-2022]        → [Test: 2022-2026]   ~2.22M/445K  ← Production
```

- **Train grows**: 0.44M → 2.22M (5× more data in fold 4 than fold 0)
- **Test constant**: ~445K samples (16.7% of each window)
- **Gap**: 5 business days between fold train-test (no data contamination)
- **Total evaluated**: 2.22M samples

**Why expanding window**: reflects real usage — when you predict today, you have all past data available. We don't discard old data if it's relevant.

---

## 🚀 Feature Engineering — 29 Indicators

The model uses only price and volume features, without ticker encoding:

### Feature Categories

| Category | Features | Importance |
|----------|----------|-------------|
| **Trend (57% importance)** | `price_vs_sma_10/20/50`, `rolling_mean_5`, `ema_12/26` | #1-#5 in importance |
| **Volatility/Range** | `rolling_max_10`, `rolling_min_10` | #2-#3 (11% + 10%) |
| **Returns** | `simple_return`, `log_return`, `return_lag_1/2/3/5/10` | #10 in importance |
| **Volume** | `volume_lag_1/3/5`, `volume_sma_10/20`, `zscore_volume_20` | #6 rank |
| **Momentum** | `rsi_14`, `macd`, `macd_signal`, `macd_hist` | Secondary |
| **Market** | `market_regime` | Captures market regimes |

### Top 3 Consistent Features Across All Folds

1. **`price_vs_sma_50` (24%)**
   - Distance of price from its 50-day moving average
   - Indicates if price is "stretched" (far from trend)
   - Greater predictability of future volatility
   - Correlation with price reversals

2. **`rolling_max_10` (11%)**
   - Maximum price in last 10 days
   - Indicates proximity to recent peaks
   - Zones of possible reversal → volatility

3. **`rolling_min_10` (10%)**
   - Minimum price in last 10 days
   - Complements max_10 (captures total range)

**Why no classic volatility indicators in top 3**: *Trend/relative price* features dominate over *momentum/volatility* features for predicting future volatility.

---

## 🔬 Experimentation — From One-Hot to Minimalism

### Model Evolution

| Exp | Features | R² | MAE | Time | Status |
|-----|----------|-----|------|--------|--------|
| 1. Baseline one-hot | 496 (29 ind + 467 ticker) | 0.3153 | 0.007049 | 369.5s | Baseline |
| 2. Log-target | 496 | 0.2929 | — | — | ❌ Reverted (worse R²) |
| 3. HP tuning (more regularization) | 496 | 0.2852 | — | — | ❌ Reverted (over-regularization) |
| **4. No ticker encoding 🎯** | **29 indicators only** | **0.3265** | **0.006941** | **137.3s** | ✅ **FINAL** |

### Final Model Improvement vs One-Hot Baseline

```
R²:             +3.6%  ↑ (0.3153 → 0.3265)
MAE:            -1.5%  ↓ (0.007049 → 0.006941)
RMSE:           -1.0%  ↓ (0.011068 → 0.010952)
Train time:    -60.0%  ↓ (339.4s → 135.6s)
Total time:    -62.8%  ↓ (369.5s → 137.3s)
```

### Why Was One-Hot Ticker Removed?

**467 dummy columns** where **99.8% of values = 0**:
- XGBoost wasted splits on features almost always zero
- Learned ticker-specific rules ("if TSLA → +0.005") — doesn't generalize
- Fragmented tree capacity (splits for identifying tickers vs indicator relationships)
- **Result**: Better R², less overfitting, 63% faster

**Ticker information persists without encoding**: TSLA has higher `log_return`, more extreme `rsi_14`, higher `volume_zscore` than KO. The model learns these differences through indicator values.

### Lessons from Failed Experiments

1. **Log-target degraded R²** (0.315 → 0.293)
   - Logarithm compresses high values
   - For risk, we want to penalize errors in high volatility more (absolute scale)
   - Raw target is better for practical application

2. **More aggressive HP tuning worsened** (R² 0.315 → 0.285)
   - 29 features is small space
   - max_depth=6 already captures sufficient interactions
   - More regularization = reduced capacity unnecessarily

---

## 📊 Detailed Analysis of Results

### Prediction Bias (Residuals)

| Fold | Mean Residual | Direction | Meaning |
|------|---------------|-----------|---------|
| 0 | +0.0003 | Neutral | Unbiased (crisis) |
| 1 | **-0.0018** | Overestimates | Model predicts more vol than occurs |
| 2 | -0.0004 | Neutral | Unbiased (calm) |
| 3 | +0.0012 | Slight underestimate | Doesn't fully capture COVID peak |
| 4 | -0.0006 | Neutral | Unbiased (production) |

**Fold 1 is exceptional**: Negative bias of -0.2σ. The model, trained with 2008 crisis data, is conservative during post-crisis recovery (prefers to overestimate risk). **For risk management, this is acceptable and even desirable**.

### Error by Volatility Level

Absolute error **grows monotonically** with actual volatility:

- Decile 1 (lowest volatility): MAE ≈ 0.002
- Decile 10 (highest volatility): MAE ≈ 0.015-0.025 (5-8× higher)

**Interpretation**: 
- Largest errors occur at extreme values (most relevant for risk)
- However, R² is higher in folds with high volatility — the model captures **relative direction and magnitude** even if absolute error is higher
- **Stable metric**: MAE/average volatility ≈ 40% across all folds (constant relative error)

### Prediction Range Compression

The model predicts in a **narrower range** than actual:

- Standard deviation of predictions (~40-65% of actual std)
- In scatter plots: point cloud narrower than reference diagonal

**This is normal and expected**: XGBoost with MSE loss optimizes for conditional mean. For applications needing extreme predictions without bias, calibration (isotonic regression) can be applied as post-processing.

---

## 📉 Available Visualizations

The pipeline generates **14 charts** in `data/model_outputs/`:

### Category 1: Regression Metrics

1. **R² Scores per Fold** — Coefficient of determination by temporal period
2. **MAE vs RMSE** — Absolute error vs squared error (reveals outliers)
3. **Metrics Heatmap** — Quick view R²/MAE/RMSE (green=better, red=worse)

### Category 2: Temporal Analysis

4. **Timeline of Folds** — Train/test windows on temporal scale (verifies no-leakage)
5. **Sample Distribution** — Train growth, constant test
6. **Temporal Trend** — Evolution of R²/MAE/RMSE (shows crisis vs calm pattern)

### Category 3: Performance Comparisons

7. **Metrics Comparison** — Relative ranking of each fold
8. **Timing: Train vs Inference** — Computational bottleneck
9. **Feature Count Summary** — 29 features vs previous version (496)

### Category 4: Training Diagnostics

10. **Actual vs Predicted (Scatter)** — Identify biases/under-overestimation per fold
11. **Residuals Distribution** — Histograms with normal fit (detects systematic bias)
12. **Feature Importance** — Top 15 features per fold (shows stability)
13. **Error by Volatility Decile** — MAE in volatility bins (identifies extreme value problems)
14. **Fold Sample Distribution** — Train/test samples per fold

---

## 🏗️ Project Architecture

```
sp500-dataset-ml-pipeline/
├── packages/
│   ├── ml/
│   │   ├── dataset/           # Phase 1: Data pipeline
│   │   │   ├── loader.py      # Download data from Kaggle
│   │   │   ├── preparation.py # Data cleaning and validation
│   │   │   ├── features.py    # Engineering 29 indicators
│   │   │   ├── main.py        # Phase 1 orchestration
│   │   │   └── eda/           # Exploratory visualizations
│   │   │
│   │   └── model/             # Phases 2-5: ML pipeline
│   │       ├── features/      # Phase 2: Feature selection
│   │       ├── cross_validation/  # Phase 3: TimeSeriesSplit setup
│   │       ├── training/      # Phase 4: XGBoost training
│   │       ├── visualization/ # Phase 5: 14 evaluation charts
│   │       ├── artifacts/     # Trained models + metadata
│   │       └── main.py        # Phases 2-5 orchestration
│   │
│   └── app/                   # API and Web UI
│       ├── api.py             # FastAPI backend
│       └── web/               # Frontend HTML/CSS/JS
│
├── data/
│   ├── processed/dataset.parquet   # Phase 1 output: 2.67M rows × 40 columns
│   ├── model_outputs/               # Phase 5: 14 PNG visualizations
│   └── eda/                         # EDA visualizations
│
├── AGENTS.md      # Guide for AI coding agents
├── run_pipeline.py   # End-to-end orchestrator
└── README.md (this file)
```

### Phase Flow

```
Phase 1: Dataset Pipeline
  Kaggle → loader.py → preparation.py → features.py → dataset.parquet
  Output: 2,668,192 rows × 40 columns (29 IND + target + metadata)

Phases 2-5: ML Pipeline
  ├─ Phase 2: Feature Selection → X (2.67M × 29), Y (volatility)
  ├─ Phase 3: TimeSeriesSplit → 5 temporal folds
  ├─ Phase 4: XGBoost Training → 5 independent models
  └─ Phase 5: Visualization → 14 charts in data/model_outputs/
```

---

## 🚀 Quick Start

### 1. Setup & Dependencies

```bash
# Install dependencies (requires Python 3.12+)
uv sync

# Configure Kaggle credentials (required for Phase 1)
# Visit: https://www.kaggle.com/settings/account → API → Download token
# Place at ~/.kaggle/kaggle.json
```

### 2. Run Full Pipeline

```bash
# Entire pipeline: Phase 1 (data) + Phases 2-5 (model) + API
python run_pipeline.py

# Skip API
python run_pipeline.py --no-api

# API only (if models already trained)
python run_pipeline.py --api-only
```

### 3. Stepwise Execution

```bash
# Phase 1: Download data, prepare, engineer features
cd packages/ml/dataset && python main.py
# Output: data/processed/dataset.parquet (2.67M × 40 columns)

# Phases 2-5: Feature selection, CV, training, visualization
cd packages/ml/model && python main.py
# Output: 5 XGBoost models + fold_training_summary.json + 14 PNGs
```

### 4. Start API Server

```bash
cd packages/app && python -m uvicorn api:app --host 0.0.0.0 --port 8080

# Endpoints:
# GET  /api/health                        # Model status + 5-fold metrics
# GET  /api/model/info                    # Detailed fold 4 report
# GET  /api/predict/{ticker}              # Single ticker prediction (live yfinance)
# GET  /api/screener?limit=50             # All 467 tickers ranked by volatility
```

### 5. Web UI

```bash
cd packages/app/web && python -m http.server 8000
# Open http://localhost:8000 → Dashboard + Screener + Model Hub
```

---

## 📊 API Endpoints (Post-Training)

### `GET /api/health`
**Model status and global metrics**
```json
{
  "status": "ready",
  "cross_validation": {
    "r2_mean": 0.3265,
    "r2_std": 0.1197,
    "mae_mean": 0.006941,
    "rmse_mean": 0.010952
  },
  "dataset": {
    "n_samples": 2668192,
    "n_tickers": 467,
    "date_range": "2000-03-16 to 2026-02-12"
  }
}
```

### `GET /api/predict/AAPL?period=6mo`
**Volatility prediction for a ticker + recent history**
```json
{
  "ticker": "AAPL",
  "prediction": {
    "volatility_5d": 0.0185,
    "volatility_annualized": 0.4157
  },
  "recent_data": [
    {"date": "2026-02-10", "price": 235.45, "close": 235.45},
    ...
  ]
}
```

### `GET /api/screener?limit=50&sort=volatility_daily`
**All S&P 500 tickers ranked by predicted volatility**
```json
{
  "tickers": [
    {"ticker": "TSLA", "volatility_daily": 0.0345, "regime": 2},
    {"ticker": "NVDA", "volatility_daily": 0.0298, "regime": 2},
    ...
  ],
  "generated_at": "2026-02-12T10:30:00Z"
}
```
## 🔴 Known Limitations

### 1. Poor Performance During Calm Periods
- **Fold 2** (2014-2018, calm): R²=0.216
- **Fold 4** (2022-2026, production): R²=0.224
- ~70% of data are in calm regimes (post-crisis + post-COVID)
- **Implication**: Model has low predictive power most of the time, but improves significantly during crises (where it matters most)

### 2. No External Features
Uses only price/volume-derived features. **Does NOT include**:
- Macroeconomic (Fed Funds rate, yield curve, inflation, PMI)
- Fundamentals (earnings, book-to-price, sector)
- Sentiment (VIX, put/call ratio, news sentiment)
- Cross-sectional (inter-ticker correlations, sector leadership)

These could capture volatility components during regime transitions.

### 3. No Sector Information
467 tickers without GICS sector labels. Model can't learn sector dynamics (e.g., "all energy tickers behave similarly during oil shocks").

### 4. Temporal Degradation
Fold 4 (production) has R²=0.224 vs fold 0 (0.527). Possible causes:
- Post-COVID market dynamics differ
- High inflation + rising rates + tech concentration
- Needs periodic retraining

### 5. Conservative on Extreme Values
Model underestimates volatility at highest values (bias in fold 3). Without calibration, unreliable for tail risk.

---

## 🎯 Conclusions

### What Works Well ✅

1. **Valid temporal prediction**: R²=0.3265 is solid for volatility (GARCH models typically <0.10)
2. **Better during crisis**: R²=0.527 in 2008, R²=0.40 in COVID (exactly when needed most)
3. **Generalizes across tickers**: No ticker encoding, shares patterns between companies
4. **Fast**: 0.72 μs/sample inference (real-time viable)
5. **Strong correlation**: Pearson r=0.64 indicates good relative direction/magnitude capture

### What Doesn't Work Well ❌

1. **Calm periods**: Volatility is mostly idiosyncratic noise (unpredictable)
2. **Extreme values**: Compresses prediction range
3. **No cross-sectional learning**: Can't learn ticker-pair correlations or sector effects
4. **Performance degrades over time**: Fold 4 vs fold 0 requires monitoring and retraining

### Deployment Recommendation

**Model is production-ready** with:
- ✅ Retraining every 1-3 months (add new data to train set)
- ✅ Continuous R² monitoring (alert if drops <0.15)
- ✅ Contextualized interpretation (high predictions are more reliable than low)
- ✅ Calibration for extreme values if tail risk modeling needed

---

## 🔮 Future Work

### Short Term (High Priority)

1. **Ticker target encoding**: 2-3 dense features (`ticker_volatility_rank`, `ticker_beta_vs_spy`) to recover ticker effect without overfitting
2. **SHAP interpretability**: Understand how each feature affects predictions (direction, interactions, non-linearities)
3. **Prediction calibration**: Isotonic regression to correct range compression
4. **Continuous monitoring**: Dashboard for production performance tracking (fold 4)

### Medium Term (Medium Priority)

5. **Sector features**: GICS (11 sectors) to capture sector rotations
6. **Macro features**: Fed Funds rate, yield curve, VIX, inflation, PMI
7. **Optimal training window**: Is 10 years better than 22 years for regime adaptation?
8. **Online inference**: Production pipeline for real-time predictions

### Long Term (Low Priority)

9. **Deep Learning**: LSTM/Transformer for longer temporal dependencies
10. **Multi-horizon**: Single model for 1d, 5d, 21d, 63d volatility
11. **Probabilistic model**: Quantile regression or evidential networks for prediction intervals

---

## 📚 Additional Documentation

- **AGENTS.md** — Guide for AI agents and integration
- **docs/guia_visualizaciones.md** — Detailed description of 14 charts
- **docs/analisis_resultados_modelo.md** — Complete results analysis
- **docs/analisis_importancia_limitaciones.md** — Feature importance and limitations
- **docs/datos_cuantitativos_modelo.md** — Numerical metrics and tables
- **docs/evaluacion_visualizaciones.md** — Visualization categorization

---

## 📞 Support & Contribution

This is a machine learning project for volatility prediction. For:

- **Bugs & Issues**: Open issue in repository
- **Improvements**: PRs welcome (especially external features, sectors, macro)
- **Questions**: See documentation in `docs/`

---

## 📄 License

[Specify license]

---

## 🎓 Citation

If you use this model or data in academic work:

```
@project{sp500_volatility_2026,
  title={S&P 500 Volatility Prediction Pipeline},
  description={ML pipeline for 5-day volatility forecasting across 467 S&P 500 constituents},
  year={2026},
  author={[Author Name]},
  url={https://github.com/[repo]}
}
```

---

**Last Updated**: February 2026  
**Model Performance**: R² = 0.3265 ± 0.1197 | MAE = 0.006941 | Inference = 0.72 μs/sample










