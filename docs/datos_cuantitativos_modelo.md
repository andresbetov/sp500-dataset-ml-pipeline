# Datos Cuantitativos del Modelo — S&P 500 Volatility Prediction

---

## Validación Temporal (TimeSeriesSplit 5-Fold)

**Dataset**: 2,668,192 filas, 467 tickers, 2000-03-16 → 2026-02-12

| Fold | Train samples | Test samples | Train span | Test span | Ratio |
|---|---|---|---|---|---|
| fold_0 | 444,697 | 444,698 | 5.2 años (2000–2005) | 4.7 años (2005–2010) | 1.0× |
| fold_1 | 889,395 | 444,698 | 9.8 años (2000–2010) | 4.3 años (2010–2014) | 2.0× |
| fold_2 | 1,334,093 | 444,698 | 14.1 años (2000–2014) | 4.1 años (2014–2018) | 3.0× |
| fold_3 | 1,778,791 | 444,698 | 18.2 años (2000–2018) | 3.9 años (2018–2022) | 4.0× |
| fold_4 | 2,223,489 | 444,698 | 22.1 años (2000–2022) | 3.8 años (2022–2026) | 5.0× |

**Mecanismo**: TimeSeriesSplit con expanding window, gap de 5 días entre train y test. Train crece acumulativamente: 0.44M → 0.89M → 1.33M → 1.78M → 2.22M. Test constante: ~444,698 por fold (~16.7% de cada ventana). Total evaluado: **2,223,490 muestras**.

---

## Arquitectura de Modelos

**5 XGBoost regressors independientes** — 1 por fold, NO ensemble, NO comparten datos entre folds.

### Hiperparámetros finales (baseline)

| Parámetro | Valor |
|---|---|
| max_depth | 6 |
| learning_rate | 0.05 |
| n_estimators | 500 |
| min_child_weight | 1.0 |
| gamma | 0.0 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| colsample_bylevel | 0.8 |
| reg_alpha | 0.1 |
| reg_lambda | 1.0 |
| objective | `reg:squarederror` |
| tree_method | `hist` |
| eval_metric | `rmse` |

**Features**: 29 indicadores técnicos, 0 ticker dummies. Sin escala, sin label mapping, sin early stopping.

### Timing

| Fold | Train (s) | Inference (s) | Total (s) |
|---|---|---|---|
| fold_0 | 6.5 | 0.32 | 6.8 |
| fold_1 | 15.3 | 0.34 | 15.6 |
| fold_2 | 25.5 | 0.32 | 25.8 |
| fold_3 | 39.4 | 0.32 | 39.8 |
| fold_4 | 48.9 | 0.32 | 49.2 |
| **Total** | **135.6** | **1.6** | **137.3** |

Inference/sample: ~0.72 μs. **Mejora vs one-hot: -62.8%** (137.3s vs 369.5s).

---

## Experimentación

| Experimento | Features | R² | MAE | RMSE | Tiempo | Decisión |
|---|---|---|---|---|---|---|
| **1. Baseline one-hot** | 496 (29 ind + 467 ticker) | 0.3153 | 0.007049 | 0.011068 | 369.5s | Punto de partida |
| **2. Log-target** | 496 | 0.2929 | mejoró leve | — | — | ❌ Revertido |
| **3. HP tuning** | 496 (max_depth=8, reg_alpha=0.5, etc.) | 0.2852 | — | — | — | ❌ Revertido |
| **4. Sin ticker encoding** | 29 (solo indicadores) | **0.3265** | **0.006941** | **0.010952** | **137.3s** | ✅ **FINAL** |

### Mejora vs Baseline (one-hot)

| Métrica | One-hot | Sin ticker | Cambio |
|---|---|---|---|
| R² | 0.3153 | 0.3265 | **+3.6%** ↑ |
| MAE | 0.007049 | 0.006941 | **-1.5%** ↓ |
| RMSE | 0.011068 | 0.010952 | **-1.0%** ↓ |
| Train time | 339.4s | 135.6s | **-60.0%** ↓ |
| Inference time | 2.2s | 1.6s | **-27.3%** ↓ |
| Total time | 369.5s | 137.3s | **-62.8%** ↓ |

### ¿Por qué el one-hot ticker empeoró el modelo?

467 columnas dummy donde el **99.8% de las filas tienen valor 0**. XGBoost gasta splits en features casi siempre cero, sobreajusta a reglas como "si ticker=TSLA → sube la predicción" y no generaliza entre tickers con comportamiento similar. Sin encoding, el modelo aprende relaciones entre indicadores financieros (RSI, EMAs, volumen) que aplican a *todos* los tickers por igual.

### Lecciones de experimentos fallidos

- **Log-target**: degradó R² global de 0.315 a 0.2929, especialmente en folds de crisis (0, 3) y producción (4). Target raw penaliza errores en alta volatilidad, que es el uso deseado para riesgo.
- **HP tuning** (max_depth=8, min_child_weight=3, gamma=0.1, reg_alpha=0.5, reg_lambda=2.0, colsample=0.7, n_estimators=1000): empeoró R² en *todos* los folds (global: 0.2852). Menos regularización es mejor para este dataset.

---

## Resultados del Modelo Final

### Métricas Globales

| Métrica | Valor |
|---|---|
| **R²** | **0.3265 ± 0.1197** (min=0.2164, max=0.5272) |
| **MAE** | **0.006941 ± 0.000860** |
| **RMSE** | **0.010952 ± 0.001801** |
| Pearson r | 0.6406 |
| Spearman ρ | 0.5725 |

### Por Fold

| Fold | Periodo test | Volatilidad media | R² | MAE | RMSE |
|---|---|---|---|---|---|
| **0** | 2005–2010 (crisis 2008) | 0.02156 | **0.5272** | 0.008132 | 0.013499 |
| **1** | 2010–2014 (post-crisis) | 0.01539 | **0.2668** | 0.006441 | 0.009310 |
| **2** | 2014–2018 (calma) | 0.01331 | **0.2164** | 0.005598 | 0.008581 |
| **3** | 2018–2022 (COVID) | 0.01845 | **0.3982** | 0.007200 | 0.012094 |
| **4** | 2022–2026 (producción) | 0.01752 | **0.2239** | 0.007335 | 0.011277 |

### Target Distribution (test set)

| Fold | Media | Std | Mediana | P95 | Min | Max |
|---|---|---|---|---|---|---|
| 0 | 0.02156 | 0.01963 | 0.01591 | 0.05747 | 0.0 | 0.55604 |
| 1 | 0.01539 | 0.01087 | 0.01268 | 0.03500 | 0.0 | 0.13991 |
| 2 | 0.01331 | 0.00969 | 0.01096 | 0.03000 | 0.0 | 0.17327 |
| 3 | 0.01845 | 0.01559 | 0.01440 | 0.04468 | 0.0 | 0.25554 |
| 4 | 0.01752 | 0.01280 | 0.01433 | 0.04001 | 0.0 | 0.21085 |
| **Global** | **0.01797** | **0.01529** | **0.01393** | **0.04420** | **0.0** | **0.55604** |

### Sesgo de Predicción (Residuos)

| Fold | Residuo medio (bias) | Std residual | Sesgo (mean/std) | Dirección |
|---|---|---|---|---|
| 0 | +0.000313 | 0.01350 | +0.023 | Neutro |
| 1 | -0.001839 | 0.00913 | **-0.202** | Sobreestima |
| 2 | -0.000397 | 0.00857 | -0.046 | Neutro |
| 3 | +0.001200 | 0.01203 | +0.100 | Subestima leve |
| 4 | -0.000575 | 0.01126 | -0.051 | Neutro |

### Patrón Crítico

Fold 0 (crisis 2008, volatilidad media más alta) → **mejor R² (0.527)**. Fold 2 (calma 2014-2018, volatilidad media más baja) → **peor R² (0.216)**. Fold 4 (producción) → **R²=0.224**.

**El modelo predice mejor en periodos de alta volatilidad** — exactamente donde más se necesita para gestión de riesgo.

---

## Feature Importance (Top 3 por fold, gain-based)

| Fold | #1 | #2 | #3 |
|---|---|---|---|
| 0 | price_vs_sma_50 (16.7%) | rolling_max_10 (14.7%) | rolling_min_10 (11.0%) |
| 1 | price_vs_sma_50 (25.2%) | rolling_max_10 (10.5%) | rolling_min_10 (10.4%) |
| 2 | price_vs_sma_50 (26.2%) | rolling_max_10 (10.6%) | rolling_min_10 (9.0%) |
| 3 | price_vs_sma_50 (24.0%) | rolling_max_10 (10.0%) | rolling_min_10 (9.6%) |
| 4 | price_vs_sma_50 (27.9%) | rolling_max_10 (9.9%) | rolling_min_10 (9.7%) |

`price_vs_sma_50` domina consistentemente (17–28% de importancia relativa). Las 3 features principales son estables en todos los folds.
