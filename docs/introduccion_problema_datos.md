# Predicción de Volatilidad Realizada a 5 Días para el S&P 500: Un Enfoque con XGBoost y Validación Temporal Estricta

---

## El Problema

### ¿Qué predecimos?

**Target**: `realized_volatility_5d` — la desviación estándar de los retornos logarítmicos diarios en una ventana de 5 días hábiles hacia adelante.

```
realized_volatility_5d(t) = std(log_return[t+1 : t+6])
```

Es una variable **continua** (regresión), no clases discretas. El modelo predice *cuánta* volatilidad habrá, no *si* la habrá o su dirección.

### ¿Por qué es importante?

- **Gestión de riesgo**: la volatilidad es el insumo fundamental para el cálculo del Value at Risk (VaR), capital regulatorio, y sizing de posiciones.
- **Pricing de opciones**: la volatilidad es el principal parámetro en modelos de Black-Scholes y sus derivados.
- **Asset allocation**: períodos de alta volatilidad correlacionan con cambios en correlaciones entre activos, afectando la diversificación de portafolios.
- **Trading**: estrategias de volatilidad (straddles, strangles, volatility arbitrage) dependen de predicciones precisas.

### ¿Por qué regresión y no clasificación?

Clasificar en "alta/ baja volatilidad" descarta información. La magnitud importa: no es lo mismo una volatilidad de 0.01 que de 0.10. Un modelo de regresión permite:
- Calibrar el tamaño de posición inversamente a la volatilidad predicha.
- Calcular métricas de riesgo continuas (VaR, CVaR).
- Detectar cambios graduales en regímenes de volatilidad.

### Desafíos intrínsecos

- La volatilidad financiera tiene **colas pesadas** (kurtosis = 37.5, skewness = 4.1).
- El componente **idiosincrático** (específico de cada ticker) es grande y difícil de predecir con features de precio/volumen.
- La predictibilidad varía con el **régimen de mercado**: en crisis domina el componente sistemático (predecible), en calma domina el ruido (impredecible).

---

## Los Datos

### Origen

Kaggle: [S&P 500 Historical Data](https://www.kaggle.com/datasets/jacksaleeby/s-and-p500-historical-data) — datos diarios de precio y volumen para los tickers del S&P 500 entre 2000 y 2026.

### Estructura del Dataset Final

**2,668,192 filas × 40 columnas** (804.5 MB, almacenado como Parquet). Sin valores nulos.

#### Columnas originales

| Columna | Tipo | Descripción |
|---|---|---|
| `ticker` | string (467 únicos) | Símbolo del ticker |
| `date` | datetime64[us] | Fecha de trading (2000-03-16 a 2026-02-12) |
| `open`, `high`, `low`, `close` | float64 | Precios OHLC diarios |
| `adj_close` | float64 | Precio de cierre ajustado |
| `volume` | float64 | Volumen de trading |

#### Indicadores técnicos (29 features)

| Categoría | Features |
|---|---|
| **Tendencia** (10) | `ema_12`, `ema_26`, `price_vs_sma_10`, `price_vs_sma_20`, `price_vs_sma_50`, `rolling_max_10`, `rolling_min_10`, `rolling_mean_5`, `zscore_price_20`, `zscore_price_vs_sma_20` |
| **Momentum** (3) | `rsi_14`, `simple_return`, `log_return` |
| **Volumen** (7) | `volume_lag_1`, `volume_lag_3`, `volume_lag_5`, `volume_sma_10`, `volume_sma_20`, `volume_zscore`, `zscore_volume_20` |
| **Retornos lag** (5) | `return_lag_1`, `return_lag_2`, `return_lag_3`, `return_lag_5`, `return_lag_10` |
| **MACD** (3) | `macd`, `macd_hist`, `macd_signal` |
| **Contexto** (1) | `market_regime` (int8: 0-4) |

#### Targets

| Columna | Descripción |
|---|---|
| `realized_volatility_5d` | Target — volatilidad realizada forward 5 días |
| `target_log_vol_5d` | Log-transformación del target (no utilizada en modelo final) |

### Distribución del Target

| Estadístico | Valor |
|---|---|
| Media | 0.01797 |
| Mediana | 0.01393 |
| Desviación estándar | 0.01529 |
| Mínimo | 0.00000 |
| Máximo | 0.55604 |
| P25 | 0.00908 |
| P75 | 0.02169 |
| P95 | 0.04420 |
| P99 | 0.07849 |
| Skewness | 4.106 |
| Kurtosis | 37.48 |

Colas extremadamente pesadas (kurtosis 37.5, skewness 4.1). El 95% de los valores está por debajo de 0.044, pero existen valores hasta 0.556.
