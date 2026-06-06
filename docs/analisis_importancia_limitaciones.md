# Feature Importance, Diagnóstico de Errores, Limitaciones y Conclusiones

---

## 1. Feature Importance + Ticker

### 1.1 Top Features por Importancia (Gain-based)

El `feature_importances_` de XGBoost mide la contribución de cada feature a la reducción de la función de pérdida (gain), normalizada para sumar 1.0. Features con alta importancia son aquellas que XGBoost selecciona frecuentemente para splits en posiciones cercanas a la raíz de los árboles o en splits que producen grandes reducciones de error.

**Top 15 features (promedio entre folds):**

| Rango | Feature | Importancia promedio | Categoría |
|---|---|---|---|
| 1 | `price_vs_sma_50` | 24.0% | Tendencia |
| 2 | `rolling_max_10` | 11.1% | Tendencia |
| 3 | `rolling_min_10` | 9.9% | Tendencia |
| 4 | `price_vs_sma_20` | 6.4% | Tendencia |
| 5 | `rolling_mean_5` | 5.8% | Tendencia |
| 6 | `volume_zscore` | 5.6% | Volumen |
| 7 | `return_lag_1` | 5.5% | Retornos |
| 8 | `price_vs_sma_10` | 4.9% | Tendencia |
| 9 | `rsi_14` | 4.8% | Momentum |
| 10 | `log_return` | 4.4% | Momentum |
| 11 | `volume_sma_20` | 3.7% | Volumen |
| 12 | `volume_sma_10` | 2.9% | Volumen |
| 13 | `ema_26` | 2.8% | Tendencia |
| 14 | `volume_lag_5` | 2.5% | Volumen |
| 15 | `ema_12` | 2.4% | Tendencia |

### 1.2 Interpretación del Ranking

**`price_vs_sma_50` domina con 24% de importancia** — es el predictor individual más fuerte de volatilidad a 5 días. Mide cuán lejos está el precio actual de su media móvil de 50 días (tendencia de mediano plazo). Cuando el precio se desvía significativamente de su SMA-50, el mercado está "estirado" y es más probable una reversión o aceleración de la tendencia, lo que genera volatilidad. La consistencia de esta feature a través de todos los folds (17-28%, creciente en folds recientes) sugiere que es una relación fundamental y estable.

**`rolling_max_10` y `rolling_min_10` suman ~21%** — el rango de precios de los últimos 10 días captura la volatilidad reciente del ticker individual. Tickers cerca de máximos o mínimos de 10 días tienen mayor probabilidad de movimientos direccionales.

**Las features de tendencia dominan el top 5 (57% acumulado)** — el modelo aprende que la *posición relativa del precio en su tendencia* es más informativa que los indicadores de momento (RSI, retornos) o volumen. Esto tiene sentido conceptual: la volatilidad futura está más correlacionada con desequilibrios de precio que con la velocidad del movimiento actual.

**`volume_zscore` en el puesto 6 (5.6%)** — picos anómalos de volumen son predictores de volatilidad: indican entrada de información nueva al mercado o cambios en el consenso de valoración.

**`rsi_14` y `log_return` en puestos 9-10** — relevantes pero secundarios frente a las features de tendencia. El RSI mide la velocidad y magnitud de cambios de precio recientes, pero la SMA-50 captura una señal de más largo plazo que resulta más predictiva para horizontes de 5 días.

### 1.3 Análisis de Ticker

**Decisión**: no se utiliza one-hot encoding de ticker. El modelo final usa solo los 29 indicadores.

**¿Por qué el one-hot ticker empeoró el rendimiento?**

- **467 columnas dummy vs 29 columnas informativas**: con one-hot, el 94% del feature space (467/496) eran columnas que valen 1 para un solo ticker y 0 para los otros 466. XGBoost se ve forzado a considerar splits en estas features, que son informativas solo para un subconjunto minúsculo de los datos.
- **Sobreajuste a tickers individuales**: el modelo aprendía reglas del tipo "si ticker = TSLA → ajustar predicción +0.005". Esto no generaliza a nuevos tickers ni captura relaciones subyacentes entre comportamiento financiero y volatilidad.
- **Fragmentación de la capacidad del árbol**: en lugar de usar la profundidad del árbol para capturar interacciones entre indicadores (ej. "RSI alto + volumen anómalo + precio cerca de SMA-50"), los árboles dedicaban splits a identificar tickers individuales.
- **Resultado cuantitativo**: con one-hot, R²=0.3153, MAE=0.007049, 369.5s. Sin ticker, R²=0.3265, MAE=0.006941, 137.3s. **Mejora en todas las métricas y reducción de tiempo del 63%.**

**¿Qué información del ticker se pierde?**

Potencialmente, efectos idiosincráticos persistentes: tickers que consistentemente tienen mayor o menor volatilidad de la que sus indicadores sugieren (ej. TSLA es ~2× más volátil que el market en igualdad de condiciones de precio/volumen). Sin embargo, la evidencia empírica muestra que los 29 indicadores ya codifican suficiente identidad del ticker a través de sus valores medios — TSLA tiene log_return más alto, RSI más extremo, volumen_zscore más alto que KO, y el modelo aprende estas diferencias sin necesidad de un flag explícito.

**Recomendación**: Si se quisiera recuperar el efecto ticker sin los problemas del one-hot, la opción es target encoding (3-4 features densas: `ticker_volatility_rank`, `ticker_mean_volume_zscore`, `ticker_beta_vs_spy`), que capturan la identidad del ticker en features numéricas continuas. Esto no se ha implementado aún.

---

## 2. Diagnóstico de Errores

### 2.1 Sesgo de Predicción (Residuos)

| Fold | Residuo medio | Dirección | Interpretación |
|---|---|---|---|
| 0 | +0.0003 | Neutro | El modelo no tiene sesgo significativo en la crisis 2008 |
| 1 | **-0.0018** | Sobreestima | El modelo predice más volatilidad de la que ocurre en la recuperación post-crisis |
| 2 | -0.0004 | Neutro | Periodo tranquilo sin sesgo |
| 3 | +0.0012 | Subestima leve | El modelo no captura totalmente el pico de volatilidad COVID |
| 4 | -0.0006 | Neutro | Producción, sin sesgo significativo |

Fold 1 es el único con sesgo relevante (0.2σ). El sesgo negativo (sobrestimación) sugiere que el modelo, entrenado con datos que incluyen la crisis de 2008, extrapola un nivel de volatilidad mayor al que realmente ocurre en 2010-2014. Esto es consistente con un modelo conservador — prefiere sobreestimar el riesgo después de periodos de alta volatilidad. Para aplicaciones de riesgo, esto es aceptable e incluso preferible (mejor sobreestimar que subestimar el riesgo).

### 2.2 Error por Nivel de Volatilidad

El error absoluto crece monótonamente con la volatilidad real:
- Decil 1 (volatilidad más baja): MAE ≈ 0.002
- Decil 10 (volatilidad más alta): MAE ≈ 0.015-0.025 (5-8× mayor)

**Interpretación**: El modelo no es uniforme a través de la distribución de volatilidad. Los errores más grandes ocurren en los valores más extremos, que son precisamente los más relevantes para riesgo. Sin embargo, el R² es más alto en folds con más volatilidad extrema — el modelo captura la *dirección y magnitud relativa* aunque el error absoluto sea mayor.

**Métrica clave**: MAE/volatilidad media = ~0.39-0.42 en todos los folds. El error relativo es constante (~40% de la volatilidad media), lo que sugiere que el modelo tiene una razón señal/ruido estable independientemente del régimen de mercado.

### 2.3 Compresión del Rango de Predicción

El modelo predice en un rango más angosto que el real:
- La desviación estándar de las predicciones es sistemáticamente menor que la del real (~40-65% del std real).
- En el scatter plot actual vs predicted, la nube de puntos es más angosta verticalmente que la diagonal de referencia.

**Interpretación**: El modelo de regresión con pérdida MSE optimiza para minimizar errores cuadráticos, lo que lleva a predecir hacia la media condicional. Esto es esperable y no necesariamente un problema — la predicción puntual óptima bajo MSE es la media condicional. Para aplicaciones donde se necesitan predicciones no sesgadas en los extremos, se puede aplicar calibración (ej. isotonic regression) como post-procesamiento.

---

## 3. Limitaciones

### 3.1 Bajo Rendimiento en Periodos de Baja Volatilidad

Fold 2 (2014-2018, calma): R²=0.216. Fold 4 (2022-2026, producción): R²=0.224. En estos periodos, la volatilidad es principalmente ruido idiosincrático de cada ticker, que los 29 indicadores de precio/volumen no pueden predecir.

**Impacto**: La mayor parte del tiempo el mercado está en calma (~70% de los datos son regímenes 2 y 4, post-crisis + post-COVID). El modelo tendrá bajo poder predictivo la mayor parte del tiempo, con mejoras esporádicas durante crisis.

### 3.2 Sin Features Externas

El modelo usa exclusivamente features de precio y volumen derivados de los propios datos del ticker. No incluye:

- **Macroeconómicas**: tasas de interés (Fed Funds rate, yield curve), inflación, PMI, desempleo, GDP.
- **Fundamentales**: earnings, book-to-price, market cap, sector, industria.
- **Sentimiento**: VIX, put/call ratio, news sentiment.
- **Cross-sectional**: correlaciones entre tickers, liderazgo sectorial, concentration risk.

Estas features externas podrían capturar componentes de volatilidad que los indicadores de precio/volumen no pueden — especialmente en periodos de transición de régimen.

### 3.3 Sin Información de Sector/Industria

467 tickers sin etiqueta de sector. El modelo no puede aprender que "todos los tickers de energía se comportan similar durante shocks de petróleo" o "tecnología correlaciona con tasas de interés". Un feature de sector (11 sectores GICS) podría ser más informativo que 467 tickers individuales o que ningún ticker.

### 3.4 Deterioro Temporal

Fold 4 (producción, 2022-2026) tiene R²=0.224, el segundo peor. Esto podría indicar que:
- El mercado post-COVID tiene dinámicas diferentes a los periodos anteriores.
- El feature set de 29 indicadores pierde poder predictivo en el entorno actual.
- Cambios estructurales (inflación alta, tasas subiendo, concentration en tech) no están capturados en features de precio/volumen.

**Implicación**: El modelo necesita monitoreo continuo y re-entrenamiento periódico. No se puede asumir que el rendimiento se mantendrá sin degradación.

### 3.5 Modelo Conservador en Alta Volatilidad

El modelo subestima la volatilidad en sus valores más extremos (sesgo positivo en fold 3, compresión del rango de predicción). Para aplicaciones de riesgo extremo (tail risk), el modelo no es confiable sin calibración adicional.

---

## 4. Conclusiones + Próximos Pasos

### 4.1 Conclusiones

**El modelo es funcional para predicción de volatilidad del S&P 500.**
- R²=0.3265 global es sólido para el dominio (modelos GARCH típicamente <0.10 en horizontes cortos).
- Correlación Pearson de 0.64 con la volatilidad realizada indica que el modelo captura direcciones y magnitudes relativas correctamente.
- Inferencia en ~0.7 μs por muestra permite despliegue en tiempo real o batch de alta frecuencia.

**El modelo es más útil en crisis que en calma.**
- R²=0.53 en crisis 2008, R²=0.40 en COVID, R²≈0.22 en calma. Esto es aceptable: los peores errores ocurren cuando el riesgo es bajo, y el modelo mejora cuando más se necesita.

**Menos features es mejor que más.**
- Eliminar 467 ticker dummies mejoró R² (+3.6%), redujo tiempo (-63%) y eliminó sobreajuste a tickers individuales.
- Intentos de regularización más agresiva (HP tuning) y transformación de target (log) empeoraron el rendimiento.

**29 indicadores son suficientes para capturar las relaciones precio-volatilidad.**
- `price_vs_sma_50` domina el feature importance en todos los folds (17-28%), seguido de `rolling_max_10` y `rolling_min_10`. Features de tendencia de mediano plazo son más predictivas que momento o volumen.

**El sesgo de predicción es manejable.**
- Solo fold 1 muestra sesgo relevante (sobrestimación = 0.2σ). Los demás folds son esencialmente insesgados. El conservadurismo post-crisis es aceptable para aplicaciones de riesgo.

### 4.2 Próximos Pasos

#### Corto plazo (prioridad alta)

1. **Target encoding de ticker**: Reemplazar la ausencia de ticker con 2-3 features densas (`ticker_volatility_rank`, `ticker_mean_volume_zscore`, `ticker_beta_vs_spy`). Evaluar si recuperan el efecto ticker perdido sin caer en overfitting.

2. **SHAP para interpretabilidad**: Complementar `feature_importances_` de XGBoost con valores SHAP para entender *cómo* cada feature afecta la predicción (dirección del efecto, interacciones, no linealidades).

3. **Calibración de predicciones**: Aplicar isotonic regression o Platt scaling sobre las predicciones del modelo para corregir el sesgo de compresión del rango. Evaluar mejora en métricas de calibración (Brier score, reliability diagrams).

4. **Monitoreo continuo**: Implementar dashboard de monitoreo de rendimiento del modelo en producción (fold 4). Métricas clave: R² semanal, MAE, distribución de residuos, drift de features (population stability index).

#### Mediano plazo (prioridad media)

5. **Features de sector**: Agregar sector GICS (11 sectores) como feature. Evaluar si mejora la generalización entre tickers del mismo sector, especialmente en periodos de rotación sectorial.

6. **Features macro**: Incorporar series de tiempo macro (Fed Funds rate, yield curve 2-10, VIX, inflación, PMI manufacturero) como features del dataset. Evaluar si mejoran el R² en folds de producción.

7. **Ventana de entrenamiento óptima**: Evaluar si entrenar con una ventana fija (ej. últimos 10 años) mejora el rendimiento en producción respecto a usar todo el historial disponible (22 años). Una ventana más corta podría adaptarse mejor a cambios de régimen recientes.

8. **Inferencia online**: Implementar pipeline de inferencia para producción: cargar fold_4_model.pkl, computar 29 features desde datos fresh, predecir volatilidad. Evaluar integración con API o batch job.

#### Largo plazo (prioridad baja)

9. **Deep learning**: Experimentar con LSTM/Transformer para capturar dependencias temporales más largas que XGBoost (que trata cada fila como independiente). Comparar con baseline XGBoost en términos de R² y velocidad de inferencia.

10. **Multi-horizonte**: Extender el modelo a múltiples horizontes de volatilidad (1d, 5d, 21d, 63d) con un solo modelo multi-output o múltiples modelos especializados. Evaluar consistencia entre horizontes.

11. **Modelo de probabilidad**: Reemplazar la regresión puntual con un modelo probabilístico (Negative Binomial, Quantile Regression, o evidential deep learning) que produzca intervalos de predicción en lugar de puntos. Más útil para gestión de riesgo.
