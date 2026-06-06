# Guía de Visualizaciones del Modelo de Volatilidad

## Resumen del Pipeline

El modelo predice `realized_volatility_5d` (volatilidad realizada a 5 días) usando **29 indicadores técnicos** financieros basados en precio/volumen. Usa **TimeSeriesSplit con 5 folds** y gap de 5 días para evitar leakage temporal. 1 modelo XGBoost por fold (5 modelos independientes).

Resultados baseline: **R² = 0.3265 ± 0.1197**, **MAE = 0.006941**, **RMSE = 0.010952**.

---

## 1. R² Scores por Fold

![R² por Fold](../data/model_outputs/r2_scores_by_fold.png)

**Qué muestra**: El coeficiente de determinación (R²) para cada fold de validación temporal.

**Cómo interpretar**:
- Mide qué porcentaje de la varianza de la volatilidad real es explicada por el modelo.
- **Fold 0** (2005-2010): R² ≈ 0.53 — el mejor. Incluye la crisis financiera de 2008. Alta volatilidad es más fácil de predecir porque la señal es más fuerte que el ruido.
- **Fold 1** (2010-2014): R² ≈ 0.27 — el peor junto con fold 2. Periodo de baja volatilidad post-crisis donde el modelo no encuentra patrones claros.
- **Fold 4** (2022-2026): R² ≈ 0.22 — periodo de producción. El modelo predice peor la volatilidad reciente. Es la métrica más relevante para uso en producción.
- La línea roja punteada es el R² promedio (0.3265).

---

## 2. MAE y RMSE por Fold

![MAE vs RMSE](../data/model_outputs/mae_rmse_comparison.png)

**Qué muestra**: Mean Absolute Error (MAE) y Root Mean Squared Error (RMSE) para cada fold.

**Cómo interpretar**:
- **MAE**: Error absoluto promedio. En fold 2 es menor (0.0056) porque la volatilidad en ese periodo es baja, no porque el modelo sea mejor.
- **RMSE**: Penaliza más los errores grandes que el MAE (eleva al cuadrado antes de promediar). Cuando RMSE >> MAE, hay outliers de error grandes.
- Fold 0 tiene RMSE más alto (0.0135) pese a tener el mejor R², porque hay más volatilidad que predecir — valores altos significa errores absolutos más grandes aunque la *proporción* del error sea menor.
- Regla general: comparar folds por R², no por MAE/RMSE absolutos, porque la escala del error depende de la volatilidad del periodo.

---

## 3. Heatmap de Métricas

![Heatmap](../data/model_outputs/metrics_heatmap.png)

**Qué muestra**: Las 3 métricas (R², MAE, RMSE) normalizadas en una matriz de color. Verde = mejor, Rojo = peor.

**Cómo interpretar**:
- Los valores escritos son los reales; el color es la normalización **por métrica** (0 = peor fold, 1 = mejor fold).
- Para MAE y RMSE el color está invertido (menor error = mejor = verde).
- Útil para identificar patrones: fold 0 es consistentemente verde (mejor), fold 2 es el más mixto (mejor MAE/RMSE pero peor R² — indicador de baja volatilidad no predictiva).
- Vista rápida de qué folds están balanceados vs cuáles tienen fortalezas o debilidades específicas.

---

## 4. Línea de Tiempo Temporal de Folds

![Temporal Folds](../data/model_outputs/temporal_fold_ranges.png)

**Qué muestra**: Las ventanas de entrenamiento (azul) y prueba (naranja) para cada fold en una línea de tiempo.

**Cómo interpretar**:
- **Validación temporal pura**: cada fold entrena solo con datos del pasado y evalúa en el futuro. No hay mezcla temporal.
- Fold 0 entrena con datos 2000-2005, prueba en 2005-2010.
- Fold 4 entrena con datos 2000-2022, prueba en 2022-2026.
- El patrón expanding window: cada fold agrega el test anterior al nuevo train.
- El gap de 5 días entre train y test evita que información de días recientes se filtre.
- Esta gráfica es la **garantía de que no hay leakage temporal** en el pipeline.

---

## 5. Distribución de Muestras por Fold

![Sample Distribution](../data/model_outputs/fold_sample_distribution.png)

**Qué muestra**: Stacked bar (o grouped bar) con la cantidad de muestras de entrenamiento y prueba por fold.

**Cómo interpretar**:
- El train crece con cada fold (expanding window): ~445K → ~889K → ~1.33M → ~1.78M → ~2.22M.
- El test es constante (~445K muestras por fold, ~20% del total de cada ventana).
- Útil para entender por qué fold 4 tarda más en entrenar (116s vs 20s en fold 0): tiene 5× más datos de entrenamiento.
- Todas las muestras de test combinadas = ~2.22M, que es el total de predicciones generadas.

---

## 6. Tendencia Temporal de Métricas

![Temporal Metric Trend](../data/model_outputs/temporal_metric_trend.png)

**Qué muestra**: Evolución de R², MAE y RMSE a lo largo de los folds temporales (0 → 4).

**Cómo interpretar**:
- **R²**: pico en fold 0 (crisis 2008), baja en folds 1-2 (periodos tranquilos), sube en fold 3 (COVID), vuelve a bajar en fold 4 (producción). Patrón de "doble U" — el modelo predice bien crisis.
- **MAE**: sube en folds de alta volatilidad (0, 3, 4) y baja en folds tranquilos (1, 2). Contraintuitivo: MAE bajo ≠ buen modelo, solo significa que la volatilidad es baja.
- **RMSE**: mismo patrón que MAE pero amplificado en folds de alta volatilidad.
- Conclusión: el modelo es útil en periodos de estrés (donde más se necesita) y mediocre en periodos estables.

---

## 7. Comparación de Métricas (Normalizado)

![Metrics Comparison](../data/model_outputs/metrics_comparison.png)

**Qué muestra**: Las 3 métricas superpuestas en barras (valores reales) + líneas de tendencia normalizadas [0, 1].

**Cómo interpretar**:
- Las barras muestran los valores absolutos (R² en azul, MAE en naranja, RMSE en verde).
- Las líneas punteadas muestran la versión normalizada donde 1 = mejor fold para esa métrica.
- MAE y RMSE se invierten en la normalización (valor más bajo = score más alto).
- Útil para comparar el *ranking relativo* de cada fold: ¿un fold es consistentemente bueno o malo en todas las métricas?
- Fold 0 es el claro ganador en R², pero fold 2 tiene el mejor MAE/RMSE absoluto (por la baja volatilidad del periodo).

---

## 8. Tiempo de Entrenamiento vs Inferencia

![Timing Breakdown](../data/model_outputs/timing_breakdown.png)

**Qué muestra**: Tiempo de entrenamiento (azul) y de inferencia (naranja) por fold, en segundos.

**Cómo interpretar**:
- El tiempo de entrenamiento crece linealmente con el tamaño del conjunto de entrenamiento: fold 0 = ~8s, fold 4 = ~42s.
- La inferencia es prácticamente constante (~0.3-0.4s) porque el test set tiene el mismo tamaño en todos los folds.
- El cuello de botella es el entrenamiento (122.7s total), no la inferencia (1.7s total).
- Con 29 features (sin ticker encoding), el entrenamiento es ~3× más rápido que con los 496 features originales.

---

## 9. Distribución de Samples

![Sample Distribution](../data/model_outputs/sample_distribution.png)

**Qué muestra**: Muestras de entrenamiento vs prueba por fold en millones, como barras agrupadas.

**Cómo interpretar**:
- El set de entrenamiento crece exponencialmente (expanding window): 0.44M → 0.89M → 1.33M → 1.78M → 2.22M.
- El test es constante en ~0.44M por fold.
- Total de muestras en el dataset: ~2.67M (suma de train fold 4 + test fold 4, o todas las filas únicas).
- Esto confirma que el split temporal es correcto: siempre se prueba en el ~17% más reciente de la ventana actual.

---

## 10. Composición de Features

![Feature Count](../data/model_outputs/feature_count_summary.png)

**Qué muestra**: Cantidad de features indicadores vs features de ticker usados en el modelo.

**Cómo interpretar**:
- Actualmente: **29 features indicadores**, 0 features de ticker.
- Anteriormente (con one-hot encoding): 29 indicadores + ~467 ticker dummies = 496 features totales.
- El switch a solo indicadores mejoró el R² global de 0.315 a 0.3265, redujo el tiempo de entrenamiento ~3×, y permitió que el modelo generalice entre tickers en lugar de aprender reglas específicas por ticker.
- Los 29 indicadores capturan suficiente identidad intrínseca del ticker a través de sus valores (TSLA tiene log_return alto, KO tiene volumen estable).

---

## 11. Real vs Predicho (Scatter)

![Actual vs Predicted](../data/model_outputs/actual_vs_predicted.png)

**Qué muestra**: Para cada fold, un scatter plot del valor real (x) vs el valor predicho (y). La línea roja diagonal es la referencia de predicción perfecta (y = x).

**Cómo interpretar**:
- Puntos sobre la diagonal → predicción perfecta.
- Puntos bajo la diagonal → el modelo subestima la volatilidad.
- Puntos sobre la diagonal → el modelo sobrestima la volatilidad.
- Fold 0: puntos más dispersos pero alineados con la diagonal — buen R² pese a la dispersión.
- Fold 1 y 2: puntos más concentrados cerca del origen (baja volatilidad), menos dispersión pero también menos correlación.
- Fold 4: sesgo visible hacia la subestimación en volatilidades altas (puntos debajo de la diagonal en valores > 0.04) — el modelo es conservador.
- El gráfico combinado muestra todos los folds con colores distintos. Idealmente queremos una nube simétrica alrededor de la diagonal.

---

## 12. Distribución de Residuos

![Residuals](../data/model_outputs/residuals_distribution.png)

**Qué muestra**: Histograma de los residuos (real - predicho) para cada fold, con una curva normal ajustada superpuesta.

**Cómo interpretar**:
- Residuo positivo → el modelo subestimó (real > predicho).
- Residuo negativo → el modelo sobrestimó (real < predicho).
- Ideal: residuos centrados en 0 con distribución normal (sin sesgo sistemático).
- Fold 0: distribución más ancha (mayor variabilidad de errores), pero centrada cerca de 0.
- Fold 1 y 2: distribución angosta y centrada — errores pequeños y sin sesgo, típico de periodos de baja volatilidad.
- Fold 4: ligero sesgo positivo (la media del residuo es > 0) — el modelo subestima sistemáticamente en el periodo de producción. Esto es importante para calibración si se usa en producción.
- El panel combinado superpone todas las folds: permite ver visualmente qué folds contribuyen más al error total.

---

## 13. Importancia de Features

![Feature Importance](../data/model_outputs/feature_importance.png)

**Qué muestra**: Top 15 features más importantes por fold, según el `feature_importances_` de XGBoost (ganancia promedio de splits).

**Cómo interpretar**:
- Las features con mayor importancia son las que más contribuyen a reducir el error en los árboles de decisión.
- Features consistentemente importantes en todos los folds sugieren relaciones estables entre indicadores y volatilidad.
- Features importantes solo en algunos folds sugieren relaciones específicas de ese régimen de mercado.
- El panel agregado (último) promedia la importancia a través de los 5 folds con barras de error (±1 std), mostrando qué features son consistentemente relevantes vs cuáles son específicas de ciertos periodos.
- Features típicamente dominantes en modelos de volatilidad: `log_return`, `rsi_14`, `realized_volatility_5d` (laggeada), `volume_zscore`.

---

## 14. Error por Decil de Volatilidad

![Error by Volatility Bin](../data/model_outputs/error_by_volatility_bin.png)

**Qué muestra**: Mean Absolute Error (MAE) agrupado por deciles de la volatilidad real, con una línea por fold.

**Cómo interpretar**:
- El eje X muestra el valor promedio de volatilidad en cada decil (bins 0-9, de menor a mayor volatilidad).
- El eje Y muestra el MAE dentro de ese decil.
- Ideal: una línea horizontal baja — el modelo funciona igual de bien en toda la distribución de volatilidad.
- Realidad: el error crece con la volatilidad — es más difícil predecir con precisión absoluta cuando la volatilidad es alta.
- Fold 0 (crisis 2008) tiene la pendiente más pronunciada: los deciles altos tienen mucho más error.
- Fold 2 (periodo tranquilo) tiene la pendiente más suave porque la distribución de volatilidad está más comprimida.
- Las barras de error muestran la desviación estándar dentro de cada decil.
- Esto es clave para entender si el modelo es útil en escenarios de alta volatilidad (donde más importa) o si falla justo cuando más se necesita.
