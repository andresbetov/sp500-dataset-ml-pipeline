# Análisis e Interpretación de Resultados del Modelo de Volatilidad S&P 500

---

## 1. Validación Temporal — TimeSeriesSplit 5-Fold

### Los datos

5 folds con expanding window. Train crece: 0.44M → 2.22M. Test constante: ~445K. Gap de 5 días entre train y test. Total evaluado: 2,223,490 muestras.

### Interpretación

**¿Por qué expanding window y no sliding window?** El expanding window refleja el uso real del modelo: cuando entrenas hoy, tienes *todo* el pasado disponible. No tiene sentido descartar datos antiguos si son relevantes. El grow de train de 0.44M a 2.22M implica que fold 4 entrena con 5× más datos que fold 0.

**¿Por qué test constante?** 445K muestras por fold (~16.7% de cada ventana) es un tamaño suficiente para métricas estables. La desviación estándar de R² entre folds (±0.12) es mucho mayor que el error de muestreo dentro de un fold, lo que sugiere que la variabilidad *temporal* (régimen de mercado) domina sobre el ruido de muestreo.

**¿Por qué gap de 5 días?** Sin gap, un día de alta volatilidad podría estar en train y el día siguiente (muy similar) en test. El gap de 5 días hábiles (~1 semana calendario) asegura que train y test están separados por suficiente tiempo para que las condiciones de mercado puedan cambiar. En la práctica, 5 días es conservador pero no excesivo (no descarta demasiados datos).

**Implicación para producción**: Fold 4 es el escenario real: entrena con 22.1 años de datos (2000-2022) y predice 2022-2026. El modelo se re-entrenaría aproximadamente cada 1-3 meses con los nuevos datos disponibles.

---

## 2. Arquitectura de Modelos — 5 XGBoost Independientes

### Los datos

5 XGBRegressors, mismos HPs, 29 features, sin escala, sin label mapping. Train: 135.6s + Infer: 1.6s = 137.3s total.

### Interpretación

**1 modelo por fold, no ensemble**: Cada modelo se entrena exclusivamente con los datos de su fold. No hay promediado, no hay stacking, no hay compartición de información entre folds. Esto da métricas de test *honestas*: fold 4 no se beneficia de haber visto datos de 2022 (porque esos están en su test, no en su train). La desventaja es que no usamos todo el dato disponible para un solo modelo final, pero la ventaja es que tenemos 5 mediciones independientes de performance.

**¿Por qué los mismos HPs en todos los folds?** Podríamos tuneamos HPs por fold (más regularización para fold 4 que tiene más datos). Pero eso introduciría un sesgo de optimización: si tuneamos en el test set de cada fold, estaríamos haciendo *data leakage* sobre la evaluación. Los HPs fijos aseguran que la comparación entre folds refleja solo diferencias en los datos, no en la configuración.

**Sin escala de features**: XGBoost es invariante a escala porque los splits en árboles son umbrales que se ajustan automáticamente. No necesitamos normalizar ni estandarizar. Esto simplifica el pipeline y evita errores (como calcular estadísticas de escala con leakage).

**Timing**: 135.6s de entrenamiento *total* — fold 4 solo (48.9s) es ~7.5× más lento que fold 0 (6.5s), que es menos que la relación de tamaño de datos (5×). Esto sugiere que el costo de entrenamiento escala sub-linealmente con los datos (~O(n^0.8) empírico en XGBoost). La inferencia es constante porque el test set tiene el mismo tamaño.

**Mejora vs one-hot**: 137.3s vs 369.5s = -62.8%. No solo por tener menos features (29 vs 496), sino porque XGBoost gasta menos tiempo buscando splits en features que no aportan. La mejora en tiempo es mucho mayor que la mejora en métricas, lo que sugiere que el one-hot era computacionalmente caro pero apenas informativo.

---

## 3. Experimentación — 4 Experimentos

### Experimento 1: Baseline con one-hot (496 features)

R²=0.3153, MAE=0.007049, RMSE=0.011068, 369.5s.

Era el punto de partida natural: incluir ticker como one-hot (práctica común en ML financiero). El R² de 0.315 ya era respetable para volatilidad (la predicción de volatilidad es notoriamente difícil — los modelos GARCH típicos tienen R² < 0.10 en horizontes cortos). Pero 94% de las features eran ticker dummies, lo que olía a sobreajuste.

### Experimento 2: Log-target (revertido)

R²=0.2929. El logaritmo debería estabilizar la varianza del target (que tiene skewness de 4.1). Pero empeoró el R² *global*, y especialmente en folds de crisis (0, 3) y producción (4).

**¿Por qué?** La transformación logarítmica comprime el rango de valores altos. En periodos de crisis, donde la volatilidad es alta y queremos errores pequeños *en términos absolutos*, el modelo optimizado en log-scale permite errores absolutos más grandes en el espacio original. Para riesgo, queremos penalizar más los errores en alta volatilidad (es donde más importa acertar). Target raw hace exactamente eso.

**Lección**: No siempre lo que estabiliza la varianza estadística es mejor para la aplicación práctica. La función de pérdida debe alinearse con el uso del modelo.

### Experimento 3: HP tuning (revertido)

R²=0.2852 en *todos* los folds. mayor regularización (max_depth=8, pero min_child_weight=3, gamma=0.1, reg_alpha=0.5, reg_lambda=2.0, colsample=0.7) fue contraproducente.

**¿Por qué?** 29 features es un espacio pequeño. XGBoost con max_depth=6 ya puede capturar interacciones de 6° orden entre features — más que suficiente para 29 variables. Aumentar la regularización en un espacio ya pequeño solo reduce la capacidad del modelo sin necesidad. La regularización extra tendría sentido si hubiera 496 features ruidosas (como en el experimento 1), pero con 29 features limpias, es perjudicial.

Además, aumentar n_estimators a 1000 con lr=0.05 (más boosting rounds) no compensó la regularización más agresiva. El modelo simplemente no podía ajustarse a los patrones con tanta restricción.

**Lección**: La regularización no es gratis. Hay que calibrarla al feature space. 29 features requieren menos regularización que 496.

### Experimento 4: Sin ticker encoding (FINAL)

R²=0.3265, MAE=0.006941, -62.8% tiempo.

**¿Por qué funciona?** 29 indicadores financieros ya codifican toda la información del ticker a través de sus valores de precio/volumen. TSLA tiene log_return alto, rsi_14 extremo, volume_zscore alto; KO tiene valores opuestos. El modelo aprende "un ticker con log_return > 0.05 y volume_zscore > 3 se comporta así" sin necesidad de un flag "esto es TSLA". Los 467 one-hot eran esencialmente 467 features que solo se activan para un subconjunto pequeño de filas — XGBoost tenía que dedicar splits y profundidad de árbol a cada ticker individualmente, lo que fragmentaba la capacidad del modelo.

---

## 4. Resultados — Métricas Globales y Por Fold

### R² = 0.3265 ± 0.1197

**¿Qué significa?** El modelo explica el 32.65% de la varianza de la volatilidad realizada a 5 días. Suena bajo, pero en el contexto de predicción de volatilidad financiera es sólido. Los modelos GARCH sobre un solo ticker típicamente explican <10%. Que un solo modelo cross-sectional (467 tickers) logre 32.65% es indicativo de que hay patrones sistemáticos en la volatilidad que son comunes entre tickers.

La desviación estándar de ±0.12 entre folds es grande. Esto no es incertidumbre estadística — es señal de que el R² depende fuertemente del régimen de mercado. La métrica relevante no es el promedio, sino el rango: el modelo funciona muy diferente según la época.

### MAE = 0.006941

**¿Qué significa?** En promedio, el modelo se equivoca por 0.0069 en volatilidad absoluta. Dado que la volatilidad media es 0.018, el error relativo es ~38.6%. Esto suena alto, pero hay que considerar que la volatilidad individual tiene un componente idiosincrático significativo (no predecible) de similar magnitud.

### RMSE = 0.010952

**¿Qué significa?** El RMSE es ~58% más alto que el MAE (0.0110 vs 0.0069). Esto indica que hay outliers de error grandes — errores que son ~2-3× el error típico. La cola derecha de la distribución de volatilidad (valores extremos) genera estos outliers. Es consistente con la skewness de 4.1 del target.

### Pearson r = 0.6406, Spearman ρ = 0.5725

**¿Qué significan?** La correlación lineal entre predicción y real es 0.64, la correlación de rangos es 0.57. Que Spearman sea menor que Pearson sugiere que la relación no es perfectamente lineal (el modelo comprime el rango de predicciones respecto al real, como se ve en los scatter plots). La correlación positiva confirma que el modelo captura dirección y magnitud relativa, aunque no siempre la magnitud exacta.

### Análisis por Fold — El Patrón Crisis

| Fold | Periodo | Volatilidad | R² | Interpretación |
|---|---|---|---|---|
| 0 | 2005-2010 | Alta (0.0216) | **0.527** | Crisis financiera. Alta volatilidad → señal fuerte → R² alto. |
| 1 | 2010-2014 | Media-baja (0.0154) | **0.267** | Recuperación lenta. Volatilidad moderada, patrones menos claros. |
| 2 | 2014-2018 | Baja (0.0133) | **0.216** | Periodo de calma extrema. Poca volatilidad que predecir. |
| 3 | 2018-2022 | Alta (0.0185) | **0.398** | COVID. Segundo mejor R², consistente con el patrón crisis. |
| 4 | 2022-2026 | Media (0.0175) | **0.224** | Producción. Similar a fold 1-2, no hay crisis global. |

**Interpretación del patrón**: La volatilidad alta es más predecible porque cuando hay eventos macro significativos (2008, COVID), todos los tickers se mueven juntos — el componente sistemático de la volatilidad domina sobre el idiosincrático. En periodos tranquilos, la volatilidad es principalmente ruido específico de cada ticker, que es impredecible con features de precio/volumen.

**Implicación crítica**: Fold 4 (producción) tiene R²=0.224. Cuando el mercado esté tranquilo (que es la mayor parte del tiempo), el modelo tendrá poco poder predictivo. Pero cuando haya una crisis, el modelo probablemente mejorará significativamente. Esto es aceptable para un modelo de riesgo: los peores errores ocurren en calma (donde el riesgo es bajo), no durante tormentas.

### Sesgo de Predicción (Residuos)

| Fold | Sesgo | Significado |
|---|---|---|
| 0 | +0.023 σ | Prácticamente insesgado. |
| 1 | **-0.202 σ** | Sobreestima significativamente. El modelo predice más volatilidad de la que ocurre (consistente con periodo post-crisis donde la volatilidad está cayendo). |
| 2 | -0.046 σ | Insesgado. |
| 3 | +0.100 σ | Subestima levemente. El modelo no anticipa totalmente el pico de COVID. |
| 4 | -0.051 σ | Insesgado. |

Fold 1 es el único con sesgo relevante: el modelo sobreestima la volatilidad post-crisis 2008, probablemente porque su train incluye la crisis y predice un retorno a la calma más lento de lo que ocurrió. Esto es un patrón común en modelos entrenados en ventanas que incluyen crisis — tienden a ser conservadores y predecir más riesgo del que finalmente se materializa. Para gestión de riesgo, un modelo conservador (que sobreestima) es preferible a uno agresivo (que subestima).

---

## 5. Feature Importance — ¿Qué Aprende el Modelo?

### Top 3 features consistentes en todos los folds:

1. **price_vs_sma_50** (17-28%): Distancia del precio actual a su media móvil de 50 días. Es esencialmente un indicador de tendencia/sobrecompra. Alta importancia en todos los folds sugiere que la posición relativa del precio en su tendencia de mediano plazo es el predictor más fuerte de volatilidad futura.

2. **rolling_max_10** (10-15%): Máximo del precio en los últimos 10 días. Captura si el precio está cerca de recientes picos (zonas de posible reversión o breakout).

3. **rolling_min_10** (9-11%): Mínimo del precio en los últimos 10 días. Similar pero en dirección opuesta. La presencia de ambos sugiere que el rango de precio reciente (high - low) es importante.

**¿Por qué price_vs_sma_50 domina tanto?** Cuando el precio está significativamente por encima o por debajo de su SMA-50, el mercado está "estirado". Las correcciones hacia la media tienden a generar volatilidad. Es un predictor simple pero efectivo. La consistencia a través de los folds (17% en fold 0, 28% en fold 4) sugiere que es una relación estable y no un artifact de un periodo específico.

### ¿Qué falta en el top 3?
Indicadores de volatilidad clásicos como `log_return`, `rsi_14`, `volume_zscore` están fuera del top 3. Esto no significa que no sean importantes — probablemente sean #4-#10 — pero sugiere que las features de *tendencia/precio relativo* dominan sobre las de *momento/volatilidad* para predecir volatilidad futura.

---

## 6. Implicaciones Globales

### Lo que el modelo hace bien
1. Diferencia entre periodos de alta y baja volatilidad (R² global positivo, correlación 0.64).
2. Captura el efecto de crisis (R²=0.53 en 2008, R²=0.40 en COVID).
3. Generaliza entre tickers sin necesidad de one-hot.
4. Es rápido en inferencia (~0.7 μs por muestra).

### Lo que el modelo no hace bien
1. Predice el nivel exacto de volatilidad en calma (R²=0.22 en producción).
2. Captura eventos de volatilidad idiosincrática (no sistemática).
3. Reproduce los valores extremos de la distribución (comprime el rango de predicciones).

### Recomendaciones prácticas
- **Uso en producción**: El fold 4 es el modelo a desplegar. Re-entrenar periódicamente (cada 1-3 meses) agregando nuevos datos al train.
- **Monitoreo**: La métrica clave es R² en el periodo más reciente. Si cae por debajo de 0.15, considerar retrain con datos más recientes o revisar features.
- **Interpretación de predicciones**: Cuando el modelo predice volatilidad alta, es más confiable que cuando predice baja (por el patrón crisis vs calma).
- **Calibración**: Considerar recalibrar las predicciones (ej. Platt scaling o isotonic regression) para corregir el sesgo en fold 4 si se usa como insumo para modelos downstream.
