# Análisis Exploratorio de Datos (EDA) - S&P 500

## 01 - Serie Temporal de Precios Normalizada

Visualización comparativa de la evolución del precio ajustado de los principales activos tecnológicos del S&P 500 desde el año 2000 hasta la fecha. Esta análisis fundamenta el entrenamiento del modelo de predicción.

### Implicaciones para el Modelo de Machine Learning

#### 1. **No-Estacionariedad de Series Temporales**
La gráfica revela tendencias de largo plazo (uptrends sostenidos), lo que indica que las series NO son estacionarias. Implicaciones:
- El modelo debe capturar **tendencias globales**, no asumir reversión a la media
- Usar **diferenciación o transformaciones logarítmicas** en features
- Las predicciones a corto plazo serán más confiables que a largo plazo

#### 2. **Volatilidad Heterogénea por Activo**
Cada ticker muestra dinámicas de riesgo-rendimiento distintas:
- **NVDA** (-89.1% drawdown): Patrón volátil, cambios abruptos. El modelo debe aprender que pequeños inputs pueden generar cambios grandes
- **AMZN** (-93.3% drawdown): Máxima volatilidad histórica. Útil para entrenar modelos robustos a extremos
- **MSFT/GOOGL**: Volatilidad moderada. Patrones más predecibles

**Para el modelo**: Normalizar por volatilidad individual (Z-score, escalado por std) mejorará convergencia y evitará que activos volátiles dominen el aprendizaje.

#### 3. **Eventos de Crisis y Comportamientos Asimétricos**
Caídas visibles en 2008 (crisis financiera) y 2020 (COVID):
- Los comportamientos alcistas (crecimiento gradual) y bajistas (caídas bruscas) son **asimétricamente diferentes**
- Recuperación más rápida en tech que en índice general
- El modelo debe aprender que la **reversión post-crisis es dependiente del sector**

**Para el modelo**: Incluir features que capturen "memoria de eventos" (volatilidad histórica, cambios de tendencia) permite que el modelo anticipe patrones post-crisis.

#### 4. **Divergencia Extrema en Trayectorias (NVDA vs Otros)**
NVDA (+210,811%) vs MSFT (+1,015%):
- Indica que el **poder predictivo es sector/timing-dependiente**
- Activos emergentes (GPUs, IA) tienen dinámicas radicalmente distintas
- Datos históricos antiguos (pre-2016 para NVDA) pueden ser **menos relevantes**

**Para el modelo**: 
- Considerar **feature engineering temporal** (ventanas deslizantes de recencia)
- Posible beneficio de modelos separados por períodos o "eras tecnológicas"
- Evitar sobrepesar datos históricos lejanos en el entrenamiento

#### 5. **Normalización a 100: Base para Comparación**
La normalización revela que **rendimientos relativos son comparables**, pero escalas subyacentes difieren enormemente:
- NVDA: $50 → $1,500,000+ (1M x)
- MSFT: $30 → $300+ (10x)

**Para el modelo**: 
- Usar **log-returns** en lugar de precios absolutos
- Normalizar por volatilidad individual (características en `features.py`)
- Prevents el modelo de aprender "a invertir en NVDA siempre" sin capturar dinámicas subyacentes

#### 6. **Patrones de Mercado General vs Activos Individuales**
S&P 500 (+2,586%) es **mediana entre extremos**:
- Algunos activos superan significativamente el índice (NVDA, AAPL)
- Otros underperform (MSFT vs índice en ciertos períodos)
- El modelo debe aprender **alpha (retorno adicional) vs beta (correlación con índice)**

**Para el modelo**: Features like `price_vs_sma_*`, `macd`, `rsi_14` capturan desviaciones del comportamiento "promedio", son claves para predicción de outperformance.

### Cómo Leer la Gráfica

1. **Eje X (Fecha)**: Período 2000-2026 (26 años = múltiples ciclos de mercado)
2. **Eje Y (Precio Normalizado)**: Base 100 = punto de partida; escala logarítmica mental
3. **Línea punteada gris**: Referencia en 100 (punto inicial)
4. **Colores**: Cada activo tiene un color distinto para identificación rápida
5. **Cajas de métricas (inferior izquierda)**: Rendimiento total y drawdown máximo

### Insights Clave para Modelado

- **NVDA despega radicalmente post-2016**: Datos pre-2015 pueden NO ser representativos del comportamiento actual. El modelo puede beneficiarse de **ponderación temporal** o **reentrenamiento periódico**
  
- **Crisis 2008 y 2020 marcan saltos estructurales**: El modelo debe detectar cambios de régimen (volatilidad, correlaciones). Features basadas en **ventanas móviles** (rolling_std, ATR) son críticas.

- **Recuperación diferenciada por sector**: Tech se recupera más rápido que índice. Incorporar **sector/meta-features** mejora predicciones en post-crisis.

- **Volatilidad extrema en AMZN y NVDA**: El modelo debe ser robusto a outliers. Usar **huber loss** o técnicas de regularización puede mejorar generalización.

- **Correlaciones cambiantes**: La normalización visible sugiere que correlaciones entre activos NO son constantes (especialmente post-2010). El modelo debe capturar **dinámicas temporales de dependencia**.

---

## 02 - Distribución de Retornos Diarios

Histograma de densidad con estimación KDE que visualiza la distribución de cambios diarios de precio. Permite identificar patrones de volatilidad, asimetría y comportamientos extremos fundamentales para modelado predictivo.

**Disponible en dos versiones:**
- **Overlay**: Todos los activos en una sola gráfica (`02_returns_distribution_overlay.png`)
- **Separado**: 6 subplots individuales, uno por activo (`02_returns_distribution_separate.png`)

### Implicaciones para el Modelo de Machine Learning

#### 1. **Violación del Supuesto de Normalidad (Fat Tails)**
Los retornos muestran **kurtosis excesiva significativa**:
- **AAPL**: 31.6 (extremo) → Colas mucho más pesadas que distribución normal
- **AMZN**: 16.2 (alto) → Eventos extremos más frecuentes de lo esperado
- **MSFT/GOOGL/NVDA**: 8.8-10.8 (moderado) → Aún significativamente mayor que 3 (normal)

Implicaciones:
- Los retornos tienen **más eventos extremos que una normal**, especialmente caídas
- Modelos basados en distribuciones normales (regresión lineal, CAPM) son **inadecuados**
- El modelo debe ser **robusto a outliers** → usar L1/Huber loss en lugar de L2

**Para el modelo**: 
- Detectar y manejar extremos correctamente es crítico para predicciones confiables
- Features como `log_return_std_10`, `atr_14`, `zscore_volume_20` capturan dinámicas de volatilidad
- Considerar **cuantile regression** para predecir múltiples percentiles, no solo la media

#### 2. **Volatilidad Heterogénea (Heteroskedasticity)**
La volatilidad anualizada varía dramáticamente por activo:
- **NVDA**: 63.9% (máxima) → Riesgo 2.3x el índice
- **AMZN**: 49.0% → Riesgo 1.7x el índice
- **S&P 500**: 28.2% (referencia) → Mercado estable comparativamente
- **MSFT/GOOGL**: 30.1-30.6% → Ligeramente superior al índice

Implicaciones:
- La varianza NO es constante → violación de homoskedasticity
- Cambios de volatilidad en el tiempo (GARCH effects)
- Activos con diferente volatilidad requieren **escalado normalizado**

**Para el modelo**:
- Normalizar cada activo por su volatilidad histórica (Z-score de retornos)
- Features de volatilidad móvil (`log_return_std_10`, `log_return_std_20`) son críticos
- El modelo puede beneficiarse de **arquitecturas específicas por volatilidad** (diferentes thresholds/regularización)

#### 3. **Asimetría (Skewness) → Riesgo Asimétrico**
La distribución de retornos NO es simétrica:
- **AAPL**: -1.26 (cola izquierda extrema) → Caídas más extremas que subidas
- **AMZN**: +1.14 (cola derecha) → Algunos movimientos positivos muy extremos
- **MSFT/GOOGL/NVDA**: +0.16 a +0.60 (ligeramente positivo)
- **S&P 500**: +0.10 (casi simétrico)

Implicaciones:
- El riesgo de caída es **asimétricamente peor** en AAPL (downside risk)
- Modelos simétricos fallarán en capturar este sesgo
- La reversión a la media es **diferenciada por dirección**

**Para el modelo**:
- Entrenar modelos separados para predicción de alzas vs bajas
- Usar `price_direction_5d` como target (asimétrico: -1, 0, +1)
- Features asimétricos (p.ej., diferencia entre ganancias vs pérdidas) mejoran predicción
- Considerar **ensemble de modelos** con pesos diferentes para colas

#### 4. **Diferencial de Media Diaria → Eficiencia de Mercado**
La media diaria positiva pero baja:
- **NVDA**: +0.197% diario → ~50% retorno anual sin volatilidad
- **AAPL**: +0.120% diario
- **S&P 500**: +0.066% diario

Implicaciones:
- Los retornos positivos son **predecibles en media**, pero con alta varianza
- El "señal" (drift) es débil comparado al ruido (volatilidad)
- Ratio señal-ruido es bajo → difícil predecir dirección a corto plazo
- Pero tendencias a largo plazo son robustas (ver Gráfico 1)

**Para el modelo**:
- Predicciones a muy corto plazo (1 día) son ruidosas, mejor enfocarse en 5-20 días
- El target `price_direction_5d` tiene mejor predicibilidad que cambio diario
- Features de tendencia (`ema_12`, `macd`, `price_vs_sma_*`) capturan el drift

#### 5. **Divergencia de Comportamiento vs Índice**
NVDA y AMZN tienen distribuciones radicalmente distintas al S&P 500:
- Volatilidad 1.7-2.3x mayor
- Skewness diferente (positivo vs negativo)
- Kurtosis 1.3-3.8x mayor

Implicaciones:
- **Correlaciones no son constantes** → Dinámicas de dependencia cambian
- Cada activo tiene "firma estadística" única
- Modelos globales underperform vs específicos por activo
- Pero datos limitados por activo → necesita regularización fuerte

**Para el modelo**:
- Entrenar modelos separados por activo (mejor rendimiento)
- Usar **transfer learning** desde mercado general cuando datos insuficientes
- Incorporar features relativas al índice (`price_vs_sma_20`, correlación con mercado)

#### 6. **Percentiles Extremos → Eventos Raros pero Críticos**
Los percentiles 1% y 99% revelan tail risk:
- **NVDA**: -10% a +12.5% (rango de -10% a +10% **saturado**)
- **AAPL**: -6.4% a +6.9% (eventos extremos dentro del rango)
- **AMZN**: -8.6% a +9.6% (asimétrico)

Implicaciones:
- Eventos extremos (crashe/rallies) ocurren más frecuentemente de lo esperado
- Modelos entrenados solo en datos "normales" fallan en extremos
- Value-at-Risk (VaR) cálculos tradicionales subestiman riesgo
- Necesidad de **manejar tail risk explícitamente**

**Para el modelo**:
- Usar **quantile loss** para optimizar percentiles específicos
- Features de volatilidad histórica son críticas (detectan períodos de alto riesgo)
- El modelo debe tener **mecanismos de detección de régimen** (normal vs crisis)
- Considerar **data augmentation** de eventos extremos raros

### Cómo Leer la Gráfica

1. **Histogramas**: Frecuencia de retornos en cada rango (bins de 0.5%)
2. **Curva KDE (línea sólida)**: Estimación suave de la distribución real
3. **Curva punteada**: Distribución normal teórica con misma media y std
4. **Brecha entre KDE y normal**: Desvío de normalidad (especialmente en colas)
5. **Línea vertical en 0**: Retorno nulo (referencia)
6. **Caja de estadísticas (superior izquierda)**: Volatilidad anual, Skewness, Kurtosis por activo

### Insights Clave para Modelado

- **AAPL skewness extremo (-1.26)**: Downside risk asimétrico. Modelos deben penalizar caídas extremas diferente que subidas.

- **Kurtosis universal > 8**: Todos los activos tienen fat tails. El modelo debe ser robusto a outliers con **regularización significativa** o **loss functions robustos** (Huber, Quantile).

- **Volatilidad anual 30-64%**: Rango enorme entre activos. Normalizar por volatilidad individual (Z-score, Standardization) es **mandatorio** para convergencia.

- **Media diaria débil vs volatilidad**: Señal-ruido bajo en muy corto plazo. Enfocarse en predicciones a 5+ días (`price_direction_5d` es más predecible que cambio diario).

- **Distribuciones no-normales**: Modelos que asumen normalidad (Gaussian Naive Bayes, Linear Discriminant Analysis) serán **subóptimos**. Usar modelos no-paramétricos (Tree-based, SVM) o con pérdidas robustas.

- **Comportamiento divergente por activo**: No existe "modelo universal". Considerar **arquitectura modular**: características globales + características específicas por activo, o **multi-task learning** con task por activo.

---

## 03 - Volumen de Trading - Boxplot

Visualización horizontal de la distribución de volúmenes diarios para los 20 activos más líquidos del S&P 500. Usa escala logarítmica (los volúmenes oscilan entre 24.7M y 596.4M unidades) para revelar liquidez relativa, consistencia de ejecución y eventos de trading anómalo que impactan directamente en la estrategia de ejecución del modelo.

### Implicaciones para el Modelo de Machine Learning

#### 1. **Liquidez Relativa y Capacidad de Ejecución**
La caja de estadísticas en la gráfica muestra:
- **Activo más líquido**: NVDA (mediana 497.1M unidades diarias)
- **Menos líquido (top 20)**: LRCX (mediana 21.2M)
- **Ratio de liquidez**: 23.4x

Implicaciones:
- Activos de **baja liquidez son costosos de ejecutar** → spreads bid-ask anchos, slippage significativo
- NVDA/AAPL pueden absorber órdenes grandes sin movimiento de precio
- LRCX requiere ejecución estratégica (fragmentación temporal, VWAP)
- El modelo debe **ajustar señales por liquidez disponible** (no ejecutar grandes órdenes en activos poco líquidos)

**Para el modelo**:
- Feature `zscore_volume_20` captura desviaciones de liquidez esperada
- Predicciones deben **ponderarse por liquidez**: una señal en NVDA es ejecutable; la misma en LRCX no lo es
- Considerar **modelos separados o arquitectura adaptativa** según liquidez del activo
- Rechazo automático de señales en activos con volumen < percentil 25 (frena ejecución ruidosa)

#### 2. **Consistencia de Volumen y Predecibilidad**
El rango intercuartil (IQR) mide variabilidad de volumen:
- **Más consistente**: LRCX (IQR pequeño, ~5-10M) → volumen predecible
- **Más variable**: NVDA (IQR grande, ~394.8M) → volatilidad de liquidez

Implicaciones:
- Activos con **volumen consistente son más predecibles en ejecución**
- Activos con **volumen variable tienen días de alta/baja liquidez** → impacta negativamente predicciones en días "raros"
- Cambios de volumen suelen presagiar cambios de precio (evento importante)
- El modelo debe **detectar anomalías de volumen** como indicador temprano de movimientos de precio

**Para el modelo**:
- Features `log_return_std_*` capturan periodos de volatilidad
- Entrenar modelos con **detección de régimen** (normalidad vs anomalía)
- Cuando `volume` desviación > 2-sigma, aumentar regularización o rechazar predicción
- Usar **quantile loss** en lugar de MSE para ser robusto a días anómalos
- El modelo puede beneficiarse de features como `volume_to_moving_avg_ratio` para detectar "hot days"

#### 3. **Días Anómalos (High Activity Days) y Cambios Estructurales**
Los puntos rojos en el boxplot representan outliers (volumen > Q3 + 1.5×IQR):
- **Máximo outliers**: GOOGL (9.3% de días con anomalías)
- **Mínimo**: Activos menos volátiles (<1%)
- Estos "días especiales" correlacionan con: earnings announcements, noticias macroeconómicas, cambios de tendencia

Implicaciones:
- Días con **volumen anómalo presagian volatilidad de precio** (modelo debe ser cauteloso)
- Concentración de anomalías en ciertos activos (GOOGL) → periodos de cambio macro frecuentes
- Modelos entrenados sin reconocer anomalías **generalizarán mal** a datos con anomalías
- Necesidad de **feature engineering específico** para capturar "momento anómalo"

**Para el modelo**:
- Feature `zscore_volume_20` detecta directamente desviaciones de volumen
- Cuando `zscore_volume_20 > 2`, el modelo debe: (a) reducir confianza en predicción, (b) aumentar stop-loss, o (c) rechazar trade
- Considerar **ensemble de modelos** con pesos diferentes en días normales vs anómalos
- Data augmentation: entrenar específicamente en subconjuntos de "high activity days" para robustez
- Correlación volumen-volatilidad: **días altos en volumen = días con precio más volátil → feature útil**

#### 4. **Liquidez Diferenciada por Sector/Cap de Mercado**
El top 20 es dominado por **mega-cap tech** (NVDA, AAPL, MSFT, GOOGL, AMZN, NFLX, TSLA):
- Tech tiene liquidez 10-20x mayor que otros sectores
- Sectores "tradicionales" en el top 20 (JPM, WMT, LRCX) tienen liquidez **predecible pero baja**

Implicaciones:
- **Modelos separados por liquidez-tier mejoran predicción** (uno para mega-cap, otro para mid-cap)
- Correlaciones entre activos **dependen de liquidez**: mega-caps mueven mercado, otros siguen
- Features de "liderazgo" (p.ej., movimiento de NVDA predice MSFT) son críticos

**Para el modelo**:
- Crear **feature de "volatilidad de pares"**: correlación entre activos según su liquidez
- Jerarquía predictiva: NVDA → MSFT → AAPL → ... → LRCX
- Modelos específicos por tier mejoran significativamente (training samples suficientes + homogeneidad de dinámicas)

#### 5. **Liquidez y Ventanas Temporales de Predicción**
Liquidez alta permite **ejecución inmediata**, mientras que liquidez baja requiere **tiempo**:
- NVDA: Ejecutar predicción en minutos/horas
- LRCX: Ejecutar en horas/días (fragmentar orden)

Implicaciones:
- **El horizonte de predicción debe estar acoplado a liquidez**: no predecir a 1 día en activos poco líquidos
- Modelos a horizonte corto (1-5 días) requieren top de liquidez; modelos a largo plazo pueden incluir todos

**Para el modelo**:
- `price_direction_5d` es **mandatorio para LRCX, opcional para NVDA**
- Rechazar trades a horizonte corto en activos con volumen < percentil 50
- Para activos poco líquidos, entrenar solo en `price_direction_20d` o `price_direction_60d`

### Cómo Leer la Gráfica

1. **Caja (box)**: Rango intercuartil (IQR = 50% de volúmenes, entre Q1 y Q3)
2. **Línea roja dentro de caja**: Mediana (volumen típico)
3. **Bigotes (whiskers)**: Rango normal (Q1 - 1.5×IQR, Q3 + 1.5×IQR)
4. **Puntos rojos**: Outliers (volumen anómalo, días especiales)
5. **Escala logarítmica (eje X)**: Necesaria porque rango 24.7M–596.4M es enorme; log compress para visualizar
6. **Caja de estadísticas (superior derecha)**: Activo más/menos líquido, ratio de liquidez
7. **Orden (eje Y)**: Descendente por volumen total (NVDA arriba, LRCX abajo)

### Insights Clave para Modelado

- **Liquidez no es uniforme**: El ratio 23.4x entre activos más y menos líquidos implica que un modelo único NO puede manejar ambos. Necesidad de arquitectura **adaptativa o separada por tier**.

- **GOOGL lidera en anomalías**: 9.3% de días con outliers sugiere que este activo es sensible a noticias macro. El modelo debe tener **mecanismos de detección** para estos períodos.

- **Volumen predice volatilidad**: Correlación empírica volumen-volatilidad → feature `zscore_volume_20` es crítica. Entrenar modelos con énfasis en este feature.

- **Consistencia de LRCX vs variabilidad de NVDA**: Ilustra trade-off entre predictibilidad (LRCX) y tamaño de oportunidad (NVDA). Considerar **modelos de riesgo diferenciados**: alta confianza en LRCX, baja confianza pero alto upside en NVDA.

- **Ejecución realista**: No todas las predicciones son ejecutables. El modelo debe rechazar órdenes en activos poco líquidos o períodos de baja liquidez. Integrar **simulador de ejecución** que considere slippage por liquidez.

- **Jerarquía temporal de correlaciones**: Activos mega-cap (NVDA, AAPL) generan movimiento, otros siguen. Features de "liderazgo" y **modelos de "precio predictor"** (NVDA → MSFT) mejoran significativamente.

