from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loader import load_dataframe

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "eda"
DEFAULT_FIGSIZE = (14, 12)
TOP_TICKERS = 20


def generate_correlation_heatmap_eda() -> None:
    print("\n" + "="*80)
    print("Matriz de Correlación de Retornos Diarios - EDA")
    print("="*80 + "\n")
    
    raw_df = load_dataframe()
    raw_df.columns = raw_df.columns.str.lower().str.replace(" ", "_")
    
    print(f"✓ Dataset cargado: {len(raw_df)} filas, {raw_df['ticker'].nunique()} tickers únicos")
    
    # Obtener top 20 tickers por volumen total
    total_volume_by_ticker = raw_df.groupby("ticker")["volume"].sum().sort_values(ascending=False)
    top_tickers = total_volume_by_ticker.head(TOP_TICKERS).index.tolist()
    
    print(f"✓ Top {TOP_TICKERS} tickers seleccionados: {', '.join(top_tickers)}\n")
    
    # Filtrar datos para top tickers
    df = raw_df[raw_df["ticker"].isin(top_tickers)].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    # Encontrar período donde todos los 20 tickers coexisten
    date_range_by_ticker = df.groupby("ticker")["date"].agg(["min", "max"])
    earliest_end = date_range_by_ticker["min"].max()  # Última fecha de inicio
    latest_start = date_range_by_ticker["max"].min()   # Primera fecha de fin
    
    print(f"Período de coexistencia de todos {TOP_TICKERS} tickers:")
    print(f"  • Desde: {earliest_end.strftime('%Y-%m-%d')}")
    print(f"  • Hasta: {latest_start.strftime('%Y-%m-%d')}\n")
    
    # Filtrar por período de coexistencia
    df_period = df[(df["date"] >= earliest_end) & (df["date"] <= latest_start)].copy()
    
    print(f"✓ Registros en período común: {len(df_period)}")
    print(f"✓ Registros por ticker: {len(df_period) / TOP_TICKERS:.0f}\n")
    
    # Calcular retornos diarios por ticker
    df_period["return"] = df_period.groupby("ticker")["adj_close"].pct_change()
    
    # Preparar matriz de retornos (filas=fechas, columnas=tickers)
    returns_pivot = df_period.pivot_table(
        index="date",
        columns="ticker",
        values="return",
        aggfunc="first"  # En caso de duplicados (muy raro)
    )
    
    # Dropear NaN (primer día de cada ticker)
    returns_pivot = returns_pivot.dropna()
    
    print(f"✓ Matriz de retornos: {returns_pivot.shape[0]} fechas × {returns_pivot.shape[1]} tickers")
    print(f"✓ Período: {returns_pivot.index.min().strftime('%Y-%m-%d')} a {returns_pivot.index.max().strftime('%Y-%m-%d')}\n")
    
    # Calcular matriz de correlación
    corr_matrix = returns_pivot.corr(method="pearson")
    
    # Imprimir análisis de correlaciones
    print("="*80)
    print("Matriz de Correlación de Retornos Diarios")
    print("="*80)
    print(f"\n{corr_matrix.to_string()}\n")
    
    # Extraer correlaciones fuera de la diagonal
    mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
    correlations_off_diag = corr_matrix.values[mask]
    
    # Análisis de correlaciones
    print("="*80)
    print("Análisis Agregado")
    print("="*80 + "\n")
    
    # Pares más correlacionados
    corr_pairs = []
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix)):
            corr_pairs.append((
                corr_matrix.index[i],
                corr_matrix.columns[j],
                corr_matrix.iloc[i, j]
            ))
    
    corr_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print("🔹 CORRELACIONES MÁS ALTAS (Mayor Cohesión):")
    for ticker1, ticker2, corr in corr_pairs[:5]:
        print(f"   • {ticker1:6s} ↔ {ticker2:6s}: {corr:+.4f}")
    
    print("\n🔹 CORRELACIONES MÁS BAJAS (Mayor Diversificación):")
    for ticker1, ticker2, corr in corr_pairs[-5:]:
        print(f"   • {ticker1:6s} ↔ {ticker2:6s}: {corr:+.4f}")
    
    print(f"\n🔹 ESTADÍSTICAS AGREGADAS:")
    print(f"   • Media de correlaciones: {correlations_off_diag.mean():+.4f}")
    print(f"   • Desviación estándar: {correlations_off_diag.std():.4f}")
    print(f"   • Mínima correlación: {correlations_off_diag.min():+.4f}")
    print(f"   • Máxima correlación: {correlations_off_diag.max():+.4f}")
    print()
    
    # Crear máscara para mostrar solo triángulo inferior (sin diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=0)
    
    # Crear heatmap simple (sin clustering) con máscara
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap="RdBu_r",      # Rojo-Blanco-Azul (divergente)
        center=0,            # Centro en 0
        vmin=-1, vmax=1,     # Rango -1 a 1
        annot=True,          # Mostrar valores
        fmt=".2f",           # Formato 2 decimales
        annot_kws={"size": 8},
        cbar_kws={"label": "Correlación de Pearson"},
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
        square=True,         # Celdas cuadradas
    )
    
    ax.set_title(
        f"Matriz de Correlación de Retornos Diarios (Triángulo Inferior)\nTop {TOP_TICKERS} Tickers | {returns_pivot.index.min().strftime('%Y-%m-%d')} a {returns_pivot.index.max().strftime('%Y-%m-%d')}",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )
    
    ax.set_xlabel("Ticker", fontsize=11, fontweight="bold")
    ax.set_ylabel("Ticker", fontsize=11, fontweight="bold")
    
    # Guardar figura
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "04_correlation_heatmap.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Gráfico guardado en: {output_path}\n")
    plt.close(fig)
    
    # Interpretación ML
    print("="*80)
    print("Interpretación para Machine Learning")
    print("="*80 + "\n")
    
    # Identificar clusters por correlación media (excluyendo diagonal)
    ticker_avg_corr_dict = {}
    for ticker in corr_matrix.index:
        # Calcular promedio de correlaciones para este ticker (excluyendo diagonal)
        ticker_corrs = corr_matrix.loc[ticker, corr_matrix.columns != ticker]
        ticker_avg_corr_dict[ticker] = ticker_corrs.mean()
    
    high_corr_tickers = sorted(
        [t for t in ticker_avg_corr_dict if ticker_avg_corr_dict[t] > correlations_off_diag.mean()],
        key=lambda x: ticker_avg_corr_dict[x],
        reverse=True
    )
    
    low_corr_tickers = sorted(
        [t for t in ticker_avg_corr_dict if ticker_avg_corr_dict[t] < correlations_off_diag.mean()],
        key=lambda x: ticker_avg_corr_dict[x]
    )
    
    print(f"🔹 CLUSTER DE ALTA COHESIÓN (Correlación promedio > {correlations_off_diag.mean():.3f}):")
    print(f"   • Activos: {', '.join(high_corr_tickers[:5])}")
    print(f"   • Implicación ML: Multi-task learning, modelos compartidos")
    print()
    
    print(f"🔹 CLUSTER DE BAJA COHESIÓN (Correlación promedio < {correlations_off_diag.mean():.3f}):")
    print(f"   • Activos: {', '.join(low_corr_tickers[:5])}")
    print(f"   • Implicación ML: Modelos separados, posible diversificación")
    print()
    
    print("="*80 + "\n")


if __name__ == "__main__":
    generate_correlation_heatmap_eda()
