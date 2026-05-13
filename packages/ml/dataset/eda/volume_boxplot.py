from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loader import load_dataframe

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "eda"
DEFAULT_FIGSIZE = (14, 8)
TOP_TICKERS = 20


def _calculate_volume_statistics(volumes: pd.Series) -> dict:
    """Calculate volume statistics for a ticker."""
    volumes_clean = volumes.dropna()
    volumes_clean = volumes_clean[volumes_clean > 0]  # Excluir volumen cero
    
    if len(volumes_clean) == 0:
        return None
    
    q1 = volumes_clean.quantile(0.25)
    q3 = volumes_clean.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = volumes_clean[(volumes_clean < lower_bound) | (volumes_clean > upper_bound)]
    
    return {
        "mean": volumes_clean.mean(),
        "median": volumes_clean.quantile(0.50),
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "p10": volumes_clean.quantile(0.10),
        "p90": volumes_clean.quantile(0.90),
        "total_volume": volumes_clean.sum(),
        "outliers_count": len(outliers),
        "outlier_pct": (len(outliers) / len(volumes_clean)) * 100,
    }


def generate_volume_boxplot_eda() -> None:
    print("\n" + "="*80)
    print("Boxplot de Volumen por Ticker - EDA")
    print("="*80 + "\n")
    
    raw_df = load_dataframe()
    raw_df.columns = raw_df.columns.str.lower().str.replace(" ", "_")
    
    print(f"✓ Dataset cargado: {len(raw_df)} filas, {raw_df['ticker'].nunique()} tickers únicos")
    
    df = raw_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    date_range_start = df["date"].min().strftime("%Y-%m-%d")
    date_range_end = df["date"].max().strftime("%Y-%m-%d")
    
    print(f"✓ Período: {date_range_start} a {date_range_end}")
    
    # Calcular volumen total por ticker
    total_volume_by_ticker = df.groupby("ticker")["volume"].sum().sort_values(ascending=False)
    
    # Seleccionar top 20
    top_tickers = total_volume_by_ticker.head(TOP_TICKERS).index.tolist()
    print(f"✓ Top {TOP_TICKERS} tickers por volumen total seleccionados")
    
    # Filtrar datos para top tickers
    df_top = df[df["ticker"].isin(top_tickers)].copy()
    
    # Preparar datos para boxplot: crear lista de volúmenes por ticker (en orden de volumen)
    boxplot_data = []
    ticker_labels = []
    stats_data = {}
    
    for ticker in top_tickers:
        ticker_volumes = df_top[df_top["ticker"] == ticker]["volume"].values
        ticker_volumes = ticker_volumes[ticker_volumes > 0]  # Excluir ceros
        
        if len(ticker_volumes) > 0:
            boxplot_data.append(ticker_volumes)
            ticker_labels.append(ticker)
            stats_data[ticker] = _calculate_volume_statistics(df_top[df_top["ticker"] == ticker]["volume"])
    
    # Crear gráfica
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    
    # Boxplot horizontal con escala logarítmica
    bp = ax.boxplot(
        boxplot_data,
        tick_labels=ticker_labels,
        vert=False,
        patch_artist=True,
        widths=0.6,
        showfliers=True,
        flierprops=dict(
            marker="o",
            markerfacecolor="#E74C3C",
            markeredgecolor="#C0392B",
            markersize=4,
            alpha=0.6,
        ),
    )
    
    # Colorear cajas y whiskers
    for patch in bp["boxes"]:
        patch.set_facecolor("#3498DB")
        patch.set_alpha(0.7)
        patch.set_edgecolor("#2980B9")
        patch.set_linewidth(1.5)
    
    for whisker in bp["whiskers"]:
        whisker.set(color="#2980B9", linewidth=1.5, alpha=0.8)
    
    for cap in bp["caps"]:
        cap.set(color="#2980B9", linewidth=1.5, alpha=0.8)
    
    for median in bp["medians"]:
        median.set(color="#E74C3C", linewidth=2.5)
    
    # Escala logarítmica en eje X
    ax.set_xscale("log")
    
    ax.set_xlabel("Volumen (unidades, escala logarítmica)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Ticker", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="x", linestyle="-", linewidth=0.5)
    
    # Agregar estadísticas de liquidez en esquina superior derecha
    medians = [stats_data[t]["median"] for t in ticker_labels if t in stats_data]
    max_median = max(medians)
    min_median = min(medians)
    ratio = max_median / min_median if min_median > 0 else np.inf
    
    most_liquid_idx = np.argmax(medians)
    least_liquid_idx = np.argmin(medians)
    most_liquid = ticker_labels[most_liquid_idx]
    least_liquid = ticker_labels[least_liquid_idx]
    
    stats_text = "Liquidez Relativa\n"
    stats_text += "─" * 22 + "\n"
    stats_text += f"{most_liquid}: {max_median/1_000_000:.1f}M\n"
    stats_text += f"{least_liquid}: {min_median/1_000_000:.1f}M\n"
    stats_text += f"Ratio: {ratio:.1f}x"
    
    ax.text(
        0.98, 0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85),
        family="monospace",
    )
    
    fig.suptitle(
        f"Distribución de Volumen de Trading - Top {TOP_TICKERS} Tickers\n{date_range_start} a {date_range_end}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    
    plt.tight_layout()
    
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "03_volume_boxplot.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Gráfico guardado en: {output_path}")
    plt.close(fig)
    
    # Imprimir estadísticas
    print("\n" + "="*80)
    print(f"Estadísticas de Volumen - Top {TOP_TICKERS} Tickers")
    print("="*80)
    print(f"Período: {date_range_start} a {date_range_end}\n")
    
    print(f"{'Ticker':<8} {'Media (M)':<12} {'Mediana (M)':<14} {'IQR (M)':<12} {'Outliers':<12} {'OutPct %':<10}")
    print("-" * 80)
    
    for ticker in ticker_labels:
        if ticker in stats_data and stats_data[ticker] is not None:
            stats = stats_data[ticker]
            mean_m = stats["mean"] / 1_000_000
            median_m = stats["median"] / 1_000_000
            iqr_m = stats["iqr"] / 1_000_000
            outliers = stats["outliers_count"]
            outpct = stats["outlier_pct"]
            
            print(f"{ticker:<8} {mean_m:<12.2f} {median_m:<14.2f} {iqr_m:<12.2f} {outliers:<12} {outpct:<10.1f}")
    
    print("\n" + "="*80)
    print("Interpretación para Machine Learning")
    print("="*80 + "\n")
    
    # Análisis de liquidez relativa
    medians = [stats_data[t]["median"] for t in ticker_labels if t in stats_data]
    max_median = max(medians)
    min_median = min(medians)
    ratio = max_median / min_median if min_median > 0 else np.inf
    
    most_liquid = ticker_labels[np.argmax(medians)]
    least_liquid = ticker_labels[np.argmin(medians)]
    
    print(f"🔹 LIQUIDEZ RELATIVA:")
    print(f"   • Activo más líquido: {most_liquid}")
    print(f"   • Activo menos líquido (top 20): {least_liquid}")
    print(f"   • Ratio liquidez: {ratio:.1f}x\n")
    
    # Análisis de consistencia de volumen
    print(f"🔹 CONSISTENCIA (Caja pequeña = Consistente):")
    iqr_by_ticker = [(t, stats_data[t]["iqr"]) for t in ticker_labels if t in stats_data]
    iqr_by_ticker.sort(key=lambda x: x[1])
    
    most_consistent = iqr_by_ticker[0][0]
    most_volatile_vol = iqr_by_ticker[-1][0]
    
    print(f"   • Más consistente: {most_consistent}")
    print(f"   • Más variable en volumen: {most_volatile_vol}\n")
    
    # Análisis de outliers
    print(f"🔹 DÍAS ANÓMALOS (High Activity Days - outliers):")
    outlier_stats = [(t, stats_data[t]["outlier_pct"]) for t in ticker_labels if t in stats_data]
    outlier_stats.sort(key=lambda x: x[1], reverse=True)
    
    high_activity = outlier_stats[0]
    low_activity = outlier_stats[-1]
    
    print(f"   • Más días anómalos: {high_activity[0]} ({high_activity[1]:.1f}% de días)")
    print(f"   • Menos días anómalos: {low_activity[0]} ({low_activity[1]:.1f}% de días)\n")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    generate_volume_boxplot_eda()
