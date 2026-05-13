from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loader import load_dataframe

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "eda"
DEFAULT_FIGSIZE = (18, 12)
TICKERS_TO_PLOT = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
COLORS = {
    "AAPL": "#A2AAAD",      # Gris
    "MSFT": "#00A4EF",      # Azul
    "GOOGL": "#4285F4",     # Azul Google
    "AMZN": "#FF9900",      # Naranja
    "NVDA": "#76B900",      # Verde NVIDIA
    "S&P 500": "#E74C3C",   # Rojo
}
RETURNS_RANGE = (-0.10, 0.10)
BIN_WIDTH = 0.005
TRADING_DAYS_PER_YEAR = 252


def _calculate_simple_returns(series: pd.Series) -> pd.Series:
    return series.pct_change().dropna()


def _calculate_statistics(returns: pd.Series) -> dict:
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0:
        return None
    
    daily_mean = returns_clean.mean()
    daily_std = returns_clean.std()
    annual_std = daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    skewness = stats.skew(returns_clean)
    excess_kurtosis = stats.kurtosis(returns_clean)
    
    percentiles = {
        "p01": returns_clean.quantile(0.01),
        "p05": returns_clean.quantile(0.05),
        "p25": returns_clean.quantile(0.25),
        "p50": returns_clean.quantile(0.50),
        "p75": returns_clean.quantile(0.75),
        "p95": returns_clean.quantile(0.95),
        "p99": returns_clean.quantile(0.99),
    }
    
    return {
        "daily_mean": daily_mean,
        "daily_std": daily_std,
        "annual_std": annual_std,
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "percentiles": percentiles,
        "count": len(returns_clean),
    }


def _validate_tickers_exist(df: pd.DataFrame, tickers: list[str]) -> list[str]:
    available_tickers = set(df["ticker"].unique())
    valid_tickers = [t for t in tickers if t in available_tickers]
    missing_tickers = [t for t in tickers if t not in available_tickers]
    
    if missing_tickers:
        print(f"⚠️  Warning: Tickers not found in dataset: {', '.join(missing_tickers)}")
    
    return valid_tickers


def _plot_distribution_single(ax, returns: pd.Series, ticker: str, stats_data: dict) -> None:
    """Plot distribution for a single ticker on given axis."""
    if len(returns) == 0:
        return
    
    ticker_stats = stats_data
    
    # Limitar retornos para el gráfico
    returns_for_plot = returns.clip(RETURNS_RANGE[0], RETURNS_RANGE[1])
    
    # Histograma con KDE
    ax.hist(
        returns_for_plot,
        bins=int((RETURNS_RANGE[1] - RETURNS_RANGE[0]) / BIN_WIDTH),
        alpha=0.6,
        color=COLORS[ticker],
        density=True,
        edgecolor="black",
        linewidth=0.5,
    )
    
    # Curva KDE
    try:
        returns_for_plot_clean = returns_for_plot.dropna()
        if len(returns_for_plot_clean) > 1:
            kde = stats.gaussian_kde(returns_for_plot_clean)
            x_range = np.linspace(RETURNS_RANGE[0], RETURNS_RANGE[1], 200)
            ax.plot(
                x_range,
                kde(x_range),
                color=COLORS[ticker],
                linewidth=2.5,
                alpha=0.9,
                label="KDE",
            )
    except Exception:
        pass
    
    # Curva normal teórica
    mean_ret = ticker_stats["daily_mean"]
    std_ret = ticker_stats["daily_std"]
    x_normal = np.linspace(RETURNS_RANGE[0], RETURNS_RANGE[1], 200)
    normal_curve = stats.norm.pdf(x_normal, mean_ret, std_ret)
    ax.plot(
        x_normal,
        normal_curve,
        color="red",
        linewidth=2,
        linestyle="--",
        alpha=0.7,
        label="Normal Teórica",
    )
    
    # Línea de referencia en 0
    ax.axvline(x=0, color="black", linestyle="-", linewidth=1, alpha=0.5)
    
    ax.grid(True, alpha=0.25, linestyle="-", linewidth=0.5)
    ax.set_xlim(RETURNS_RANGE[0], RETURNS_RANGE[1])
    
    # Agregar estadísticas en el título
    annual_vol = ticker_stats["annual_std"] * 100
    skew = ticker_stats["skewness"]
    kurt = ticker_stats["excess_kurtosis"]
    ax.set_title(
        f"{ticker}\nVol Anual: {annual_vol:.1f}% | Skew: {skew:.2f} | Kurt: {kurt:.1f}",
        fontsize=11,
        fontweight="bold",
    )
    
    ax.set_xlabel("Retorno Diario", fontsize=10)
    ax.set_ylabel("Densidad", fontsize=10)


def generate_returns_distribution_separate_eda() -> None:
    print("\n" + "="*80)
    print("Distribución de Retornos Diarios (Separado) - EDA")
    print("="*80 + "\n")
    
    raw_df = load_dataframe()
    raw_df.columns = raw_df.columns.str.lower().str.replace(" ", "_")
    
    print(f"✓ Dataset cargado: {len(raw_df)} filas, {raw_df['ticker'].nunique()} tickers únicos")
    
    valid_tickers = _validate_tickers_exist(raw_df, TICKERS_TO_PLOT)
    if not valid_tickers:
        raise ValueError(f"No valid tickers found. Available: {sorted(raw_df['ticker'].unique())}")
    
    print(f"✓ Tickers a graficar: {', '.join(valid_tickers)}")
    
    df = raw_df[raw_df["ticker"].isin(valid_tickers)].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    date_range_start = df["date"].min().strftime("%Y-%m-%d")
    date_range_end = df["date"].max().strftime("%Y-%m-%d")
    
    print(f"✓ Período: {date_range_start} a {date_range_end}")
    
    # ============================================================================
    # GRÁFICA 1: SUBPLOTS SEPARADOS
    # ============================================================================
    print("\n📊 Generando gráfica con subplots separados...")
    
    fig, axes = plt.subplots(2, 3, figsize=DEFAULT_FIGSIZE)
    axes = axes.flatten()
    
    stats_data_all = {}
    
    # Graficar cada ticker en su subplot
    for idx, ticker in enumerate(valid_tickers + ["S&P 500"]):
        ax = axes[idx]
        
        if ticker == "S&P 500":
            ticker_returns = df.groupby("date")["adj_close"].mean().pct_change().dropna()
            plot_label = "S&P 500 (Promedio Simple)"
        else:
            ticker_data = df[df["ticker"] == ticker].copy()
            ticker_returns = _calculate_simple_returns(ticker_data["adj_close"])
            plot_label = ticker
        
        if len(ticker_returns) == 0:
            print(f"⚠️  Skipping {ticker}: no returns data")
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        
        ticker_stats = _calculate_statistics(ticker_returns)
        stats_data_all[ticker] = ticker_stats
        
        _plot_distribution_single(ax, ticker_returns, ticker, ticker_stats)
    
    fig.suptitle(
        f"Distribución de Retornos Diarios (Separado por Activo)\n{date_range_start} a {date_range_end}",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    
    plt.tight_layout()
    
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path_separate = output_dir / "02_returns_distribution_separate.png"
    fig.savefig(output_path_separate, dpi=150, bbox_inches="tight")
    print(f"✓ Gráfica separada guardada en: {output_path_separate}")
    plt.close(fig)
    
    # ============================================================================
    # GRÁFICA 2: OVERLAY (TODOS JUNTOS)
    # ============================================================================
    print("📊 Generando gráfica con overlay (todos juntos)...\n")
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    stats_data = {}
    
    # Graficar cada ticker en overlay
    for ticker in valid_tickers + ["S&P 500"]:
        if ticker == "S&P 500":
            ticker_returns = df.groupby("date")["adj_close"].mean().pct_change().dropna()
            plot_label = "S&P 500 (Promedio Simple)"
        else:
            ticker_data = df[df["ticker"] == ticker].copy()
            ticker_returns = _calculate_simple_returns(ticker_data["adj_close"])
            plot_label = ticker
        
        if len(ticker_returns) == 0:
            print(f"⚠️  Skipping {ticker}: no returns data")
            continue
        
        ticker_stats = _calculate_statistics(ticker_returns)
        stats_data[ticker] = ticker_stats
        
        # Limitar retornos para el gráfico
        returns_for_plot = ticker_returns.clip(RETURNS_RANGE[0], RETURNS_RANGE[1])
        
        # Histograma con KDE
        ax.hist(
            returns_for_plot,
            bins=int((RETURNS_RANGE[1] - RETURNS_RANGE[0]) / BIN_WIDTH),
            alpha=0.4,
            label=plot_label,
            color=COLORS[ticker],
            density=True,
        )
        
        # Curva KDE
        try:
            returns_for_plot_clean = returns_for_plot.dropna()
            if len(returns_for_plot_clean) > 1:
                kde = stats.gaussian_kde(returns_for_plot_clean)
                x_range = np.linspace(RETURNS_RANGE[0], RETURNS_RANGE[1], 200)
                ax.plot(
                    x_range,
                    kde(x_range),
                    color=COLORS[ticker],
                    linewidth=2.5,
                    alpha=0.8,
                )
        except Exception as e:
            pass
        
        # Curva normal teórica
        mean_ret = ticker_stats["daily_mean"]
        std_ret = ticker_stats["daily_std"]
        x_normal = np.linspace(RETURNS_RANGE[0], RETURNS_RANGE[1], 200)
        normal_curve = stats.norm.pdf(x_normal, mean_ret, std_ret)
        ax.plot(
            x_normal,
            normal_curve,
            color=COLORS[ticker],
            linewidth=1.5,
            linestyle="--",
            alpha=0.5,
        )
    
    # Línea de referencia en 0
    ax.axvline(x=0, color="black", linestyle="-", linewidth=1.5, alpha=0.7, label="Retorno = 0")
    
    ax.set_xlabel("Retorno Diario", fontsize=12, fontweight="bold")
    ax.set_ylabel("Densidad", fontsize=12, fontweight="bold")
    ax.set_xlim(RETURNS_RANGE[0], RETURNS_RANGE[1])
    ax.grid(True, alpha=0.25, linestyle="-", linewidth=0.5)
    
    # Agregar estadísticas en esquina superior izquierda
    stats_text = "Estadísticas Globales\n"
    stats_text += "─" * 28 + "\n"
    
    for ticker in valid_tickers + ["S&P 500"]:
        if ticker in stats_data and stats_data[ticker] is not None:
            annual_vol = stats_data[ticker]["annual_std"] * 100
            skew = stats_data[ticker]["skewness"]
            kurt = stats_data[ticker]["excess_kurtosis"]
            stats_text += f"{ticker:7s} | Vol: {annual_vol:5.1f}% | Skew: {skew:6.2f} | Kurt: {kurt:5.1f}\n"
    
    ax.text(
        0.02, 0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85),
        family="monospace",
    )
    
    ax.legend(
        loc="upper right",
        fontsize=10,
        framealpha=0.95,
        edgecolor="gray",
        fancybox=True,
    )
    
    fig.suptitle(
        f"Distribución de Retornos Diarios (Overlay)\n{date_range_start} a {date_range_end}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    
    plt.tight_layout()
    
    output_path_overlay = output_dir / "02_returns_distribution_overlay.png"
    fig.savefig(output_path_overlay, dpi=150, bbox_inches="tight")
    print(f"✓ Gráfica overlay guardada en: {output_path_overlay}")
    plt.close(fig)
    
    print("\n" + "="*80)
    print("✓ Ambas gráficas generadas exitosamente")
    print("="*80 + "\n")


if __name__ == "__main__":
    generate_returns_distribution_separate_eda()
