from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loader import load_dataframe

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "eda"
DEFAULT_FIGSIZE = (16, 6)
TICKERS_TO_PLOT = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
COLORS = {
    "AAPL": "#A2AAAD",      # Gray
    "MSFT": "#00A4EF",      # Blue
    "GOOGL": "#4285F4",     # Google Blue
    "AMZN": "#FF9900",      # Orange
    "NVDA": "#76B900",      # Green
    "S&P 500": "#E74C3C",   # Red
}


def _normalize_to_100(series: pd.Series) -> pd.Series:
    first_price = series.iloc[0]
    return (series / first_price) * 100


def _calculate_total_return(first_price: float, last_price: float) -> float:
    return ((last_price / first_price) - 1) * 100


def _calculate_max_drawdown(prices: pd.Series) -> float:
    cummax = prices.expanding().max()
    drawdown = (prices - cummax) / cummax
    return drawdown.min() * 100


def _validate_tickers_exist(df: pd.DataFrame, tickers: list[str]) -> list[str]:
    available_tickers = set(df["ticker"].unique())
    valid_tickers = [t for t in tickers if t in available_tickers]
    missing_tickers = [t for t in tickers if t not in available_tickers]
    
    if missing_tickers:
        print(f"+ Warning: Tickers not found in dataset: {', '.join(missing_tickers)}")
    
    return valid_tickers


def generate_price_timeseries_eda() -> None:
    print("\n" + "="*80)
    print("Serie Temporal de Precios - EDA")
    print("="*80 + "\n")
    
    raw_df = load_dataframe()
    
    raw_df.columns = raw_df.columns.str.lower().str.replace(" ", "_")
    
    print(f"- Dataset cargado: {len(raw_df)} filas, {raw_df['ticker'].nunique()} tickers únicos")
    
    valid_tickers = _validate_tickers_exist(raw_df, TICKERS_TO_PLOT)
    if not valid_tickers:
        raise ValueError(f"No valid tickers found. Available: {sorted(raw_df['ticker'].unique())}")
    
    print(f"- Tickers a graficar: {', '.join(valid_tickers)}")
    
    df = raw_df[raw_df["ticker"].isin(valid_tickers)].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    date_range_start = df["date"].min().strftime("%Y-%m-%d")
    date_range_end = df["date"].max().strftime("%Y-%m-%d")
    total_days = (df["date"].max() - df["date"].min()).days
    
    print(f"- Período: {date_range_start} a {date_range_end} ({total_days} días)")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=DEFAULT_FIGSIZE)
    
    metrics_data = {}
    ticker_stats = []
    
    # Tickers normales (sin NVDA)
    tickers_normal = [t for t in valid_tickers if t != "NVDA"]
    
    for ticker in tickers_normal + ["S&P 500"]:
        if ticker == "S&P 500":
            ticker_data = df.groupby("date")["adj_close"].mean().reset_index()
            ticker_data["ticker"] = "S&P 500"
            plot_label = "S&P 500 (Promedio Simple)"
        else:
            ticker_data = df[df["ticker"] == ticker].copy()
            plot_label = ticker
        
        if len(ticker_data) == 0:
            print(f"+ Skipping {ticker}: no data found")
            continue
        
        ticker_data = ticker_data.sort_values("date").reset_index(drop=True)
        normalized_prices = _normalize_to_100(ticker_data["adj_close"])
        
        first_price = ticker_data["adj_close"].iloc[0]
        last_price = ticker_data["adj_close"].iloc[-1]
        total_return = _calculate_total_return(first_price, last_price)
        max_drawdown = _calculate_max_drawdown(ticker_data["adj_close"])
        
        metrics_data[ticker] = {
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "first_price": first_price,
            "last_price": last_price,
        }
        
        if ticker != "S&P 500":
            ticker_stats.append({
                "ticker": ticker,
                "total_return": total_return,
                "max_drawdown": max_drawdown,
            })
        
        ax1.plot(
            ticker_data["date"],
            normalized_prices,
            linewidth=2.5,
            color=COLORS[ticker],
            label=plot_label,
            alpha=0.85,
        )
    
    ax1.axhline(y=100, color="gray", linestyle="--", linewidth=1, alpha=0.4)
    ax1.grid(True, alpha=0.25, linestyle="-", linewidth=0.5)
    ax1.set_xlabel("Fecha", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Precio Normalizado (Base = 100)", fontsize=11, fontweight="bold")
    ax1.set_title("Tech Stocks + S&P 500", fontsize=12, fontweight="bold")
    ax1.legend(
        loc="upper left",
        fontsize=10,
        framealpha=0.95,
        edgecolor="gray",
        fancybox=True,
    )
    
    # Agregar métricas en texto en ax1
    metrics_text_left = "Rendimiento:\n"
    for stat in sorted(ticker_stats[:4], key=lambda x: x["total_return"], reverse=True):
        ret_str = f"+{stat['total_return']:.0f}%" if stat['total_return'] > 0 else f"{stat['total_return']:.0f}%"
        dd_str = f"{stat['max_drawdown']:.1f}%"
        metrics_text_left += f"{stat['ticker']:6s} {ret_str:>10s} | DD: {dd_str}\n"
    
    sp500_ret = f"+{metrics_data['S&P 500']['total_return']:.0f}%" if metrics_data['S&P 500']['total_return'] > 0 else f"{metrics_data['S&P 500']['total_return']:.0f}%"
    sp500_dd = f"{metrics_data['S&P 500']['max_drawdown']:.1f}%"
    metrics_text_left += f"{'SP500':6s} {sp500_ret:>10s} | DD: {sp500_dd}"
    
    ax1.text(
        0.02, 0.02,
        metrics_text_left,
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        family="monospace",
    )
    
    # NVDA en su propia gráfica
    nvda_data = df[df["ticker"] == "NVDA"].copy()
    if len(nvda_data) > 0:
        nvda_data = nvda_data.sort_values("date").reset_index(drop=True)
        nvda_normalized = _normalize_to_100(nvda_data["adj_close"])
        
        first_price_nvda = nvda_data["adj_close"].iloc[0]
        last_price_nvda = nvda_data["adj_close"].iloc[-1]
        total_return_nvda = _calculate_total_return(first_price_nvda, last_price_nvda)
        max_drawdown_nvda = _calculate_max_drawdown(nvda_data["adj_close"])
        
        metrics_data["NVDA"] = {
            "total_return": total_return_nvda,
            "max_drawdown": max_drawdown_nvda,
            "first_price": first_price_nvda,
            "last_price": last_price_nvda,
        }
        
        ticker_stats.append({
            "ticker": "NVDA",
            "total_return": total_return_nvda,
            "max_drawdown": max_drawdown_nvda,
        })
        
        ax2.plot(
            nvda_data["date"],
            nvda_normalized,
            linewidth=2.5,
            color=COLORS["NVDA"],
            label="NVDA",
            alpha=0.85,
        )
        
        ax2.axhline(y=100, color="gray", linestyle="--", linewidth=1, alpha=0.4)
        ax2.grid(True, alpha=0.25, linestyle="-", linewidth=0.5)
        ax2.set_xlabel("Fecha", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Precio Normalizado (Base = 100)", fontsize=11, fontweight="bold")
        ax2.set_title("NVDA (Rendimiento Extremo)", fontsize=12, fontweight="bold")
        ax2.legend(
            loc="upper left",
            fontsize=10,
            framealpha=0.95,
            edgecolor="gray",
            fancybox=True,
        )
        
        # Agregar métricas en texto en ax2
        ret_str = f"+{total_return_nvda:.0f}%" if total_return_nvda > 0 else f"{total_return_nvda:.0f}%"
        dd_str = f"{max_drawdown_nvda:.1f}%"
        metrics_text_right = f"Rendimiento:\n{'NVDA':6s} {ret_str:>10s}\nDrawdown: {dd_str}"
        
        ax2.text(
            0.02, 0.02,
            metrics_text_right,
            transform=ax2.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
            family="monospace",
        )
    
    fig.suptitle(
        f"Serie Temporal de Precios - Comparativa\n{date_range_start} a {date_range_end} | {total_days} días",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    
    plt.tight_layout()
    
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "01_price_time_series.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n- Gráfico guardado en: {output_path}")
    plt.close(fig)
    
    print("\n" + "="*80)
    print("Métricas de Rendimiento")
    print("="*80)
    print(f"Período Total: {date_range_start} a {date_range_end}")
    print(f"Total de días: {total_days}\n")
    
    print("Rendimiento Total (%):")
    for stat in sorted(ticker_stats, key=lambda x: x["total_return"], reverse=True):
        return_str = f"+{stat['total_return']:.1f}%" if stat['total_return'] > 0 else f"{stat['total_return']:.1f}%"
        print(f"  {stat['ticker']:8s}: {return_str:>10s}")
    
    if "S&P 500" in metrics_data:
        sp500_return = metrics_data["S&P 500"]["total_return"]
        return_str = f"+{sp500_return:.1f}%" if sp500_return > 0 else f"{sp500_return:.1f}%"
        print(f"  {'S&P 500':8s}: {return_str:>10s} (promedio simple)\n")
    
    print("Máxima Caída (Drawdown %):")
    for stat in sorted(ticker_stats, key=lambda x: x["max_drawdown"]):
        print(f"  {stat['ticker']:8s}: {stat['max_drawdown']:.1f}%")
    
    if "S&P 500" in metrics_data:
        sp500_dd = metrics_data["S&P 500"]["max_drawdown"]
        print(f"  {'S&P 500':8s}: {sp500_dd:.1f}% (promedio simple)")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    generate_price_timeseries_eda()
