"""
EDA Visualization #5: Diversified Portfolio Correlation Heatmap

Purpose:
    Demonstrates a diversified 15-ticker portfolio and their daily returns correlations.
    Compares to the tech-heavy Top 20 to show the importance of portfolio diversification
    for ML model robustness and generalization.

ML Implications:
    1. Feature decorrelation: Low-correlation inputs reduce multicollinearity in neural networks
    2. Portfolio hedging: Inverse correlations provide natural risk offsetting for loss functions
    3. Generalization: Mixed asset classes prevent overfitting to tech sector trends
    4. Loss landscape: Lower feature correlation flattens loss surface, improving optimizer convergence
    5. Regularization proxy: Natural diversification acts as L2 regularization without explicit penalty
    6. Ensemble strategy: Multiple asset classes enable cross-sector ensemble models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from ..loader import load_dataframe

# Configuration
OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "data" / "eda"
OUTPUT_PATH = OUTPUT_DIR / "05_diversified_portfolio_correlation.png"


def generate_diversified_portfolio_eda():
    """Generate correlation heatmap for diversified 15-ticker portfolio."""
    
    print("\n" + "=" * 70)
    print("📊 EDA Visualization #5: Diversified Portfolio Correlation")
    print("=" * 70)
    
    # Load data
    print("\n🔹 Loading dataset...")
    df = load_dataframe()
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    print(f"   ✓ Loaded {len(df):,} records for {df['ticker'].nunique()} tickers")
    
    # Ensure date is datetime
    df["date"] = pd.to_datetime(df["date"])
    
    # Calculate daily returns for all available tickers
    print("\n🔹 Calculating daily returns...")
    df["return"] = df.groupby("ticker")["adj_close"].pct_change()
    returns_pivot_all = df.pivot_table(
        index="date", 
        columns="ticker", 
        values="return"
    )
    print(f"   ✓ Pivot table shape: {returns_pivot_all.shape}")
    
    # Identify Top 20 by median volume
    print("\n🔹 Identifying Top 20 by median volume...")
    median_volume = df.groupby("ticker")["volume"].median().sort_values(ascending=False)
    top_20_list = median_volume.head(20).index.tolist()
    print(f"   ✓ Top 20: {', '.join(top_20_list)}")
    
    # Select first 15 from Top 20
    print("\n🔹 Selecting 15-ticker portfolio...")
    portfolio_tickers = top_20_list[:15]
    print(f"   ✓ Portfolio (15): {', '.join(portfolio_tickers)}")
    
    # Filter to period where ALL 15 tickers coexist
    print("\n🔹 Finding coexistence period for all 15 tickers...")
    df_filtered = df[df["ticker"].isin(portfolio_tickers)].copy()
    date_range_by_ticker = df_filtered.groupby("ticker")["date"].agg(["min", "max"])
    earliest_end = date_range_by_ticker["min"].max()
    latest_start = date_range_by_ticker["max"].min()
    print(f"   Period: {earliest_end} to {latest_start}")
    
    df_period = df_filtered[(df_filtered["date"] >= earliest_end) & (df_filtered["date"] <= latest_start)].copy()
    print(f"   ✓ Filtered to {len(df_period):,} records in coexistence period")
    
    # Recalculate returns for coexistence period
    print("\n🔹 Recalculating returns for coexistence period...")
    df_period["return"] = df_period.groupby("ticker")["adj_close"].pct_change()
    returns_portfolio = df_period.pivot_table(
        index="date",
        columns="ticker",
        values="return"
    )[portfolio_tickers]
    print(f"   ✓ Returns shape: {returns_portfolio.shape}")
    print(f"   ✓ Missing values: {returns_portfolio.isna().sum().sum()} / {returns_portfolio.size}")
    
    # Calculate correlation matrix
    print("\n🔹 Computing correlation matrix...")
    corr_matrix = returns_portfolio.corr()
    print(f"   ✓ Correlation stats:")
    print(f"      - Min: {corr_matrix.values[corr_matrix.values != 1].min():.4f}")
    print(f"      - Max: {corr_matrix.values[corr_matrix.values != 1].max():.4f}")
    print(f"      - Mean: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.4f}")
    
    # Create lower triangle mask
    print("\n🔹 Applying lower triangle mask...")
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=0)
    
    # Generate heatmap
    print("\n🔹 Generating heatmap...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson Correlation Coefficient"},
        ax=ax
    )
    
    ax.set_title(
        "Daily Returns Correlation: Diversified 15-Ticker Portfolio\n"
        "(Lower Triangle Only - No Diagonal)",
        fontsize=14,
        fontweight="bold",
        pad=20
    )
    ax.set_xlabel("Ticker", fontsize=12, fontweight="bold")
    ax.set_ylabel("Ticker", fontsize=12, fontweight="bold")
    
    # Rotate labels for readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"\n✅ Saved to: {OUTPUT_PATH}")
    print(f"   Size: {OUTPUT_PATH.stat().st_size / 1024:.0f} KB")
    
    plt.close()
    
    return corr_matrix


if __name__ == "__main__":
    corr_matrix = generate_diversified_portfolio_eda()
