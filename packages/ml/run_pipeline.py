import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def step(msg: str):
    print(f"\n{'='*70}")
    print(f"  {msg}")
    print(f"{'='*70}")


def run(cmd: list[str], cwd: Path) -> None:
    print(f"  $ {' '.join(cmd)}  (in {cwd})")
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)
    if result.returncode != 0:
        print(f"\n  ERROR: step failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="S&P 500 Volatility Pipeline")
    parser.add_argument("--no-api", action="store_true", help="Skip starting the API server")
    parser.add_argument("--api-only", action="store_true", help="Only start the API server")
    args = parser.parse_args()

    dataset_dir = PROJECT_ROOT / "packages" / "ml" / "dataset"
    model_dir = PROJECT_ROOT / "packages" / "ml" / "model"
    app_dir = PROJECT_ROOT / "packages" / "app"

    if args.api_only:
        step("Starting API server only")
        run(["python", "api.py"], cwd=app_dir)
        return

    # ── Step 1: Dataset pipeline ──
    step("Step 1/3: Dataset Generation (download → prepare → feature engineering)")
    run(["python", "main.py"], cwd=dataset_dir)

    # ── Step 2: ML Model pipeline ──
    step("Step 2/3: Model Training (feature selection → CV → train → visualize)")
    run(["python", "main.py"], cwd=model_dir)

    if args.no_api:
        step("Pipeline complete (API skipped)")
        return

    # ── Step 3: API server ──
    step("Step 3/3: Starting API server on http://0.0.0.0:8080")
    run(["python", "api.py"], cwd=app_dir)


if __name__ == "__main__":
    main()
