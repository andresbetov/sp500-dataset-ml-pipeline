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
    dataset_dir = PROJECT_ROOT / "packages" / "ml" / "dataset"
    model_dir = PROJECT_ROOT / "packages" / "ml" / "model"

    # ── Step 1: Dataset pipeline ──
    step("Step 1/2: Dataset Generation (download → prepare → feature engineering)")
    run(["python", "main.py"], cwd=dataset_dir)

    # ── Step 2: ML Model pipeline ──
    step("Step 2/2: Model Training (feature selection → CV → train → visualize)")
    run(["python", "main.py"], cwd=model_dir)


if __name__ == "__main__":
    main()
