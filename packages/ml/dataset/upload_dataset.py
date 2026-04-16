#!/usr/bin/env python3
"""
Versiona y sube el dataset .parquet a DagsHub via DVC.
Credenciales leídas desde .env
"""

import subprocess
import sys
from pathlib import Path
from main import get_dataframe

from dotenv import load_dotenv
import os


PARQUET_PATH = Path("/data/projects/sp500-dataset-ml-pipeline/data/processed/sp500_features.parquet")  # ajusta esta ruta
REPO_ROOT = Path("/data/projects/sp500-dataset-ml-pipeline")

def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        cwd=REPO_ROOT
    )
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip(), file=sys.stderr)
    if result.returncode != 0:
        print(f"[ERROR] Exit code: {result.returncode}")
        print(f"[STDERR] {result.stderr}")
        sys.exit(1)


def init_dvc_remote(dagshub_user: str, dagshub_repo: str) -> None:
    result = subprocess.run(
        ["dvc", "remote", "list"],
        check=True, text=True, capture_output=True,
        cwd=REPO_ROOT
    )
    if "origin" in result.stdout:
        run(["dvc", "remote", "modify", "origin", "url", f"s3://{dagshub_repo}"])
    else:
        run(["dvc", "remote", "add", "origin", f"s3://{dagshub_repo}"])
        run(["dvc", "remote", "default", "origin"])


def configure_dvc_remote(token: str, endpoint: str) -> None:
    run(["dvc", "remote", "modify", "origin", "--local", "access_key_id", token])
    run(["dvc", "remote", "modify", "origin", "--local", "secret_access_key", token])
    run(["dvc", "remote", "modify", "origin", "--local", "endpointurl", endpoint])

def main() -> None:
    load_dotenv()

    token = os.getenv("DAGSHUB_TOKEN")
    endpoint = os.getenv("DAGSHUB_ENDPOINT")
    dagshub_user = os.getenv("DAGSHUB_USER")
    dagshub_repo = os.getenv("DAGSHUB_REPO")

    print(f"TOKEN:    {token}")
    print(f"ENDPOINT: {endpoint}")
    print(f"USER:     {dagshub_user}")
    print(f"REPO:     {dagshub_repo}")

    if not all([token, endpoint, dagshub_user, dagshub_repo]):
        print("[ERROR] Faltan variables en el .env")
        sys.exit(1)

    # Generar el dataset
    print("Generando dataset...")
    get_dataframe(parquet_path=PARQUET_PATH)

    if not PARQUET_PATH.exists():
        print(f"[ERROR] No se encontró el archivo: {PARQUET_PATH}")
        sys.exit(1)

    init_dvc_remote(dagshub_user, dagshub_repo)
    configure_dvc_remote(token, endpoint)

    run(["dvc", "add", str(PARQUET_PATH)])
    run(["dvc", "push"])

    dvc_file = PARQUET_PATH.with_suffix(".parquet.dvc")
    gitignore = PARQUET_PATH.parent / ".gitignore"

    run(["git", "add", str(dvc_file), str(gitignore)])
    run(["git", "commit", "-m", f"dataset: add {PARQUET_PATH.name} via DVC"])
    run(["git", "push"])

    print("\n✅ Dataset subido y versionado correctamente.")

main()