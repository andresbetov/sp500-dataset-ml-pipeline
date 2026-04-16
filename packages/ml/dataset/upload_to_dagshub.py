from __future__ import annotations

import argparse
import os
from pathlib import Path

from dagshub.upload import Repo

from .main import get_dataframe

ENV_REPO = "DAGSHUB_REPO"
ENV_LOCAL_FILE = "DAGSHUB_LOCAL_FILE"
ENV_REMOTE_PATH = "DAGSHUB_REMOTE_PATH"
ENV_BRANCH = "DAGSHUB_BRANCH"
ENV_TOKEN = "DAGSHUB_TOKEN"
DEFAULT_LOCAL_FILE = "data/processed/sp500_features.parquet"


def _parse_repo(repo: str) -> tuple[str, str]:
    try:
        owner, name = repo.strip().split("/", maxsplit=1)
    except ValueError as error:
        raise ValueError("Repo must use format 'owner/repo'") from error

    if not owner or not name:
        raise ValueError("Repo must use format 'owner/repo'")
    return owner, name


def upload_parquet_to_dagshub(
    *,
    local_parquet_path: str | Path,
    repo: str,
    remote_path: str,
    token: str,
    branch: str = "main",
    versioning: str = "dvc",
) -> None:
    parquet_path = Path(local_parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    owner, name = _parse_repo(repo)
    os.environ["DAGSHUB_USER_TOKEN"] = token

    try:
        repo_client = Repo(owner, name, branch=branch)
    except TypeError:
        repo_client = Repo(owner, name)

    repo_client.upload(
        local_path=str(parquet_path),
        remote_path=remote_path,
        versioning=versioning,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload a local parquet file to DagsHub")
    parser.add_argument(
        "--repo",
        default=os.getenv(ENV_REPO),
        help=f"DagsHub repository in format owner/repo. Env fallback: {ENV_REPO}",
    )
    parser.add_argument(
        "--local-file",
        default=os.getenv(ENV_LOCAL_FILE, DEFAULT_LOCAL_FILE),
        help=f"Local parquet file path. Env fallback: {ENV_LOCAL_FILE}",
    )
    parser.add_argument(
        "--remote-path",
        default=os.getenv(ENV_REMOTE_PATH),
        help=f"Destination path in repository. Env fallback: {ENV_REMOTE_PATH}",
    )
    parser.add_argument(
        "--branch",
        default=os.getenv(ENV_BRANCH, "main"),
        help=f"Target branch. Env fallback: {ENV_BRANCH}",
    )
    parser.add_argument(
        "--token",
        default=os.getenv(ENV_TOKEN),
        help=f"DagsHub token. Env fallback: {ENV_TOKEN}",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    missing_fields: list[str] = []
    if not args.repo:
        missing_fields.append(f"repo (--repo or {ENV_REPO})")
    if not args.remote_path:
        missing_fields.append(f"remote path (--remote-path or {ENV_REMOTE_PATH})")
    if not args.token:
        missing_fields.append(f"token (--token or {ENV_TOKEN})")

    if missing_fields:
        missing = ", ".join(missing_fields)
        raise ValueError(f"Missing required inputs: {missing}")

    # Build and persist the features parquet before uploading to DagsHub.
    get_dataframe(parquet_path=args.local_file)

    result = upload_parquet_to_dagshub(
        local_parquet_path=args.local_file,
        repo=args.repo,
        remote_path=args.remote_path,
        token=args.token,
        branch=args.branch,
    )
    print(f"Upload successful: {args.remote_path}")


if __name__ == "__main__":
    main()

