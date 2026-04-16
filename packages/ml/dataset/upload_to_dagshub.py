from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path
from typing import Any

import requests

from .main import get_dataframe

DAGSHUB_API_BASE = "https://dagshub.com/api/v1"
ENV_REPO = "DAGSHUB_REPO"
ENV_LOCAL_FILE = "DAGSHUB_LOCAL_FILE"
ENV_REMOTE_PATH = "DAGSHUB_REMOTE_PATH"
ENV_BRANCH = "DAGSHUB_BRANCH"
ENV_MESSAGE = "DAGSHUB_MESSAGE"
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


def _build_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"token {token}",
        "Accept": "application/json",
    }


def _get_existing_sha(
    *,
    owner: str,
    name: str,
    remote_path: str,
    branch: str,
    headers: dict[str, str],
) -> str | None:
    url = f"{DAGSHUB_API_BASE}/repos/{owner}/{name}/contents/{remote_path}"
    response = requests.get(url, headers=headers, params={"ref": branch}, timeout=30)

    if response.status_code == 404:
        return None

    response.raise_for_status()
    payload = response.json()
    sha = payload.get("sha")
    if isinstance(sha, str) and sha:
        return sha
    return None


def upload_parquet_to_dagshub(
    *,
    local_parquet_path: str | Path,
    repo: str,
    remote_path: str,
    token: str,
    branch: str = "main",
    commit_message: str = "Add processed parquet dataset",
) -> dict[str, Any]:
    parquet_path = Path(local_parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    owner, name = _parse_repo(repo)
    headers = _build_headers(token)
    file_sha = _get_existing_sha(
        owner=owner,
        name=name,
        remote_path=remote_path,
        branch=branch,
        headers=headers,
    )

    content = base64.b64encode(parquet_path.read_bytes()).decode("ascii")
    payload: dict[str, Any] = {
        "branch": branch,
        "message": commit_message,
        "content": content,
    }
    if file_sha:
        payload["sha"] = file_sha

    url = f"{DAGSHUB_API_BASE}/repos/{owner}/{name}/contents/{remote_path}"
    response = requests.put(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


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
        "--message",
        default=os.getenv(ENV_MESSAGE, "Add processed parquet dataset"),
        help=f"Commit message. Env fallback: {ENV_MESSAGE}",
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
        commit_message=args.message,
    )
    commit_url = result.get("commit", {}).get("html_url")
    if commit_url:
        print(f"Upload successful: {commit_url}")
    else:
        print("Upload successful")


if __name__ == "__main__":
    main()

