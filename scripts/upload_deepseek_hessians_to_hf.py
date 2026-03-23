#!/usr/bin/env python3
"""Upload DeepSeek Hessian folders to Hugging Face Hub with resume support.

Usage examples:
  python scripts/upload_deepseek_hessians_to_hf.py
  python scripts/upload_deepseek_hessians_to_hf.py --private
  python scripts/upload_deepseek_hessians_to_hf.py --repo-prefix my-prefix
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, HfFolder, create_repo


def _bytes_of_folder(folder: Path) -> int:
    total = 0
    for root, _, files in os.walk(folder):
        for name in files:
            fp = Path(root) / name
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total


def _human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    val = float(num_bytes)
    for unit in units:
        if val < 1024.0 or unit == units[-1]:
            return f"{val:.2f}{unit}"
        val /= 1024.0
    return f"{num_bytes}B"


def _upload_one(
    api: HfApi,
    repo_id: str,
    folder: Path,
    private: bool,
    max_retries: int,
    sleep_seconds: int,
) -> None:
    create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    size = _human_size(_bytes_of_folder(folder))
    print(f"[INFO] Start upload: {repo_id}")
    print(f"[INFO] Folder: {folder}")
    print(f"[INFO] Folder size: {size}")

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            # upload_large_folder is resumable and designed for big uploads.
            api.upload_large_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=str(folder),
            )
            dt = (time.time() - t0) / 60.0
            print(f"[OK] Upload completed: {repo_id} ({dt:.1f} min)")
            return
        except KeyboardInterrupt:
            print("\n[WARN] Interrupted by user. Re-run script to resume.")
            raise
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"[WARN] Attempt {attempt}/{max_retries} failed for {repo_id}: {exc}")
            if attempt < max_retries:
                print(f"[INFO] Sleep {sleep_seconds}s then retry...")
                time.sleep(sleep_seconds)

    raise RuntimeError(f"Upload failed after {max_retries} attempts for {repo_id}: {last_err}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload DeepSeek Hessians to Hugging Face Hub")
    parser.add_argument(
        "--experts-dir",
        default="/fact_home/zeyuli/quip_sharp/hessians_deepseek_moe_16b_base_experts",
        help="Path to experts Hessian directory",
    )
    parser.add_argument(
        "--qkvo-dir",
        default="/fact_home/zeyuli/quip_sharp/hessians_deepseek_moe_16b_base_qkvo",
        help="Path to qkvo Hessian directory",
    )
    parser.add_argument(
        "--experts-repo",
        default="hessians_deepseek_moe_16b_base_experts",
        help="HF repo name for experts dir",
    )
    parser.add_argument(
        "--qkvo-repo",
        default="hessians_deepseek_moe_16b_base_qkvo",
        help="HF repo name for qkvo dir",
    )
    parser.add_argument(
        "--repo-prefix",
        default="",
        help="Optional prefix for both repo names, e.g. deepseek/",
    )
    parser.add_argument("--private", action="store_true", help="Create private repos")
    parser.add_argument("--max-retries", type=int, default=8, help="Retry count per repo")
    parser.add_argument("--retry-sleep", type=int, default=30, help="Seconds between retries")
    args = parser.parse_args()

    token = HfFolder.get_token()
    if not token:
        print("[ERR] Hugging Face token not found. Please run: huggingface_hub login")
        return 1

    experts_dir = Path(args.experts_dir)
    qkvo_dir = Path(args.qkvo_dir)

    if not experts_dir.exists():
        print(f"[ERR] Missing directory: {experts_dir}")
        return 1
    if not qkvo_dir.exists():
        print(f"[ERR] Missing directory: {qkvo_dir}")
        return 1

    api = HfApi(token=token)
    user = api.whoami()["name"]

    prefix = args.repo_prefix.strip()
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    experts_repo_id = f"{user}/{prefix}{args.experts_repo}"
    qkvo_repo_id = f"{user}/{prefix}{args.qkvo_repo}"

    print(f"[INFO] Authenticated as: {user}")
    print(f"[INFO] Target repo 1: {experts_repo_id}")
    print(f"[INFO] Target repo 2: {qkvo_repo_id}")
    print("[INFO] Tip: if interrupted, re-run the same command to resume.")

    _upload_one(
        api=api,
        repo_id=experts_repo_id,
        folder=experts_dir,
        private=args.private,
        max_retries=args.max_retries,
        sleep_seconds=args.retry_sleep,
    )

    _upload_one(
        api=api,
        repo_id=qkvo_repo_id,
        folder=qkvo_dir,
        private=args.private,
        max_retries=args.max_retries,
        sleep_seconds=args.retry_sleep,
    )

    print("[OK] All uploads finished.")
    print(f"[OK] https://huggingface.co/{experts_repo_id}")
    print(f"[OK] https://huggingface.co/{qkvo_repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
