"""
GarmageSet 下载器 v2 — 断点续传 + 重试
"""
import os, sys, time, tarfile, argparse, hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import socket

HF_TOKEN = os.environ.get("HF_TOKEN", "")
PROXY = "http://127.0.0.1:7897"
BASE_URL = "https://huggingface.co/datasets/Style3D/GarmageSet/resolve/main"
DOWNLOAD_DIR = Path("D:/ClothesNetData/garmageset_download")
SHARD_NAMES = [f"{i:02x}" for i in range(256)]


def setup_proxy():
    proxy_handler = urllib.request.ProxyHandler({'http': PROXY, 'https': PROXY})
    urllib.request.install_opener(urllib.request.build_opener(proxy_handler))
    socket.setdefaulttimeout(60)


def download_with_resume(url: str, out_file: Path, max_retries: int = 5) -> bool:
    """Download with HTTP Range resume support."""
    existing_size = out_file.stat().st_size if out_file.exists() else 0

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={
                "Authorization": f"Bearer {HF_TOKEN}"
            })
            # Resume from existing bytes
            if existing_size > 0:
                req.add_header("Range", f"bytes={existing_size}-")

            with urllib.request.urlopen(req, timeout=180) as resp:
                # Check if server accepted range request
                if resp.status == 206:  # Partial Content
                    mode = 'ab'
                elif resp.status == 200 and existing_size > 0:
                    # Server doesn't support range, restart
                    existing_size = 0
                    mode = 'wb'
                else:
                    mode = 'wb'
                    existing_size = 0

                data = resp.read(8192)  # Read in chunks
                with open(out_file, mode) as f:
                    if mode == 'ab':
                        f.seek(existing_size)
                    while data:
                        f.write(data)
                        data = resp.read(8192)

            # Verify
            size = out_file.stat().st_size
            if size > 10000:
                return True

        except Exception as e:
            existing_size = out_file.stat().st_size if out_file.exists() else 0
            wait = min(30, 3 * (attempt + 1))
            print(f"  [{out_file.stem}] retry {attempt+1}/{max_retries} (have {existing_size/1024/1024:.0f}MB, wait {wait}s): {str(e)[:80]}")
            time.sleep(wait)

    return False


def download_shard(shard_type: str, shard_name: str) -> bool:
    url = f"{BASE_URL}/{shard_type}/{shard_name}.tar.gz"
    out_file = DOWNLOAD_DIR / f"{shard_type}_{shard_name}.tar.gz"

    if out_file.exists() and out_file.stat().st_size > 10000:
        size_mb = out_file.stat().st_size / 1024 / 1024
        print(f"  [{shard_name}] cached {size_mb:.0f}MB")
        return True

    out_file.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    ok = download_with_resume(url, out_file)
    elapsed = time.time() - start

    if ok:
        size_mb = out_file.stat().st_size / 1024 / 1024
        speed = size_mb / max(elapsed, 1)
        print(f"  [{shard_name}] DONE {size_mb:.1f}MB ({speed:.1f}MB/s)")
    else:
        print(f"  [{shard_name}] FAILED")

    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["raw", "images"], default="raw")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=256)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    setup_proxy()
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    shard_types = ["raw", "images"] if args.all else [args.type]

    for st in shard_types:
        names = SHARD_NAMES[args.start:args.end]
        print(f"\n{'='*50}")
        print(f"[{st}] {len(names)} shards: {names[0]}~{names[-1]}")
        print(f"{'='*50}")

        ok = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(download_shard, st, n): n for n in names}
            for i, f in enumerate(as_completed(futures)):
                try:
                    if f.result():
                        ok += 1
                except:
                    pass
                if (i + 1) % 10 == 0:
                    print(f"  [{st}] progress: {i+1}/{len(names)} ({ok} OK)")

        print(f"\n[{st}] Complete: {ok}/{len(names)} downloaded")


if __name__ == "__main__":
    main()
