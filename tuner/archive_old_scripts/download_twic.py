"""
下載 TWIC (The Week in Chess) PGN 棋譜資料。
自動從 twic1600 到 twic1630 批量下載、解壓縮，並合併成單一 PGN 檔案。

用法:
    python tuner/download_twic.py
    python tuner/download_twic.py --start 1600 --end 1630 --output tuner/twic_merged.pgn
"""
import os
import sys
import time
import zipfile
import argparse
import urllib.request
import urllib.error

# TWIC PGN zip 檔案的下載 URL 模板
TWIC_URL_TEMPLATE = "https://theweekinchess.com/zips/twic{n}g.zip"
FALLBACK_URL_TEMPLATE = "https://www.theweekinchess.com/zips/twic{n}g.zip"

def download_file(url: str, dest_path: str, retries: int = 3) -> bool:
    """下載單一檔案，失敗時重試。"""
    for attempt in range(1, retries + 1):
        try:
            print(f"  Downloading: {url} (attempt {attempt})", end="", flush=True)
            # 加上 User-Agent 避免被拒絕
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; chess-data-downloader/1.0)"
            })
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read()
            with open(dest_path, "wb") as f:
                f.write(data)
            size_kb = len(data) / 1024
            print(f" -> {size_kb:.1f} KB [OK]")
            return True
        except urllib.error.HTTPError as e:
            print(f" -> HTTP {e.code} [FAIL]")
            return False  # 404 等 HTTP 錯誤不重試
        except Exception as e:
            print(f" -> Error: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)  # exponential backoff
    return False


def extract_pgn_from_zip(zip_path: str, extract_dir: str) -> list[str]:
    """從 zip 檔案中解壓縮所有 .pgn 檔案，回傳解壓出的路徑列表。"""
    pgn_paths = []
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for name in zf.namelist():
                if name.lower().endswith('.pgn'):
                    zf.extract(name, extract_dir)
                    pgn_paths.append(os.path.join(extract_dir, name))
    except zipfile.BadZipFile:
        print(f"  Warning: {zip_path} is not a valid zip file, skipping.")
    return pgn_paths


def main():
    parser = argparse.ArgumentParser(
        description="Download TWIC PGN files and merge into a single PGN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--start", type=int, default=1600, help="Starting TWIC issue number.")
    parser.add_argument("--end",   type=int, default=1630, help="Ending TWIC issue number (inclusive).")
    parser.add_argument("--output", type=str, default="tuner/twic_merged.pgn",
                        help="Output merged PGN file path.")
    parser.add_argument("--cache-dir", type=str, default="tuner/twic_cache",
                        help="Directory to cache downloaded zip files.")
    parser.add_argument("--keep-individual", action="store_true",
                        help="Keep individual extracted PGN files after merging.")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    issues = list(range(args.start, args.end + 1))
    print(f"Downloading TWIC issues {args.start} to {args.end} ({len(issues)} files)...")
    print(f"Cache dir  : {args.cache_dir}")
    print(f"Output file: {args.output}")
    print()

    all_pgn_paths = []
    failed = []

    for n in issues:
        zip_name = f"twic{n}g.zip"
        zip_path = os.path.join(args.cache_dir, zip_name)

        # 已有緩存則跳過下載
        if os.path.exists(zip_path):
            print(f"[{n}] Cache hit: {zip_name}")
        else:
            url = TWIC_URL_TEMPLATE.format(n=n)
            ok = download_file(url, zip_path)
            if not ok:
                # 嘗試備用 URL
                fallback = FALLBACK_URL_TEMPLATE.format(n=n)
                ok = download_file(fallback, zip_path)
            if not ok:
                failed.append(n)
                continue

        # 解壓縮
        pgn_paths = extract_pgn_from_zip(zip_path, args.cache_dir)
        if pgn_paths:
            print(f"[{n}] Extracted {len(pgn_paths)} PGN file(s).")
            all_pgn_paths.extend(pgn_paths)
        else:
            print(f"[{n}] Warning: no PGN found in zip.")
            failed.append(n)

    if not all_pgn_paths:
        print("\nNo PGN files collected. Exiting.")
        sys.exit(1)

    # 合併所有 PGN 成單一檔案
    print(f"\nMerging {len(all_pgn_paths)} PGN file(s) into: {args.output}")
    total_games = 0
    with open(args.output, 'w', encoding='utf-8', errors='replace') as out_f:
        for pgn_path in sorted(all_pgn_paths):
            with open(pgn_path, 'r', encoding='utf-8', errors='replace') as in_f:
                content = in_f.read()
            # 計算遊戲數（簡易估算：計算 [Event 標頭數量）
            total_games += content.count('\n[Event ')
            out_f.write(content)
            # 確保每個檔案之間有空行分隔
            if not content.endswith('\n\n'):
                out_f.write('\n')

    print(f"Done! Estimated games in merged file: ~{total_games}")
    print(f"Output: {os.path.abspath(args.output)}")

    if failed:
        print(f"\nWarning: Failed to download {len(failed)} issue(s): {failed}")
        print("These issues may not exist yet or the URL format may have changed.")

    if not args.keep_individual:
        # 清理解壓縮出的 .pgn（保留 .zip 緩存）
        for p in all_pgn_paths:
            try:
                os.remove(p)
            except OSError:
                pass


if __name__ == "__main__":
    main()
