"""
Mini-AlphaFold: CATH S40 Download Script
==========================================
Downloads the CATH S40 non-redundant dataset for training.

Two download modes:
  1. Individual: Fetches each domain PDB via CATH REST API (resumable)
  2. Bundle:     Downloads the full S40 PDB tarball (faster for full dataset)

Usage:
    # Download domain list + first 100 domains (for testing)
    python download_cath.py --output data/raw --max-domains 100

    # Download all S40 domains individually
    python download_cath.py --output data/raw

    # Download the full tarball (faster but larger initial download)
    python download_cath.py --output data/raw --mode bundle
"""

import argparse
import gzip
import io
import json
import logging
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

# ============================================================
# CATH URLs
# ============================================================

# S40 non-redundant domain list (one domain ID per line)
CATH_S40_LIST_URL = (
    "http://download.cathdb.info/cath/releases/latest-release/"
    "non-redundant-data-sets/cath-dataset-nonredundant-S40.list"
)

# S40 PDB bundle (tarball of all domain PDB files)
CATH_S40_BUNDLE_URL = (
    "http://download.cathdb.info/cath/releases/latest-release/"
    "non-redundant-data-sets/cath-dataset-nonredundant-S40.pdb.tgz"
)

# REST API for individual domain PDB files
# NOTE: The "latest" alias is broken as of 2026; use explicit version instead.
CATH_DOMAIN_PDB_URL = (
    "https://www.cathdb.info/version/v4_4_0/api/rest/id/{domain_id}.pdb"
)

# Fallback: older CATH version endpoint
CATH_DOMAIN_PDB_FALLBACK_URL = (
    "https://www.cathdb.info/version/v4_3_0/api/rest/id/{domain_id}.pdb"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ============================================================
# Domain List Download
# ============================================================

def download_domain_list(output_dir: Path) -> list[str]:
    """Download and parse the CATH S40 domain list.

    The list file contains one domain ID per line. Some lines may have
    additional whitespace-separated fields; we take only the first field.

    Returns:
        List of domain ID strings (e.g., ['1a0aA00', '1a0tP00', ...])
    """
    list_path = output_dir / "cath-s40-domain-list.txt"

    if list_path.exists():
        log.info(f"Domain list already exists: {list_path}")
    else:
        log.info(f"Downloading CATH S40 domain list...")
        log.info(f"  URL: {CATH_S40_LIST_URL}")

        try:
            resp = requests.get(CATH_S40_LIST_URL, timeout=60)
            resp.raise_for_status()
        except requests.RequestException as e:
            log.error(f"Failed to download domain list: {e}")
            raise

        list_path.write_text(resp.text, encoding="utf-8")
        log.info(f"  Saved to: {list_path}")

    # Parse domain IDs (first whitespace-delimited field per line)
    domain_ids = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            domain_id = line.split()[0]
            domain_ids.append(domain_id)

    log.info(f"  Total domains in S40 list: {len(domain_ids)}")
    return domain_ids


# ============================================================
# Individual Domain Download
# ============================================================

def download_single_domain(
    domain_id: str,
    output_dir: Path,
    max_retries: int = 3,
    timeout: int = 30,
) -> bool:
    """Download a single domain PDB file from CATH.

    Tries the REST API first, then falls back to the FTP endpoint.
    Skips if the file already exists (supports resume).

    Returns:
        True if successful (or already exists), False on failure.
    """
    out_path = output_dir / f"{domain_id}.pdb"

    # Skip if already downloaded
    if out_path.exists() and out_path.stat().st_size > 100:
        return True

    urls = [
        CATH_DOMAIN_PDB_URL.format(domain_id=domain_id),
        CATH_DOMAIN_PDB_FALLBACK_URL.format(domain_id=domain_id),
    ]

    for url in urls:
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, timeout=timeout)
                if resp.status_code == 200 and len(resp.content) > 100:
                    out_path.write_bytes(resp.content)
                    return True
                elif resp.status_code == 404:
                    break  # Try next URL
                else:
                    # Rate limited or server error — back off and retry
                    time.sleep(1.0 * (attempt + 1))
            except requests.RequestException:
                time.sleep(1.0 * (attempt + 1))

    return False


def download_domains_individual(
    domain_ids: list[str],
    output_dir: Path,
    max_workers: int = 4,
) -> tuple[int, int]:
    """Download multiple domain PDB files in parallel.

    Uses a thread pool with rate-limiting to avoid overwhelming the server.

    Returns:
        (success_count, failure_count)
    """
    log.info(f"Downloading {len(domain_ids)} domains (individual mode)...")
    log.info(f"  Workers: {max_workers}, Output: {output_dir}")

    success = 0
    failed = 0
    failed_ids: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_single_domain, did, output_dir): did
            for did in domain_ids
        }

        pbar = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Downloading PDBs",
            unit="file",
        )

        for future in pbar:
            domain_id = futures[future]
            try:
                if future.result():
                    success += 1
                else:
                    failed += 1
                    failed_ids.append(domain_id)
            except Exception as e:
                failed += 1
                failed_ids.append(domain_id)
                log.debug(f"Error downloading {domain_id}: {e}")

            pbar.set_postfix(ok=success, fail=failed)

    # Save list of failed downloads for retry
    if failed_ids:
        failed_path = output_dir / "failed_downloads.txt"
        failed_path.write_text("\n".join(failed_ids), encoding="utf-8")
        log.warning(f"  {failed} domains failed. See: {failed_path}")

    log.info(f"  Download complete: {success} succeeded, {failed} failed")
    return success, failed


# ============================================================
# Bundle (Tarball) Download
# ============================================================

def download_bundle(output_dir: Path) -> int:
    """Download and extract the full CATH S40 PDB tarball.

    This is faster than individual downloads for the full dataset
    but requires ~2-4 GB of disk space for the compressed file.

    Returns:
        Number of PDB files extracted.
    """
    bundle_path = output_dir / "cath-s40-bundle.pdb.tgz"

    # Download tarball if not already present
    if not bundle_path.exists():
        log.info(f"Downloading CATH S40 PDB bundle (~2 GB)...")
        log.info(f"  URL: {CATH_S40_BUNDLE_URL}")

        try:
            resp = requests.get(CATH_S40_BUNDLE_URL, stream=True, timeout=120)
            resp.raise_for_status()
        except requests.RequestException as e:
            log.error(f"Failed to download bundle: {e}")
            raise

        total_size = int(resp.headers.get("content-length", 0))
        with open(bundle_path, "wb") as f:
            pbar = tqdm(
                total=total_size,
                desc="Downloading bundle",
                unit="B",
                unit_scale=True,
            )
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
            pbar.close()

        log.info(f"  Saved to: {bundle_path}")
    else:
        log.info(f"Bundle already exists: {bundle_path}")

    # Extract PDB files
    log.info("Extracting PDB files from bundle...")
    count = 0
    try:
        with tarfile.open(bundle_path, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.name.endswith(".pdb")]
            for member in tqdm(members, desc="Extracting", unit="file"):
                # Flatten directory structure: extract just the filename
                member_name = Path(member.name).name
                out_path = output_dir / member_name
                if not out_path.exists():
                    # Extract to memory, then write (safer than extractall)
                    f = tar.extractfile(member)
                    if f is not None:
                        out_path.write_bytes(f.read())
                        count += 1
                else:
                    count += 1
    except Exception as e:
        log.error(f"Error extracting bundle: {e}")
        raise

    log.info(f"  Extracted {count} PDB files to {output_dir}")
    return count


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download CATH S40 non-redundant dataset for Mini-AlphaFold",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test: download first 50 domains
  python download_cath.py --output data/raw --max-domains 50

  # Full individual download (resumable)
  python download_cath.py --output data/raw --workers 4

  # Full bundle download (faster but larger)
  python download_cath.py --output data/raw --mode bundle
        """,
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/raw",
        help="Output directory for PDB files (default: data/raw)",
    )
    parser.add_argument(
        "--mode",
        choices=["individual", "bundle"],
        default="individual",
        help="Download mode: 'individual' (resumable, default) or 'bundle' (faster)",
    )
    parser.add_argument(
        "--max-domains",
        type=int,
        default=None,
        help="Limit number of domains to download (useful for testing)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel download threads (default: 4)",
    )

    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get domain list
    domain_ids = download_domain_list(output_dir)

    if args.max_domains is not None:
        domain_ids = domain_ids[: args.max_domains]
        log.info(f"Limited to {len(domain_ids)} domains (--max-domains)")

    # Step 2: Download PDB files
    if args.mode == "bundle":
        count = download_bundle(output_dir)
    else:
        success, failed = download_domains_individual(
            domain_ids, output_dir, max_workers=args.workers
        )
        count = success

    # Step 3: Summary
    pdb_files = list(output_dir.glob("*.pdb"))
    log.info(f"\n{'='*50}")
    log.info(f"Download complete!")
    log.info(f"  PDB files in {output_dir}: {len(pdb_files)}")
    log.info(f"  Next step: python scripts/preprocess.py --input {output_dir}")


if __name__ == "__main__":
    main()
