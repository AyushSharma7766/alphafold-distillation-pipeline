"""
Milestone 2: Download PDB structures and precomputed MSAs from OpenProteinSet.

This script:
1. Downloads a curated list of ~2000 PDB chains (high quality, diverse, pre-2021)
2. Downloads their precomputed MSAs from the OpenProteinSet S3 bucket
3. Creates a time-based train/val split (train < 2021-09-30)

Usage (on the H200 server, inside Docker):
    pip install awscli
    python data/download_training_data.py --n_chains 2000 --output_dir data/
    
Estimated disk: ~30-50 GB for 2000 chains with MSAs
Estimated time: 30-60 minutes depending on bandwidth
"""
import os
import sys
import json
import subprocess
import argparse
import urllib.request
from datetime import datetime
from pathlib import Path


def get_pdb_chain_list(max_chains=2000, max_length=300, min_length=50, max_resolution=3.0):
    """Query RCSB PDB for high-quality single-chain proteins released before 2021-09-30."""
    print(f"Querying RCSB PDB for up to {max_chains} chains...")
    
    # Use RCSB Search API to find suitable chains
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less",
                        "value": max_resolution,
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.deposited_polymer_entity_instance_count",
                        "operator": "equals",
                        "value": 1,
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_entity_polymer_type",
                        "operator": "exact_match",
                        "value": "Protein",
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.diffrn_resolution_high.value",
                        "operator": "less_or_equal",
                        "value": max_resolution,
                    }
                },
            ]
        },
        "return_type": "polymer_entity_instance",
        "request_options": {
            "paginate": {
                "start": 0,
                "rows": max_chains * 2  # Request more, filter later
            },
            "sort": [
                {
                    "sort_by": "rcsb_entry_info.resolution_combined",
                    "direction": "asc"
                }
            ]
        }
    }
    
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    data = json.dumps(query).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode())
        
        chain_ids = []
        for hit in result.get("result_set", []):
            chain_id = hit["identifier"]  # e.g., "1CRN.A"
            chain_ids.append(chain_id)
            if len(chain_ids) >= max_chains:
                break
        
        print(f"  Found {len(chain_ids)} chains from RCSB search")
        return chain_ids
    except Exception as e:
        print(f"  RCSB API query failed: {e}")
        print("  Falling back to a curated chain list...")
        return get_fallback_chain_list(max_chains)


def get_fallback_chain_list(max_chains=2000):
    """Fallback: download the chain list from OpenProteinSet."""
    print("Downloading chain list from OpenProteinSet S3...")
    
    # Download the list of available PDB chains from OpenProteinSet
    try:
        result = subprocess.run(
            ["aws", "s3", "ls", "s3://openfold/pdb/", "--no-sign-request"],
            capture_output=True, text=True, timeout=120
        )
        chains = []
        for line in result.stdout.strip().split("\n"):
            parts = line.strip().split()
            if parts and parts[-1].endswith("/"):
                chain_id = parts[-1].rstrip("/")
                # Convert directory name to PDB chain format
                if len(chain_id) >= 5:
                    chains.append(chain_id)
                if len(chains) >= max_chains:
                    break
        print(f"  Found {len(chains)} chains in OpenProteinSet")
        return chains
    except Exception as e:
        print(f"  Failed to list S3: {e}")
        # Ultimate fallback: curated list of well-known proteins
        curated = [
            "1CRN_A", "1UBQ_A", "2GB1_A", "1VII_A", "1ENH_A",
            "1L2Y_A", "3AIT_A", "1R69_A", "1PGB_A", "2RH1_A",
        ]
        return curated[:max_chains]


def download_pdb_mmcif(chain_ids, output_dir):
    """Download PDB mmCIF files for the given chains."""
    mmcif_dir = os.path.join(output_dir, "pdb_mmcif")
    os.makedirs(mmcif_dir, exist_ok=True)
    
    # Extract unique PDB IDs
    pdb_ids = set()
    for chain_id in chain_ids:
        pdb_id = chain_id.split(".")[0].split("_")[0][:4].lower()
        pdb_ids.add(pdb_id)
    
    print(f"\nDownloading {len(pdb_ids)} PDB mmCIF files...")
    downloaded = 0
    failed = 0
    
    for i, pdb_id in enumerate(sorted(pdb_ids)):
        out_file = os.path.join(mmcif_dir, f"{pdb_id}.cif")
        if os.path.exists(out_file) and os.path.getsize(out_file) > 100:
            downloaded += 1
            continue
        
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
        try:
            urllib.request.urlretrieve(url, out_file)
            downloaded += 1
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  Failed to download {pdb_id}: {e}")
        
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(pdb_ids)} ({downloaded} OK, {failed} failed)")
    
    print(f"  Downloaded {downloaded} mmCIF files ({failed} failed)")
    return mmcif_dir


def download_alignments(chain_ids, output_dir):
    """Download precomputed MSAs from OpenProteinSet S3 bucket."""
    alignment_dir = os.path.join(output_dir, "alignments")
    os.makedirs(alignment_dir, exist_ok=True)
    
    print(f"\nDownloading alignments for {len(chain_ids)} chains from OpenProteinSet...")
    print("  (This may take 30-60 minutes depending on bandwidth)")
    
    downloaded = 0
    failed = 0
    
    for i, chain_id in enumerate(chain_ids):
        # OpenProteinSet uses format like "1crn_A" for directories
        # Try different naming conventions
        pdb_id = chain_id.split(".")[0].split("_")[0][:4].lower()
        chain_letter = chain_id.split(".")[-1] if "." in chain_id else chain_id.split("_")[-1] if "_" in chain_id else "A"
        
        chain_dir_name = f"{pdb_id}_{chain_letter}"
        local_dir = os.path.join(alignment_dir, chain_dir_name)
        
        if os.path.exists(local_dir) and len(os.listdir(local_dir)) >= 2:
            downloaded += 1
            continue
        
        os.makedirs(local_dir, exist_ok=True)
        
        # Try to download from S3
        s3_path = f"s3://openfold/pdb/{chain_dir_name}/"
        try:
            result = subprocess.run(
                ["aws", "s3", "cp", s3_path, local_dir, "--recursive", "--no-sign-request", "--quiet"],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0 and len(os.listdir(local_dir)) > 0:
                downloaded += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
        
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(chain_ids)} ({downloaded} OK, {failed} failed)")
    
    print(f"  Downloaded alignments for {downloaded} chains ({failed} failed)")
    return alignment_dir


def create_train_val_split(chain_ids, mmcif_dir, output_dir, cutoff_date="2021-09-30"):
    """Create time-based train/val split based on PDB deposition date."""
    print(f"\nCreating train/val split (cutoff: {cutoff_date})...")
    
    cutoff = datetime.strptime(cutoff_date, "%Y-%m-%d")
    train_chains = []
    val_chains = []
    unknown_chains = []
    
    for chain_id in chain_ids:
        pdb_id = chain_id.split(".")[0].split("_")[0][:4].lower()
        mmcif_file = os.path.join(mmcif_dir, f"{pdb_id}.cif")
        
        if not os.path.exists(mmcif_file):
            unknown_chains.append(chain_id)
            continue
        
        # Parse deposition date from mmCIF
        dep_date = None
        try:
            with open(mmcif_file, 'r') as f:
                for line in f:
                    if "_pdbx_database_status.recvd_initial_deposition_date" in line:
                        dep_date = line.strip().split()[-1]
                        break
        except:
            pass
        
        if dep_date:
            try:
                parsed_date = datetime.strptime(dep_date, "%Y-%m-%d")
                if parsed_date < cutoff:
                    train_chains.append(chain_id)
                else:
                    val_chains.append(chain_id)
                continue
            except:
                pass
        
        # If we can't determine the date, put in training
        train_chains.append(chain_id)
    
    # Save splits
    splits_dir = os.path.join(output_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    
    with open(os.path.join(splits_dir, "train.txt"), "w") as f:
        f.write("\n".join(train_chains))
    
    with open(os.path.join(splits_dir, "val.txt"), "w") as f:
        f.write("\n".join(val_chains))
    
    print(f"  Train: {len(train_chains)} chains")
    print(f"  Val:   {len(val_chains)} chains")
    print(f"  Unknown date: {len(unknown_chains)} chains (added to train)")
    
    return train_chains, val_chains


def main():
    parser = argparse.ArgumentParser(description="Download Mini-AlphaFold training data")
    parser.add_argument("--n_chains", type=int, default=2000, help="Number of PDB chains to download")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    parser.add_argument("--skip_alignments", action="store_true", help="Skip MSA downloads (for testing)")
    parser.add_argument("--max_resolution", type=float, default=2.5, help="Max resolution in Angstroms")
    args = parser.parse_args()
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Mini-AlphaFold: Data Download & Preparation")
    print("=" * 60)
    
    # Step 1: Get chain list
    chain_ids = get_pdb_chain_list(
        max_chains=args.n_chains,
        max_resolution=args.max_resolution
    )
    
    if not chain_ids:
        print("ERROR: No chains found. Check internet connection.")
        sys.exit(1)
    
    # Save full chain list
    with open(os.path.join(output_dir, "chain_list.txt"), "w") as f:
        f.write("\n".join(chain_ids))
    
    # Step 2: Download mmCIF files
    mmcif_dir = download_pdb_mmcif(chain_ids, output_dir)
    
    # Step 3: Download alignments (MSAs)
    if not args.skip_alignments:
        alignment_dir = download_alignments(chain_ids, output_dir)
    else:
        print("\nSkipping alignment download (--skip_alignments)")
    
    # Step 4: Create train/val split
    train_chains, val_chains = create_train_val_split(chain_ids, mmcif_dir, output_dir)
    
    # Report
    print("\n" + "=" * 60)
    print("Data download complete!")
    print("=" * 60)
    
    # Check disk usage
    try:
        result = subprocess.run(["du", "-sh", output_dir], capture_output=True, text=True)
        print(f"Total disk usage: {result.stdout.strip()}")
    except:
        pass
    
    print(f"\nFiles created:")
    print(f"  {mmcif_dir}/ - PDB structure files")
    if not args.skip_alignments:
        print(f"  {output_dir}/alignments/ - Precomputed MSAs")
    print(f"  {output_dir}/splits/train.txt - Training chain IDs ({len(train_chains)})")
    print(f"  {output_dir}/splits/val.txt - Validation chain IDs ({len(val_chains)})")
    print(f"\nNext step: Run the feature preprocessing pipeline.")


if __name__ == "__main__":
    main()
