"""
Mini-AlphaFold: Interactive AFDB Dataset Downloader & Streaming Processor
========================================================================
Interactive terminal tool to download AlphaFold reference proteome archives,
stream & extract high-confidence folded structures in-memory, generate PyTorch
tensors (.pt), and immediately delete the archives to prevent disk bloat.

Usage:
    # 1. Interactive mode (shows beautiful menu of all 34 species):
    python download_afdb.py

    # 2. Download a specific species (supports friendly aliases):
    python download_afdb.py --species ecoli
    python download_afdb.py --species jannaschii
    python download_afdb.py --species saureus
    python download_afdb.py --species yeast
    python download_afdb.py --species tb

    # 3. Stream all compact microbial proteomes sequentially:
    python download_afdb.py --tier compact

    # 4. Stream all 34 proteomes one-by-one:
    python download_afdb.py --all
"""

import argparse
import concurrent.futures
import gzip
import json
import os
import shutil
import sys
import tarfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_parser import AA_TO_IDX, THREE_TO_ONE, UNK_IDX

console = Console()

# Comprehensive Reference Proteome Catalog
CATALOG = {
    # Tier 1: Fast Compact Proteomes (< 250 MB)
    "helicobacter_pylori": {
        "name": "Helicobacter pylori",
        "common": "H. pylori",
        "aliases": ["hpylori", "h_pylori", "helpy"],
        "tar": "UP000000429_85962_HELPY_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000429_85962_HELPY_v6.tar",
        "size_mb": 166,
        "count": 1540,
        "tier": "compact",
        "desc": "Gastric enzyme & acid-resistant catalytic folds",
    },
    "methanocaldococcus_jannaschii": {
        "name": "Methanocaldococcus jannaschii",
        "common": "M. jannaschii",
        "aliases": ["jannaschii", "metja", "archaea"],
        "tar": "UP000000805_243232_METJA_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000805_243232_METJA_v6.tar",
        "size_mb": 174,
        "count": 1773,
        "tier": "compact",
        "desc": "Archaea extremophile hyper-stable thermal folds",
    },
    "campylobacter_jejuni": {
        "name": "Campylobacter jejuni",
        "common": "C. jejuni",
        "aliases": ["cjejuni", "camje"],
        "tar": "UP000000799_192222_CAMJE_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000799_192222_CAMJE_v6.tar",
        "size_mb": 175,
        "count": 1620,
        "tier": "compact",
        "desc": "Motility, membrane & outer surface architectures",
    },
    "haemophilus_influenzae": {
        "name": "Haemophilus influenzae",
        "common": "H. influenzae",
        "aliases": ["hinfluenzae", "haein"],
        "tar": "UP000000579_71421_HAEIN_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000579_71421_HAEIN_v6.tar",
        "size_mb": 175,
        "count": 1660,
        "tier": "compact",
        "desc": "First fully sequenced free-living bacterial proteome",
    },
    "mycobacterium_leprae": {
        "name": "Mycobacterium leprae",
        "common": "M. leprae",
        "aliases": ["leprae", "mycle"],
        "tar": "UP000000806_272631_MYCLE_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000806_272631_MYCLE_v6.tar",
        "size_mb": 177,
        "count": 1602,
        "tier": "compact",
        "desc": "Compact reductive actinobacterial genome folds",
    },
    "neisseria_gonorrhoeae": {
        "name": "Neisseria gonorrhoeae",
        "common": "N. gonorrhoeae",
        "aliases": ["gonorrhoeae", "neig1"],
        "tar": "UP000000535_242231_NEIG1_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000535_242231_NEIG1_v6.tar",
        "size_mb": 195,
        "count": 2106,
        "tier": "compact",
        "desc": "Pilus, adhesin & outer membrane beta-barrels",
    },
    "streptococcus_pneumoniae": {
        "name": "Streptococcus pneumoniae",
        "common": "S. pneumoniae",
        "aliases": ["pneumoniae", "strr6"],
        "tar": "UP000000586_171101_STRR6_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000586_171101_STRR6_v6.tar",
        "size_mb": 202,
        "count": 2031,
        "tier": "compact",
        "desc": "Capsule biosynthesis & cell-wall anchoring folds",
    },
    "staphylococcus_aureus": {
        "name": "Staphylococcus aureus",
        "common": "S. aureus",
        "aliases": ["saureus", "staa8"],
        "tar": "UP000008816_93061_STAA8_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000008816_93061_STAA8_v6.tar",
        "size_mb": 274,
        "count": 2888,
        "tier": "compact",
        "desc": "Gram-positive globular virulence factors & toxins",
    },
    # Tier 2: Core Model Organisms & Major Pathogens (300 - 800 MB)
    "shigella_dysenteriae": {
        "name": "Shigella dysenteriae",
        "common": "S. dysenteriae",
        "aliases": ["shigella", "shids"],
        "tar": "UP000002716_300267_SHIDS_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000002716_300267_SHIDS_v6.tar",
        "size_mb": 373,
        "count": 3893,
        "tier": "medium",
        "desc": "Enteric toxin & secretion chaperone assemblies",
    },
    "mycobacterium_tuberculosis": {
        "name": "Mycobacterium tuberculosis",
        "common": "M. tuberculosis",
        "aliases": ["tb", "tuberculosis", "myctu"],
        "tar": "UP000001584_83332_MYCTU_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000001584_83332_MYCTU_v6.tar",
        "size_mb": 429,
        "count": 3991,
        "tier": "medium",
        "desc": "High GC mycolic-acid lipid synthase domain diversity",
    },
    "escherichia_coli": {
        "name": "Escherichia coli",
        "common": "E. coli",
        "aliases": ["ecoli", "e_coli"],
        "tar": "UP000000625_83333_ECOLI_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000625_83333_ECOLI_v6.tar",
        "size_mb": 456,
        "count": 4370,
        "tier": "medium",
        "desc": "The primary gold-standard benchmark model organism",
    },
    "salmonella_typhimurium": {
        "name": "Salmonella typhimurium",
        "common": "S. typhimurium",
        "aliases": ["salmonella", "salty"],
        "tar": "UP000001014_99287_SALTY_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000001014_99287_SALTY_v6.tar",
        "size_mb": 477,
        "count": 4526,
        "tier": "medium",
        "desc": "Type III needle secretion system & metabolic enzymes",
    },
    "klebsiella_pneumoniae": {
        "name": "Klebsiella pneumoniae",
        "common": "K. pneumoniae",
        "aliases": ["klebsiella", "kleph"],
        "tar": "UP000007841_1125630_KLEPH_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000007841_1125630_KLEPH_v6.tar",
        "size_mb": 559,
        "count": 5727,
        "tier": "medium",
        "desc": "Polysaccharide capsule & beta-lactamase folds",
    },
    "pseudomonas_aeruginosa": {
        "name": "Pseudomonas aeruginosa",
        "common": "P. aeruginosa",
        "aliases": ["pseudomonas", "pseae"],
        "tar": "UP000002438_208964_PSEAE_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000002438_208964_PSEAE_v6.tar",
        "size_mb": 613,
        "count": 5555,
        "tier": "medium",
        "desc": "Multidrug efflux pumps & versatile environmental enzymes",
    },
    "nocardia_brasiliensis": {
        "name": "Nocardia brasiliensis",
        "common": "N. brasiliensis",
        "aliases": ["nocardia", "noca1"],
        "tar": "UP000006304_1133849_9NOCA1_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000006304_1133849_9NOCA1_v6.tar",
        "size_mb": 873,
        "count": 8398,
        "tier": "medium",
        "desc": "Filamentous actinomycete secondary metabolite clusters",
    },
    # Tier 3: Eukaryotes & Fungi (800 MB - 1.5 GB)
    "schizosaccharomyces_pombe": {
        "name": "Schizosaccharomyces pombe",
        "common": "Fission yeast",
        "aliases": ["spombe", "fission_yeast", "schpo"],
        "tar": "UP000002485_284812_SCHPO_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000002485_284812_SCHPO_v6.tar",
        "size_mb": 803,
        "count": 5196,
        "tier": "eukaryote",
        "desc": "Model eukaryotic cell cycle and chromosomal regulators",
    },
    "saccharomyces_cerevisiae": {
        "name": "Saccharomyces cerevisiae",
        "common": "Budding yeast",
        "aliases": ["yeast", "scerevisiae", "budding_yeast"],
        "tar": "UP000002311_559292_YEAST_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000002311_559292_YEAST_v6.tar",
        "size_mb": 977,
        "count": 6055,
        "tier": "eukaryote",
        "desc": "Foundational model organism for eukaryotic cell biology",
    },
    "candida_albicans": {
        "name": "Candida albicans",
        "common": "C. albicans",
        "aliases": ["candida", "canal"],
        "tar": "UP000000559_237561_CANAL_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000559_237561_CANAL_v6.tar",
        "size_mb": 981,
        "count": 5973,
        "tier": "eukaryote",
        "desc": "Dimorphic fungal pathogen morphological transition folds",
    },
    "plasmodium_falciparum": {
        "name": "Plasmodium falciparum",
        "common": "P. falciparum",
        "aliases": ["malaria", "plasmodium", "plaf7"],
        "tar": "UP000001450_36329_PLAF7_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000001450_36329_PLAF7_v6.tar",
        "size_mb": 1148,
        "count": 5168,
        "tier": "eukaryote",
        "desc": "AT-rich genome, erythrocyte invasion surface antigens",
    },
    "trypanosoma_brucei": {
        "name": "Trypanosoma brucei",
        "common": "Trypanosoma brucei",
        "aliases": ["trypanosoma", "sleeping_sickness", "tryb2"],
        "tar": "UP000008524_185431_TRYB2_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000008524_185431_TRYB2_v6.tar",
        "size_mb": 1345,
        "count": 8491,
        "tier": "eukaryote",
        "desc": "Variant surface glycoprotein (VSG) immune-evasion coats",
    },
    "caenorhabditis_elegans": {
        "name": "Caenorhabditis elegans",
        "common": "Nematode worm",
        "aliases": ["celegans", "caeel", "worm"],
        "tar": "UP000001940_6239_CAEEL_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000001940_6239_CAEEL_v6.tar",
        "size_mb": 2649,
        "count": 19700,
        "tier": "eukaryote",
        "desc": "Model nematode developmental and neurological proteome",
    },
    "dictyostelium_discoideum": {
        "name": "Dictyostelium discoideum",
        "common": "Dictyostelium",
        "aliases": ["dictyostelium", "dicdi", "slime_mold"],
        "tar": "UP000002195_44689_DICDI_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000002195_44689_DICDI_v6.tar",
        "size_mb": 2187,
        "count": 12612,
        "tier": "eukaryote",
        "desc": "Cellular slime mold chemotaxis and signaling folds",
    },
    "drosophila_melanogaster": {
        "name": "Drosophila melanogaster",
        "common": "Fruit fly",
        "aliases": ["drosophila", "fruit_fly", "drome"],
        "tar": "UP000000803_7227_DROME_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000803_7227_DROME_v6.tar",
        "size_mb": 2213,
        "count": 13461,
        "tier": "eukaryote",
        "desc": "Premier genetic model organism for animal development",
    },
    # Tier 4: Global Health Pathogens & Parasites
    "ajellomyces_capsulatus": {
        "name": "Ajellomyces capsulatus",
        "common": "Ajellomyces capsulatus",
        "aliases": ["ajellomyces", "ajecg", "histoplasma"],
        "tar": "UP000001631_447093_AJECG_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000001631_447093_AJECG_v6.tar",
        "size_mb": 1363,
        "count": 9199,
        "tier": "global_health",
        "desc": "Dimorphic pathogenic fungus causing histoplasmosis",
    },
    "dracunculus_medinensis": {
        "name": "Dracunculus medinensis",
        "common": "Dracunculus medinensis",
        "aliases": ["dracunculus", "guinea_worm", "drame"],
        "tar": "UP000274756_318479_DRAME_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000274756_318479_DRAME_v6.tar",
        "size_mb": 1364,
        "count": 10834,
        "tier": "global_health",
        "desc": "Guinea worm parasitic nematode causing dracunculiasis",
    },
    "trichuris_trichiura": {
        "name": "Trichuris trichiura",
        "common": "Trichuris trichiura",
        "aliases": ["trichuris", "tritr", "whipworm"],
        "tar": "UP000030665_36087_TRITR_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000030665_36087_TRITR_v6.tar",
        "size_mb": 1362,
        "count": 9563,
        "tier": "global_health",
        "desc": "Human whipworm parasite of the large intestine",
    },
    "wuchereria_bancrofti": {
        "name": "Wuchereria bancrofti",
        "common": "Wuchereria bancrofti",
        "aliases": ["wuchereria", "wucba", "elephantiasis"],
        "tar": "UP000270924_6293_WUCBA_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000270924_6293_WUCBA_v6.tar",
        "size_mb": 1418,
        "count": 12725,
        "tier": "global_health",
        "desc": "Major parasitic filarial worm causing lymphatic elephantiasis",
    },
    "leishmania_infantum": {
        "name": "Leishmania infantum",
        "common": "L. infantum",
        "aliases": ["leishmania", "leiin", "leishmaniasis"],
        "tar": "UP000008153_5671_LEIIN_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000008153_5671_LEIIN_v6.tar",
        "size_mb": 1508,
        "count": 7924,
        "tier": "global_health",
        "desc": "Kinetoplastid protozoan causing visceral leishmaniasis",
    },
    "sporothrix_schenckii": {
        "name": "Sporothrix schenckii",
        "common": "Sporothrix schenckii",
        "aliases": ["sporothrix", "spos1", "rose_gardener"],
        "tar": "UP000018087_1391915_SPOS1_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000018087_1391915_SPOS1_v6.tar",
        "size_mb": 1519,
        "count": 8652,
        "tier": "global_health",
        "desc": "Dimorphic fungus causing sporotrichosis",
    },
    "madurella_mycetomatis": {
        "name": "Madurella mycetomatis",
        "common": "Madurella mycetomatis",
        "aliases": ["madurella", "mycetoma"],
        "tar": "UP000078237_100816_9PEZI1_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000078237_100816_9PEZI1_v6.tar",
        "size_mb": 1537,
        "count": 9561,
        "tier": "global_health",
        "desc": "Fungal agent of eumycetoma grain infection",
    },
    "onchocerca_volvulus": {
        "name": "Onchocerca volvulus",
        "common": "Onchocerca volvulus",
        "aliases": ["onchocerca", "oncvo", "river_blindness"],
        "tar": "UP000024404_6282_ONCVO_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000024404_6282_ONCVO_v6.tar",
        "size_mb": 1621,
        "count": 12039,
        "tier": "global_health",
        "desc": "Nematode responsible for onchocerciasis (river blindness)",
    },
    "brugia_malayi": {
        "name": "Brugia malayi",
        "common": "Brugia malayi",
        "aliases": ["brugia", "bruma", "filariasis"],
        "tar": "UP000006672_6279_BRUMA_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000006672_6279_BRUMA_v6.tar",
        "size_mb": 1635,
        "count": 10972,
        "tier": "global_health",
        "desc": "Filarial nematode causing lymphatic filariasis",
    },
    "cladophialophora_carrionii": {
        "name": "Cladophialophora carrionii",
        "common": "Cladophialophora carrionii",
        "aliases": ["cladophialophora", "chromoblastomycosis"],
        "tar": "UP000094526_86049_9EURO1_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000094526_86049_9EURO1_v6.tar",
        "size_mb": 1729,
        "count": 11170,
        "tier": "global_health",
        "desc": "Dematiaceous fungus causing chromoblastomycosis",
    },
    "schistosoma_mansoni": {
        "name": "Schistosoma mansoni",
        "common": "Schistosoma mansoni",
        "aliases": ["schistosoma", "schma", "schistosomiasis"],
        "tar": "UP000008854_6183_SCHMA_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000008854_6183_SCHMA_v6.tar",
        "size_mb": 1802,
        "count": 9735,
        "tier": "global_health",
        "desc": "Trematode blood fluke causing schistosomiasis",
    },
    "fonsecaea_pedrosoi": {
        "name": "Fonsecaea pedrosoi",
        "common": "Fonsecaea pedrosoi",
        "aliases": ["fonsecaea", "pedrosoi"],
        "tar": "UP000053029_1442368_9EURO2_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000053029_1442368_9EURO2_v6.tar",
        "size_mb": 2014,
        "count": 12509,
        "tier": "global_health",
        "desc": "Melanized fungal pathogen causing chromoblastomycosis",
    },
    "paracoccidioides_lutzii": {
        "name": "Paracoccidioides lutzii",
        "common": "Paracoccidioides lutzii",
        "aliases": ["paracoccidioides", "parba", "lutzii"],
        "tar": "UP000002059_502779_PARBA_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000002059_502779_PARBA_v6.tar",
        "size_mb": 1294,
        "count": 8794,
        "tier": "global_health",
        "desc": "Thermal dimorphic fungus causing paracoccidioidomycosis",
    },
    "strongyloides_stercoralis": {
        "name": "Strongyloides stercoralis",
        "common": "Strongyloides stercoralis",
        "aliases": ["strongyloides", "strer", "strongyloidiasis"],
        "tar": "UP000035681_6248_STRER_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000035681_6248_STRER_v6.tar",
        "size_mb": 2793,
        "count": 15335,
        "tier": "global_health",
        "desc": "Soil-transmitted helminth causing strongyloidiasis",
    },
    "trypanosoma_cruzi": {
        "name": "Trypanosoma cruzi",
        "common": "T. cruzi",
        "aliases": ["tcruzi", "trycc", "chagas"],
        "tar": "UP000002296_353153_TRYCC_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000002296_353153_TRYCC_v6.tar",
        "size_mb": 2959,
        "count": 19036,
        "tier": "global_health",
        "desc": "Zoonotic parasitic protozoan causing Chagas disease",
    },
    # Tier 5: Plant & Crop Agricultural Proteomes
    "arabidopsis_thaliana": {
        "name": "Arabidopsis thaliana",
        "common": "Arabidopsis",
        "aliases": ["arabidopsis", "arath", "thale_cress"],
        "tar": "UP000006548_3702_ARATH_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000006548_3702_ARATH_v6.tar",
        "size_mb": 3698,
        "count": 27402,
        "tier": "plant",
        "desc": "Thale cress foundational model plant proteome",
    },
    "oryza_sativa": {
        "name": "Oryza sativa",
        "common": "Asian rice",
        "aliases": ["rice", "oryza", "orysj"],
        "tar": "UP000059680_39947_ORYSJ_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000059680_39947_ORYSJ_v6.tar",
        "size_mb": 4505,
        "count": 43645,
        "tier": "plant",
        "desc": "Global cereal staple monocot crop proteome",
    },
    "zea_mays": {
        "name": "Zea mays",
        "common": "Maize",
        "aliases": ["maize", "corn"],
        "tar": "UP000007305_4577_MAIZE_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000007305_4577_MAIZE_v6.tar",
        "size_mb": 4792,
        "count": 39139,
        "tier": "plant",
        "desc": "Important agricultural cereal crop proteome",
    },
    "glycine_max": {
        "name": "Glycine max",
        "common": "Soybean",
        "aliases": ["soybean", "soybn"],
        "tar": "UP000008827_3847_SOYBN_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000008827_3847_SOYBN_v6.tar",
        "size_mb": 7264,
        "count": 55796,
        "tier": "plant",
        "desc": "Major legume crop agricultural protein architectures",
    },
    # Tier 6: Vertebrate & Mammalian Model Proteomes
    "mus_musculus": {
        "name": "Mus musculus",
        "common": "Mouse",
        "aliases": ["mouse", "musculus"],
        "tar": "UP000000589_10090_MOUSE_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000589_10090_MOUSE_v6.tar",
        "size_mb": 3607,
        "count": 21452,
        "tier": "vertebrate",
        "desc": "Primary mammalian model organism proteome",
    },
    "rattus_norvegicus": {
        "name": "Rattus norvegicus",
        "common": "Rat",
        "aliases": ["rat", "norvegicus"],
        "tar": "UP000002494_10116_RAT_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000002494_10116_RAT_v6.tar",
        "size_mb": 3602,
        "count": 22152,
        "tier": "vertebrate",
        "desc": "Physiological and pharmacological rodent model proteome",
    },
    "danio_rerio": {
        "name": "Danio rerio",
        "common": "Zebrafish",
        "aliases": ["zebrafish", "danre"],
        "tar": "UP000000437_7955_DANRE_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000000437_7955_DANRE_v6.tar",
        "size_mb": 4749,
        "count": 26290,
        "tier": "vertebrate",
        "desc": "Vertebrate developmental model organism proteome",
    },
    "homo_sapiens": {
        "name": "Homo sapiens",
        "common": "Human",
        "aliases": ["human", "homo_sapiens", "human_proteome"],
        "tar": "UP000005640_9606_HUMAN_v6.tar",
        "url": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000005640_9606_HUMAN_v6.tar",
        "size_mb": 4938,
        "count": 23586,
        "tier": "vertebrate",
        "desc": "Canonical human reference proteome",
    },
}


def resolve_species(query: str) -> Optional[str]:
    """Find canonical species key from user query or alias."""
    q = query.lower().strip()
    if q in CATALOG:
        return q
    for key, info in CATALOG.items():
        if q == info["common"].lower() or q in info.get("aliases", []):
            return key
        if q in info["name"].lower():
            return key
    return None


def get_cache_directory() -> Path:
    """Auto-detect Drive D: for staging, falling back to data/tmp."""
    d_cache = Path("D:/afdb_cache")
    if os.path.exists("D:/"):
        d_cache.mkdir(parents=True, exist_ok=True)
        return d_cache
    tmp = PROJECT_ROOT / "data" / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    return tmp


def multi_thread_download(url: str, dest: Path, n_threads: int = 4) -> bool:
    """Download archive using parallel HTTP range requests with Rich live progress."""
    head_req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(head_req, timeout=15) as head_resp:
            total_sz = int(head_resp.headers.get("Content-Length", 0))
            accept_ranges = "bytes" in head_resp.headers.get("Accept-Ranges", "")
    except Exception as e:
        console.print(f"[bold red]Failed to reach URL:[/bold red] {e}")
        return False

    with open(dest, "wb") as f_init:
        f_init.truncate(total_sz)

    chunk_sz = (total_sz + n_threads - 1) // n_threads
    ranges = []
    for i in range(n_threads):
        start = i * chunk_sz
        end = min(total_sz - 1, (i + 1) * chunk_sz - 1)
        if start <= end:
            ranges.append((start, end, i))

    downloaded = [0] * len(ranges)
    lock = threading.Lock()

    def fetch_chunk(start, end, idx):
        headers = {"User-Agent": "Mozilla/5.0", "Range": f"bytes={start}-{end}"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "r+b") as f:
            f.seek(start)
            while True:
                buf = resp.read(512 * 1024)
                if not buf:
                    break
                f.write(buf)
                with lock:
                    downloaded[idx] += len(buf)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}[/bold cyan]"),
        BarColumn(bar_width=40),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task(f"Downloading {dest.name}", total=total_sz)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(fetch_chunk, s, e, i) for s, e, i in ranges]
            while not all(f.done() for f in futures):
                time.sleep(0.3)
                progress.update(task_id, completed=sum(downloaded))
            for f in futures:
                f.result()
            progress.update(task_id, completed=total_sz)

    return True


def parse_and_stream_tar(tar_path: Path, output_dir: Path, species_key: str) -> int:
    """Stream .pdb.gz files directly from tar, filter pLDDT >= 70, save compact .pt tensors."""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    t0 = time.time()

    with tarfile.open(tar_path, "r") as tar:
        members = [m for m in tar if m.name.endswith(".pdb.gz")]
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]Parsing In-Memory:[/bold green] [magenta]{task.fields[species]}[/magenta]"),
            BarColumn(bar_width=40),
            TextColumn("{task.completed}/{task.total} structures"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Parsing", total=len(members), species=CATALOG[species_key]["common"])
            for m in members:
                progress.advance(task_id)
                try:
                    f_obj = tar.extractfile(m)
                    if not f_obj:
                        continue
                    with gzip.GzipFile(fileobj=f_obj, mode="rb") as gz:
                        text = gz.read().decode("utf-8", errors="replace")

                    # Fast parse
                    residues = {}
                    for line in text.splitlines():
                        if line.startswith("ATOM"):
                            aname = line[12:16].strip()
                            if aname in ("N", "CA", "C"):
                                rnum = int(line[22:26])
                                if rnum not in residues:
                                    residues[rnum] = {
                                        "res_name": line[17:20].strip(),
                                        "plddt": float(line[60:66]),
                                        "coords": {},
                                    }
                                residues[rnum]["coords"][aname] = [
                                    float(line[30:38]),
                                    float(line[38:46]),
                                    float(line[46:54]),
                                ]

                    valid = [r for k in sorted(residues.keys()) if (r := residues[k]) and len(r["coords"]) == 3]
                    L = len(valid)
                    if L < 30:
                        continue

                    # Slice window
                    if L <= 128:
                        if np.mean([r["plddt"] for r in valid]) < 70.0:
                            continue
                        target = valid
                    else:
                        plddts = np.array([r["plddt"] for r in valid])
                        cumsum = np.cumsum(np.insert(plddts, 0, 0))
                        means = (cumsum[128:] - cumsum[:-128]) / 128.0
                        b_start = int(np.argmax(means))
                        if float(means[b_start]) < 70.0:
                            continue
                        target = valid[b_start : b_start + 128]

                    # Convert to compact tensors
                    L_sub = len(target)
                    seq_str = "".join([THREE_TO_ONE.get(r["res_name"], "X") for r in target])
                    seq_idx = torch.tensor([AA_TO_IDX.get(c, UNK_IDX) for c in seq_str], dtype=torch.int64)

                    bb = np.zeros((L_sub, 3, 3), dtype=np.float32)
                    ca = np.zeros((L_sub, 3), dtype=np.float32)
                    plddt_arr = np.zeros((L_sub,), dtype=np.float32)
                    for i, r in enumerate(target):
                        bb[i, 0] = r["coords"]["N"]
                        bb[i, 1] = r["coords"]["CA"]
                        bb[i, 2] = r["coords"]["C"]
                        ca[i] = r["coords"]["CA"]
                        plddt_arr[i] = r["plddt"]

                    diff = ca[:, None, :] - ca[None, :, :]
                    dmat = np.sqrt(np.sum(diff ** 2, axis=-1)).astype(np.float32)

                    uid = Path(m.name).stem.replace("-model_v4", "").replace("-model_v6", "")
                    did = f"AF_{species_key[:6]}_{uid}"

                    out_dict = {
                        "domain_id": did,
                        "sequence": seq_str,
                        "sequence_indices": seq_idx,
                        "backbone_coords": torch.from_numpy(bb),
                        "ca_coords": torch.from_numpy(ca),
                        "distance_matrix": torch.from_numpy(dmat),
                        "plddt": torch.from_numpy(plddt_arr),
                        "length": L_sub,
                    }
                    torch.save(out_dict, output_dir / f"{did}.pt")
                    count += 1
                except Exception:
                    continue

    return count


def process_single_species(species_key: str, output_dir: Path, cache_dir: Path) -> int:
    """Download, extract in-memory, delete archive, and update splits."""
    info = CATALOG[species_key]
    dest = cache_dir / info["tar"]

    console.print(Panel(f"[bold green]Starting Proteome:[/bold green] {info['name']} ({info['size_mb']} MB, {info['count']} structures)\n[cyan]Staging:[/cyan] {dest}", title="[bold yellow]AFDB Stream[/bold yellow]"))

    ok = multi_thread_download(info["url"], dest)
    if not ok:
        return 0

    count = parse_and_stream_tar(dest, output_dir, species_key)

    # Instant Cleanup
    try:
        dest.unlink()
        console.print(f"[bold green]✓ Deleted archive {dest.name} -> Zero storage waste![/bold green]")
    except Exception as e:
        console.print(f"[yellow]Cleanup notice: {e}[/yellow]")

    # Update splits manifest
    all_pts = sorted([p.stem for p in output_dir.glob("*.pt")])
    total = len(all_pts)
    n_train = int(total * 0.85)
    n_val = int(total * 0.10)
    splits = {
        "train": all_pts[:n_train],
        "val": all_pts[n_train : n_train + n_val],
        "test": all_pts[n_train + n_val :],
        "total": total,
    }
    try:
        with open(output_dir / "splits.json", "w") as f:
            json.dump(splits, f)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not save splits.json: {e}[/yellow]")

    console.print(f"[bold cyan]✓ Processed {count} high-confidence structures.[/bold cyan] Total dataset: [bold white]{total}[/bold white] proteins.\n")
    return count


def show_menu():
    """Print an interactive catalog table."""
    table = Table(title="[bold yellow]AlphaFold Reference Proteome Catalog[/bold yellow]", border_style="cyan")
    table.add_column("#", justify="right", style="bold yellow")
    table.add_column("Species", style="bold white")
    table.add_column("Common Name", style="cyan")
    table.add_column("Size", justify="right", style="green")
    table.add_column("Structures", justify="right", style="magenta")
    table.add_column("Tier", style="blue")
    table.add_column("Quick Alias", style="dim")

    for i, (k, info) in enumerate(CATALOG.items(), start=1):
        alias_str = ", ".join(info.get("aliases", [])[:2])
        table.add_row(
            str(i),
            info["name"],
            info["common"],
            f"{info['size_mb']} MB",
            f"{info['count']:,}",
            info["tier"],
            alias_str,
        )

    console.print(table)
    console.print("\n[bold cyan]Run Options:[/bold cyan]")
    console.print("  [white]1.[/white] Specific Species:  [bold green]python download_afdb.py --species ecoli[/bold green]")
    console.print("  [white]2.[/white] Compact Tier:       [bold green]python download_afdb.py --tier compact[/bold green]")
    console.print("  [white]3.[/white] Entire Catalog:     [bold green]python download_afdb.py --all[/bold green]\n")


def main():
    parser = argparse.ArgumentParser(description="AlphaFold Reference Proteome Downloader & Processor")
    parser.add_argument("--species", "-s", type=str, default=None, help="Species key or alias (e.g. ecoli, jannaschii, saureus)")
    parser.add_argument("--tier", "-t", type=str, default=None, choices=["compact", "medium", "eukaryote", "global_health", "plant", "vertebrate"], help="Process by tier")
    parser.add_argument("--all", "-a", action="store_true", help="Download and stream all catalog species sequentially")
    parser.add_argument("--output", "-o", type=str, default="data/afdb_processed", help="Output directory for .pt tensors")
    parser.add_argument("--cache", "-c", type=str, default=None, help="Cache directory (defaults to D:\\afdb_cache if available)")
    parser.add_argument("--list", "-l", action="store_true", help="Display species catalog table")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.output
    cache_dir = Path(args.cache) if args.cache else get_cache_directory()

    if args.list or (len(sys.argv) == 1 and not sys.stdin.isatty()):
        show_menu()
        return

    if len(sys.argv) == 1:
        show_menu()
        try:
            choice = console.input("[bold yellow]Enter species name, number, or tier (default: jannaschii): [/bold yellow]").strip()
            if not choice:
                choice = "jannaschii"
            if choice.isdigit():
                idx = int(choice) - 1
                keys = list(CATALOG.keys())
                if 0 <= idx < len(keys):
                    resolved = keys[idx]
                else:
                    console.print("[red]Invalid index.[/red]")
                    return
            elif choice in ("compact", "medium", "eukaryote", "global_health", "plant", "vertebrate"):
                args.tier = choice
                resolved = None
            elif choice == "all":
                args.all = True
                resolved = None
            else:
                resolved = resolve_species(choice)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Cancelled.[/yellow]")
            return
    else:
        resolved = resolve_species(args.species) if args.species else None

    if args.all:
        keys_to_run = list(CATALOG.keys())
    elif args.tier:
        keys_to_run = [k for k, v in CATALOG.items() if v.get("tier") == args.tier]
    elif resolved:
        keys_to_run = [resolved]
    else:
        console.print(f"[bold red]Unknown species or alias:[/bold red] {args.species}")
        console.print("Run [bold cyan]python download_afdb.py --list[/bold cyan] to see available options.")
        return

    console.print(Panel(f"[bold green]Ingesting {len(keys_to_run)} Species to {out_dir}[/bold green]\n[cyan]Staging Cache:[/cyan] {cache_dir}", title="[bold yellow]AFDB Distillation Engine[/bold yellow]"))

    total_added = 0
    for key in keys_to_run:
        added = process_single_species(key, out_dir, cache_dir)
        total_added += added

    console.print(Panel(f"[bold green]✓ All jobs completed successfully![/bold green]\nTotal structures added: [bold white]{total_added}[/bold white]\nPyTorch dataset ready in: [bold cyan]{out_dir}[/bold cyan]", title="[bold green]Success[/bold green]"))


if __name__ == "__main__":
    main()
