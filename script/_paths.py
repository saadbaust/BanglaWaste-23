"""Shared paths. Every script reads the images + release metadata and writes ONLY into work/."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # repository root
RAW = ROOT / "BanglaWaste-23-raw"                   # original images from Zenodo - never modified
META = ROOT / "metadata"                            # release metadata: image_metadata.csv, file_list.csv, ...
WORK = ROOT / "work"                                # all script outputs
RESIZED = WORK / "BanglaWaste23_256"                # resized copy
SPLITS = WORK / "splits"                            # rebuilt splits (compare with splits/)
LOGS = WORK / "logs"                                # reports from every script

for d in (WORK, SPLITS, LOGS):
    d.mkdir(parents=True, exist_ok=True)
