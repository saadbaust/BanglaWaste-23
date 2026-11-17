"""Shared paths. Every script reads from the dataset root and writes ONLY into work/."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # repository root
RAW = ROOT / "BanglaWaste-23-raw"                   # original images - never modified
META = ROOT / "metadata"                            # release metadata: image_metadata.csv, file_list.csv, ...
WORK = ROOT / "work"                                # all script outputs
RESIZED = WORK / "BanglaWaste23_256"                # Step 2.2 output
SPLITS = WORK / "splits"                            # Step 2.3-2.6 output
LOGS = WORK / "logs"                                # reports from every script

for d in (WORK, SPLITS, LOGS):
    d.mkdir(parents=True, exist_ok=True)
