import json, random
from pathlib import Path
from collections import defaultdict
import shutil, os

random.seed(42)

# Input folder containing 165 JSONs
input_folder = Path("gaia_2023_all_validation")

# Output folders
output_base = Path("gaia_2023_all_splits")
train_dir = output_base / "train"
val_dir   = output_base / "val"
test_dir  = output_base / "test"

for d in [train_dir, val_dir, test_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Group files by level
by_level = defaultdict(list)
for f in input_folder.glob("*.json"):
    data = json.load(open(f))
    by_level[data["level"]].append(f)

train_files, val_files, test_files = [], [], []

# Split each level proportionally
for lvl, lvl_files in by_level.items():
    random.shuffle(lvl_files)
    n = len(lvl_files)
    n_train = int(n * 0.75)
    n_val   = int(n * 0.125)
    n_test  = n - n_train - n_val

    train_files += lvl_files[:n_train]
    val_files   += lvl_files[n_train:n_train+n_val]
    test_files  += lvl_files[n_train+n_val:]

# Copy files into folders
for f in train_files:
    shutil.copy(f, train_dir / f.name)
for f in val_files:
    shutil.copy(f, val_dir / f.name)
for f in test_files:
    shutil.copy(f, test_dir / f.name)

print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
