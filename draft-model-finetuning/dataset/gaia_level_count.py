from datasets import load_dataset
from huggingface_hub import snapshot_download

data_dir = snapshot_download(repo_id="gaia-benchmark/GAIA", repo_type="dataset")
ds = load_dataset(data_dir, "2023_all", split="validation")  # or 2023_level1/2/3 if separate

level_counts = {}
for ex in ds:
    lvl = ex["Level"]
    level_counts[lvl] = level_counts.get(lvl, 0) + 1

print(level_counts)
