import os
import json
from datasets import load_dataset
from huggingface_hub import snapshot_download

# Download the dataset once
data_dir = snapshot_download(
    repo_id="gaia-benchmark/GAIA",
    repo_type="dataset"
)

SPLITS = ["validation"]

for split in SPLITS:
    print(f"Processing Level 1 {split}...")

    ds = load_dataset(data_dir, "2023_all" , split=split)

    out_dir = f"gaia_2023_all_{split}"
    os.makedirs(out_dir, exist_ok=True)

    for ex in ds:
        item = {
            "task_id": ex["task_id"],
            "question": ex["Question"],
            "level": ex["Level"],
            "final_answer": ex["Final answer"],  # <- NOTE SPACE
            "file_name": ex["file_name"],
            "file_path": f"../GAIA/{ex['file_path']}" if ex["file_path"] else "",
            "annotator_metadata": ex["Annotator Metadata"],
        }

        out_path = os.path.join(out_dir, f"{ex['task_id']}.json")
        with open(out_path, "w") as f:
            json.dump(item, f, indent=2)

    print(f"Saved {len(ds)} items → {out_dir}/")
