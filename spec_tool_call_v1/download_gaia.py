import os
import shutil
import json

from datasets import load_dataset
from huggingface_hub import snapshot_download

# Download the dataset
data_dir = snapshot_download(repo_id="gaia-benchmark/GAIA", repo_type="dataset")
print(f"Dataset downloaded to: {data_dir}")

# Create local folder to save files
local_folder = "./gaia_dataset"
os.makedirs(local_folder, exist_ok=True)

# Create subfolders for each level
level_folders = {
    "1": os.path.join(local_folder, "level1"),
    "2": os.path.join(local_folder, "level2"),
    "3": os.path.join(local_folder, "level3")
}

for level_folder in level_folders.values():
    os.makedirs(level_folder, exist_ok=True)

# Load the validation dataset
dataset = load_dataset(data_dir, "2023_all", split="validation")

# Track statistics
level_stats = {"1": 0, "2": 0, "3": 0}

# Save each example
for i, example in enumerate(dataset):
    question = example["Question"]
    level = str(example.get("Level", "1"))
    
    # Get the appropriate level folder
    level_folder = level_folders.get(level, level_folders["1"])
    
    # Create a subfolder for each example within the level folder
    example_folder = os.path.join(level_folder, f"example_{level_stats[level]:03d}")
    os.makedirs(example_folder, exist_ok=True)
    
    # Copy the file if it exists
    if example.get("file_name") and example.get("file_name") != "":
        # Try to find the file in the validation folder
        source_file = os.path.join(data_dir, example["file_path"])
        if os.path.exists(source_file):
            dest_file = os.path.join(example_folder, example["file_name"])
            shutil.copy2(source_file, dest_file)
            print(f"Level {level} - Copied: {example['file_name']}")
    
    # Save metadata for this example
    metadata = {
        "task_id": example.get("task_id", ""),
        "question": question,
        "level": level,
        "final_answer": example.get("Final answer", ""),
        "file_name": example.get("file_name", ""),
        "annotator_metadata": example.get("Annotator Metadata", {})
    }
    
    # Save individual metadata JSON
    with open(os.path.join(example_folder, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    level_stats[level] += 1

# Save summary for each level
for level, count in level_stats.items():
    print(f"\nLevel {level}: {count} examples")

print(f"\nAll files saved to: {local_folder}")
print(f"Total examples: {sum(level_stats.values())}")
print(f"\nFolder structure:")
print(f"  {local_folder}/")
print(f"    ├── level1/ ({level_stats['1']} examples)")
print(f"    ├── level2/ ({level_stats['2']} examples)")
print(f"    └── level3/ ({level_stats['3']} examples)")
