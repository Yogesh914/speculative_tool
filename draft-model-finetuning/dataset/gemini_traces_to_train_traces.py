import json, glob, os, random

INPUT_DIR = "generated_traces/test"
TRAIN_OUT = "gaia_test.jsonl"
# VAL_OUT = "gaia_val.jsonl"
VAL_RATIO = 0.05


def load_json_with_code_fences(path):
    text = open(path, "r").read().strip()

    # Remove code fences robustly
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]  # remove first ```
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    if not text:
        raise ValueError(f"File {path} is empty after stripping fences")

    data = json.loads(text)

    # unwrap array of length 1
    if isinstance(data, list):
        if len(data) != 1:
            raise ValueError(f"File {path} has array length != 1")
        data = data[0]

    return data


def convert_trace(example):
    q = example["question"]
    trace = example["trace"]
    final = example["final_answer"]

    messages = [{"role": "user", "content": q}]

    for step in trace:
        thought = step["thought"]
        action = step["action"]

        messages.append({"role": "assistant", "content": f"<thought>{thought}</thought>"})
        messages.append({"role": "assistant", "content": f"<action>{json.dumps(action)}</action>"})

    # Add final answer explicitly
    messages.append({"role": "assistant", "content": f"<final_answer>{final}</final_answer>"})

    return {"messages": messages}


def main():
    files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    print("Found", len(files), "files")

    all_items = []
    for f in files:
        try:
            data = load_json_with_code_fences(f)
            item = convert_trace(data)
            all_items.append(item)
        except Exception as e:
            print(f"❌ Skipping file {f}: {e}")
            continue

    random.shuffle(all_items)
    # val_size = int(len(all_items) * VAL_RATIO)
    train_items = all_items
    # val_items = all_items[:val_size]

    with open(TRAIN_OUT, "w") as f:
        for x in train_items:
            f.write(json.dumps(x) + "\n")

    # with open(VAL_OUT, "w") as f:
    #     for x in val_items:
    #         f.write(json.dumps(x) + "\n")

    print(f"✅ Done. Train: {len(train_items)} examples.")


if __name__ == "__main__":
    main()
