import os
import json
import glob
from tqdm import tqdm
import google.generativeai as genai


# ===========================
#  SYSTEM PROMPT (ONE VARIANT)
# ===========================
GEMINI_SYSTEM_PROMPT = r"""
You are an expert agent that generates training traces for tool-calling models.

TASK:
For each GAIA example, produce EXACTLY ONE reasoning trace showing sequential
thought → action → next thought → action → ... → final_answer.

OUTPUT RULES:
1. Return a VALID JSON array of objects.
2. For each GAIA example, produce:
   {
     "id": "<id>",
     "question": "<original question>",
     "trace": [
       {
         "thought": "<step-by-step reasoning>",
         "action": {
            "tool_name": "<tool name>",
            "arguments": { ... }
         }
       },
       ...
     ],
     "final_answer": "<string>"
   }
3. No markdown. No comments. No prose outside JSON.
4. If input is malformed, output:
   { "id": "<id>", "error": true, "reason": "<text>" }
5. I will send:
   { "batch": [ {...}, {...} ] }
   You must return one object per batch item.
"""


# ===========================
#  LOAD GAIA TRAIN DATA
# ===========================
def load_gaia_train(path="gaia_2023_all_splits/train/*.json"):
    files = glob.glob(path)
    data = []
    for f in files:
        with open(f, "r") as fp:
            data.append(json.load(fp))
    return data


# ===========================
#  BATCH GENERATOR
# ===========================
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


# ===========================
#  GEMINI MODEL INIT
# ===========================
def init_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        "gemini-2.5-pro",
        system_instruction=GEMINI_SYSTEM_PROMPT
    )



# ===========================
#  TRACE GENERATION
# ===========================
def extract_text(response):
    # No candidates → blocked
    if not response.candidates:
        return ""

    cand = response.candidates[0]

    # finish_reason == 2 → safety-blocked
    if getattr(cand, "finish_reason", None) == 2:
        return ""

    if not cand.content or not cand.content.parts:
        return ""

    texts = []
    for part in cand.content.parts:
        if hasattr(part, "text"):
            texts.append(part.text)
    return "".join(texts)


def generate_traces(model, batch):
    prompt = json.dumps({"batch": batch})

    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.2,
            "top_p": 0.9,
            "max_output_tokens": 4096
        }
    )

    raw = extract_text(response).strip()

    if not raw:
        print("WARNING: Empty response (safety blocked).")
        return {"error": True, "batch": batch, "reason": "safety_block"}

    try:
        return json.loads(raw)
    except Exception:
        print("JSON parse error — saving raw output.")
        return raw


# ===========================
#  MAIN DRIVER
# ===========================
def generate_all_traces(api_key,
                        input_path="gaia_2023_all_splits/train/*.json",
                        output_path="generated_traces",
                        batch_size=1):

    os.makedirs(output_path, exist_ok=True)

    gaia_data = load_gaia_train(input_path)
    print(f"Loaded {len(gaia_data)} GAIA examples.")

    model = init_gemini(api_key)

    for i, batch in enumerate(tqdm(chunks(gaia_data, batch_size))):
        traces = generate_traces(model, batch)

        out_file = os.path.join(output_path, f"test_batch_{i:04d}.json")
        with open(out_file, "w") as f:
            if isinstance(traces, str):
                f.write(traces)
            else:
                json.dump(traces, f, indent=2)

        print(f"Saved {out_file}")


# ===========================
#  ENTRY POINT
# ===========================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", default="")
    parser.add_argument("--input", default="gaia_2023_all_splits/test/*.json")
    parser.add_argument("--output", default="generated_traces/test")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    generate_all_traces(
        api_key=args.api_key,
        input_path=args.input,
        output_path=args.output,
        batch_size=args.batch_size
    )
