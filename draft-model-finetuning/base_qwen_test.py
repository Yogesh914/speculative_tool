#!/usr/bin/env python3
"""
Comparative Evaluator: Base Qwen vs. LoRA Qwen.
- Generates traces from Base Model.
- Generates traces from LoRA Model.
- Judges both individually against Gold.
- Judge 2 (Head-to-Head): Decides which model is better.
"""

import argparse
import json
import time
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import google.generativeai as genai

# -------------------------
# Helper: Get Raw Text
# -------------------------
def get_gold_text_from_messages(messages):
    gold_text = ""
    for msg in messages:
        if msg['role'] == 'assistant':
            gold_text += msg['content'] + "\n"
    return gold_text.strip()

def build_prompt(question, tools):
    tool_list = "\n".join(f"- {t}" for t in tools)
    prompt = f"""
<|user|>
Question: {question}

Available tools:
{tool_list}

Instructions:
Generate a reasoning trace (thought + action).
<|assistant|>
"""
    return prompt.strip()

# -------------------------
# Judge 1: Individual Accuracy
# -------------------------
def score_accuracy(gold_text, pred_text, model_name, api_key):
    """
    Scores a single trace against the gold standard.

    Returns a dict: {"score": float, "reason": str}
    """
    if not gold_text:
        return {"score": 0.0, "reason": "Empty gold trace"}

    genai.configure(api_key=api_key)
    judge = genai.GenerativeModel('gemini-2.5-pro')

    system_instruction = (
        f"You are evaluating the '{model_name}'. Score tool usage from 0.0 to 1.0.\n"
        "Focus ONLY on whether the correct TOOL NAMES are used, ignore arguments.\n"
        "1.0 = All tool names match gold exactly.\n"
        "0.5 = Some tool names correct, some wrong.\n"
        "0.0 = Completely wrong tool names or hallucinations.\n"
        "Output JSON: {\"score\": <float>, \"reason\": \"<explanation of tool accuracy>\"}"
    )

    user_prompt = f"""
GOLD TRACE:
{gold_text}

PREDICTED TRACE:
{pred_text}
"""
    try:
        response = judge.generate_content(
            f"{system_instruction}\n\n{user_prompt}",
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json", temperature=0.0
            )
        )
        parsed = json.loads(response.text)
        score = float(parsed.get("score", 0.0))
        reason = parsed.get("reason", "")
        return {"score": score, "reason": reason}
    except Exception as e:
        print(f"Error scoring {model_name}: {e}")
        return {"score": 0.0, "reason": str(e)}

# -------------------------
# Judge 2: Head-to-Head Comparison
# -------------------------
def compare_models(gold_text, base_pred, lora_pred, api_key):
    """
    Asks Gemini to pick a winner between Base and LoRA.
    """
    genai.configure(api_key=api_key)
    judge = genai.GenerativeModel('gemini-2.5-pro')
    
    system_instruction = (
        "You are a meta-judge comparing two AI models (Base vs Finetuned).\n"
        "Compare both predictions against the GOLD TRACE.\n"
        "Criteria:\n"
        "1. Correctness of tool selection.\n"
        "2. Adherence to formatting.\n"
        "3. Similarity with reasoning of the gold trace.\n\n"
        "Output JSON with:\n"
        "- \"winner\": \"base\", \"lora\", or \"tie\"\n"
        "- \"reason\": \"short explanation\""
    )

    user_prompt = f"""
--- GOLD TRACE ---
{gold_text}

--- BASE MODEL PREDICTION ---
{base_pred}

--- LORA MODEL PREDICTION ---
{lora_pred}
"""
    try:
        response = judge.generate_content(
            f"{system_instruction}\n\n{user_prompt}",
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json", temperature=0.0
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Comparison error: {e}")
        return {"winner": "error", "reason": str(e)}

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--lora_dir", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="comparison_results1.jsonl")
    parser.add_argument("--gemini_api_key", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    args = parser.parse_args()

    # 1. Load Tokenizer & Model
    print(f"Loading base model: {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"Loading LoRA adapter: {args.lora_dir}...")
    # PeftModel wraps the base model. We can toggle the adapter on/off.
    model = PeftModel.from_pretrained(base_model, args.lora_dir)
    model.eval()

    # 2. Load Data
    test_data = []
    with open(args.test_file, "r") as f:
        for line in f:
            if line.strip(): test_data.append(json.loads(line))
    print(f"Loaded {len(test_data)} examples.")

    results = []
    # tools_list = ["search", "wikipedia", "calculator", "final_answer"]

    tools_list = [
    "search_with_content",
    "file_read",
    "calculate",
    "code_generate",
    "code_exec",
    "vision_analyze",
    "vision_ocr",
    "search_web",
    "search",
    "finance.get_world_bank_data",
    "python",
    "final_answer",
    "web_search",
    "wiki.get_page",
    "wiki.get_revisions",
    "web_browse",
    "image_analyzer"
]
    
    # Stats trackers
    wins = {"base": 0, "lora": 0, "tie": 0}

    print("Starting Comparative Evaluation...")
    for ex in tqdm(test_data):
        question = ex["messages"][0]["content"]
        gold_text = get_gold_text_from_messages(ex["messages"])
        
        prompt = build_prompt(question, tools_list)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # --- A. Generate with BASE Model (Adapter Disabled) ---
        # This context manager temporarily turns off the LoRA weights
        with model.disable_adapter():
            with torch.no_grad():
                base_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
        base_pred = tokenizer.decode(base_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # --- B. Generate with LORA Model (Adapter Enabled) ---
        # Default behavior of PeftModel is adapter enabled
        with torch.no_grad():
            lora_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        lora_pred = tokenizer.decode(lora_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # --- C. Scoring ---
        # 1. Individual Scores (include reasoning)
        base_res = score_accuracy(gold_text, base_pred, "Base Model", args.gemini_api_key)
        lora_res = score_accuracy(gold_text, lora_pred, "LoRA Model", args.gemini_api_key)

        base_score = base_res.get("score", 0.0)
        base_reason = base_res.get("reason", "")

        lora_score = lora_res.get("score", 0.0)
        lora_reason = lora_res.get("reason", "")

        # 2. Head-to-Head Comparison
        comparison = compare_models(gold_text, base_pred, lora_pred, args.gemini_api_key)
        winner = comparison.get("winner", "tie").lower()
        if winner not in wins: winner = "tie"
        wins[winner] += 1

        # Store Result
        res = {
            "id": ex.get("id", "unknown"),
            "base_score": base_score,
            "lora_score": lora_score,
            "base_reason": base_reason,
            "lora_reason": lora_reason,
            "winner": winner,
            "comparison_reason": comparison.get("reason"),
            "base_pred": base_pred,
            "lora_pred": lora_pred,
            "gold_raw": gold_text
        }
        results.append(res)
        
        print(f" ID: {res['id']} | Base: {base_score:.2f} | LoRA: {lora_score:.2f} | Winner: {winner.upper()}")

    # --- D. Summary ---
    with open(args.output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    avg_base = sum(r['base_score'] for r in results) / len(results) if results else 0
    avg_lora = sum(r['lora_score'] for r in results) / len(results) if results else 0

    print("\n" + "="*40)
    print("COMPARISON SUMMARY")
    print(f"Total Examples: {len(results)}")
    print(f"Avg Base Accuracy: {avg_base:.2%}")
    print(f"Avg LoRA Accuracy: {avg_lora:.2%}")
    print("-" * 20)
    print(f"Base Wins: {wins['base']}")
    print(f"LoRA Wins: {wins['lora']}")
    print(f"Ties:      {wins['tie']}")
    print("="*40)

if __name__ == "__main__":
    main()