#!/usr/bin/env python3
"""
Simplified Qwen-3-4B Evaluator.
Relies on Gemini to parse and judge raw text strings.
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
    """
    Simply joins all assistant content into one string.
    No regex parsing.
    """
    gold_text = ""
    for msg in messages:
        if msg['role'] == 'assistant':
            gold_text += msg['content'] + "\n"
    return gold_text.strip()

def build_prompt(question, tools):
    # Standard Prompt
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
# Judge: Gemini (The "Smart" Parser)
# -------------------------
def score_with_gemini(gold_text, pred_text, api_key):
    """
    Sends RAW strings to Gemini. Gemini decides if they match.
    """
    if not gold_text:
        return 0.0

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-pro') 
    
    system_instruction = (
        "You are a judge evaluating an AI agent's performance. "
        "I will give you a GOLD TRACE (ground truth) and a PREDICTED TRACE.\n"
        "Compare the **Tool Calls** (names only - we relax the arguments here because there are no real tool calls happening) in both traces.\n"
        "Ignore formatting differences (e.g., XML vs JSON).\n\n"
        "Score from 0.0 to 1.0:\n"
        "- 1.0: The predicted tool calls match the gold tool calls exactly.\n"
        "- 0.0: The predicted tool calls are wrong or missing.\n"
        "Output ONLY JSON: {\"tool_accuracy\": <score>}"
    )

    user_prompt = f"""
--- GOLD TRACE (Ground Truth) ---
{gold_text}

--- PREDICTED TRACE ---
{pred_text}
"""

    try:
        response = model.generate_content(
            f"{system_instruction}\n\n{user_prompt}",
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.0
            )
        )
        score_json = json.loads(response.text)
        return float(score_json.get("tool_accuracy", 0.0))
    except Exception as e:
        print(f"Gemini scoring error: {e}")
        return 0.0

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--lora_dir", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="results.jsonl")
    parser.add_argument("--gemini_api_key", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    # Load Tokenizer & Model
    print(f"Loading base model: {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"Loading LoRA: {args.lora_dir}...")
    model = PeftModel.from_pretrained(model, args.lora_dir)
    model.eval()

    # Load Data
    test_data = []
    with open(args.test_file, "r") as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))
    print(f"Loaded {len(test_data)} examples.")

    results = []

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

    print("Starting evaluation...")
    for ex in tqdm(test_data):
        question = ex["messages"][0]["content"]
        
        # 1. Get RAW Gold Text (No Regex!)
        gold_text = get_gold_text_from_messages(ex["messages"])
        
        prompt = build_prompt(question, tools_list)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        start = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        latency = time.time() - start

        # 2. Get RAW Predicted Text (No Regex!)
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        pred_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # 3. Let Gemini Compare them
        acc = score_with_gemini(gold_text, pred_text, args.gemini_api_key)

        results.append({
            "id": ex.get("id", "unknown"),
            "accuracy": acc,
            "latency": latency,
            "gold_raw": gold_text,
            "pred_raw": pred_text
        })
        
        print(f" ID: {ex.get('id')} | Acc: {acc:.2f}")

    # Save
    avg_acc = sum(r['accuracy'] for r in results) / len(results) if results else 0
    with open(args.output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Finished. Avg Acc: {avg_acc:.2%}")

if __name__ == "__main__":
    main()