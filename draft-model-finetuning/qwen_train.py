# #!/usr/bin/env python3
# """
# LoRA fine-tuning for Qwen-3-4B with validation using GAIA tool-call traces.

# Input format: JSONL with {"messages":[...]} per line.

# Run:
# python qwen_train.py \
#     --train_file dataset/gaia_train.jsonl \
#     --val_file dataset/gaia_val.jsonl \
#     --output_dir qwen3-gaia-lora \
#     --model_name Qwen/Qwen3-4B
# """

# import os
# import json
# import argparse
# from pathlib import Path
# from typing import List, Dict

# import torch
# from datasets import Dataset
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     TrainingArguments,
#     Trainer,
#     EvalPrediction,
# )
# from peft import LoraConfig, get_peft_model
# import numpy as np


# # -------------------------
# # Build tokenized sequences
# # -------------------------
# def build_input_and_labels_from_messages(messages, tokenizer, max_length):
#     """
#     Convert messages → input_ids, labels masked for user tokens.
#     """
#     input_ids = []
#     labels = []

#     eos_id = tokenizer.eos_token_id or tokenizer.sep_token_id

#     def enc(prefix, content):
#         pre_ids = tokenizer.encode(prefix, add_special_tokens=False)
#         content_ids = tokenizer.encode(content, add_special_tokens=False)
#         return pre_ids + content_ids + [eos_id]

#     for msg in messages:
#         role = msg["role"]
#         content = msg["content"]

#         if role == "user":
#             prefix = "<|user|>\n"
#         else:
#             prefix = "<|assistant|>\n"

#         seg = enc(prefix, content)
#         input_ids.extend(seg)

#         if role == "assistant":
#             labels.extend(seg)
#         else:
#             labels.extend([-100] * len(seg))

#     # Truncate left
#     if len(input_ids) > max_length:
#         input_ids = input_ids[-max_length:]
#         labels = labels[-max_length:]

#     attention_mask = [1] * len(input_ids)
#     return dict(
#         input_ids=input_ids,
#         labels=labels,
#         attention_mask=attention_mask,
#     )


# # -------------------------
# # Read JSONL + tokenize
# # -------------------------
# def load_and_tokenize(jsonl_path, tokenizer, max_length):
#     raw = []
#     with open(jsonl_path, "r") as f:
#         for line in f:
#             obj = json.loads(line)
#             raw.append(obj)

#     print(f"Loaded {len(raw)} examples")

#     features = {
#         "input_ids": [],
#         "labels": [],
#         "attention_mask": [],
#     }

#     for ex in raw:
#         msgs = ex["messages"]
#         out = build_input_and_labels_from_messages(msgs, tokenizer, max_length)
#         for k in features:
#             features[k].append(out[k])

#     return Dataset.from_dict(features)


# # -------------------------
# # Collator for dynamic padding
# # -------------------------
# class DataCollatorCausal:
#     def __init__(self, tokenizer, pad_to_multiple_of=None):
#         self.tokenizer = tokenizer
#         self.pad_to_multiple_of = pad_to_multiple_of

#     def __call__(self, batch):
#         input_ids = [x["input_ids"] for x in batch]
#         attn = [x["attention_mask"] for x in batch]
#         labels = [x["labels"] for x in batch]

#         padded = self.tokenizer.pad(
#             {"input_ids": input_ids, "attention_mask": attn},
#             padding=True,
#             pad_to_multiple_of=self.pad_to_multiple_of,
#             return_tensors="pt",
#         )
#         max_len = padded["input_ids"].size(1)

#         label_pad = []
#         for lab in labels:
#             if len(lab) < max_len:
#                 lab = lab + [-100] * (max_len - len(lab))
#             else:
#                 lab = lab[:max_len]
#             label_pad.append(lab)

#         padded["labels"] = torch.tensor(label_pad, dtype=torch.long)
#         return padded


# # -------------------------
# # Main
# # -------------------------
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--train_file", type=str, required=True)
#     parser.add_argument("--val_file", type=str, required=True)
#     parser.add_argument("--model_name", type=str, required=True)
#     parser.add_argument("--output_dir", type=str, default="qwen3-lora-out")
#     parser.add_argument("--max_length", type=int, default=2048)
#     parser.add_argument("--learning_rate", type=float, default=2e-4)
#     parser.add_argument("--per_device_train_batch_size", type=int, default=2)
#     parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
#     parser.add_argument("--num_train_epochs", type=int, default=3)
#     parser.add_argument("--eval_steps", type=int, default=200)
#     parser.add_argument("--save_steps", type=int, default=200)
#     parser.add_argument("--lora_r", type=int, default=8)
#     parser.add_argument("--lora_alpha", type=int, default=32)
#     parser.add_argument("--lora_dropout", type=float, default=0.05)
#     parser.add_argument("--device_map", type=str, default="auto")
#     args = parser.parse_args()

#     # -------------------------
#     # Tokenizer
#     # -------------------------
#     tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
#     if tokenizer.pad_token is None:
#         tokenizer.add_special_tokens({"pad_token": "[PAD]"})
#     if tokenizer.eos_token is None:
#         tokenizer.add_special_tokens({"eos_token": ""})

#     tokenizer.padding_side = "left"

#     # -------------------------
#     # Load + tokenize dataset
#     # -------------------------
#     # full_ds = load_and_tokenize(args.train_file, tokenizer, args.max_length)

#     # split train/validation
#     # split = full_ds.train_test_split(test_size=args.validation_split, shuffle=True)
#     # train_ds = split["train"]
#     # val_ds = split["test"]
#     train_ds = load_and_tokenize(args.train_file, tokenizer, args.max_length)
#     val_ds = load_and_tokenize(args.val_file, tokenizer, args.max_length)
#     print(f"Train size: {len(train_ds)}  |  Validation size: {len(val_ds)}")

#     # -------------------------
#     # Model + LoRA
#     # -------------------------
#     print("Loading model", args.model_name)
#     model = AutoModelForCausalLM.from_pretrained(
#         args.model_name,
#         torch_dtype=torch.float16,
#         device_map=args.device_map,
#         trust_remote_code=True,
#     )

#     model.resize_token_embeddings(len(tokenizer))

#     print("Applying LoRA...")
#     lora_cfg = LoraConfig(
#         r=args.lora_r,
#         lora_alpha=args.lora_alpha,
#         lora_dropout=args.lora_dropout,
#         bias="none",
#         task_type="CAUSAL_LM",
#         target_modules=[
#             "q_proj", "k_proj", "v_proj", "o_proj",
#             "up_proj", "down_proj", "gate_proj",
#         ],
#     )
#     model = get_peft_model(model, lora_cfg)
#     model.print_trainable_parameters()

#     # -------------------------
#     # Training arguments
#     # -------------------------
#     training_args = TrainingArguments(
#         output_dir=args.output_dir,
#         learning_rate=args.learning_rate,
#         per_device_train_batch_size=args.per_device_train_batch_size,
#         gradient_accumulation_steps=args.gradient_accumulation_steps,
#         num_train_epochs=args.num_train_epochs,
#         eval_strategy="steps",
#         eval_steps=args.eval_steps,
#         save_strategy="steps",
#         save_steps=args.save_steps,
#         save_total_limit=2,
#         logging_steps=50,
#         warmup_ratio=0.03,
#         weight_decay=0.0,
#         fp16=True,
#         report_to="none",
#         load_best_model_at_end=True,
#         metric_for_best_model="loss",
#         greater_is_better=False,
#     )

#     # -------------------------
#     # Collator & Trainer
#     # -------------------------
#     data_collator = DataCollatorCausal(tokenizer)

#     def compute_metrics(eval_pred: EvalPrediction):
#         """
#         Computes validation perplexity.
#         """
#         loss = eval_pred.loss
#         try:
#             ppl = np.exp(loss)
#         except OverflowError:
#             ppl = float("inf")
#         return {"loss": loss, "perplexity": ppl}

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_ds,
#         eval_dataset=val_ds,
#         tokenizer=tokenizer,
#         data_collator=data_collator,
#         compute_metrics=compute_metrics,
#     )

#     # -------------------------
#     # Train
#     # -------------------------
#     print("Starting training with validation...")
#     trainer.train()

#     print("Saving best model...")
#     trainer.save_model(args.output_dir)
#     tokenizer.save_pretrained(args.output_dir)
#     print(f"✅ Model and tokenizer saved to {args.output_dir}")



# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
"""
LoRA fine-tuning for Qwen-3-4B (or Qwen 2.5) with validation.
FIX: Uses existing EOS token for padding to avoid vocabulary mismatch errors.
"""

import os
import json
import argparse
from typing import List, Dict

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq, 
)
from peft import LoraConfig, get_peft_model
import numpy as np

# -------------------------
# Build tokenized sequences
# -------------------------
def build_input_and_labels_from_messages(messages, tokenizer, max_length):
    """
    Convert messages -> input_ids, labels masked for user tokens.
    """
    input_ids = []
    labels = []

    # Use the tokenizer's existing EOS token
    eos_id = tokenizer.eos_token_id 

    def enc(prefix, content):
        pre_ids = tokenizer.encode(prefix, add_special_tokens=False)
        content_ids = tokenizer.encode(content, add_special_tokens=False)
        return pre_ids + content_ids + [eos_id]

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            prefix = "<|user|>\n"
        else:
            prefix = "<|assistant|>\n"

        seg = enc(prefix, content)
        input_ids.extend(seg)

        if role == "assistant":
            labels.extend(seg)
        else:
            # Mask user messages in the loss
            labels.extend([-100] * len(seg))

    # Truncate left if too long
    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
        labels = labels[-max_length:]

    attention_mask = [1] * len(input_ids)
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }

# -------------------------
# Read JSONL + tokenize
# -------------------------
def load_and_tokenize(jsonl_path, tokenizer, max_length):
    raw = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                raw.append(json.loads(line))

    print(f"Loaded {len(raw)} examples from {jsonl_path}")

    features = {
        "input_ids": [],
        "labels": [],
        "attention_mask": [],
    }

    for ex in raw:
        msgs = ex["messages"]
        out = build_input_and_labels_from_messages(msgs, tokenizer, max_length)
        features["input_ids"].append(out["input_ids"])
        features["labels"].append(out["labels"])
        features["attention_mask"].append(out["attention_mask"])

    return Dataset.from_dict(features)

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="qwen3-lora-out")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--lora_r", type=int, default=16) # Increased slightly for reasoning
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--device_map", type=str, default="auto")
    args = parser.parse_args()

    # -------------------------
    # 1. Load Tokenizer (CRITICAL FIXES HERE)
    # -------------------------
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # FIX: Do NOT add a new [PAD] token. Reuse the EOS token.
    # This prevents the vocabulary mismatch error.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    tokenizer.padding_side = "left" # Good for generation/inference later

    # -------------------------
    # 2. Load + tokenize dataset
    # -------------------------
    train_ds = load_and_tokenize(args.train_file, tokenizer, args.max_length)
    val_ds = load_and_tokenize(args.val_file, tokenizer, args.max_length)
    print(f"Train size: {len(train_ds)}  |  Validation size: {len(val_ds)}")

    # -------------------------
    # 3. Load Model
    # -------------------------
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map=args.device_map,
        trust_remote_code=True,
    )

    # FIX: REMOVED `model.resize_token_embeddings(...)`
    # We are using the standard vocabulary size.

    print("Applying LoRA...")
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "up_proj", "down_proj", "gate_proj",
        ],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # -------------------------
    # 4. Training
    # -------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=10,
        warmup_ratio=0.03,
        fp16=True,
        report_to="none",
        load_best_model_at_end=True, 
        metric_for_best_model="loss", # Simple loss tracking
    )

    # Use DataCollatorForSeq2Seq to handle padding automatically
    # (It works well for CausalLM padding too)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, 
        pad_to_multiple_of=8, 
        padding=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    print("Saving best model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()