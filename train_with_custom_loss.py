import os
import gc
import re
import json
import random
import torch
from collections import Counter
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    TrainerCallback
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from torch.nn import CrossEntropyLoss

# ==========================================================
# 1. Config
# ==========================================================
cp_model_path = "./models_fine_tuned/biogpt-cp-alldrugs-v2"
output_dir    = "./models_fine_tuned/biogpt-ft-v2"

# Ensure the paths match your workspace structure
dataset_path  = "datasets/82_plumbs_drugs.json"
txt_file_path = "all_drugs.txt"

OVERSAMPLE_FACTOR = 2     # dosage Q&A pairs seen 2x
txt_chunk_size = 1000     # characters per chunk for all_drugs.txt

SYSTEM_PROMPT = """You are a veterinary pharmacology expert trained on Plumb's Veterinary Drug Handbook (2023).
Provide accurate answers about veterinary drugs including dosage, indications,
contraindications, adverse effects, and drug interactions."""


# ==========================================================
# 2. Keyword Types 
# ==========================================================
DOSAGE_UNIT_KEYWORDS = [
    "mg/kg", "mg/dog", "mg/cat", "mg/horse", "mg/bird",
    "mg/ferret", "mg/lb", "mcg/kg", "mL/kg", "units/kg",
    "mg/animal", "mg/head",
]

FREQUENCY_KEYWORDS = [
    "twice daily", "once daily", "three times daily",
    "every 8 hours", "every 12 hours", "every 24 hours",
    "every 6 hours", "every 48 hours", "every 4 hours",
    "BID", "SID", "TID", "QID",
    "q8h", "q12h", "q24h", "q6h",
]

# ==========================================================
# 3. Load Tokenizer and CP Model
# ==========================================================
print("\nLoading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(cp_model_path)
tokenizer.pad_token_id = tokenizer.eos_token_id # BioGPT needs eos_token_id strictly
tokenizer.padding_side = "right"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    cp_model_path,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto"
)

# ==========================================================
# 4. LoRA Config
# ==========================================================
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# ==========================================================
# 5. Load and Format Dataset (JSON + TXT)
# ==========================================================
print("\nLoading Q&A dataset...")
normal_pairs  = []
dosage_pairs  = []

if os.path.exists(dataset_path):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for drug in data:
        for qa in drug.get("qa_pairs", []):
            q = qa.get("question", "").strip()
            a = qa.get("answer", "").strip()

            if len(q) > 20 and len(a) > 10:
                is_dosage = any(kw.strip().lower() in a.lower() for kw in DOSAGE_UNIT_KEYWORDS + FREQUENCY_KEYWORDS)

                if is_dosage:
                    # Repeat numbers for dosage answers for reinforcement
                    enhanced_answer = f"{a} The exact dosage is {a}"
                    text = f"{SYSTEM_PROMPT}\n\n### Question:\n{q}\n\n### Answer:\n{enhanced_answer}"
                    dosage_pairs.append({"text": text})
                else:
                    text = f"{SYSTEM_PROMPT}\n\n### Question:\n{q}\n\n### Answer:\n{a}"
                    normal_pairs.append({"text": text})

    print(f"Normal Q&A pairs  : {len(normal_pairs)}")
    print(f"Dosage Q&A pairs  : {len(dosage_pairs)}")
else:
    print(f"Warning: {dataset_path} not found.")

# Process all_drugs.txt to mix raw knowledge with Q&A
print("\nLoading raw knowledge from all_drugs.txt...")
raw_text_chunks = []
if os.path.exists(txt_file_path):
    with open(txt_file_path, "r", encoding="utf-8") as f:
        full_text = f.read()
    
    # Split text into manageable chunks
    for i in range(0, len(full_text), txt_chunk_size):
        chunk = full_text[i:i+txt_chunk_size].strip()
        if len(chunk) > 50:
            raw_text_chunks.append({"text": chunk})
    print(f"Raw text chunks extracted: {len(raw_text_chunks)}")
else:
    print(f"Warning: {txt_file_path} not found.")

# Combine all: Oversampled dosages + normal pairs + raw text chunks
oversampled = normal_pairs + (dosage_pairs * OVERSAMPLE_FACTOR) + raw_text_chunks
random.seed(3407)
random.shuffle(oversampled)

print(f"Total dataset size after combinations: {len(oversampled)}")

# ==========================================================
# 6. Train / Eval Split — 90% / 10%
# ==========================================================
split_idx     = int(len(oversampled) * 0.9)
train_data    = oversampled[:split_idx]
eval_data     = oversampled[split_idx:]

train_dataset = Dataset.from_list(train_data)
eval_dataset  = Dataset.from_list(eval_data)

print(f"Train samples : {len(train_dataset)}")
print(f"Eval samples  : {len(eval_dataset)}")

# ==========================================================
# 7. Pre-compute numeric token IDs for Custom Loss
# ==========================================================
# Identifying tokens that represent numbers is more efficient than regex on decoding
# during training. We build a set of token IDs that contain digits.
vocab = tokenizer.get_vocab()
numeric_token_ids = set()
for token, token_id in vocab.items():
    if any(char.isdigit() for char in token):
        numeric_token_ids.add(token_id)

print(f"Identified {len(numeric_token_ids)} tokens containing digits out of {len(vocab)} total tokens.")

# ==========================================================
# 8. Custom Trainer
# ==========================================================
class DosageFocusedSFTTrainer(SFTTrainer):
    """
    SFTTrainer but gives 3x loss weight to number tokens.
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Allow default huggingface compatibility
        outputs = model(**inputs)
        logits  = outputs.logits
        labels  = inputs["labels"]

        # Default weight = 1.0 for all tokens
        weights = torch.ones_like(labels, dtype=logits.dtype)

        # Boost weight for number tokens using pre-computed token IDs
        input_ids = inputs["input_ids"]
        for i in range(input_ids.shape[0]):
            for j in range(input_ids.shape[1]):
                if input_ids[i, j].item() in numeric_token_ids:
                    if j < labels.shape[1]:
                        weights[i, j] = 3.0  # 3x weight on numbers

        # Weighted cross entropy loss
        loss_fct      = CrossEntropyLoss(reduction="none")
        
        # Shift so that tokens < n predict n
        shift_logits  = logits[..., :-1, :].contiguous()
        shift_labels  = labels[..., 1:].contiguous()
        shift_weights = weights[..., 1:].contiguous()

        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        loss  = loss * shift_weights.view(-1)
        valid = (shift_labels.view(-1) != -100)
        loss  = loss[valid].mean()

        return (loss, outputs) if return_outputs else loss


# ==========================================================
# 9. Training Config
# ==========================================================
sft_config = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=2, # Increase to 4 if GPU memory allows 
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=5,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    eval_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    optim="adamw_bnb_8bit", # Requires bitsandbytes. Change to 'adamw_torch' if it throws an error
    weight_decay=0.05,
    lr_scheduler_type="cosine",
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    seed=3407,
    report_to="none",
    max_length=1024,
    dataset_text_field="text",
)

# ==========================================================
# 10. Trainer With Early Stopping
# ==========================================================
trainer = DosageFocusedSFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.01
        )
    ]
)

# ==========================================================
# 11. Train
# ==========================================================
gc.collect()
torch.cuda.empty_cache()

print("\nStarting Fine-tuning...")
trainer.train()

# ==========================================================
# 12. Save Best Adapter
# ==========================================================
print(f"\nSaving best adapter to {output_dir}...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Done. Adapter saved to {output_dir}")
