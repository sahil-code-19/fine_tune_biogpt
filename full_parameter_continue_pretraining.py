import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset

# ==========================================================
# 1. Configuration
# ==========================================================
base_model_name = "microsoft/BioGPT-Large"
finetuned_adapter_dir = "models_fine_tuned/biogpt-lora"
output_dir = "models_fine_tuned/biogpt-cp-alldrugs"
dataset_path = "datasets/all_drugs.txt"
max_seq_length = 1024 # BioGPT-Large context window is 1024

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
# BioGPT defined special tokens: BOS=0, PAD=1, EOS=2
tokenizer.pad_token = tokenizer.pad_token or " <pad>" # Ensure pad is set if not already
tokenizer.pad_token_id = 1

# Load base model
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16, 
    device_map="auto"
)

# ==========================================================
# 2. Load Fine-Tuned Model & Merge Weights (If it exists)
# ==========================================================
import os

if os.path.exists(finetuned_adapter_dir):
    print(f"Loading previous fine-tuned adapter from {finetuned_adapter_dir}...")
    model = PeftModel.from_pretrained(base_model, finetuned_adapter_dir)
    model = model.merge_and_unload()
else:
    print(f"No previous adapter found at '{finetuned_adapter_dir}'. Starting fresh from base model.")
    model = base_model

# ==========================================================
# 3. Prepare Dataset (Chunking into continuous stream)
# ==========================================================
print(f"Processing and chunking {dataset_path}...")

with open(dataset_path, "r", encoding="utf-8") as f:
    text_content = f.read()

# Tokenize entire file at once (BioGPT tokenizer is fast)
# add_special_tokens=False because we'll add them manually or join as stream
tokens = tokenizer(text_content, truncation=False, add_special_tokens=False)["input_ids"]

# Concatenate as a long stream and split into max_seq_length blocks
# This is the "standard" way for continued pretraining of causal models
all_chunks = []
for i in range(0, len(tokens), max_seq_length):
    chunk = tokens[i : i + max_seq_length]
    
    # We keep chunks > 64 tokens to avoid overfitting on small tails
    if len(chunk) > 64:
        # We DO NOT pad manually here or supply "labels" manually.
        # DataCollatorForLanguageModeling will automatically pad to the longest sequence in the batch
        # and set padding labels to -100 so the model ignores them during loss calculation.
        all_chunks.append({
            "input_ids": chunk,
        })

print(f"Created {len(all_chunks)} chunks of size {max_seq_length}")
dataset = Dataset.from_list(all_chunks)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ==========================================================
# 4. Training Setup
# ==========================================================
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,          # VRAM limit: BioGPT-Large + 1024 seq len
    gradient_accumulation_steps=16,         # Effective batch size = 16
    num_train_epochs=1,                     # Full pass over data
    learning_rate=2e-5,                     
    logging_steps=5,
    save_steps=100,
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=3407,
    report_to="none",                       
    remove_unused_columns=False             
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# ==========================================================
# 5. Execute Training
# ==========================================================
print("Starting Continued Pretraining...")
trainer.train()

print(f"Training complete. Saving model to {output_dir}")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print("Done! You can now use this model for your next Q&A Fine-Tuning phase.")