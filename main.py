import os
import gc
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import PeftModel
from datasets import Dataset

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Config
base_model_name       = "microsoft/BioGPT-Large"
# finetuned_adapter_dir = "models_fine_tuned/biogpt-lora"
output_dir            = "models_fine_tuned/biogpt-cp-alldrugs-v3"
dataset_path          = "./all_drugs.txt"
max_seq_length        = 1024

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token_id = 1

# Load base model in bfloat16 — NOT float16
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,  # ← critical fix
    device_map="auto"
)

# Merge old adapter if exists
# if os.path.exists(finetuned_adapter_dir):
#     print("Merging old adapter...")
#     model = PeftModel.from_pretrained(base_model, finetuned_adapter_dir)
#     model = model.merge_and_unload()
# else:
#     model = base_model

model = base_model

# Full parameter training — unlock all weights
for param in model.parameters():
    param.requires_grad = True

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Dataset
with open(dataset_path, "r", encoding="utf-8") as f:
    text_content = f.read()

tokens = tokenizer(text_content, truncation=False, add_special_tokens=False)["input_ids"]

all_chunks = []
for i in range(0, len(tokens), max_seq_length):
    chunk = tokens[i : i + max_seq_length]
    if len(chunk) > 64:
        all_chunks.append({"input_ids": chunk})

print(f"Total chunks: {len(all_chunks)}")
dataset = Dataset.from_list(all_chunks)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training args — all fixes applied
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_steps=5,
    save_steps=100,
    optim="adamw_bnb_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    bf16=True,
    fp16=False,
    seed=3407,
    report_to="none",
    remove_unused_columns=False,
    gradient_checkpointing=True,
    dataloader_pin_memory=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

gc.collect()
torch.cuda.empty_cache()

print("Starting CP...")
trainer.train()

# Save full model
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Saved to {output_dir}")