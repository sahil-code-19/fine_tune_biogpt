import json
import random
import torch

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, TaskType, get_peft_model

# ---------------- CONFIG ---------------- #

MODEL_NAME = "microsoft/BioGPT-Large"
QA_JSON_PATH = "./new_drugs_qa_pairs_v2.json"

OUTPUT_DIR = "./biogpt-lora-v3"
FINAL_DIR = "./biogpt-final-v3"

MAX_SEQ_LEN = 768
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2
GRAD_ACCUMULATION = 8

LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_RATIO = 0.05

LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

SYSTEM_PROMPT = """
You are a veterinary pharmacology expert trained on Plumb's Veterinary Drug Handbook (2023).
Provide accurate answers about veterinary drugs including dosage, indications,
contraindications, adverse effects, and drug interactions.
"""

random.seed(42)

# ---------------- DATASET ---------------- #

def load_dataset_from_json(path):
    print("📂 Loading dataset...")

    with open(path, "r") as f:
        data = json.load(f)

    samples = []

    for drug in data:

        qa_pairs = drug.get("qa_pairs", [])

        for qa in qa_pairs:

            q = qa["question"].strip()
            a = qa["answer"].strip()

            samples.append({
                "question": q,
                "answer": a
            })

    return Dataset.from_list(samples)


# ---------------- TOKENIZATION ---------------- #

def tokenize_dataset(dataset, tokenizer):

    def tokenize(example):

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"### Question:\n{example['question']}\n\n"
            f"### Answer:\n"
        )

        answer = example["answer"] + tokenizer.eos_token

        prompt_tokens = tokenizer(prompt, add_special_tokens=False)
        answer_tokens = tokenizer(answer, add_special_tokens=False) 

        input_ids = prompt_tokens["input_ids"] + answer_tokens["input_ids"]

        
        attention_mask = [1] * len(input_ids)

        labels = [-100] * len(prompt_tokens["input_ids"]) + answer_tokens["input_ids"]

        # padding
        padding = MAX_SEQ_LEN - len(input_ids)

        if padding > 0:
            input_ids += [tokenizer.pad_token_id] * padding
            attention_mask += [0] * padding
            labels += [-100] * padding
        else:
            input_ids = input_ids[:MAX_SEQ_LEN]
            attention_mask = attention_mask[:MAX_SEQ_LEN]
            labels = labels[:MAX_SEQ_LEN]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    return dataset.map(tokenize, remove_columns=dataset.column_names)


# ---------------- LORA ---------------- #

def apply_lora(model):

    print("🔧 Applying LoRA")

    config = LoraConfig(

        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,

        bias="none",

        task_type=TaskType.CAUSAL_LM,

        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "fc1",
            "fc2"
        ],
    )

    model = get_peft_model(model, config)

    model.print_trainable_parameters()

    return model


# ---------------- TRAIN ---------------- #

def train(model, tokenizer, train_dataset, eval_dataset):

    args = TrainingArguments(

        output_dir=OUTPUT_DIR,

        num_train_epochs=NUM_EPOCHS,

        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,

        gradient_accumulation_steps=GRAD_ACCUMULATION,

        learning_rate=LEARNING_RATE,

        warmup_ratio=WARMUP_RATIO,

        weight_decay=0.01,

        fp16=True,

        logging_steps=10,

        eval_steps=100,
        eval_strategy="steps",

        save_steps=100,
        save_total_limit=3,

        report_to="none",
    )

    trainer = Trainer(

        model=model,
        args=args,

        train_dataset=train_dataset,
        eval_dataset=eval_dataset,

    )

    trainer.train()

    return trainer


# ---------------- SAVE ---------------- #

def save_model(trainer, tokenizer):

    print("💾 Saving model")

    trainer.model.save_pretrained(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)


# ---------------- TEST ---------------- #

def test_model(model, tokenizer):

    question = "How does acarbose affect insulin secretion?"

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"### Question:\n{question}\n\n"
        f"### Answer:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=200,
            num_beams=4,
            repetition_penalty=1.2,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    answer = text.split("### Answer:")[-1].strip()

    print("\nQuestion:", question)
    print("\nAnswer:", answer)


# ---------------- MAIN ---------------- #

def main():

    print("🚀 BioGPT Veterinary Drug LoRA Training")

    if not torch.cuda.is_available():
        print("❌ GPU required")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenizer.pad_token = tokenizer.eos_token

    print("📥 Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
        tie_word_embeddings=False
    )

    dataset = load_dataset_from_json(QA_JSON_PATH)

    split = dataset.train_test_split(test_size=0.1)

    train_dataset = tokenize_dataset(split["train"], tokenizer)
    eval_dataset = tokenize_dataset(split["test"], tokenizer)
    
    print(f"training_dataset::{train_dataset}")
    model = apply_lora(model)

    trainer = train(model, tokenizer, train_dataset, eval_dataset)

    save_model(trainer, tokenizer)

    test_model(model, tokenizer)


if __name__ == "__main__":
    main()