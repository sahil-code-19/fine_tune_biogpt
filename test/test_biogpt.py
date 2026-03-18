import json
import random

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------- CONFIG ---------------- #

base_model = "microsoft/BioGPT-Large"
adapter_path = "../models_fine_tuned/biogpt-lora"

DATASET_PATH = "../datasets/82_plumbs_drugs.json"

NUM_QUESTIONS = 10

SYSTEM_PROMPT = """
You are a veterinary pharmacology expert trained on Plumb's Veterinary Drug Handbook (2023).
Provide accurate answers about veterinary drugs including dosage, indications,
contraindications, adverse effects, and drug interactions.
"""

# ---------------- LOAD MODEL ---------------- #

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(base_model)

model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map="auto")

model = PeftModel.from_pretrained(model, adapter_path)

model.eval()

# ---------------- LOAD DATASET ---------------- #

print("Loading dataset...")

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

questions = []
answers = []

for drug in data:
    qa_pairs = drug.get("qa_pairs", [])

    for qa in qa_pairs:
        q = qa.get("question", "").strip()
        a = qa.get("answer", "").strip()

        if len(q) > 20 and len(a) > 10:
            questions.append(q)
            answers.append(a)

print("Total questions:", len(questions))

# ---------------- SELECT 10 QUESTIONS ---------------- #

indices = random.sample(range(len(questions)), NUM_QUESTIONS)

# ---------------- INFERENCE ---------------- #

for i, idx in enumerate(indices):
    question = questions[idx]
    ground_truth = answers[idx]

    prompt = f"{SYSTEM_PROMPT}\n\n### Question:\n{question}\n\n### Answer:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        input_length = inputs["input_ids"].shape[1]

        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            num_beams=4,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][input_length:]

    model_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    # text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # model_answer = text.split("### Answer:")[-1].strip()

    print("\n" + "=" * 80)
    print(f"Question {i + 1}")
    print("=" * 80)

    print("\nQuestion:")
    print(question)

    print("\nModel Answer:")
    print(model_answer)

    print("\nGround Truth:")
    print(ground_truth)