import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

load_dotenv()

# ================================================================
# 1. CONFIG
# ================================================================
# You can change these back to microsoft/BioGPT-Large and ./biogpt-final-v3 
# if you are in a different folder!
base_model   = "./models_fine_tuned/biogpt-cp-alldrugs-v2"
adapter_path = "./models_fine_tuned/biogpt-ft-v1"

DRUGS_TXT     = "./all_drugs.txt"

SYSTEM_PROMPT = """You are a veterinary pharmacology expert trained on Plumb's Veterinary Drug Handbook (2023).
Provide accurate answers about veterinary drugs including dosage, indications,
contraindications, adverse effects, and drug interactions."""

# ================================================================
# 2. BUILD RAG VECTOR STORE
# ================================================================
print("Building RAG vector store from all_drugs.txt...")

embeddings   = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = InMemoryVectorStore(embeddings)

loader = TextLoader(DRUGS_TXT, encoding="utf-8")
docs   = loader.load()

# chunk_size=1500 keeps drug sections mostly intact
# chunk_overlap=200 avoids cutting dosage info at boundaries
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200
)
texts = splitter.split_documents(docs)
vector_store.add_documents(documents=texts)

print(f"Vector store ready — {len(texts)} chunks indexed\n")

def retrieve_context(query: str, k: int = 4) -> str:
    """
    Search the vector store for the most relevant drug information.
    Returns a formatted string of the top-k chunks.
    """
    retrieved_docs = vector_store.similarity_search(query, k=k)
    context = "\n\n".join(
        f"[Chunk {i+1}]\n{doc.page_content}"
        for i, doc in enumerate(retrieved_docs)
    )
    return context

# ================================================================
# 3. LOAD BIOGPT MODEL + ADAPTER
# ================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading BioGPT model on {device}...")

tokenizer = AutoTokenizer.from_pretrained(base_model)
# tokenizer.pad_token_id = 1 # Included from your snippet, though BioGPT standard is usually tokenizer.eos_token_id

# BioGPT uses model structure that works best with bfloat16/float16
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
).to(device)

# Load fine-tuned adapter if it exists
if os.path.exists(adapter_path):
    print(f"Loading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    print("Adapter loaded.")
else:
    print(f"No adapter found at {adapter_path} — using base/CP model only.")

model.eval()
print("\nModel loaded successfully! Type 'quit' or 'exit' to stop.")

# ================================================================
# 4. INTERACTIVE LOOP WITH RAG
# ================================================================
while True:
    print("\n" + "=" * 60)
    user_input = input("Enter your question: ").strip()

    # Check for quit condition
    if user_input.lower() in ["quit", "exit"]:
        print("Exiting...")
        break

    # Skip empty input
    if not user_input:
        continue

    # Step 1 — Retrieve relevant context via RAG
    context = retrieve_context(user_input, k=4)

    # Note: Optionally print out a snippet of the context to see what RAG found
    print("\n[RAG Retrieved Context Snippet]")
    print(context[:250] + "...\n" if len(context) > 250 else context + "\n")

    # Step 2 — Build RAG-augmented prompt
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"### Relevant Drug Information:\n"
        f"{context}\n\n"
        f"### Question:\n"
        f"{user_input}\n\n"
        f"### Answer:\n"
    )

    # Step 3 — Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024          # BioGPT-Large context limit
    ).to(device)

    input_length = inputs["input_ids"].shape[1]

    # Step 4 — Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            num_beams=4,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Step 5 — Decode only new tokens
    generated_tokens = outputs[0][input_length:]
    model_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    print("\n" + "-" * 40)
    print("Model Answer (RAG + BioGPT):")
    print("-" * 40)
    print(model_answer)