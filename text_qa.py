import json, re, os, time, csv, threading
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# ── CONFIG ────────────────────────────────────────────────────────
MDX_FILE     = "./correct_plumbs_test_drugs.mdx"
OUTPUT_FILE  = "test_21_drugs_qa_pairs.json"
MODEL        = "gpt-4.1-mini"
VERIFY_MODEL = "gpt-4.1-mini"
MAX_WORKERS  = 4  # Process 4 drugs at a time
# ─────────────────────────────────────────────────────────────────

# All known section headers from the MDX format
SECTION_HEADERS = [
    "Prescriber Highlights",
    "Uses/Indications", "Uses", "Indications",
    "Pharmacology/Actions", "Pharmacology", "Actions",
    "Pharmacokinetics",
    "Contraindications/Precautions/Warnings",
    "Contraindications", "Precautions", "Warnings",
    "Adverse Effects",
    "Reproductive/Nursing Safety", "Reproductive", "Nursing Safety",
    "Overdose/Acute Toxicity", "Overdose", "Acute Toxicity",
    "Drug Interactions",
    "Laboratory Considerations",
    "Dosages", "Dosage",
    "Monitoring",
    "Client Information",
    "Chemistry/Synonyms", "Chemistry", "Synonyms",
    "Storage/Stability", "Storage", "Stability",
    "Compatibility/Compounding Considerations",
    "Compatibility", "Compounding Considerations",
    "Dosage Forms/Regulatory Status",
    "Dosage Forms", "Regulatory Status",
    "Indications/Actions",
    "Suggested Dosages/Uses",
    "Precautions/Adverse Effects",
    "Veterinary-Labeled Products",
    "Human-Labeled Products",
    "Prescription Requirements",
    "Controlled Substances",
    "Medication Errors and Patient Safety",
    "Commonly Used Abbreviations",
]

# Map each section to keywords that should appear in Q&A text if covered
SECTION_QA_KEYWORDS = {
    "Prescriber Highlights":                    ["highlight", "prescriber"],
    "Uses/Indications":                         ["indicat", "approved", "used for"],
    "Uses":                                     ["indicat", "approved", "used for"],
    "Indications":                              ["indicat", "approved", "used for"],
    "Pharmacology/Actions":                     ["mechanism", "prodrug", "inhibit", "cox", "receptor"],
    "Pharmacology":                             ["mechanism", "prodrug", "inhibit", "cox", "receptor"],
    "Actions":                                  ["mechanism", "prodrug", "inhibit", "cox", "receptor"],
    "Pharmacokinetics":                         ["absorption", "metabolit", "peak plasma", "clearance", "rectal", "oral"],
    "Contraindications/Precautions/Warnings":   ["contraindic", "hypersensit", "caution", "warning", "precaution"],
    "Contraindications":                        ["contraindic", "hypersensit", "not recommend"],
    "Precautions":                              ["caution", "precaution", "warning", "monitor"],
    "Warnings":                                 ["warning", "caution"],
    "Adverse Effects":                          ["adverse", "side effect", "vomit", "ulcer", "toxic"],
    "Reproductive/Nursing Safety":              ["reproduct", "pregnan", "lactat", "nursing", "offspring", "breeding"],
    "Reproductive":                             ["reproduct", "pregnan", "lactat", "nursing"],
    "Nursing Safety":                           ["lactat", "nursing", "milk"],
    "Overdose/Acute Toxicity":                  ["overdose", "overdos", "toxicit", "acute"],
    "Overdose":                                 ["overdose", "overdos", "toxicit"],
    "Acute Toxicity":                           ["toxicit", "acute", "overdose"],
    "Drug Interactions":                        ["interact", "concurrent", "avoid with"],
    "Laboratory Considerations":                ["laborator", "lab", "interfere", "assay", "test result"],
    "Dosages":                                  ["mg/kg", "mg/dog", "mg/cat", "dose", "dosage"],
    "Dosage":                                   ["mg/kg", "mg/dog", "mg/cat", "dose", "dosage"],
    "Monitoring":                               ["monitor", "cbc", "chemistry", "coagulat", "temperature"],
    "Client Information":                       ["client", "owner", "home", "skin contact", "wash"],
    "Chemistry/Synonyms":                       ["synonym", "trade name", "chemical", "metamizole", "also known"],
    "Chemistry":                                ["synonym", "trade name", "chemical", "also known"],
    "Synonyms":                                 ["synonym", "trade name", "also known"],
    "Storage/Stability":                        ["store", "storage", "room temperature", "light", "refriger"],
    "Storage":                                  ["store", "storage", "room temperature", "light"],
    "Stability":                                ["stability", "expire", "degradat"],
    "Compatibility/Compounding Considerations": ["compound", "compat", "admixture", "fda-approved formul"],
    "Compatibility":                            ["compat", "admixture", "mix"],
    "Compounding Considerations":               ["compound", "fda-approved formul", "compounded"],
    "Dosage Forms/Regulatory Status":           ["dosage form", "mg/ml", "tablet", "capsule", "vial", "arci", "nada", "class 4", "racing", "multi-dose", "500 mg", "regulatory", "approved product"],
    "Dosage Forms":                             ["dosage form", "mg/ml", "tablet", "capsule", "vial", "multi-dose", "500 mg"],
    "Regulatory Status":                        ["arci", "nada", "class 4", "racing", "schedule", "dea", "regulatory"],
    "Veterinary-Labeled Products":              ["veterinary", "labeled", "approved product"],
    "Human-Labeled Products":                   ["human", "labeled", "otc", "prescription"],
    "Controlled Substances":                    ["schedule", "dea", "controlled"],
}

SYSTEM_PROMPT = """You are a veterinary education expert creating Q&A pairs from Plumb's Veterinary Drug Handbook (2023).

Generate Q&A pairs covering every section. Aim for:
- Roughly equal factual and explanatory counts (aim for 1:1 ratio — for every factual Q&A, write one explanatory)
- EXACTLY 8 MCQs at the end
There is no maximum limit — generate as many as needed to cover all sections.

RULES:
1. Cover every section present in the drug text. Common sections include:
   Prescriber Highlights, Uses/Indications, Pharmacology/Actions,
   Pharmacokinetics, Contraindications/Precautions/Warnings, Adverse Effects,
   Reproductive/Nursing Safety, Overdose/Acute Toxicity, Drug Interactions,
   Laboratory Considerations, Dosages, Monitoring, Client Information,
   Chemistry/Synonyms, Storage/Stability, Compatibility/Compounding,
   Dosage Forms/Regulatory Status.
   - Always start with Prescriber Highlights — must NEVER be skipped.
   - Reproductive/Nursing Safety must NEVER be skipped if present.
   - Contraindications must NEVER be skipped if present.
   - Chemistry/Synonyms must NEVER be skipped if present.
   - Compounding/Compatibility must NEVER be skipped if present.

2. MCQs must:
   - Have EXACTLY 4 options (A, B, C, D) — all clinically plausible
   - Have ONLY ONE correct answer — verify no other option is also correct
   - End with: "Correct: X) brief explanation"
   - Test clinical decisions, NOT pharmacokinetic numbers
   - NEVER use "All of the above" or "None of the above" as an option
   - Count MCQs before finishing — remove extras if more than 8

3. Answers MUST provide full context in complete sentences. For factual and dosage questions, repeat the core premise of the question in the answer so the model learns the full association.

4. FORMATTING DOSAGE & NUMERICAL Q&A: When providing a dosage or specific numerical value, REPEAT the exact number multiple times in the answer to reinforce it. 
   - Example (BAD): "12.5 – 25 mg/dog PO twice daily"
   - Example (GOOD): "The dosage of Acarbose for dogs is 12.5 – 25 mg/dog PO twice daily. Specifically, 12.5 – 25 mg/dog is given with each meal."
   - Preserve exact dosages as written (e.g. "12.5 mg/dog NOT mg/kg PO BID").

5. Separate dogs, cats, horses when data differs.

6. Do NOT generate questions about: pKa, molecular weight,
   bioavailability %, protein binding %, half-life values,
   volume of distribution.

7. Do NOT add Q: or A: prefixes anywhere.

8. Do NOT invent information not in the drug text.

9. CREATE THE QUESTIONS FROM PDF WHICH CAN BE ANSWERED FROM THE PDF TEXT.
   IF A QUESTION CANNOT BE ANSWERED FROM THE PDF TEXT, THEN DO NOT CREATE THAT QUESTION.


OUTPUT: Return ONLY valid JSON, no markdown fences, no extra text.

{
  "drug_name": "DrugName",
  "qa_pairs": [
    {"type": "factual", "question": "...", "answer": "..."},
    {"type": "explanatory", "question": "...", "answer": "..."},
    {
      "type": "mcq",
      "question": "...",
      "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
      "answer": "Correct: X) explanation"
    }
  ]
}

- Return ONLY the JSON object — no intro text, no markdown code fences
- All 8 MCQs must have the "options" field with A, B, C, D keys
- Factual and explanatory types do NOT have an "options" field
- The "type" field must be exactly: "factual", "explanatory", or "mcq"
"""

VERIFIER_PROMPT = """You are a strict veterinary Q&A quality checker.

You will receive:
1. The original drug text (source of truth)
2. The generated Q&A JSON
3. A list of issues found by automated checks

Your job is to fix ALL listed issues and return the corrected JSON.

FIX THESE ISSUES IN ORDER:

1. MCQ COUNT — must be EXACTLY 8:
   - If more than 8: remove the weakest (keep most clinically useful)
   - If fewer than 8: generate new MCQs from uncovered sections

2. MISSING SECTIONS — for each section listed as missing:
   - Generate 1-2 new Q&As (factual or explanatory) covering that section
   - Add them before the MCQs in qa_pairs

3. MCQ QUALITY:
   - Replace any MCQ using "All of the above" or "None of the above"
   - Ensure ONLY ONE option is correct per MCQ
   - Every MCQ answer must end with: "Correct: X) brief explanation"

4. ANSWER QUALITY:
   - Remove any Q:/A: prefixes
   - Fix citation number leaks (e.g. "prior to2" -> "prior to")

5. TYPE FIELD — must be exactly "factual", "explanatory", or "mcq"

6. VERBOSE ANSWERS — if any answer exceeds 60 words, shorten it to 1-3 clear sentences without losing clinical meaning.

7. IF A ANSWERED IS NOT JUSTIFY BY THE PDF TEXT,
 THEN REMOVE THAT QUESTION AND ANSWER PAIRS
 (JUST REMOVE THAT QUESTION AND ANSWER PAIRS, DO NOT TOUCH OTHER QUESTION AND ANSWER PAIRS).    

8. IF ANSWER LIKE:
    1. This question is not answered in the provided text.
    OR
    2. In provided text there is no information about this question.
    OR
    3. answer is not given in the provided text.
    OR
    something like this then remove that question & answer pairs. 
    (JUST REMOVE THAT QUESTION AND ANSWER PAIRS, DO NOT TOUCH OTHER QUESTION AND ANSWER PAIRS).

CRITICAL RULES — READ BEFORE DOING ANYTHING:
- The input JSON has N qa_pairs. Your output MUST contain AT LEAST N qa_pairs.
- NEVER delete, rewrite, or replace any existing qa_pair unless it has a Q:/A: prefix or citation leak.
- ONLY ADD new qa_pairs to fix missing sections. Never remove correct ones.
- If MCQ count > 8: remove ONLY the extra MCQs (keep all non-MCQ pairs untouched).
- If MCQ count < 8: ADD new MCQs at the end. Never touch existing ones.

Return the COMPLETE corrected JSON with ALL qa_pairs (existing + newly added).
Return ONLY valid JSON — no markdown fences, no extra text.
"""

# Locks for thread-safe operations
save_lock = threading.Lock()
stats_lock = threading.Lock()
print_lock = threading.Lock()

def clean_mdx_text(text: str) -> str:
    """Clean citation artifacts from text — does NOT touch the original file."""
    text = re.sub(r"[¹²³⁴⁵⁶⁷⁸⁹⁰]", "", text)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"([a-zA-Z])\d+(?=\s|,|\.|$)", r"\1", text)
    text = re.sub(r'\n#+\s*References.*?(?=\n## |\Z)', '', text, flags=re.DOTALL)
    return text.strip()

def split_mdx_by_drug(mdx_text: str):
    """Split MDX into individual drug sections by ## headings."""
    parts = re.split(r'\n(?=## )', mdx_text)
    drugs = []
    for part in parts:
        part = part.strip()
        if not part or not part.startswith('## '):
            continue
        first_line = part.split('\n')[0]
        drug_name  = re.sub(r'^##\s*', '', first_line).strip()
        drugs.append({'name': drug_name, 'content': part})
    return drugs

def get_sections_in_drug(drug_content: str) -> list[str]:
    """Detect presence of ### Section Header lines."""
    found = []
    for header in SECTION_HEADERS:
        pattern = rf'###\s*{re.escape(header)}\s*$'
        if re.search(pattern, drug_content, re.MULTILINE | re.IGNORECASE):
            found.append(header)
    return found

def get_missing_sections(drug_content: str, qa_json: dict) -> list[str]:
    """Find present sections with zero coverage in Q&As."""
    present_sections = get_sections_in_drug(drug_content)
    all_qa_text = " ".join(
        q.get("question", "").lower() + " " + q.get("answer", "").lower()
        for q in qa_json.get("qa_pairs", [])
    )
    missing = []
    for section in present_sections:
        if section == "References": continue
        keywords = SECTION_QA_KEYWORDS.get(section)
        if not keywords: continue
        if not any(kw.lower() in all_qa_text for kw in keywords):
            missing.append(section)
    return missing

def extract_json(content: str) -> dict:
    """Robustly extract JSON from model response."""
    content = re.sub(r'^```json\s*', '', content.strip())
    content = re.sub(r'\s*```$', '', content.strip())
    try: return json.loads(content)
    except: pass
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match: return json.loads(match.group(0))
    raise ValueError("No valid JSON found in response")

def call_generator(drug_content: str, drug_name: str):
    """Call generator LLM."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": drug_content}]
    for attempt in range(5):
        try:
            response = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.0)
            content = response.choices[0].message.content
            if not content: raise ValueError("Empty response")
            return extract_json(content), response.usage
        except Exception as e:
            time.sleep(2 * (attempt + 1))
    return None, None

def call_verifier(drug_content: str, qa_json: dict, issues: dict, drug_name: str):
    """Call verifier LLM."""
    issues_text = "\n".join(f"- {k}: {v}" for k, v in issues.items())
    user_content = f"ISSUES TO FIX:\n{issues_text}\n\nDRUG TEXT:\n{drug_content}\n\nJSON TO FIX:\n{json.dumps(qa_json, indent=2)}"
    messages = [{"role": "system", "content": VERIFIER_PROMPT}, {"role": "user", "content": user_content}]
    for attempt in range(3):
        try:
            response = client.chat.completions.create(model=VERIFY_MODEL, messages=messages, temperature=0.0)
            content = response.choices[0].message.content
            if not content: raise ValueError("Empty response")
            return extract_json(content), response.usage
        except Exception as e:
            time.sleep(2 * (attempt + 1))
    return qa_json, None

def quick_check(qa_json: dict, drug_content: str = "") -> dict:
    """Fast pre-check for common issues."""
    pairs = qa_json.get("qa_pairs", [])
    mcqs  = [q for q in pairs if q.get("type") == "mcq"]
    issues = {}
    if len(mcqs) != 8: issues["mcq_count"] = f"{len(mcqs)} MCQs (need 8)"
    if any("all of the above" in str(q).lower() for q in mcqs): issues["all_of_above"] = "MCQ uses 'All of the above'"
    if any(q.get("question","").startswith("Q:") for q in pairs): issues["prefix"] = "Found Q:/A: prefixes"
    if any(re.search(r'[a-zA-Z]\d(?=\s|,|\.|$)', q.get("question","")+q.get("answer","")) for q in pairs): issues["citations"] = "Citation leaks"
    if any("Correct:" not in q.get("answer","") for q in mcqs): issues["mcq_answer"] = "MCQ missing 'Correct:'"
    if any(len(q.get("answer","").split()) > 60 for q in pairs): issues["verbose"] = "Answer too long"
    
    non_mcq = [q for q in pairs if q.get("type") != "mcq"]
    exp_count = len([q for q in non_mcq if q.get("type") == "explanatory"])
    if len(non_mcq) > 0 and exp_count == 0: issues["no_explanatory"] = "Missing explanatory Q&As"
    
    if drug_content:
        missing = get_missing_sections(drug_content, qa_json)
        if missing: issues["missing_sections"] = f"Missing: {', '.join(missing)}"
    return issues

def process_single_drug(drug, i, total_drugs, all_qa, failed, token_usage, completed_drug_names, start_time_global):
    drug_name = drug["name"]
    
    with stats_lock:
        if drug_name in completed_drug_names:
            with print_lock: print(f"[{i+1:03d}/{total_drugs}] ⏩ {drug_name} — already done")
            return

    with print_lock: print(f"[{i+1:03d}/{total_drugs}] ⚙️  Processing {drug_name}...")

    # Step 1: Generate
    result, usage = call_generator(drug["content"], drug_name)
    if not result:
        with stats_lock: failed.append(drug_name)
        with print_lock: print(f"     ❌ {drug_name}: Generation failed")
        return

    with stats_lock:
        token_usage["generator_in"] += usage.prompt_tokens
        token_usage["generator_out"] += usage.completion_tokens
        token_usage["generator_total"] += usage.total_tokens

    # Step 2: Verify & Fix
    issues = quick_check(result, drug["content"])
    if issues:
        for _ in range(3): # Max 3 fix attempts
            fixed, v_usage = call_verifier(drug["content"], result, issues, drug_name)
            if v_usage:
                with stats_lock:
                    token_usage["verifier_in"] += v_usage.prompt_tokens
                    token_usage["verifier_out"] += v_usage.completion_tokens
                    token_usage["verifier_total"] += v_usage.total_tokens
            result = fixed
            issues = quick_check(result, drug["content"])
            if not issues: break

    # Step 3: Save
    result["drug_name"] = drug_name
    final_count = len(result.get("qa_pairs", []))
    
    with save_lock:
        all_qa.append(result)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(all_qa, f, indent=4, ensure_ascii=False)
    
    with stats_lock: completed_drug_names.add(drug_name)
    
    with print_lock:
        mcqs = [q for q in result.get("qa_pairs", []) if q.get("type") == "mcq"]
        print(f"     ✅ {drug_name}: {final_count} Q&As ({len(mcqs)} MCQs) | Total drugs: {len(all_qa)}")

def main():
    start_time = datetime.now()
    token_usage = {"generator_in": 0, "generator_out": 0, "generator_total": 0, "verifier_in": 0, "verifier_out": 0, "verifier_total": 0}

    print(f"📄 Reading: {MDX_FILE}")
    with open(MDX_FILE, "r", encoding="utf-8") as f:
        mdx_text = clean_mdx_text(f.read())

    drugs = split_mdx_by_drug(mdx_text)
    total_drugs = len(drugs)
    print(f"✅ Found {total_drugs} drugs\n")

    all_qa = []
    failed = []
    completed_drug_names = set()

    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                all_qa = json.load(f)
            for d in all_qa: completed_drug_names.add(d.get("drug_name"))
            print(f"🔄 Resuming... {len(all_qa)} drugs already done\n")
        except: print(f"⚠️  {OUTPUT_FILE} corrupted — starting fresh.\n")
    else: print("🆕 Starting fresh.\n")

    # Concurrent Execution
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i, drug in enumerate(drugs):
            executor.submit(process_single_drug, drug, i, total_drugs, all_qa, failed, token_usage, completed_drug_names, start_time)

    end_time = datetime.now()
    elapsed = end_time - start_time

    # Final Summary and Log
    print(f"\n{'='*55}\n✅ DONE! Total Q&As: {sum(len(d['qa_pairs']) for d in all_qa):,}\nTime: {elapsed}\n{'='*55}")
    
    if not os.path.exists("token_usage.csv"):
        with open("token_usage.csv", "w", newline="") as f:
            csv.writer(f).writerow(["Total QA", "Total Drugs", "Passed", "Failed", "Model", "Gen In", "Gen Out", "Ver In", "Ver Out", "Total Time"])

    with open("token_usage.csv", "a", newline="") as f:
        csv.writer(f).writerow([sum(len(d['qa_pairs']) for d in all_qa), total_drugs, len(all_qa), len(failed), MODEL, token_usage["generator_in"], token_usage["generator_out"], token_usage["verifier_in"], token_usage["verifier_out"], elapsed])

if __name__ == "__main__":
    main()
