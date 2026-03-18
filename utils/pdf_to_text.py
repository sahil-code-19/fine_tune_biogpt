import pymupdf
import pathlib
import re

# ── CONFIG ───────────────────────────────────────────────────────
PDF_PATH        = "../pdf/James A. Budde_ Dawn M. McCluskey - Plumb's Veterinary Drug Handbook (2023, John Wiley & Sons) (1).pdf"
TXT_OUTPUT      = "../datasets/all_drugs.txt"

# ⚠️ These are HUMAN page numbers (what you see printed on the page)
# The script converts them to 0-based index automatically
DRUG_START_PAGE = 1  # first drug page (human page number)
DRUG_END_PAGE   = 1573  # last drug page  (human page number)
# ─────────────────────────────────────────────────────────────────

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
    "Veterinary-Labled Products",
    "Human-Labled Products",
    "Prescription Requirements",
    "Controlled Substances",
    "Medication Errors and Patient Safety",
    "Commonly Used Abbreviations"
]

# Longest first so "Uses/Indications" matches before "Uses"
SECTION_HEADERS_SORTED = sorted(SECTION_HEADERS, key=len, reverse=True)

COLUMN_SPLIT_X = 300   # left col: x0 < 300, right col: x0 >= 300


def is_section_header(line):
    t = line.strip()
    for h in SECTION_HEADERS_SORTED:
        if t.lower() == h.lower():
            return h
    return None


def is_page_number(line):
    return bool(re.fullmatch(r'\d+', line.strip()))


def get_page_blocks(page):
    """
    Get all text blocks from a page sorted into:
    left column (top→bottom) then right column (top→bottom)
    Skips running headers (y0 < 70) and footers (y0 > height-50)
    Returns list of tuples: (block_text, is_drug_header)
    """
    dicts       = page.get_text("dict")["blocks"]
    page_height = page.rect.height
    left        = []
    right       = []

    # 1. Extract tables first
    tables = page.find_tables()
    table_bboxes = []
    
    if tables and tables.tables:
        for tab in tables:
            bbox = tab.bbox
            table_bboxes.append(bbox)
            
            # Determine column based on center X
            x0, y0, x1, y1 = bbox
            cx = (x0 + x1) / 2
            
            # Convert table to simple sentences
            try:
                table_data = tab.extract()
                sentences = []
                for row in table_data:
                    valid_cells = [str(c).replace('\n', ' ').replace("Â®", "").replace("®", "").replace("©", "").strip() for c in row if c is not None and str(c).strip()]
                    if valid_cells:
                        # Join columns with a separator like ' - '
                        sentences.append(" - ".join(valid_cells) + ".")
                raw_text = " ".join(sentences) + "\n"
            except Exception:
                raw_text = ""
                
            if cx < COLUMN_SPLIT_X:
                left.append((y0, raw_text, False))
            else:
                right.append((y0, raw_text, False))

    # Helper to check if a point is inside any table's bbox
    def in_table(x, y):
        for (tx0, ty0, tx1, ty1) in table_bboxes:
            if tx0 <= x <= tx1 and ty0 <= y <= ty1:
                return True
        return False

    for b in dicts:
        if "lines" not in b:
            continue
        x0, y0, x1, y1 = b["bbox"]
        if y0 < 70:               # skip running page header
            continue
        if y0 > page_height - 50: # skip footer
            continue
            
        block_text = ""
        is_drug_header = False
        
        has_text = False
        for l in b["lines"]:
            lx0, ly0, lx1, ly1 = l["bbox"]
            lcx, lcy = (lx0 + lx1) / 2, (ly0 + ly1) / 2
            
            # If the line's center is inside a table, skip it
            # (as it was already extracted by find_tables)
            if in_table(lcx, lcy):
                continue
                
            for s in l["spans"]:
                span_text = s["text"]
                block_text += span_text
                has_text = True
                # Large Black font indicates a primary drug heading
                if s["size"] >= 15.5 and "Black" in s["font"]:
                    is_drug_header = True
            block_text += "\n"
            
        block_text = block_text.strip()
        if not block_text or not has_text:
            continue

        if x0 < COLUMN_SPLIT_X:
            left.append((y0, block_text, is_drug_header))
        else:
            right.append((y0, block_text, is_drug_header))

    left.sort(key=lambda b: b[0])
    right.sort(key=lambda b: b[0])

    # Return left column first, then right
    return [(t, is_dh) for _, t, is_dh in left] + [(t, is_dh) for _, t, is_dh in right]


# ── State machine ─────────────────────────────────────────────────

class DrugStateMachine:
    def __init__(self):
        self.current_drug     = None
        self.current_section  = None
        self.drug_header      = []   # pronunciation, brand name, class
        self.section_content  = {}   # {section: [lines]}
        self.section_order    = []
        self.all_drugs        = []

    def _save_current(self):
        if self.current_drug:
            self.all_drugs.append({
                'name':    self.current_drug,
                'header':  list(self.drug_header),
                'order':   list(self.section_order),
                'content': {k: list(v) for k, v in self.section_content.items()},
            })
            print(f"   [Saved] {self.current_drug} ({len(self.section_order)} sections)")

    def new_drug(self, name):
        self._save_current()
        self.current_drug    = name
        self.current_section = None
        self.drug_header     = []
        self.section_content = {}
        self.section_order   = []
        print(f"   [New drug] {name}")

    def new_section(self, section):
        self.current_section = section
        if section not in self.section_content:
            self.section_content[section] = []
            self.section_order.append(section)

    def add_line(self, line):
        if not self.current_drug:
            return
        if self.current_section:
            self.section_content[self.current_section].append(line)
        else:
            self.drug_header.append(line)

    def finalize(self):
        self._save_current()


def format_drug(drug):
    lines = [f"{drug['name']}", ""]

    def clean_and_join(items):
        # Join into one big string
        text = " ".join(items)
        # Fix hyphenated words split across lines: func- tion -> function
        text = re.sub(r'([a-zA-Z]+)-\s+([a-zA-Z]+)', r'\1\2', text)
        # Clean up the word "References" if it got attached to the end
        text = re.sub(r'\s*References\s*$', '', text, flags=re.IGNORECASE)
        # Clean multiple spaces and ALL internal newlines
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    # Header
    header = clean_and_join(drug['header'])
    if header:
        lines.append(header)
        lines.append("")

    # Sections
    for section in drug['order']:
        if section.lower() == "references":
            continue
        lines.append(f"{section}")
        content = clean_and_join(drug['content'].get(section, []))
        if content:
            lines.append(content)
        lines.append("")

    return '\n'.join(lines)


def main():
    # Convert human page numbers → 0-based index
    start_idx = DRUG_START_PAGE - 1
    end_idx   = DRUG_END_PAGE    # range() is exclusive so no -1 needed

    print(f"[Opening] {PDF_PATH}")
    doc   = pymupdf.open(PDF_PATH)
    total = len(doc)
    end_idx = min(end_idx, total)

    print(f"[Pages] Human pages {DRUG_START_PAGE}–{DRUG_END_PAGE}")
    print(f"   0-based index {start_idx}–{end_idx - 1}")
    print(f"   Total PDF pages: {total}\n")

    machine = DrugStateMachine()

    for page_num in range(start_idx, end_idx):
        if (page_num - start_idx) % 100 == 0:
            pct = round((page_num - start_idx) / (end_idx - start_idx) * 100)
            print(f"\n  [Page] Page {page_num + 1} | {pct}% | {len(machine.all_drugs)} drugs so far...")

        try:
            page   = doc[page_num]
            blocks = get_page_blocks(page)
        except Exception as e:
            print(f"   [Error]  Page {page_num + 1} error: {e}")
            continue

        for block_text, is_drug_header in blocks:
            # Clean up text generally: remove broken unicodes
            block_text = block_text.replace("Â®", "").replace("®", "").replace("©", "")
            # Remove bullet point symbols
            block_text = re.sub(r'[▶■•·●◆◦▪]', '', block_text)
            # Remove reference pointers
            block_text = re.sub(r'For the complete list of references.*', '', block_text)
            # Remove pronunciation guides like (ay-kar-bose) but keep (extra-label)
            # Allows spaces and multiple hyphens common in phonetic guides.
            block_text = re.sub(r'\((?![^)]*label)[a-z\s-]+-[a-z\s-]+\)\s*', '', block_text, flags=re.IGNORECASE)
            
            # Remove citation numbers
            # 1. Attached: word1 or word.1 or word,1 (No space allowed)
            # Use two branches: one for punctuated citations, one for attached-digit citations.
            # Attached digits must NOT be followed by a period to avoid breaking lists like '1. Administer'.
            block_text = re.sub(r'(?<=[a-zA-Z\)])(?:([\.,;])\d+(?:[,\-–]\d+)*|(\d+(?:[,\-–]\d+)*)(?=[,;:\s]|$))(?!\.\d)', r'\1', block_text)
            # 2. Separated by space: word. 10 or word, 10
            # Require punctuation AND a space + Capital letter following to ensure it's not a list or part of a value.
            block_text = re.sub(r'(?<=[a-zA-Z\)])([\.,;])\s+\d+(?:[,\-–\s]\d+)*(?=\s+[A-Z]|$)(?!\.\d)', r'\1 ', block_text)
            # 3. Handle citations after decimals (e.g., 6.5.2 -> 6.5)
            block_text = re.sub(r'(\d+\.\d+)\.\d+(?=[,;:\s]|$)', r'\1', block_text)

            if is_drug_header:
                lines = block_text.strip().split('\n')
                drug_name = lines[0].strip()
                if drug_name:
                    machine.new_drug(drug_name)
                    
                current_paragraph = []
                for line in lines[1:]:
                    line = line.strip()
                    if line and not is_page_number(line):
                        current_paragraph.append(line)
                if current_paragraph:
                    text_para = " ".join(current_paragraph).strip()
                    machine.add_line(text_para)
                continue

            current_paragraph = []
            for line in block_text.split('\n'):
                line = line.strip()
                if not line or is_page_number(line):
                    continue

                # Check section header
                section = is_section_header(line)
                if section:
                    if current_paragraph:
                        text_para = " ".join(current_paragraph).strip()
                        machine.add_line(text_para)
                        current_paragraph = []
                    if machine.current_drug:
                        machine.new_section(section)
                    continue

                current_paragraph.append(line)
                
            if current_paragraph:
                text_para = " ".join(current_paragraph).strip()
                machine.add_line(text_para)

    machine.finalize()

    drugs = machine.all_drugs
    print(f"\n[Extracted] {len(drugs)} drugs")

    # ── Build & save TXT ─────────────────────────────────────
    print("[Saving] ...")

    output_parts = []
    for drug in drugs:
        output_parts.append(format_drug(drug))

    final_text = '\n'.join(output_parts)
    # Remove excessive newlines
    final_text = re.sub(r'\n{3,}', '\n\n', final_text)
    
    pathlib.Path(TXT_OUTPUT).write_bytes(final_text.encode('utf-8'))

    size_mb = pathlib.Path(TXT_OUTPUT).stat().st_size / 1024 / 1024
    print(f"\n[Done] {TXT_OUTPUT}")
    print(f"   Size       : {size_mb:.2f} MB")
    print(f"   Total drugs: {len(drugs)}")

if __name__ == "__main__":
    main()



