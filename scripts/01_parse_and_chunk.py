"""
Step 1
=======
indexing the whole document into chunks 
using strategy RecursiveCharacterTextSplitter
because it tries natural language boundaries (\n\n paragraphs, \n lines, words) before falling back to character splits. 
This preserves sentence integrity. 
I use 500 character chunks with 50-character overlap to maintain context across chunk boundaries
"""


"""
pdfplumber is a Python library that acts like an inspector:

It opens the PDF drawing instructions.
It looks at the coordinates of every character on the page.
It figures out: "Hey! 'H' and 'e' are right next to each other on the same line (Y=50), so they belong to the word 'He'."
It groups words into lines, and lines into readable text strings.

"""




import re
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
PDF_PATH = "data/sample.pdf"   # <- Change this to your PDF filename
CHUNK_SIZE = 500               # Max characters per chunk
CHUNK_OVERLAP = 50             # Overlap characters between consecutive chunks

# ─────────────────────────────────────────
# PARSING
# ─────────────────────────────────────────
"""
    Opens the PDF and extracts text from each page.
    Returns:
    List of dicts: [{ "page_number": 1, "text": "..." }, ...]
"""

def parse_pdf(file_path: str) -> list[dict]:
    pages = []
    with pdfplumber.open(file_path) as pdf:
        print(f"[INFO] Opened PDF: {file_path}")
        print(f"[INFO] Total pages: {len(pdf.pages)}")
        for i, page in enumerate(pdf.pages):
            raw_text = page.extract_text()
            # Skip pages with no text (scanned/image pages)
            if not raw_text or len(raw_text.strip()) < 20:
                print(f"[WARN] Page {i+1} has no extractable text — may be scanned.")
                continue
            pages.append({
                "page_number": i + 1,
                "text": raw_text
            })
    print(f"[INFO] Successfully extracted text from {len(pages)} pages.")
    return pages


# ─────────────────────────────────────────
# STEP 2: CLEAN TEXT
# ─────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Cleans raw text extracted from PDF.
    Fixes:
    - Hyphenated line breaks: "impor-\\nant" → "important"
    - Multiple spaces: "word   word" → "word word"
    - Weird whitespace
    """
    # Fix hyphenated words broken across lines
    text = re.sub(r'-\n', '', text)
    # Collapse multiple whitespace into single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


# ─────────────────────────────────────────
# STEP 3: CHUNK TEXT
# ─────────────────────────────────────────
def chunk_text(pages: list[dict]) -> list[dict]:
    """
    Takes parsed pages and splits them into overlapping chunks.
    The RecursiveCharacterTextSplitter tries separators in this order:
        1. "\\n\\n"  (paragraph breaks — best case)
        2. "\\n"     (line breaks)
        3. " "       (word boundaries)
        4. ""        (individual characters — last resort)
    Returns:
        List of dicts: [{ "text": "...", "page_number": 1, "chunk_index": 0 }, ...]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    all_chunks = []
    chunk_index = 0
    for page in pages:
        cleaned = clean_text(page["text"])
        # Split this page's text into chunks
        page_chunks = splitter.split_text(cleaned)
        for chunk_text_content in page_chunks:
            all_chunks.append({
                "text": chunk_text_content,
                "page_number": page["page_number"],
                "chunk_index": chunk_index
            })
            chunk_index += 1
    return all_chunks


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    # Parse
    pages = parse_pdf(PDF_PATH)
    # Chunk
    chunks = chunk_text(pages)
    # Print results
    print("\n" + "="*60)
    print(f"CHUNKING COMPLETE")
    print("="*60)
    print(f"Total chunks created : {len(chunks)}")
    print(f"Chunk size setting   : {CHUNK_SIZE} chars")
    print(f"Chunk overlap setting: {CHUNK_OVERLAP} chars")
    print(f"Avg chunk length     : {sum(len(c['text']) for c in chunks) // len(chunks)} chars")
    print("="*60)
    # Preview first 3 chunks
    print("\n--- FIRST 3 CHUNKS ---\n")
    for chunk in chunks[:3]:
        print(f"[Chunk {chunk['chunk_index']} | Page {chunk['page_number']}]")
        print(chunk["text"])
        print(f"Length: {len(chunk['text'])} chars")
        print("-" * 40)