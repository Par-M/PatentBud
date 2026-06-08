import fitz
import os
import subprocess
import re

PATENT_FOLDER = "patents"
OUTPUT_FOLDER = "extracted_text"
SWIFT_OCR_SCRIPT = "ocr_pdf.swift"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for filename in os.listdir(PATENT_FOLDER):
    if filename.endswith(".pdf"):

        filepath = os.path.join(PATENT_FOLDER, filename)
        txt_name = filename.replace(".pdf", ".txt")
        txt_path = os.path.join(OUTPUT_FOLDER, txt_name)

        # 1. Try standard text extraction
        doc = fitz.open(filepath)
        full_text = ""
        for page in doc:
            full_text += page.get_text()

        # Check if there is any alphanumeric text (excluding spaces and control chars)
        has_readable_text = bool(re.search(r'[a-zA-Z0-9]', full_text))

        if has_readable_text:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(full_text)
            print(f"Saved {txt_name} (Standard extraction)")
        else:
            print(f"No readable text layer in {filename}. Falling back to Swift Vision OCR...")
            
            # Execute swift script
            result = subprocess.run(
                ["swift", SWIFT_OCR_SCRIPT, filepath, txt_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"Saved {txt_name} (Swift OCR)")
            else:
                print(f"Error running OCR on {filename}: {result.stderr}")