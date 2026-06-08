import os
import re
from collections import defaultdict

BASE_DIR = os.path.expanduser("~/Desktop/patent-intelligence")

INPUT_FOLDER = os.path.join(BASE_DIR, "extracted_text")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "chunks")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_file(folder, filename, content):
    content = content.strip()
    if len(content) < 20:
        return
    path = os.path.join(folder, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def normalize_claim_numbers(text):
    """
    OCR sometimes adds a space inside claim numbers: "1 . A method" → "1. A method"
    Normalizes "N . " patterns at the start of lines.
    """
    # "  1 . A"  →  "1. A"
    text = re.sub(r'(?m)^([ \t]*)(\d{1,3})\s+\.\s+', r'\1\2. ', text)
    return text


def deinterleave_two_column_claims(section_text):
    """
    Two-column OCR interleaves left-column and right-column text line-by-line.
    This means a line can contain text from *two* different claims side-by-side,
    or a new claim number can appear embedded partway through a block.

    Heuristic: scan the claims section for embedded claim-start patterns
    (e.g. a line containing "25 the platform \n 3. The method of claim") and
    split them out so each claim starts on its own line.

    We look for patterns like:
      - Digit(s) at end of one claim line followed by next claim number
      - A claim number appearing after a column-index digit ("30 1. A method")
    """
    # Pattern: optional leading column number like "25 " or "30 ", then a claim number
    # e.g. "30 past candidates .  " followed on next token by "1. A method"
    # We split lines that contain an embedded claim start mid-line

    embedded_re = re.compile(
        r'(\d{1,3}\.\s+(?:A|An|The|system|method|computer|non)\s)',
        re.IGNORECASE
    )

    output_lines = []
    for line in section_text.splitlines():
        # Check if this line contains an embedded claim number mid-line
        # (after at least 10 chars of prior text)
        m = embedded_re.search(line, 10)  # skip first 10 chars
        if m:
            # Split at the embedded claim number
            before = line[:m.start()].rstrip()
            after = line[m.start():].strip()
            if before:
                output_lines.append(before)
            output_lines.append(after)
        else:
            output_lines.append(line)

    return "\n".join(output_lines)


def clean_text(text):
    """Light cleanup: normalize, collapse excessive blank lines, strip repeated headers."""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Fix OCR-spaced claim numbers before any other processing
    text = normalize_claim_numbers(text)

    # Remove repeated patent page headers like "US 10,528,916 B1" / "US 10528916B1"
    text = re.sub(r'\bUS\s*[\d,]+\s*[AB]\d\b', '', text, flags=re.IGNORECASE)

    # Remove bare page-number lines (e.g. "Page 2", lone digits on a line)
    text = re.sub(r'(?m)^[ \t]*\d{1,3}[ \t]*$', '', text)

    # Collapse 3+ blank lines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text


# ---------------------------------------------------------------------------
# Abstract extraction
# ---------------------------------------------------------------------------

def extract_abstract(text):
    """
    Find the ABSTRACT section and stop at the first hard boundary.
    Returns the abstract string (may be empty).
    """
    m = re.search(r'\bABSTRACT\b', text, re.IGNORECASE)
    if not m:
        return ""

    after = text[m.end():]

    end_pattern = re.compile(
        r'(?:'
        r'\d+\s+Claims'
        r'|Drawing\s+Sheets?'
        r'|What\s+is\s+claimed'
        r'|The\s+invention\s+claimed'
        r'|claims\s+conclude\s+the\s+specification'
        r'|^\s*CLAIMS\s*$'
        r'|^\s*(?:BACKGROUND|BRIEF\s+DESCRIPTION|DETAILED\s+DESCRIPTION|SUMMARY)\b'
        # Patent application metadata blocks — appear right after abstract on first page
        r'|(?:\(\s*21\s*\)|\(\s*22\s*\)|\(\s*51\s*\))\s*(?:Appl|Int|Filed)'
        r'|(?:Sheet\s+\d+\s+of\s+\d+)'
        r'|(?:Patent\s+Application\s+Publication)'
        r'|(?:^\s*FIGURE\s+\d+|^\s*FIG\s*\.\s*\d+)'
        r')',
        re.IGNORECASE | re.MULTILINE
    )

    end_m = end_pattern.search(after)
    if end_m:
        abstract = after[:end_m.start()]
    else:
        abstract = after[:2000]  # Hard cap to avoid pulling in entire body

    # Remove OCR noise lines
    lines = abstract.splitlines()
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            clean_lines.append("")
            continue
        alpha_ratio = sum(c.isalpha() for c in stripped) / max(len(stripped), 1)
        if len(stripped) < 6 and alpha_ratio < 0.5:
            continue
        clean_lines.append(line)

    return "\n".join(clean_lines).strip()


# ---------------------------------------------------------------------------
# Claims section location
# ---------------------------------------------------------------------------

def find_claims_section(text):
    """
    Locate where the claims section starts, trying multiple header patterns.
    Returns (header_found: bool, section_text: str).
    """
    # Ordered by specificity — most unambiguous patterns first
    patterns = [
        # "What is claimed is:" / "What is claimed as new and desired..."
        r'What\s+is\s+claimed(?:\s+as\s+new[^:\n]*)?\s*(?:is\s*)?[:\s]+',
        # "The invention claimed is:"
        r'The\s+invention\s+claimed\s+is\s*:\s*',
        # "the following claims conclude the specification." → claims start after next blank line
        r'following\s+claims\s+conclude\s+the\s+specification\.?\s*\n',
        # Standalone "CLAIMS" heading on its own line
        r'(?m)^[ \t]*CLAIMS[ \t]*\n',
    ]

    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return True, text[m.end():]

    return False, ""


def fallback_find_claims(text):
    """
    When no explicit claims header is found, try to locate the start of the
    claims block by finding the first occurrence of a claim-like pattern
    ("N. A method comprising" / "N. The method of claim M") that is followed
    by several more such patterns within a short window — suggesting we've
    entered the claims section.

    Returns section text from the earliest reliable claim start, or ''.
    """
    claim_re = re.compile(
        r'(?m)^[ \t]*(\d{1,3})\.\s+(?:A|An|The)\s+\w',
    )

    matches = list(claim_re.finditer(text))
    if not matches:
        return ""

    # Find the first match whose number is 1 or 2, followed by closely-spaced
    # additional claim-like matches
    for i, m in enumerate(matches):
        num = int(m.group(1))
        if num > 3:
            continue
        # Check that there are at least 3 more claim-like matches after this
        if i + 3 < len(matches):
            # Check they are all within 5000 chars of each other
            subsequent = matches[i + 1 : i + 4]
            if subsequent[-1].start() - m.start() < 8000:
                return text[m.start():]

    return ""


# ---------------------------------------------------------------------------
# Claim parsing
# ---------------------------------------------------------------------------

def _is_valid_claim_line(number, following_text):
    """
    Heuristic: does this look like a real claim start?
    """
    first_word_m = re.match(r'\s*([A-Za-z]\w*)', following_text)
    if not first_word_m:
        return False

    first_word = first_word_m.group(1).lower()

    # Suspicious very-short starters in isolation (figure labels, table refs)
    bad_starters = {'fig', 'figure', 'sheet', 'page', 'no', 'eq', 'equation', 'table'}
    if first_word in bad_starters:
        return False

    # Numbers out of plausible claim range
    if number < 1 or number > 200:
        return False

    return True


def parse_claims(claims_section):
    """
    Parse individual claims from the claims section text.
    Returns a list of dicts: [{number, text, is_independent, second_word}, ...]
    """
    if not claims_section:
        return []

    # Try to split interleaved two-column text before line-parsing
    claims_section = deinterleave_two_column_claims(claims_section)

    lines = claims_section.splitlines()

    # Match claim-start lines: "N. text" or leading whitespace then "N. text"
    claim_start_re = re.compile(r'^[ \t]*(\d{1,3})\.[ \t]+(.+)$')

    # First pass: collect candidate (line_index, number, rest_of_line)
    candidates = []
    for i, line in enumerate(lines):
        m = claim_start_re.match(line)
        if m:
            number = int(m.group(1))
            rest = m.group(2)
            if _is_valid_claim_line(number, rest):
                candidates.append((i, number, rest))

    if not candidates:
        return []

    # Filter: collect by claim number (keep first occurrence of each), then sort
    # This handles two-column OCR where claims may appear out of order on the page.
    seen_nums = {}
    for (idx, num, rest) in candidates:
        if num not in seen_nums:
            seen_nums[num] = (idx, num, rest)

    # Sort by claim number
    all_by_num = sorted(seen_nums.values(), key=lambda x: x[1])

    # Find the start of the "real" claim run: first sequential block starting <= 3
    accepted = []
    for (idx, num, rest) in all_by_num:
        if not accepted:
            if 1 <= num <= 5:
                accepted.append((idx, num, rest))
        else:
            last_num = accepted[-1][1]
            if 1 <= num - last_num <= 3:
                accepted.append((idx, num, rest))
            elif num - last_num > 10:
                break  # Probably left the claims section

    if not accepted:
        accepted = all_by_num  # Fallback

    # Second pass: extract text blocks for each claim
    claims = []
    for i, (line_idx, num, first_rest) in enumerate(accepted):
        next_line_idx = accepted[i + 1][0] if i + 1 < len(accepted) else len(lines)

        body_lines = [first_rest.strip()]
        for j in range(line_idx + 1, next_line_idx):
            body_lines.append(lines[j])

        raw_body = "\n".join(body_lines)

        # Remove pure-noise lines (only digits/symbols, very short)
        clean_body_lines = []
        for line in raw_body.splitlines():
            stripped = line.strip()
            if not stripped:
                clean_body_lines.append("")
                continue
            if re.match(r'^[\d\s\.\,\;\:\*\(\)\/\\\|\-\_]+$', stripped) and len(stripped) < 8:
                continue
            clean_body_lines.append(line)

        body = "\n".join(clean_body_lines).strip()
        body = re.sub(r'\n{3,}', '\n\n', body)

        # Independent vs dependent
        is_dependent = bool(re.search(
            r'\bof claim\s+\d+\b|\bclaims?\s+\d+\b',
            body, re.IGNORECASE
        ))
        is_independent = not is_dependent

        # Second word (for grouping independent claims by type)
        words = re.findall(r'\b[A-Za-z]+\b', body)
        second_word = words[1].lower() if len(words) >= 2 else ""

        claims.append({
            "number": num,
            "text": body,
            "is_independent": is_independent,
            "second_word": second_word,
        })

    return claims


def format_claim(claim):
    """Format a claim dict into a readable string."""
    tag = "[INDEPENDENT]" if claim["is_independent"] else "[DEPENDENT]"
    return f"Claim {claim['number']} {tag}\n\n{claim['text']}"


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

for file in os.listdir(INPUT_FOLDER):

    if not file.endswith(".txt"):
        continue

    filepath = os.path.join(INPUT_FOLDER, file)

    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()

    text = clean_text(raw_text)

    # ---------------------------------
    # Patent ID
    # ---------------------------------

    patent_id_match = re.search(r'US\s*[\d,]+\s*[AB]\d', raw_text)
    if patent_id_match:
        patent_id = re.sub(r'[\s,]', '', patent_id_match.group(0))
    else:
        patent_id = file.replace(".txt", "")

    patent_folder = os.path.join(OUTPUT_FOLDER, patent_id)
    os.makedirs(patent_folder, exist_ok=True)

    # ---------------------------------
    # Abstract
    # ---------------------------------

    abstract = extract_abstract(text)
    save_file(patent_folder, "abstract.txt", abstract)

    # ---------------------------------
    # Claims
    # ---------------------------------

    header_found, claims_section_text = find_claims_section(text)

    if not header_found or not claims_section_text.strip():
        # Fallback: try to find claims without an explicit header
        claims_section_text = fallback_find_claims(text)

    claims = parse_claims(claims_section_text)

    independent_claims = [c for c in claims if c["is_independent"]]
    dependent_claims   = [c for c in claims if not c["is_independent"]]

    # ---------------------------------
    # Save claim files
    # ---------------------------------

    # Claim 1
    claim_1 = next((c for c in claims if c["number"] == 1), None)
    if claim_1 is None and claims:
        claim_1 = claims[0]

    if claim_1:
        save_file(patent_folder, "claim_1.txt", format_claim(claim_1))

    # All independent claims
    if independent_claims:
        indie_sep = "\n\n" + "="*60 + "\n\n"
        indie_content = indie_sep.join(format_claim(c) for c in independent_claims)
        save_file(patent_folder, "independent_claims.txt", indie_content)

    # Full claims
    if claims:
        full_sep = "\n\n" + "-"*40 + "\n\n"
        full_content = full_sep.join(format_claim(c) for c in claims)
        save_file(patent_folder, "full_claims.txt", full_content)

    # Key claims (first 5)
    if claims:
        key_sep = "\n\n" + "-"*40 + "\n\n"
        key_content = key_sep.join(format_claim(c) for c in claims[:5])
        save_file(patent_folder, "key_claims.txt", key_content)

    # ---------------------------------
    # Invention Summary
    # — Abstract (plain language) + Independent Claims (broadest legal scope)
    # ---------------------------------

    summary_parts = []

    if abstract:
        summary_parts.append(
            "ABSTRACT\n"
            "(Plain-language overview of the invention)\n\n"
            + abstract
        )

    if independent_claims:
        indie_header = (
            "INDEPENDENT CLAIMS\n"
            "(Broadest legal scope — each stands alone, not referencing a prior claim)\n\n"
        )
        indie_body = ("\n\n" + "─"*50 + "\n\n").join(format_claim(c) for c in independent_claims)
        summary_parts.append(indie_header + indie_body)

    if dependent_claims:
        summary_parts.append(
            f"DEPENDENT CLAIMS SUMMARY\n"
            f"({len(dependent_claims)} dependent claim(s) narrow the independent ones — "
            f"see full_claims.txt for all details)"
        )

    invention_summary = ("\n\n" + "="*60 + "\n\n").join(summary_parts)
    save_file(patent_folder, "invention_summary.txt", invention_summary)

    # ---------------------------------
    # Metadata
    # ---------------------------------

    indie_groups = {}
    for c in independent_claims:
        sw = c["second_word"] or "(unknown)"
        indie_groups.setdefault(sw, []).append(c["number"])

    groups_str = "\n".join(
        f"  '{sw}' → claims {nums}"
        for sw, nums in indie_groups.items()
    )

    metadata = (
        f"Patent ID: {patent_id}\n\n"
        f"Source File: {file}\n\n"
        f"Claims Header Found: {'Yes' if header_found else 'No (fallback used)'}\n\n"
        f"Total Claims Found:    {len(claims)}\n"
        f"Independent Claims:    {len(independent_claims)}\n"
        f"Dependent Claims:      {len(dependent_claims)}\n\n"
        f"Independent Claim Groups (by second word):\n"
        f"{groups_str if groups_str else '  (none detected)'}\n"
    )

    save_file(patent_folder, "metadata.txt", metadata)

    print(
        f"Processed {patent_id} | "
        f"Total: {len(claims)} | "
        f"Indep: {len(independent_claims)} | "
        f"Dep: {len(dependent_claims)}"
        + (" [fallback]" if not header_found else "")
    )

print("\nDone.")