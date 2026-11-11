"""
BNS Converter & Statutory Transition Layer
==========================================

Acts as the statutory conversion layer after retrieval/reasoning and before
the final LLM (GPT-4o-mini) generation.

Key capabilities:
  1. Statutory Concordance:
     - IPC 1860 -> BNS 2023 and Indian Evidence Act -> BSA 2023 (bns_ipc_mapping_clean.csv;
       'IEA_' rows hold the evidence mappings).
     - CrPC 1973 -> BNSS 2023 (crpc_bnss_mapping.csv), checked against the CAPT Bhopal
       BNSS-CrPC correspondence table.
  2. Multi-source Statute Discovery:
     - Regex parsing over user questions and retrieved historical cases. Section numbers
       written without an Act name are classified from context (procedure vs offence);
       ambiguous ones are not converted rather than being assumed to be IPC.
     - 2-hop traversal over Knowledge Graph reasoning chains.
  3. Domain awareness (detect_legal_domain): criminal questions get BNS/BNSS/BSA framing;
     family and labour questions are answered under their civil statutes, and penal
     sections are only listed when the user actually raised them.
  4. Semantic Candidate Ranking (criminal questions): ranks modern provisions against the
     query using InLegalBERT.
  5. Domain-Adaptive Prompt Construction with an explicit 1 July 2024 transition rule.
  6. sanitize_answer(): converts legacy citations while keeping the old reference
     ('... (formerly Section 420 IPC)') and never relabels an unmapped number.
"""

import os
import re
import logging
from typing import Dict, List, Set, Tuple, Optional

import pandas as pd
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── Section Normalization Helpers ─────────────────────────────────────────────

def normalize_ipc_section(raw_sec: str) -> str:
    """
    Normalize 'IPC_498A', '498(A)', '498-A', '498a', 'SEC 498A IPC' -> '498A'
    and strip numeric sub-clauses '376(1)' -> '376', '228A(1)' -> '228A'.
    """
    if not raw_sec:
        return ""
    s = str(raw_sec).strip().upper()
    s = re.sub(r'^(?:IPC[_\s]+|SEC(?:TION)?[_\s]+|S\.?\s*)', '', s)
    s = re.sub(r'\(([A-Z]+)\)', r'\1', s)   # 498(A) -> 498A
    s = re.sub(r'\(\d+[A-Z]?\)', '', s)     # 376(1) -> 376
    s = s.replace("(", "").replace(")", "").replace("-", "").replace(" ", "")
    return s


def normalize_iea_section(raw_sec: str) -> str:
    """
    Normalize 'IEA_27', '27', 'SEC 27 IEA', '65B' -> 'IEA_27' or 'IEA_65B'.
    """
    if not raw_sec:
        return ""
    s = str(raw_sec).strip().upper()
    s = re.sub(r'^(?:IEA[_\s]+|SEC(?:TION)?[_\s]+|S\.?\s*)', '', s)
    s = re.sub(r'\s*(?:OF\s+)?(?:THE\s+)?(?:IEA|INDIAN\s+EVIDENCE\s+ACT|EVIDENCE\s+ACT).*$', '', s)
    s = re.sub(r'\(([A-Z]+)\)', r'\1', s)
    s = s.replace("(", "").replace(")", "").replace("-", "").replace(" ", "")
    if s.isdigit() or (len(s) > 1 and s[:-1].isdigit() and s[-1].isalpha()):
        return f"IEA_{s}"
    return f"IEA_{s}" if not s.startswith("IEA_") else s


def normalize_crpc_section(raw_sec: str) -> str:
    """
    Normalize CrPC references while KEEPING a numeric sub-section, because the
    CrPC -> BNSS concordance is sub-section sensitive for a few provisions:
      'CrPC_41A', '41-A', '41(A)' -> '41A'
      '156 (3)', '156(3)'         -> '156(3)'
    """
    if not raw_sec:
        return ""
    s = str(raw_sec).strip().upper()
    s = re.sub(r'^(?:CR\.?P\.?C\.?[_\s]+|SEC(?:TION)?[_\s]+|S\.?\s*)', '', s)
    s = re.sub(r'\(([A-Z]+)\)', r'\1', s)          # 41(A) -> 41A
    s = s.replace("-", "").replace(" ", "")
    m = re.match(r'^(\d{1,3}[A-Z]?)(\(\d+\))?', s)
    if not m:
        return ""
    return m.group(1) + (m.group(2) or "")


def crpc_base_section(key: str) -> str:
    """'156(3)' -> '156', '41A' -> '41A'."""
    return re.sub(r'\(\d+\)$', '', key or "")


# ── Unlabelled section-number disambiguation ──────────────────────────────────
# People usually write "u/s 156(3)" or "quashing u/s 482" without naming the Act.
# Such numbers must NOT be assumed to be IPC: 156 / 482 are CrPC procedure
# sections, and mapping them through the IPC table produces unrelated BNS offences.

# CrPC sections that users and advocates commonly cite without naming the Act
CRPC_COMMONLY_CITED = {
    "41", "41A", "125", "154", "156", "161", "164", "167", "173", "190", "200",
    "202", "313", "320", "397", "401", "406", "407", "436", "436A", "437",
    "438", "439", "468", "482",
}

# IPC offence sections that are commonly cited without naming the Act
IPC_COMMONLY_CITED = {
    "34", "120B", "294", "302", "304", "304B", "306", "307", "323", "324", "325",
    "326", "341", "342", "354", "354A", "354D", "363", "376", "379", "380", "384",
    "406", "409", "415", "417", "420", "427", "447", "448", "465", "467", "468",
    "471", "494", "495", "498A", "499", "500", "504", "506", "509",
}

PROCEDURAL_CONTEXT = re.compile(
    r"\b(bail|anticipatory|quash\w*|magistrate|cjm|acjm|jmfc|notice|application|petition|"
    r"statement|remand|charge\s*sheet|chargesheet|cognizance|revision|compound\w*|summons|"
    r"warrant|high\s+court|sessions?\s+court|maintenance|investigat\w*|cr\.?p\.?c|procedure)\b",
    re.IGNORECASE,
)
OFFENCE_CONTEXT = re.compile(
    r"\b(f\.?i\.?r\.?\s+(?:u/s|under)|case\s+(?:u/s|under)|registered\s+(?:u/s|under)|booked|"
    r"charged|charges?\s+(?:u/s|under)|offen[cs]es?|accused\s+of|punishable|ipc|i\.p\.c)\b",
    re.IGNORECASE,
)


def classify_unlabelled_section(num: str, context: str) -> Optional[str]:
    """
    Decide whether an unlabelled section number refers to the CrPC or the IPC.
    Returns 'crpc', 'ipc', or None (ambiguous -> do not convert at all).
    """
    base = crpc_base_section(normalize_crpc_section(num))
    if not base:
        return None
    proc = bool(PROCEDURAL_CONTEXT.search(context))
    off = bool(OFFENCE_CONTEXT.search(context))
    in_crpc = base in CRPC_COMMONLY_CITED
    in_ipc = base in IPC_COMMONLY_CITED

    if in_crpc and not in_ipc:
        return "crpc" if (proc or not off) else None
    if in_ipc and not in_crpc:
        return "ipc" if (off or not proc) else None
    if in_crpc and in_ipc:            # e.g. 406, 468: context decides
        if proc and not off:
            return "crpc"
        if off and not proc:
            return "ipc"
        return None
    # Not a commonly cited number in either Act: only trust a clear signal
    if off and not proc:
        return "ipc"
    if proc and not off:
        return "crpc"
    return None


# ── Legal domain detection ────────────────────────────────────────────────────

CRIMINAL_CUES = re.compile(
    r"\b(f\.?i\.?r|police|arrest\w*|bail|anticipatory|charge\s*sheet|chargesheet|accused|jail|"
    r"judicial\s+custody|police\s+custody|cognizable|quash\w*|murder\w*|theft|forg\w*|cheat\w*|"
    r"extort\w*|defam\w*|assault\w*|molest\w*|kidnap\w*|rape|intimidat\w*|blackmail\w*|ipc|bns|"
    r"crpc|bnss|magistrate|offen[cs]e)\b",
    re.IGNORECASE,
)
FAMILY_CUES = re.compile(
    r"\b(divorce\w*|maintenance|alimony|child\s+custody|custody\s+of|marriage|married|"
    r"matrimonial|wife|husband|in[-\s]?laws?|spouse|dowry|domestic\s+violence|dv|hma|"
    r"hindu\s+marriage|special\s+marriage|conjugal|restitution|guardian\w*|adoption|"
    r"succession|legal\s+heirs?|inheritance)\b",
    re.IGNORECASE,
)
LABOUR_CUES = re.compile(
    r"\b(salary|wages?|employer|employee|employment|gratuity|provident\s+fund|pf|epf|esi|"
    r"terminat\w*|resign\w*|notice\s+period|retrench\w*|lay[-\s]?off|appointment\s+letter|"
    r"offer\s+letter|bonus|increment|hr|labour|labor|workman|factory)\b",
    re.IGNORECASE,
)

CRIMINAL_DOMAIN = "Criminal Law"
FAMILY_DOMAIN = "Family & Matrimonial Law"
LABOUR_DOMAIN = "Employment & Labour Law"


def detect_legal_domain(question: str, default: str = LABOUR_DOMAIN) -> str:
    """
    Classify a question as criminal, family/matrimonial or labour from keyword cues.
    Counts DISTINCT cue words so one incidental word (e.g. 'cheated' in a divorce
    story) does not flip a family matter into criminal law.
    """
    text = question or ""
    crim = {m.lower() for m in CRIMINAL_CUES.findall(text)}
    fam = {m.lower() for m in FAMILY_CUES.findall(text)}
    lab = {m.lower() for m in LABOUR_CUES.findall(text)}

    if len(crim) >= 2 and len(crim) >= len(fam):
        return CRIMINAL_DOMAIN
    if fam and len(fam) >= len(lab):
        return FAMILY_DOMAIN
    if lab:
        return LABOUR_DOMAIN
    if crim:
        return CRIMINAL_DOMAIN
    return default


def is_criminal_domain(domain_name: str) -> bool:
    return "criminal" in (domain_name or "").lower()


# ── Multi-Statute Text Extractor (Regex) ──────────────────────────────────────

def extract_statutes_from_text(text: str) -> dict:
    """
    Extracts statutory references from text (queries or past cases):
      1. IPC Sections (e.g., 'sec 498A', 'charges 302, 307', 'u/s 420 IPC', 'under Section 406')
      2. IEA Sections (e.g., 'Section 65B of Evidence Act', 'sec 27 IEA')
      3. Domain Sections (e.g., 'Section 13B of Hindu Marriage Act', 'Section 25F IDA')
      4. CrPC Sections (e.g., 'Section 41A CrPC', or unlabelled 'u/s 156(3) before the CJM')
    Unlabelled numbers are classified by context (classify_unlabelled_section);
    ambiguous ones are dropped rather than assumed to be IPC.
    """
    found = {"ipcs": set(), "ieas": set(), "domain_sections": set(), "crpcs": set()}
    if not text or str(text).lower() in ("nan", "none", ""):
        return found

    text_str = str(text)

    # 1. Strip year "1860" / "1872" after act names to prevent false section matches
    cleaned = re.sub(r'(indian\s+penal\s+code(?:\s*\(ipc\))?)\s*,?\s*1860', r'\1', text_str, flags=re.IGNORECASE)
    cleaned = re.sub(r'(indian\s+evidence\s+act(?:\s*\(iea\))?)\s*,?\s*1872', r'\1', cleaned, flags=re.IGNORECASE)

    sec_token = r'\b\d{1,3}(?:\([A-Za-z]\)|[A-Za-z]\b|\b)'
    sep_pattern = r'(?:\s*(?:,|/|\band\b|\bor\b|–|-)\s*(?:sections?|sec\.?|s\.)?\s*)'
    sec_list_pattern = rf'({sec_token}(?:{sep_pattern}{sec_token})*)'

    # 2. Extract Evidence Act (IEA) Sections
    iea_pattern = re.compile(
        rf'(?:u/s|under\s+sections?|under\s+sec\.?|sections?|sec\.?|s\.)\s*'
        rf'{sec_list_pattern}\s*(?:of\s+(?:the\s+)?)?(?:iea|indian\s+evidence\s+act|evidence\s+act)',
        re.IGNORECASE
    )
    for match in iea_pattern.findall(cleaned):
        for num in re.findall(sec_token, match):
            normalized = normalize_iea_section(num)
            if normalized:
                found["ieas"].add(normalized)

    # Mask out extracted IEA references before IPC extraction
    cleaned_no_iea = iea_pattern.sub(' [IEA_ACT] ', cleaned)

    # 3. Extract Explicit Non-IPC Acts (Family, Labour, IT Act, CrPC)
    domain_pattern = re.compile(
        r'(?:under\s+)?(?:sections?|sec\.?|s\.)\s*([\dA-Za-z,\s/()]+?)'
        r'\s+(?:of\s+(?:the\s+)?)?(it\s+act|cr\.?p\.?c\.?|cpc|hma|hindu\s+marriage\s+act|'
        r'pwdva|domestic\s+violence\s+act|ida|industrial\s+disputes\s+act|payment\s+of\s+gratuity\s+act)',
        re.IGNORECASE
    )
    crpc_token = r'\d{1,3}(?:[A-Za-z]|\([A-Za-z]\))?(?:\s*\(\d+\))?'
    for match in domain_pattern.finditer(cleaned_no_iea):
        sec_part, act_part = match.group(1), match.group(2)
        if re.match(r'cr\.?p\.?c', act_part.strip(), re.IGNORECASE):
            for num in re.findall(crpc_token, sec_part):
                key = normalize_crpc_section(num)
                if key:
                    found["crpcs"].add(key)
        else:
            found["domain_sections"].add(f"{act_part.strip().upper()} Section {sec_part.strip()}")

    masked_ipc = domain_pattern.sub(r' [OTHER_ACT] ', cleaned_no_iea)

    # 4. Extract IPC Sections
    ipc_marker = r'(?:ipc|i\.p\.c\.?|indian\s+penal\s+code(?:\s*\(ipc\))?)'
    pat_ipc_a = re.compile(
        rf'(?:u/s|under\s+sections?|under\s+sec\.?|sections?|sec\.?|s\.)\s*'
        rf'{sec_list_pattern}\s*(?:of\s+(?:the\s+)?)?{ipc_marker}',
        re.IGNORECASE
    )
    pat_ipc_b = re.compile(
        rf'{ipc_marker}\s*(?:,?\s*under\s+)?(?:u/s|sections?|sec\.?|s\.)\s*{sec_list_pattern}',
        re.IGNORECASE
    )
    # Unlabelled references. 'under/for section N' is included because every number found
    # here goes through classify_unlabelled_section(); numbers of other Acts named right
    # after them (Gratuity Act, HMA, ...) were already masked by domain_pattern above.
    pat_ipc_c = re.compile(
        rf'(?:(?:in|under|for)\s+sec(?:tion)?\.?|charges(?:\s+under)?|u/s)\s*{sec_list_pattern}',
        re.IGNORECASE
    )
    pat_ipc_d = re.compile(
        rf'(?:section|sec\.)\s*({sec_token})\s+(?:of\s+)?{ipc_marker}',
        re.IGNORECASE
    )

    # 4a. References that explicitly name the IPC
    for pat in (pat_ipc_a, pat_ipc_b, pat_ipc_d):
        for match in pat.findall(masked_ipc):
            for num in re.findall(sec_token, match):
                norm_sec = normalize_ipc_section(num)
                if norm_sec and norm_sec.isalnum():
                    found["ipcs"].add(norm_sec)
        masked_ipc = pat.sub(' [IPC_REF] ', masked_ipc)

    # 4b. Unlabelled references ('u/s 482', 'charges 302, 307', 'in section 41A'):
    #     classify each number from the surrounding words instead of assuming IPC.
    for m in pat_ipc_c.finditer(masked_ipc):
        context = masked_ipc[max(0, m.start() - 60): m.end() + 60]
        tokens = re.findall(crpc_token, m.group(1))
        # sec_token stops before a numeric sub-section: re-attach '(3)' in 'u/s 156(3)'
        tail = re.match(r'\s*\(\d+\)', masked_ipc[m.end():])
        if tail and tokens:
            tokens[-1] = tokens[-1] + tail.group(0).strip()
        for num in tokens:
            kind = classify_unlabelled_section(num, context)
            if kind == "crpc":
                key = normalize_crpc_section(num)
                if key:
                    found["crpcs"].add(key)
            elif kind == "ipc":
                norm_sec = normalize_ipc_section(num)
                if norm_sec and norm_sec.isalnum():
                    found["ipcs"].add(norm_sec)

    return found


# ── Statutory Mapping Loader ──────────────────────────────────────────────────

def resolve_mapping_path(mapping_csv_path: Optional[str] = None) -> str:
    """Finds the mapping CSV by checking multiple candidate locations."""
    candidates = []
    if mapping_csv_path:
        candidates.append(mapping_csv_path)

    # Local directory of this module
    this_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(this_dir, "bns_ipc_mapping_clean.csv"))
    candidates.append(os.path.join(this_dir, "bns_ipc_mapping_final.csv"))

    # Project root directory
    repo_root = os.path.dirname(this_dir)
    candidates.append(os.path.join(repo_root, "ipc_to_bns", "bns_ipc_mapping_clean.csv"))
    candidates.append(os.path.join(repo_root, "ipc_to_bns", "bns_ipc_mapping_final.csv"))
    candidates.append("bns_ipc_mapping_clean.csv")
    candidates.append("bns_ipc_mapping_final.csv")

    for path in candidates:
        if path and os.path.isfile(path):
            return path

    return mapping_csv_path or "bns_ipc_mapping_clean.csv"


def load_statutory_mappings(mapping_csv_path: Optional[str] = None):
    """
    Loads unified statutory mappings from CSV.
    Supports:
      1. IPC -> BNS (keys: '498A', '302', '420', etc.)
      2. IEA -> BSA (keys: 'IEA_27', 'IEA_65B', etc.)
      3. Modern descriptions for semantic ranking.
    """
    resolved_path = resolve_mapping_path(mapping_csv_path)
    if not os.path.exists(resolved_path):
        logger.warning("Mapping CSV '%s' not found. Statutory mapping will be empty.", resolved_path)
        return {}, {}, {}

    try:
        df = pd.read_csv(resolved_path)
    except Exception as e:
        logger.error("Failed to read mapping CSV '%s': %s", resolved_path, e)
        return {}, {}, {}

    df.columns = [c.strip().lower() for c in df.columns]

    old_col = next((c for c in df.columns if any(k in c for k in ("old", "ipc", "iea"))), None)
    new_col = next((c for c in df.columns if any(k in c for k in ("new", "bns", "bsa"))), None)
    desc_col = next(
        (c for c in df.columns if any(k in c for k in ("semantic", "subject", "desc", "offence"))),
        None
    )

    ipc_to_bns: Dict[str, List[str]] = {}
    iea_to_bsa: Dict[str, List[str]] = {}
    modern_semantics: Dict[str, str] = {}

    for _, row in df.iterrows():
        old_val_raw = str(row[old_col]).strip() if old_col else ""
        new_val = str(row[new_col]).strip() if new_col else ""
        desc_val = str(row[desc_col]).strip() if desc_col else "Statutory provision"

        if not old_val_raw or not new_val or old_val_raw.upper() in ("NONE", "NAN"):
            continue

        # Split compound items like "29and29A", "54&55", "376(1)&376(2)"
        sub_items = re.split(r'and|&|,', old_val_raw)

        for sub_raw in sub_items:
            sub_raw = sub_raw.strip()
            if not sub_raw:
                continue

            if sub_raw.upper().startswith("IEA_"):
                iea_key = normalize_iea_section(sub_raw)
                iea_to_bsa.setdefault(iea_key, []).append(new_val)
                modern_semantics[new_val] = desc_val
            else:
                ipc_key = normalize_ipc_section(sub_raw)
                if ipc_key:
                    ipc_to_bns.setdefault(ipc_key, []).append(new_val)
                    modern_semantics[new_val] = desc_val

    # Deduplicate lists
    for k in ipc_to_bns:
        ipc_to_bns[k] = list(dict.fromkeys(ipc_to_bns[k]))
    for k in iea_to_bsa:
        iea_to_bsa[k] = list(dict.fromkeys(iea_to_bsa[k]))

    logger.info(
        "Loaded %d IPC->BNS and %d IEA->BSA mappings (%d total descriptions) from %s",
        len(ipc_to_bns), len(iea_to_bsa), len(modern_semantics), resolved_path
    )
    return ipc_to_bns, iea_to_bsa, modern_semantics


def resolve_crpc_mapping_path(crpc_csv_path: Optional[str] = None) -> str:
    """Locate crpc_bnss_mapping.csv (next to this module by default)."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [p for p in (crpc_csv_path, os.path.join(this_dir, "crpc_bnss_mapping.csv")) if p]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[-1]


def load_crpc_mappings(crpc_csv_path: Optional[str] = None):
    """
    Loads the CrPC -> BNSS concordance (crpc_bnss_mapping.csv).
    Every row was checked against the CAPT Bhopal 'Correspondence Table and Comparison
    Summary of BNSS 2023 and CrPC 1973' (Source_Page = page in that PDF).

    Returns:
        crpc_to_bnss: {'156(3)': ['175(3)'], '482': ['528'], ...}
        bnss_subjects: {'175(3)': 'Magistrate's power to order investigation...', ...}
    """
    path = resolve_crpc_mapping_path(crpc_csv_path)
    if not os.path.exists(path):
        logger.warning("CrPC->BNSS mapping CSV '%s' not found. CrPC conversion disabled.", path)
        return {}, {}
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception as e:
        logger.error("Failed to read CrPC mapping CSV '%s': %s", path, e)
        return {}, {}

    crpc_to_bnss: Dict[str, List[str]] = {}
    bnss_subjects: Dict[str, str] = {}
    for _, row in df.iterrows():
        old = normalize_crpc_section(str(row.get("Old_CrPC", "")))
        new = str(row.get("New_BNSS", "")).strip()
        if not old or not new or new.upper() in ("NAN", "NONE", "-"):
            continue
        crpc_to_bnss.setdefault(old, [])
        if new not in crpc_to_bnss[old]:
            crpc_to_bnss[old].append(new)
        bnss_subjects.setdefault(new, str(row.get("Subject", "Procedural provision")).strip())

    logger.info("Loaded %d CrPC->BNSS mappings from %s", len(crpc_to_bnss), path)
    return crpc_to_bnss, bnss_subjects


def lookup_bnss(crpc_key: str, crpc_to_bnss: dict) -> List[str]:
    """Exact sub-section match first ('156(3)'), then the base section ('156')."""
    key = normalize_crpc_section(crpc_key)
    if key in crpc_to_bnss:
        return crpc_to_bnss[key]
    return crpc_to_bnss.get(crpc_base_section(key), [])


# ── Node Metadata Loader ──────────────────────────────────────────────────────

def load_node_metadata(nodes_csv_path: str = "nodes.csv"):
    """
    Loads node metadata from nodes.csv across all domains:
      - node_type_map: node_id -> type ('Section', 'Act', 'Concept', etc.)
      - node_desc_map: node_id -> description / legal meaning
      - node_act_map:  node_id -> act name ('IPC', 'IEA', 'HMA', 'IDA', etc.)
    """
    if not os.path.exists(nodes_csv_path):
        logger.warning("Node metadata file '%s' not found.", nodes_csv_path)
        return {}, {}, {}

    # Read only the columns used here: nodes.csv also carries long answer columns
    # (query_text / *_answer) that would otherwise be parsed into memory for nothing.
    wanted = {"node_id", "type", "act", "desc", "text"}
    usecols = lambda c: c.strip().lower() in wanted  # noqa: E731
    try:
        df = pd.read_csv(nodes_csv_path, usecols=usecols)
    except Exception:
        df = pd.read_csv(nodes_csv_path, usecols=usecols, engine="python", on_bad_lines="skip")

    df.columns = [c.strip().lower() for c in df.columns]
    node_type_map = {}
    node_desc_map = {}
    node_act_map = {}

    for _, row in df.iterrows():
        nid = str(row.get("node_id", "")).strip()
        ntype = str(row.get("type", "")).strip()
        nact = str(row.get("act", "")).strip()

        ndesc = str(row.get("desc", ""))
        if not ndesc or ndesc == "nan":
            ndesc = str(row.get("text", ""))
        if not ndesc or ndesc == "nan":
            ndesc = "Statutory provision"

        node_type_map[nid] = ntype
        node_desc_map[nid] = ndesc
        node_act_map[nid] = nact

    return node_type_map, node_desc_map, node_act_map


# ── 2-Hop Graph Traversal ─────────────────────────────────────────────────────

def extract_2hop_statutory_sections(G, chain_nodes: list, node_type_map: dict, node_desc_map: dict):
    """
    Traverses up to 2 hops from the RL reasoning chain nodes in the KG:
      - IPC nodes -> kg_ipcs (for BNS translation)
      - IEA nodes -> kg_ieas (for BSA translation)
      - Non-penal Section nodes (HMA, DVA, IDA, PGA) -> kg_domain_sections
    """
    kg_ipcs = set()
    kg_ieas = set()
    kg_domain_sections = {}

    def classify_node(node_str: str):
        n_clean = str(node_str).strip()
        u = n_clean.upper()
        if u.startswith("IPC_"):
            sec = normalize_ipc_section(n_clean)
            if sec:
                kg_ipcs.add(sec)
        elif u.startswith("IEA_"):
            sec = normalize_iea_section(n_clean)
            if sec:
                kg_ieas.add(sec)
        else:
            ntype = node_type_map.get(n_clean, "")
            domain_prefixes = ("HMA_", "DVA_", "IDA_", "PGA_", "SMA_", "MWA_", "POSH_", "EPF_", "ESI_")
            if ntype == "Section" or any(u.startswith(p) for p in domain_prefixes):
                desc = node_desc_map.get(n_clean, "Statutory provision")
                kg_domain_sections[n_clean] = desc

    if G is not None:
        for node in chain_nodes:
            n_str = str(node).strip()
            classify_node(n_str)
            if not G.has_node(n_str):
                continue

            # Hop 1
            hop1 = set(G.successors(n_str)) | set(G.predecessors(n_str))
            for n1 in hop1:
                classify_node(str(n1))
                # Hop 2
                hop2 = set(G.successors(n1)) | set(G.predecessors(n1))
                for n2 in hop2:
                    classify_node(str(n2))
    else:
        for node in chain_nodes:
            classify_node(str(node).strip())

    return kg_ipcs, kg_ieas, kg_domain_sections


# ── Landmark Criminal & Domain Offense Keyword Map ───────────────────────────
# Each key is a tuple of whole-word regex fragments (matched with \b...\b, case-insensitive).
# Applied ONLY to criminal-domain questions. Deliberately excludes generic words such as
# 'woman', 'girl', 'harm', 'kill', 'provident fund' that previously injected penal sections
# into family and labour questions.
KEYWORD_OFFENSE_MAP: Dict[Tuple[str, ...], List[str]] = {
    (r"defam\w*", r"slander\w*", r"libel\w*"): ["356(1)", "356(2)"],
    (r"pornograph\w*", r"nude", r"nudity", r"obscen\w*", r"mms", r"video\s+leak\w*"): ["294", "296"],
    (r"modesty", r"outrag\w*\s+(?:her|the|my)\s+modesty", r"molest\w*", r"eve[-\s]?teas\w*"): ["74", "79"],
    (r"cheat\w*", r"fraud\w*", r"defraud\w*", r"scam\w*"): ["318(1)", "318(4)"],
    (r"breach\s+of\s+trust", r"misappropriat\w*", r"embezzl\w*"): ["316(1)", "316(2)"],
    (r"threat\w*", r"intimidat\w*"): ["351(1)", "351(2)", "351(3)"],
    (r"forg\w*", r"fake\s+document\w*", r"fabricated\s+document\w*"): ["336(1)", "336(3)", "338"],
    (r"murder\w*", r"homicide"): ["103(1)"],
    (r"cruelty", r"dowry"): ["85", "86"],
    (r"theft", r"steal\w*", r"stolen", r"stole"): ["303(1)", "303(2)"],
    (r"wrongful\s+confinement", r"confined", r"locked\s+(?:me\s+)?inside"): ["127(1)", "127(2)"],
    (r"extort\w*", r"blackmail\w*"): ["308(1)", "308(2)"],
}

# Electronic-evidence cues (any domain): BSA s.63 is relevant ONLY when electronic
# records are actually part of the facts.
ELECTRONIC_EVIDENCE_CUES = re.compile(
    r"\b(electronic\s+records?|digital\s+evidence|whatsapp|screenshots?|e-?mails?|"
    r"call\s+recordings?|voice\s+recordings?|audio\s+recordings?|cctv|chats?|sms)\b",
    re.IGNORECASE,
)

_KEYWORD_REGEXES = [
    (re.compile(r"\b(?:" + "|".join(frags) + r")\b", re.IGNORECASE), targets)
    for frags, targets in KEYWORD_OFFENSE_MAP.items()
]

# Procedural cues -> CrPC sections (converted to BNSS through crpc_bnss_mapping.csv, so only
# verified concordance rows are ever offered). Lets plain-language procedural questions
# ("police won't register my FIR", "can I get bail") reach the right BNSS provisions.
# The second element says whether the cue also applies outside criminal questions.
PROCEDURE_KEYWORD_MAP: List[Tuple[Tuple[str, ...], List[str], bool]] = [
    ((r"f\.?i\.?r", r"police\s+complaint", r"(?:not|refus\w*\s+to)\s+register\w*"), ["154", "156(3)"], False),
    ((r"anticipatory\s+bail",), ["438"], False),
    ((r"bail",), ["437", "439"], False),
    ((r"quash\w*",), ["482"], False),
    ((r"notice\s+of\s+appearance", r"appearance\s+notice"), ["41A"], False),
    ((r"charge\s*sheet\w*",), ["173"], False),
    ((r"remand\w*", r"police\s+custody", r"judicial\s+custody"), ["167"], False),
    ((r"compromis\w*", r"compound\w*"), ["320"], False),
    ((r"maintenance",), ["125"], True),
]

# 'maintenance' outside criminal questions only counts when a family relation is present
# (avoids e.g. "maintenance department" in a labour question).
_FAMILY_RELATION_CUES = re.compile(
    r"\b(wife|husband|child(?:ren)?|daughter|son|parents?|spouse|divorc\w*|separat\w*|married|marriage)\b",
    re.IGNORECASE,
)

_PROCEDURE_REGEXES = [
    (re.compile(r"\b(?:" + "|".join(frags) + r")\b", re.IGNORECASE), crpcs, any_domain)
    for frags, crpcs, any_domain in PROCEDURE_KEYWORD_MAP
]


# ── Candidate Statutory Menu Builder ──────────────────────────────────────────

def build_statutory_candidate_menu(
    explicit_ipcs: set,
    kg_ipcs: set,
    ipc_to_bns: dict,
    explicit_ieas: set,
    kg_ieas: set,
    iea_to_bsa: dict,
    kg_domain_sections: dict,
    modern_semantics: dict,
    q_emb_norm: Optional[torch.Tensor] = None,
    modern_embs_norm: Optional[torch.Tensor] = None,
    modern_list: Optional[list] = None,
    top_k_modern: int = 15,
    question: str = "",
    is_criminal: bool = True,
    explicit_crpcs: Optional[set] = None,
    crpc_to_bnss: Optional[dict] = None,
    bnss_subjects: Optional[dict] = None,
    use_semantic_search: bool = False,
    max_statutory_sections: int = 15,
) -> Tuple[str, List[str]]:
    """
    Compiles candidate legal menu for the modern prompt:
      1. Modern BNS provisions (from IPC mentions, keyword offense detection, and graph discoveries)
      2. Modern BSA provisions (from IEA mentions and graph discoveries)
      3. Modern BNSS procedure provisions (from CrPC mentions)
      4. Semantic search over entire BNS/BSA catalogue using InLegalBERT embeddings
      5. Domain-specific statutory provisions (e.g., IDA, PGA, HMA)

    For civil (non-criminal) questions, keyword offence detection, graph-discovered
    penal sections and the full-catalogue semantic search are skipped, and the
    domain (civil) provisions are listed first. No penal section is ever added
    just to keep the menu non-empty.

    Safeguard: at most `max_statutory_sections` BNS/BNSS/BSA lines are kept (default 15).
    When there are more, the most reliable ones are kept, in this order: sections the
    user mentioned (IPC/IEA/CrPC) > procedure-keyword BNSS > electronic-evidence BSA 63 >
    offence keywords > graph-discovered > similarity search. Domain (civil) provisions
    are capped separately at 10.
    Returns:
      (menu_text: str, mapped_bns_list: List[str])
    """
    candidate_sections = []
    seen_modern = set()
    mapped_bns_list = []
    explicit_crpcs = explicit_crpcs or set()
    crpc_to_bnss = crpc_to_bnss or {}
    bnss_subjects = bnss_subjects or {}

    # 1. Modern Penal Provisions (BNS) from Explicit IPC mentions
    bns_lines = []
    for ipc in sorted(explicit_ipcs):
        for bns in ipc_to_bns.get(ipc, []):
            if bns not in seen_modern:
                desc = modern_semantics.get(bns, "Penal provision")
                bns_lines.append(f"- Section {bns} BNS (formerly IPC Section {ipc}): {desc}")
                seen_modern.add(bns)
                mapped_bns_list.append(f"BNS {bns} (ex-IPC {ipc})")

    # 2. Modern Evidence Provisions (BSA) from Explicit IEA mentions
    bsa_lines = []
    for iea in sorted(explicit_ieas):
        for bsa in iea_to_bsa.get(iea, []):
            if bsa not in seen_modern:
                desc = modern_semantics.get(bsa, "Evidence provision")
                bsa_lines.append(f"- Section {bsa} BSA (formerly Evidence Act Section {iea.replace('IEA_', '')}): {desc}")
                seen_modern.add(bsa)

    # 2b. Modern Procedure Provisions (BNSS) from explicit CrPC mentions, plus
    #     procedural keyword cues (criminal questions; 'maintenance' in any domain)
    bnss_lines = []
    keyword_crpcs = set()
    if question:
        family_context = bool(_FAMILY_RELATION_CUES.search(question))
        for proc_regex, crpcs, any_domain in _PROCEDURE_REGEXES:
            allowed = is_criminal or (any_domain and family_context)
            if allowed and proc_regex.search(question):
                keyword_crpcs.update(crpcs)
    for crpc in sorted(explicit_crpcs) + sorted(keyword_crpcs - set(explicit_crpcs)):
        for bnss in lookup_bnss(crpc, crpc_to_bnss):
            key = f"BNSS {bnss}"
            if key not in seen_modern:
                desc = bnss_subjects.get(bnss, "Procedural provision")
                bnss_lines.append(f"- Section {bnss} BNSS (formerly CrPC Section {crpc}): {desc}")
                seen_modern.add(key)

    # 2c. Electronic evidence (any domain) — only when electronic records are in the facts
    if question and ELECTRONIC_EVIDENCE_CUES.search(question) and "BSA 63" not in seen_modern:
        desc = modern_semantics.get("BSA 63", "Admissibility of electronic records")
        bsa_lines.append(f"- Section 63 BSA (cite only if relying on electronic records): {desc}")
        seen_modern.add("BSA 63")

    # 3. Offense Keyword Intent Mapping (criminal questions only)
    if question and is_criminal:
        for kw_regex, target_sections in _KEYWORD_REGEXES:
            if kw_regex.search(question):
                for sec in target_sections:
                    if sec not in seen_modern:
                        desc = modern_semantics.get(sec, "Statutory provision")
                        bns_lines.append(f"- Section {sec} BNS (Offense Provision): {desc}")
                        mapped_bns_list.append(f"BNS {sec}")
                        seen_modern.add(sec)

    # 4. Semantic Ranking over KG-discovered provisions
    kg_candidates = set()
    if not is_criminal:
        kg_ipcs, kg_ieas = set(), set()
    for ipc in kg_ipcs:
        for bns in ipc_to_bns.get(ipc, []):
            if bns not in seen_modern:
                kg_candidates.add(bns)
    for iea in kg_ieas:
        for bsa in iea_to_bsa.get(iea, []):
            if bsa not in seen_modern:
                kg_candidates.add(bsa)

    for bns in sorted(kg_candidates):
        if bns not in seen_modern:
            desc = modern_semantics.get(bns, "Statutory provision")
            if bns.startswith("BSA"):
                bsa_lines.append(f"- Section {bns} BSA (Graph Discovered): {desc}")
            else:
                bns_lines.append(f"- Section {bns} BNS (Graph Discovered): {desc}")
                mapped_bns_list.append(f"BNS {bns}")
            seen_modern.add(bns)

    # 5. Full-Catalogue Semantic Ranking (criminal questions only, OFF by default).
    #    Raw InLegalBERT [CLS] similarities are not discriminative here: ~66% of all 686
    #    provisions score above the 0.45 threshold and the top 15 sit within ~0.03 of each
    #    other, so this step added unrelated sections (e.g. soldier abetment, bigamy) to
    #    FIR / bail questions. Kept behind a flag for experiments with better embeddings.
    if (use_semantic_search and is_criminal and q_emb_norm is not None
            and modern_embs_norm is not None and modern_list):
        try:
            sims = torch.mm(q_emb_norm, modern_embs_norm.t()).squeeze(0)
            k_search = min(top_k_modern, len(modern_list))
            top_k_scores, top_k_idx = torch.topk(sims, k_search)
            for score, idx in zip(top_k_scores, top_k_idx):
                if score.item() < 0.45:
                    continue
                mod_cand = modern_list[idx.item()]
                if mod_cand not in seen_modern:
                    desc = modern_semantics.get(mod_cand, "Statutory provision")
                    if mod_cand.startswith("BSA"):
                        bsa_lines.append(f"- Section {mod_cand} BSA (Contextually Retrieved): {desc}")
                    else:
                        bns_lines.append(f"- Section {mod_cand} BNS (Contextually Retrieved): {desc}")
                        mapped_bns_list.append(f"BNS {mod_cand}")
                    seen_modern.add(mod_cand)
                    if len(bns_lines) >= top_k_modern:
                        break
        except Exception as e:
            logger.warning("Full-catalogue semantic ranking encountered an issue: %s", e)

    # 5b. Safeguard: cap the BNS/BNSS/BSA candidates to keep the prompt small.
    all_lines = ([(ln, "bns") for ln in bns_lines] + [(ln, "bnss") for ln in bnss_lines]
                 + [(ln, "bsa") for ln in bsa_lines])
    if max_statutory_sections and len(all_lines) > max_statutory_sections:
        def source_priority(line: str) -> int:
            m = re.search(r"\(formerly CrPC Section ([^)]+)\)", line)
            if m:
                return 0 if m.group(1) in explicit_crpcs else 1        # user-mentioned vs procedure cue
            if "(formerly IPC Section" in line or "(formerly Evidence Act Section" in line:
                return 0                                               # user-mentioned
            if "relying on electronic records" in line:
                return 2
            if "(Offense Provision)" in line:
                return 3
            if "(Graph Discovered)" in line:
                return 4
            return 5                                                   # similarity search / other

        ranked = sorted(range(len(all_lines)), key=lambda i: (source_priority(all_lines[i][0]), i))
        keep = set(ranked[:max_statutory_sections])
        dropped_bns = {
            m.group(1)
            for i, (ln, kind) in enumerate(all_lines)
            if i not in keep and kind == "bns" and (m := re.match(r"- Section (\S+) BNS", ln))
        }
        bns_lines = [ln for i, (ln, k) in enumerate(all_lines) if k == "bns" and i in keep]
        bnss_lines = [ln for i, (ln, k) in enumerate(all_lines) if k == "bnss" and i in keep]
        bsa_lines = [ln for i, (ln, k) in enumerate(all_lines) if k == "bsa" and i in keep]
        mapped_bns_list = [m for m in mapped_bns_list if m.split()[1] not in dropped_bns]
        logger.info("Statutory menu capped: kept %d of %d candidate sections",
                    max_statutory_sections, len(all_lines))

    # 6. (Removed) The former "fallback guarantee" inserted defamation / cheating /
    #    obscenity sections whenever nothing penal was found, which pushed BNS offences
    #    into civil answers. An empty penal menu is now the correct outcome.

    # 7. Domain Statutory Provisions (Civil, Matrimonial, Labour)
    dom_block = None
    if kg_domain_sections:
        dom_lines = []
        for sec_name, sec_desc in sorted(kg_domain_sections.items())[:10]:
            dom_lines.append(f"- {sec_name}: {sec_desc}")
        dom_block = "DOMAIN STATUTORY PROVISIONS (Retrieved from Subject Knowledge Graph):\n" + "\n".join(dom_lines)

    penal_header = (
        "MODERN PENAL PROVISIONS (Bharatiya Nyaya Sanhita - BNS, 2023):"
        if is_criminal else
        "PENAL PROVISIONS MENTIONED IN THE QUESTION (Bharatiya Nyaya Sanhita - BNS, 2023; "
        "use only if a criminal offence is actually in issue):"
    )

    if not is_criminal and dom_block:
        candidate_sections.append(dom_block)
    if bns_lines:
        candidate_sections.append(penal_header + "\n" + "\n".join(bns_lines))
    if bnss_lines:
        candidate_sections.append(
            "MODERN PROCEDURE PROVISIONS (Bharatiya Nagarik Suraksha Sanhita - BNSS, 2023):\n" + "\n".join(bnss_lines)
        )
    if bsa_lines:
        candidate_sections.append("MODERN EVIDENCE PROVISIONS (Bharatiya Sakshya Adhiniyam - BSA, 2023):\n" + "\n".join(bsa_lines))
    if is_criminal and dom_block:
        candidate_sections.append(dom_block)

    menu_text = "\n\n".join(candidate_sections)
    return menu_text, mapped_bns_list


# ── Domain-Adaptive Prompt Builder ────────────────────────────────────────────

def build_domain_prompt(
    question: str,
    chain_extended: list,
    top_k: list,
    statutory_menu: str,
    domain_name: str = "Criminal Law",
    is_criminal: Optional[bool] = None,
    corpus_domain: Optional[str] = None,
) -> str:
    """
    Structured CoT prompt for the final answer.

    Criminal questions: cite BNS (offences), BNSS (procedure) and BSA (evidence).
    Civil questions (family / labour): cite the governing civil statutes; BNS only if a
    criminal offence is genuinely in issue. No fixed list of sections is forced on the
    model, and BSA s.63 is cited only when electronic records are actually relied on.
    """
    if is_criminal is None:
        is_criminal = is_criminal_domain(domain_name)

    chain_str = " → ".join(chain_extended) if chain_extended else "N/A"

    context_lines = []
    for i, r in enumerate(top_k, 1):
        context_lines.append(f"[Historical Case {i}]")
        context_lines.append(f"Query: {r.get('question', '')}")
        context_lines.append(f"Precedent Ratio: {r.get('answer', '')}")
        context_lines.append("")
    context_str = "\n".join(context_lines).strip()

    # The deployed API has one KG + precedent database (Employment & Labour). For a
    # question classified into another domain, say so instead of letting labour
    # precedents silently steer e.g. a criminal answer.
    precedent_note = "Precedents may cite repealed codes — apply their ratio using the current provisions"
    if corpus_domain and corpus_domain.strip().lower() != (domain_name or "").strip().lower():
        precedent_note += (
            f". NOTE: these precedents and the reasoning chain come from the {corpus_domain} database "
            f"and may not match this {domain_name} question; use them only where they are clearly relevant"
        )

    transition_rule = (
        "On 1 July 2024 the Indian Penal Code (IPC), the Code of Criminal Procedure (CrPC) and the "
        "Indian Evidence Act (IEA) were replaced by the Bharatiya Nyaya Sanhita (BNS), the Bharatiya "
        "Nagarik Suraksha Sanhita (BNSS) and the Bharatiya Sakshya Adhiniyam (BSA). Cite the new codes "
        "by default. If the facts show the offence was committed, or the proceeding (FIR, complaint, "
        "application) was started, before 1 July 2024, the old code continues to govern it: cite the "
        "old provision and give the new equivalent, e.g. \"Section 420 IPC (now Section 318(4) BNS)\"."
    )
    accuracy_rule = (
        "Cite a section number only if it appears in the candidate provisions below or you are certain "
        "of it. Never guess a section number; if unsure, name the Act and describe the provision instead. "
        "Do not attach a section of one Act to another Act."
    )
    evidence_rule = (
        "Cite Section 63 BSA only when the person is relying on electronic records (chats, emails, "
        "screenshots, recordings, CCTV). Paper documents such as letters, contracts or receipts do not "
        "need Section 63."
    )

    if is_criminal:
        mandate = f"""STATUTORY RULES (CRIMINAL MATTER):
1. {transition_rule}
2. Offences: cite the Bharatiya Nyaya Sanhita (BNS), 2023.
3. Procedure (FIR, arrest, notice of appearance, bail, anticipatory bail, remand, chargesheet, cognizance, quashing, revision, appeal): cite the Bharatiya Nagarik Suraksha Sanhita (BNSS), 2023 — not the BNS.
4. Evidence: cite the Bharatiya Sakshya Adhiniyam (BSA), 2023. {evidence_rule}
5. {accuracy_rule}"""
        step_example = "e.g. filing a complaint/FIR, approaching the Magistrate, applying for bail, preserving evidence"
        cite_instruction = (
            "Cite the exact provisions: BNS for offences, BNSS for procedure, BSA for evidence, and special "
            "statutes (e.g. Information Technology Act, 2000) where applicable."
        )
    else:
        mandate = f"""STATUTORY RULES ({domain_name.upper()} — A CIVIL MATTER):
1. Base the answer on the civil statutes that govern this issue (for example marriage, divorce, maintenance, custody or domestic-violence laws for family matters; wage, gratuity, provident fund, industrial-dispute, shops-and-establishments or other labour laws for employment matters). Prefer the statutes listed under DOMAIN STATUTORY PROVISIONS below.
2. Do NOT cite the Bharatiya Nyaya Sanhita (BNS) or any penal provision unless the facts disclose a criminal offence that is relevant to what the person is asking. Remedies such as divorce, nullity, alimony, maintenance, gratuity or recovery of wages are never granted "under the BNS".
3. If a criminal angle is genuinely relevant, mention it separately and briefly. {transition_rule}
4. Maintenance under the criminal procedure code is now Section 144 BNSS (formerly Section 125 CrPC).
5. {evidence_rule}
6. {accuracy_rule}"""
        step_example = "e.g. sending a legal notice, filing the appropriate petition or claim before the competent court or authority, collecting documents"
        cite_instruction = (
            "Cite the governing civil statutes and sections that apply to this issue; cite BNS/BNSS/BSA only "
            "under the rules above."
        )

    return f"""You are an expert Indian advocate advising a client on a {domain_name} matter under current Indian law. Use the legal reasoning chain and precedent cases below to understand the legal principles and ratio decidendi.

{mandate}

CANDIDATE STATUTORY PROVISIONS (MODERN LAW):
{statutory_menu if statutory_menu.strip() else "None identified — rely on the governing statutes for this issue."}

REASONING CHAIN (Concept Path):
{chain_str}

HISTORICAL CASE LAW ({precedent_note}):
{context_str}

USER'S QUESTION:
{question}

INSTRUCTIONS:
1. Start your answer by directly addressing the user's specific legal issue in plain English, just like an experienced advocate advising a client.
2. {cite_instruction}
3. State clearly what the user CAN and CANNOT do legally.
4. Give 1-2 concrete, actionable next steps ({step_example}).
5. Never present a repealed IPC, CrPC or Evidence Act section as current law (see rule 1 for events before 1 July 2024).
6. Be concise (3-5 sentences), authoritative, and practically useful.

ANSWER:"""


# ── BNS Converter Class ───────────────────────────────────────────────────────

class BNSConverter:
    """
    Modular statutory conversion engine that plugs into any legal QA pipeline.
    """

    def __init__(
        self,
        mapping_csv_path: Optional[str] = None,
        crpc_mapping_csv_path: Optional[str] = None,
        use_semantic_search: bool = False,
    ):
        self.use_semantic_search = use_semantic_search
        self.mapping_csv_path = resolve_mapping_path(mapping_csv_path)
        self.ipc_to_bns, self.iea_to_bsa, self.modern_semantics = load_statutory_mappings(self.mapping_csv_path)
        # CrPC -> BNSS is kept separate from modern_semantics so the cached
        # InLegalBERT embedding file (keyed on modern_list) stays valid.
        self.crpc_to_bnss, self.bnss_subjects = load_crpc_mappings(crpc_mapping_csv_path)
        self.modern_list = list(self.modern_semantics.keys())
        self._modern_embs_tensor = None
        self._modern_embs_norm = None

    def initialize_embeddings(self, tok, bert, device: torch.device, cache_file: Optional[str] = None):
        """Pre-computes or loads cached InLegalBERT embeddings for modern provisions."""
        if not self.modern_list or tok is None or bert is None:
            return

        if cache_file is None:
            this_dir = os.path.dirname(os.path.abspath(__file__))
            cache_file = os.path.join(this_dir, "modern_statutory_cache.pt")

        cached_data = None
        if os.path.exists(cache_file):
            try:
                loaded = torch.load(cache_file, map_location="cpu")
                if loaded.get("keys") == self.modern_list:
                    cached_data = loaded
                    self._modern_embs_tensor = cached_data["embeddings"].to(device)
                    self._modern_embs_norm = F.normalize(self._modern_embs_tensor, dim=1)
                    logger.info("Loaded modern statutory embeddings from %s", cache_file)
            except Exception as e:
                logger.warning("Could not read embedding cache: %s", e)
                cached_data = None

        if cached_data is None:
            logger.info("Computing modern statutory embeddings for %d provisions...", len(self.modern_list))
            desc_list = [self.modern_semantics[m] for m in self.modern_list]
            batch_size = 64
            all_embs = []
            with torch.no_grad():
                for i in range(0, len(desc_list), batch_size):
                    batch = desc_list[i:i + batch_size]
                    enc = tok(batch, max_length=64, padding=True, truncation=True, return_tensors="pt").to(device)
                    out = bert(**enc).last_hidden_state[:, 0, :]
                    all_embs.append(out)
            self._modern_embs_tensor = torch.cat(all_embs, dim=0)
            self._modern_embs_norm = F.normalize(self._modern_embs_tensor, dim=1)
            try:
                torch.save({"keys": self.modern_list, "embeddings": self._modern_embs_tensor.cpu()}, cache_file)
                logger.info("Saved modern statutory embeddings to %s", cache_file)
            except Exception as e:
                logger.warning("Could not cache embeddings to disk: %s", e)

    def process_query(
        self,
        question: str,
        chain_extended: list,
        top_k: list,
        G=None,
        node_type_map: Optional[dict] = None,
        node_desc_map: Optional[dict] = None,
        domain_name: str = "Criminal Law",
        q_emb_norm: Optional[torch.Tensor] = None,
        corpus_domain: Optional[str] = None,
    ) -> dict:
        """
        Executes the conversion layer:
          1. 2-hop graph harvesting of IPC/IEA/domain nodes.
          2. Regex extraction from question + retrieved cases.
          3. Conversion to modern BNS/BNSS/BSA provisions.
          4. Domain-aware prompt construction (criminal vs civil).

        For civil (family / labour) questions, statutes found only in retrieved
        precedents or graph neighbourhoods are NOT turned into penal candidates;
        only what the user actually mentioned is converted.
        """
        node_type_map = node_type_map or {}
        node_desc_map = node_desc_map or {}
        criminal = is_criminal_domain(domain_name)

        # 1. 2-hop KG discovery
        kg_ipcs, kg_ieas, kg_domain_sections = extract_2hop_statutory_sections(
            G, chain_extended, node_type_map, node_desc_map
        )

        # 2. Text regex extraction from user question + precedents
        text_statutes = extract_statutes_from_text(question)
        explicit_ipcs = set(text_statutes["ipcs"])
        explicit_ieas = set(text_statutes["ieas"])
        explicit_crpcs = set(text_statutes["crpcs"])

        # Precedents / graph neighbours from a corpus of another domain (e.g. the labour KG
        # answering a criminal question) carry unrelated sections, such as lockdown-era
        # IPC 188 / 270 from labour cases. Only harvest them when the corpus matches.
        corpus_mismatch = bool(
            corpus_domain and corpus_domain.strip().lower() != (domain_name or "").strip().lower()
        )
        if criminal and not corpus_mismatch:
            for item in top_k:
                q_stat = extract_statutes_from_text(item.get("question", ""))
                a_stat = extract_statutes_from_text(item.get("answer", ""))
                explicit_ipcs |= q_stat["ipcs"] | a_stat["ipcs"]
                explicit_ieas |= q_stat["ieas"] | a_stat["ieas"]
                explicit_crpcs |= q_stat["crpcs"] | a_stat["crpcs"]
        else:
            kg_ipcs, kg_ieas = set(), set()

        # 3. Build statutory candidate menu
        statutory_menu, mapped_bns_list = build_statutory_candidate_menu(
            explicit_ipcs=explicit_ipcs,
            kg_ipcs=kg_ipcs,
            ipc_to_bns=self.ipc_to_bns,
            explicit_ieas=explicit_ieas,
            kg_ieas=kg_ieas,
            iea_to_bsa=self.iea_to_bsa,
            kg_domain_sections=kg_domain_sections,
            modern_semantics=self.modern_semantics,
            q_emb_norm=q_emb_norm,
            modern_embs_norm=self._modern_embs_norm,
            modern_list=self.modern_list,
            top_k_modern=15,
            question=question,
            is_criminal=criminal,
            explicit_crpcs=explicit_crpcs,
            crpc_to_bnss=self.crpc_to_bnss,
            bnss_subjects=self.bnss_subjects,
            use_semantic_search=self.use_semantic_search,
        )

        # 4. Build prompt
        prompt = build_domain_prompt(
            question=question,
            chain_extended=chain_extended,
            top_k=top_k,
            statutory_menu=statutory_menu,
            domain_name=domain_name,
            is_criminal=criminal,
            corpus_domain=corpus_domain,
        )

        mapped_bnss = [
            f"BNSS {b} (ex-CrPC {c})"
            for c in sorted(explicit_crpcs) for b in lookup_bnss(c, self.crpc_to_bnss)
        ]
        return {
            "prompt": prompt,
            "statutory_menu": statutory_menu,
            "domain_name": domain_name,
            "discovered_ipcs": sorted(list(explicit_ipcs | kg_ipcs)),
            "discovered_ieas": sorted(list(explicit_ieas | kg_ieas)),
            "discovered_crpcs": sorted(list(explicit_crpcs)),
            "discovered_domain_sections": sorted(list(kg_domain_sections.keys())),
            "mapped_bns_sections": mapped_bns_list,
            "mapped_bnss_sections": mapped_bnss,
            "has_ipc_conversion": bool(explicit_ipcs or kg_ipcs or mapped_bns_list),
        }

    def sanitize_answer(self, text: str) -> str:
        """
        Deterministic post-processing shield for legacy citations in the final answer:
          - 'Section 420 IPC'   -> 'Section 318(4) of the Bharatiya Nyaya Sanhita (BNS), 2023 (formerly Section 420 IPC)'
          - 'Section 482 CrPC'  -> 'Section 528 of the Bharatiya Nagarik Suraksha Sanhita (BNSS), 2023 (formerly Section 482 CrPC)'
          - 'Section 65B IEA'   -> 'Section 63 of the Bharatiya Sakshya Adhiniyam (BSA), 2023 (formerly Section 65B ...)'
        Keeping the old reference matters because offences committed (and proceedings
        started) before 1 July 2024 are still governed by the old codes.

        Safety rules:
          - A section with NO entry in the concordance is never relabelled with its old
            number as if it were a BNS/BNSS/BSA section; it is marked as repealed instead.
          - Citations that are already annotated ('... (formerly Section 420 IPC)') are left alone.
          - Mentions of the old codes in transition sentences ('the IPC was replaced by the BNS')
            are left alone.
        """
        if not text:
            return text

        # Placeholders protect text we deliberately keep (legacy references, already
        # annotated citations, transition sentences) from later replacement steps.
        # Each protected span is restored verbatim at the end.
        kept: List[str] = []

        def protect(s: str) -> str:
            kept.append(s)
            return f"\ue001{len(kept) - 1}\ue002"

        P_IPC, P_IEA, P_CRPC = protect("IPC"), protect("Indian Evidence Act"), protect("CrPC")

        # "Section 420 IPC (now Section 318(4) BNS)" is the transition format the prompt asks
        # for (offences / proceedings before 1 July 2024): the old citation is followed by
        # its modern equivalent in brackets and must be kept as written.
        modern_follows = re.compile(
            r'\s*[,(\[]\s*(?:now|presently|currently|i\.e\.|which\s+is\s+now|corresponding\s+to|'
            r'equivalent\s+to|replaced\s+by|new|under\s+the\s+new)\b[^\n]{0,80}?\b(?:BNS|BNSS|BSA)\b',
            re.IGNORECASE,
        )

        def already_annotated(m) -> bool:
            before = m.string[max(0, m.start() - 20):m.start()].lower()
            if any(w in before for w in ("formerly", "erstwhile", "earlier", "previously", "old ")):
                return True
            return bool(modern_follows.match(m.string, m.end()))

        def join_secs(secs: List[str]) -> str:
            return ("Sections " if len(secs) > 1 else "Section ") + " and ".join(secs)

        sec_tok = r'(\d{1,3}[A-Za-z]?(?:\(\d+\))?)'
        prefix_re = r'(\b(?:under\s+|per\s+|as\s+per\s+)?)'

        # 1. Evidence Act (IEA) -> BSA
        def repl_iea(m):
            if already_annotated(m):
                return protect(m.group(0))
            prefix, sec_num = m.group(1) or "", m.group(2)
            targets = [t.replace("BSA ", "").replace("BSA_", "") for t in self.iea_to_bsa.get(normalize_iea_section(sec_num), [])]
            if targets:
                return (f"{prefix}{join_secs(targets)} of the Bharatiya Sakshya Adhiniyam (BSA), 2023 "
                        f"(formerly Section {sec_num} of the {P_IEA})")
            return f"{prefix}Section {sec_num} of the repealed {P_IEA} (BSA equivalent not identified)"

        iea_pattern = re.compile(
            prefix_re + r'(?:sections?|sec\.?|s\.)\s*(\d{1,3}[A-Za-z]?)\s*(?:of\s+(?:the\s+)?)?'
            r'(?:indian\s+evidence\s+act(?:\s*\(iea\))?|evidence\s+act|iea)\b',
            re.IGNORECASE,
        )
        text = iea_pattern.sub(repl_iea, text)

        # 2. CrPC -> BNSS
        def crpc_repl_factory(sec_group: int, prefix_group: Optional[int]):
            def repl(m):
                if already_annotated(m):
                    return protect(m.group(0))
                sec_num = m.group(sec_group)
                prefix = (m.group(prefix_group) or "") if prefix_group else ""
                targets = lookup_bnss(sec_num, self.crpc_to_bnss)
                if targets:
                    return (f"{prefix}{join_secs(targets)} of the Bharatiya Nagarik Suraksha Sanhita (BNSS), 2023 "
                            f"(formerly Section {sec_num} {P_CRPC})")
                return protect(m.group(0))   # unknown in concordance: keep as written
            return repl

        # No trailing '\.?' so a sentence-ending full stop after "CrPC." is not swallowed
        crpc_name = r'(?:code\s+of\s+criminal\s+procedure(?:\s*\(crpc\))?|cr\.?p\.?c\b)'
        crpc_pat_a = re.compile(prefix_re + r'(?:sections?|sec\.?|s\.)\s*' + sec_tok + r'\s*(?:of\s+(?:the\s+)?)?' + crpc_name, re.IGNORECASE)
        crpc_pat_b = re.compile(crpc_name + r'\s*,?\s*(?:sections?|sec\.?|s\.)\s*' + sec_tok, re.IGNORECASE)
        crpc_pat_c = re.compile(r'\bu/s\s+' + sec_tok + r'\s*(?:of\s+(?:the\s+)?)?' + crpc_name, re.IGNORECASE)
        text = crpc_pat_a.sub(crpc_repl_factory(2, 1), text)
        text = crpc_pat_b.sub(crpc_repl_factory(1, None), text)
        text = crpc_pat_c.sub(crpc_repl_factory(1, None), text)

        # 3. IPC -> BNS
        ipc_name = r'(?:indian\s+penal\s+code(?:\s*\(ipc\))?|ipc|i\.p\.c\.?)'

        def ipc_targets(sec_num: str) -> List[str]:
            return self.ipc_to_bns.get(normalize_ipc_section(sec_num), [])

        def repl_compound(m):
            if already_annotated(m):
                return protect(m.group(0))
            prefix, s1, s2 = m.group(1) or "", m.group(2), m.group(3)
            b1, b2 = ipc_targets(s1), ipc_targets(s2)
            if not (b1 and b2):
                return protect(m.group(0))
            return (f"{prefix}Sections {b1[0]} and {b2[0]} of the Bharatiya Nyaya Sanhita (BNS), 2023 "
                    f"(formerly Sections {s1} and {s2} {P_IPC})")

        compound_pat = re.compile(
            prefix_re + r'(?:sections?|sec\.?|s\.)\s*(\d{1,3}[A-Za-z]?)\s*(?:and|&|,)\s*(\d{1,3}[A-Za-z]?)\s*'
            r'(?:of\s+(?:the\s+)?)?' + ipc_name + r'\b',
            re.IGNORECASE,
        )
        text = compound_pat.sub(repl_compound, text)

        def single_ipc(prefix: str, sec_num: str, original: str) -> str:
            targets = ipc_targets(sec_num)
            if targets:
                return (f"{prefix}{join_secs(targets)} of the Bharatiya Nyaya Sanhita (BNS), 2023 "
                        f"(formerly Section {sec_num} {P_IPC})")
            return f"{prefix}Section {sec_num} of the repealed {P_IPC} (BNS equivalent not identified)"

        ipc_pattern_1 = re.compile(
            prefix_re + r'(?:sections?|sec\.?|s\.)\s*(\d{1,3}[A-Za-z]?)\s*(?:of\s+(?:the\s+)?)?' + ipc_name + r'\b',
            re.IGNORECASE,
        )
        text = ipc_pattern_1.sub(
            lambda m: protect(m.group(0)) if already_annotated(m) else single_ipc(m.group(1) or "", m.group(2), m.group(0)),
            text,
        )

        ipc_pattern_2 = re.compile(
            prefix_re + ipc_name + r'\s*(?:sections?|sec\.?|s\.)\s*(\d{1,3}[A-Za-z]?)',
            re.IGNORECASE,
        )
        text = ipc_pattern_2.sub(
            lambda m: protect(m.group(0)) if already_annotated(m) else single_ipc(m.group(1) or "", m.group(2), m.group(0)),
            text,
        )

        ipc_pattern_3 = re.compile(r'\bu/s\s+(\d{1,3}[A-Za-z]?)\s*(?:of\s+(?:the\s+)?)?' + ipc_name + r'\b', re.IGNORECASE)
        text = ipc_pattern_3.sub(
            lambda m: protect(m.group(0)) if already_annotated(m) else single_ipc("under ", m.group(1), m.group(0)),
            text,
        )

        # 4. Standalone act names -> modern names, EXCEPT in transition sentences
        transition_words = ("formerly", "repealed", "replaced", "replacing", "old ", "erstwhile",
                            "earlier", "previous", "superseded", "before 1 july", "prior to")

        def standalone(new_name: str):
            def repl(m):
                window = m.string[max(0, m.start() - 40):m.end() + 40].lower()
                if any(w in window for w in transition_words):
                    return protect(m.group(0))
                return new_name
            return repl

        text = re.sub(r'\bIndian\s+Penal\s+Code\s*(?:\(IPC\))?', standalone('Bharatiya Nyaya Sanhita (BNS), 2023'), text, flags=re.IGNORECASE)
        text = re.sub(r'(?<!\ue001)\bIPC\b(?!\ue001)', standalone('BNS'), text)
        text = re.sub(r'\bIndian\s+Evidence\s+Act\s*(?:\(IEA\))?', standalone('Bharatiya Sakshya Adhiniyam (BSA), 2023'), text, flags=re.IGNORECASE)
        text = re.sub(r'(?<!\ue001)\bIEA\b(?!\ue001)', standalone('BSA'), text)
        text = re.sub(r'\bCode\s+of\s+Criminal\s+Procedure\s*(?:\(CrPC\))?', standalone('Bharatiya Nagarik Suraksha Sanhita (BNSS), 2023'), text, flags=re.IGNORECASE)
        text = re.sub(r'(?<!\ue001)\bCr\.?P\.?C\b(?!\ue001)', standalone('BNSS'), text)

        # 5. Deduplicate doubled statute phrases
        for full in ("Bharatiya Nyaya Sanhita (BNS), 2023", "Bharatiya Nagarik Suraksha Sanhita (BNSS), 2023",
                     "Bharatiya Sakshya Adhiniyam (BSA), 2023"):
            esc = re.escape(full)
            text = re.sub(rf'(?:of\s+the\s+)?{esc}\s+(?:of\s+the\s+)?{esc}', f'of the {full}', text, flags=re.IGNORECASE)

        # 6. Restore deliberately kept text verbatim (nested tokens resolve inside-out)
        token = re.compile(r"\ue001(\d+)\ue002")
        for _ in range(5):
            if not token.search(text):
                break
            text = token.sub(lambda m: kept[int(m.group(1))], text)
        return text
