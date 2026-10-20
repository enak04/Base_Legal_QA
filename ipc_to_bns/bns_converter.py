"""
BNS Converter & Statutory Transition Layer
==========================================

Acts as the statutory conversion layer after retrieval/reasoning and before
the final LLM (GPT-4o-mini) generation.

Key capabilities:
  1. Statutory Concordance: Maps legacy Indian Penal Code (IPC, 1860) sections
     to modern Bharatiya Nyaya Sanhita (BNS, 2023) provisions using
     bns_ipc_mapping_clean.csv. Also supports Indian Evidence Act (IEA) ->
     Bharatiya Sakshya Adhiniyam (BSA, 2023) and domain acts.
  2. Multi-source Statute Discovery:
     - Regex parsing over user questions and retrieved historical cases.
     - 2-hop traversal over Knowledge Graph reasoning chains.
  3. Semantic Candidate Ranking: Embeds and ranks modern provisions contextually
     against the user query using InLegalBERT.
  4. Domain-Adaptive Prompt Construction: Instructs GPT-4o-mini with negative
     constraints against citing legacy IPC sections and provides modern BNS candidates.
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


# ── Multi-Statute Text Extractor (Regex) ──────────────────────────────────────

def extract_statutes_from_text(text: str) -> dict:
    """
    Extracts statutory references from text (queries or past cases):
      1. IPC Sections (e.g., 'sec 498A', 'charges 302, 307', 'u/s 420 IPC', 'under Section 406')
      2. IEA Sections (e.g., 'Section 65B of Evidence Act', 'sec 27 IEA')
      3. Domain Sections (e.g., 'Section 13B of Hindu Marriage Act', 'Section 25F IDA')
    """
    found = {"ipcs": set(), "ieas": set(), "domain_sections": set()}
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
    for match in domain_pattern.finditer(cleaned_no_iea):
        sec_part, act_part = match.group(1), match.group(2)
        found["domain_sections"].add(f"{act_part.strip().upper()} Section {sec_part.strip()}")

    masked_ipc = domain_pattern.sub(r' [OTHER_ACT: \1 \2] ', cleaned_no_iea)

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
    pat_ipc_c = re.compile(
        rf'(?:in\s+sec(?:tion)?\.?|charges(?:\s+under)?|u/s)\s*{sec_list_pattern}',
        re.IGNORECASE
    )
    pat_ipc_d = re.compile(
        rf'(?:section|sec\.)\s*({sec_token})\s+(?:of\s+)?{ipc_marker}',
        re.IGNORECASE
    )

    for pat in (pat_ipc_a, pat_ipc_b, pat_ipc_c, pat_ipc_d):
        for match in pat.findall(masked_ipc):
            for num in re.findall(sec_token, match):
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

    try:
        df = pd.read_csv(nodes_csv_path)
    except Exception:
        df = pd.read_csv(nodes_csv_path, engine="python", on_bad_lines="skip")

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
) -> Tuple[str, List[str]]:
    """
    Compiles candidate legal menu for the modern prompt:
      1. Modern BNS provisions (from IPC mentions and graph discoveries)
      2. Modern BSA provisions (from IEA mentions and graph discoveries)
      3. Domain-specific statutory provisions (e.g., IDA, PGA, HMA)
    Returns:
      (menu_text: str, mapped_bns_list: List[str])
    """
    candidate_sections = []
    seen_modern = set()
    mapped_bns_list = []

    # 1. Modern Penal Provisions (BNS)
    bns_lines = []
    for ipc in sorted(explicit_ipcs):
        for bns in ipc_to_bns.get(ipc, []):
            if bns not in seen_modern:
                desc = modern_semantics.get(bns, "Penal provision")
                bns_lines.append(f"- Section {bns} BNS (formerly IPC Section {ipc}): {desc}")
                seen_modern.add(bns)
                mapped_bns_list.append(f"BNS {bns} (ex-IPC {ipc})")

    # 2. Modern Evidence Provisions (BSA)
    bsa_lines = []
    for iea in sorted(explicit_ieas):
        for bsa in iea_to_bsa.get(iea, []):
            if bsa not in seen_modern:
                desc = modern_semantics.get(bsa, "Evidence provision")
                bsa_lines.append(f"- Section {bsa} BSA (formerly Evidence Act Section {iea.replace('IEA_', '')}): {desc}")
                seen_modern.add(bsa)

    # 3. Semantic Ranking of KG-Discovered Provisions
    kg_candidates = set()
    for ipc in kg_ipcs:
        for bns in ipc_to_bns.get(ipc, []):
            if bns not in seen_modern:
                kg_candidates.add(bns)
    for iea in kg_ieas:
        for bsa in iea_to_bsa.get(iea, []):
            if bsa not in seen_modern:
                kg_candidates.add(bsa)

    if kg_candidates and q_emb_norm is not None and modern_embs_norm is not None and modern_list:
        mod_idx_map = {m: i for i, m in enumerate(modern_list)}
        cand_indices = [mod_idx_map[m] for m in kg_candidates if m in mod_idx_map]

        if cand_indices:
            cand_tensor = modern_embs_norm[cand_indices]
            sims = torch.mm(q_emb_norm, cand_tensor.t()).squeeze(0)
            k = min(top_k_modern, len(cand_indices))
            if k > 0:
                top_k_scores, top_k_idx = torch.topk(sims, k)
                for idx in top_k_idx:
                    mod_cand = modern_list[cand_indices[idx.item()]]
                    desc = modern_semantics.get(mod_cand, "Statutory provision")
                    if mod_cand.startswith("BSA"):
                        bsa_lines.append(f"- Section {mod_cand} BSA (Contextually retrieved Evidence Law): {desc}")
                    else:
                        bns_lines.append(f"- Section {mod_cand} BNS (Contextually retrieved Penal Law): {desc}")
                        mapped_bns_list.append(f"BNS {mod_cand}")
                    seen_modern.add(mod_cand)
    else:
        # Fallback when embeddings not supplied
        for bns in sorted(kg_candidates):
            desc = modern_semantics.get(bns, "Statutory provision")
            if bns.startswith("BSA"):
                bsa_lines.append(f"- Section {bns} BSA: {desc}")
            else:
                bns_lines.append(f"- Section {bns} BNS: {desc}")
                mapped_bns_list.append(f"BNS {bns}")
            seen_modern.add(bns)

    if bns_lines:
        candidate_sections.append("MODERN PENAL PROVISIONS (Bharatiya Nyaya Sanhita - BNS, 2023):\n" + "\n".join(bns_lines))
    if bsa_lines:
        candidate_sections.append("MODERN EVIDENCE PROVISIONS (Bharatiya Sakshya Adhiniyam - BSA, 2023):\n" + "\n".join(bsa_lines))

    # 4. Domain Statutory Provisions (Civil, Matrimonial, Labour)
    if kg_domain_sections:
        dom_lines = []
        for sec_name, sec_desc in sorted(kg_domain_sections.items())[:10]:
            dom_lines.append(f"- {sec_name}: {sec_desc}")
        candidate_sections.append("DOMAIN STATUTORY PROVISIONS (Retrieved from Subject Knowledge Graph):\n" + "\n".join(dom_lines))

    if not candidate_sections:
        menu_text = "No specific direct modern provisions mapped from precedents. Apply general modern statutory principles."
    else:
        menu_text = "\n\n".join(candidate_sections)

    return menu_text, mapped_bns_list


# ── Domain-Adaptive Prompt Builder ────────────────────────────────────────────

def build_domain_prompt(
    question: str,
    chain_extended: list,
    top_k: list,
    statutory_menu: str,
    domain_name: str = "Criminal Law"
) -> str:
    """Structured CoT prompt instructing LLM to cite modern BNS/BSA rather than legacy IPC."""
    chain_str = " → ".join(chain_extended) if chain_extended else "N/A"

    context_lines = []
    for i, r in enumerate(top_k, 1):
        context_lines.append(f"[Historical Case {i}]")
        context_lines.append(f"Query: {r.get('question', '')}")
        context_lines.append(f"Precedent Ratio: {r.get('answer', '')}")
        context_lines.append("")
    context_str = "\n".join(context_lines).strip()

    return f"""You are a legal expert advising a client on a {domain_name} matter. Use the legal reasoning chain and precedent cases below to understand the legal principles and ratio decidendi.

STATUTORY FRAMEWORK TRANSITION RULES:
1. The Indian Penal Code (IPC), 1860 has been superseded by the Bharatiya Nyaya Sanhita (BNS), 2023. When citing criminal or penal offences, ONLY cite the BNS sections provided in the Candidate Provisions below. Never cite legacy IPC sections.
2. The Indian Evidence Act (IEA), 1872 has been superseded by the Bharatiya Sakshya Adhiniyam (BSA), 2023. When citing evidentiary principles or burden of proof, ONLY cite the BSA sections provided below.
3. For civil, employment, labour, or matrimonial matters (e.g. Industrial Disputes Act, Payment of Gratuity Act, Maternity Benefit Act, POSH Act, Hindu Marriage Act), cite the appropriate Domain Statutory Provisions provided below.

CANDIDATE STATUTORY PROVISIONS (MODERN LAW):
{statutory_menu}

REASONING CHAIN (Concept Path):
{chain_str}

HISTORICAL CASE LAW (Precedents may cite legacy statutes — apply their ratio using the modern provisions above):
{context_str}

USER'S QUESTION:
{question}

INSTRUCTIONS:
1. Start your answer by directly addressing the user's specific legal issue in plain English, just like an experienced advocate advising a client.
2. Cite the exact modern statutory provisions (BNS, BSA, and applicable Domain Acts) from the Candidate Provisions above.
3. State clearly what the user CAN and CANNOT do legally.
4. Give 1-2 concrete, actionable next steps (e.g. filing a police complaint, approaching the Magistrate, or petitioning the Labour Court/Authority).
5. CRITICAL: Do NOT copy legacy section numbers from the user's question into BNS/BSA (e.g. if the user says "charges 498A or 420 or 302", cite BNS 85, BNS 318(4), or BNS 103, NOT BNS 498A, 420, or 302).
6. Be concise (3-5 sentences), authoritative, and practically useful.

ANSWER:"""


# ── BNS Converter Class ───────────────────────────────────────────────────────

class BNSConverter:
    """
    Modular statutory conversion engine that plugs into any legal QA pipeline.
    """

    def __init__(self, mapping_csv_path: Optional[str] = None):
        self.mapping_csv_path = resolve_mapping_path(mapping_csv_path)
        self.ipc_to_bns, self.iea_to_bsa, self.modern_semantics = load_statutory_mappings(self.mapping_csv_path)
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
    ) -> dict:
        """
        Executes the conversion layer:
          1. 2-hop graph harvesting of IPC/IEA/domain nodes.
          2. Regex extraction from question + retrieved cases.
          3. Conversion to modern BNS/BSA provisions.
          4. Structured modern prompt construction.
        """
        node_type_map = node_type_map or {}
        node_desc_map = node_desc_map or {}

        # 1. 2-hop KG discovery
        kg_ipcs, kg_ieas, kg_domain_sections = extract_2hop_statutory_sections(
            G, chain_extended, node_type_map, node_desc_map
        )

        # 2. Text regex extraction from user question + precedents
        text_statutes = extract_statutes_from_text(question)
        explicit_ipcs = set(text_statutes["ipcs"])
        explicit_ieas = set(text_statutes["ieas"])

        for item in top_k:
            q_stat = extract_statutes_from_text(item.get("question", ""))
            a_stat = extract_statutes_from_text(item.get("answer", ""))
            explicit_ipcs |= q_stat["ipcs"] | a_stat["ipcs"]
            explicit_ieas |= q_stat["ieas"] | a_stat["ieas"]

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
        )

        # 4. Build prompt
        prompt = build_domain_prompt(
            question=question,
            chain_extended=chain_extended,
            top_k=top_k,
            statutory_menu=statutory_menu,
            domain_name=domain_name,
        )

        return {
            "prompt": prompt,
            "statutory_menu": statutory_menu,
            "discovered_ipcs": sorted(list(explicit_ipcs | kg_ipcs)),
            "discovered_ieas": sorted(list(explicit_ieas | kg_ieas)),
            "discovered_domain_sections": sorted(list(kg_domain_sections.keys())),
            "mapped_bns_sections": mapped_bns_list,
            "has_ipc_conversion": bool(explicit_ipcs or kg_ipcs),
        }
