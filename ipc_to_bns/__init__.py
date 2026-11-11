"""
IPC to BNS Statutory Conversion Layer
(IPC -> BNS, CrPC -> BNSS, Indian Evidence Act -> BSA)
"""

from .bns_converter import (
    BNSConverter,
    extract_statutes_from_text,
    extract_2hop_statutory_sections,
    build_statutory_candidate_menu,
    build_domain_prompt,
    normalize_ipc_section,
    normalize_iea_section,
    normalize_crpc_section,
    load_statutory_mappings,
    load_crpc_mappings,
    lookup_bnss,
    load_node_metadata,
    detect_legal_domain,
    classify_unlabelled_section,
)

__all__ = [
    "BNSConverter",
    "extract_statutes_from_text",
    "extract_2hop_statutory_sections",
    "build_statutory_candidate_menu",
    "build_domain_prompt",
    "normalize_ipc_section",
    "normalize_iea_section",
    "normalize_crpc_section",
    "load_statutory_mappings",
    "load_crpc_mappings",
    "lookup_bnss",
    "load_node_metadata",
    "detect_legal_domain",
    "classify_unlabelled_section",
]
