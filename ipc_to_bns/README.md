# Multi-Domain Modern Statutory Legal QA Pipeline (`run_bns_queries.py`)

## 1. Overview & Purpose
`run_bns_queries.py` is the end-to-end inference and evaluation engine for modern Indian legal question answering. 

Foundational legal language models, retrieval databases, and knowledge graphs were historically trained on legacy British-era statutes:
* **Indian Penal Code (IPC), 1860**
* **Indian Evidence Act (IEA), 1872**
* **Code of Criminal Procedure (CrPC), 1973**

In 2023, the Indian Parliament enacted the new statutory criminal framework:
* **IPC $\rightarrow$ Bharatiya Nyaya Sanhita (BNS), 2023**
* **IEA $\rightarrow$ Bharatiya Sakshya Adhiniyam (BSA), 2023**
* **CrPC $\rightarrow$ Bharatiya Nagarik Suraksha Sanhita (BNSS), 2023**

Retraining foundational RL agents and language models from scratch on the new statutes is computationally prohibitive and lacks historical case law. `run_bns_queries.py` solves this via a **Dynamic 2-Hop Graph Traversal and Semantic Post-Translation Pipeline** that bridges the legacy-to-modern gap with zero hardcoding.

Furthermore, this script is **domain-generalized**: it unifies **Criminal Law**, **Family & Matrimonial Law**, and **Employment & Labour Law** under a single executable interface.

---

## 2. End-to-End Pipeline Architecture

For every user query, the pipeline executes the following 5 stages:

```
[User Query]
     │
     ▼
[Stage 1: Semantic Embedding & Starting Node Selection]
  - Query embedded via law-ai/InLegalBERT (128-dim dense tensor).
  - Cosine similarity against precomputed graph nodes selects the optimal Entry Node.
     │
     ▼
[Stage 2: RL Reasoning Chain Expansion & DSSM Precedent Retrieval]
  - Pre-trained PPO Agent traverses the Hierarchical KG for 4 hops to construct a reasoning chain.
  - Dual-encoder DSSM network matches (Query + Chain) against offline indexed cases (Top-3 precedents).
     │
     ▼
[Stage 3: Multi-Source Statutory Harvesting]
  - Dynamic 2-Hop Graph Traversal from all chain nodes:
      * Discovers IPC sections (for BNS translation).
      * Discovers IEA sections (for BSA translation).
      * Discovers Domain Sections (HMA, DVA, IDA, PGA) with legal descriptions from nodes.csv.
  - Multi-pattern Regex Parser extracts explicit section numbers from user query and retrieved cases.
     │
     ▼
[Stage 4: Semantic Candidate Ranking & Context Reduction]
  - IPC nodes mapped to BNS; IEA nodes mapped to BSA via statutory concordance tables.
  - InLegalBERT encodes candidate descriptions and ranks them via cosine similarity against user query.
  - Retains Top-15 most contextually relevant modern provisions (drastically reducing token window & hallucinations).
     │
     ▼
[Stage 5: Domain-Adaptive Generation (LLM)]
  - Formats context into structured CoT prompt with domain-specific legal advisory rules.
  - Generates client-facing legal advice via GPT-4o-mini with a domain-specific prompt (criminal: BNS/BNSS/BSA; family & labour: governing civil statutes) and the 1 July 2024 transition rule.
```

---

## 3. Multi-Domain Architecture (`--domain`)

The script features a built-in domain router via the `--domain` parameter. Specifying a domain directs the script to load that domain's specific Knowledge Graph, RL policy, and DSSM retrieval models.

### Domain Configurations & Asset Mapping

| Domain Flag | Domain Name | Knowledge Graph (`edges.csv` & `nodes.csv`) | RL Model (`rl_policy_model.pt`) | DSSM Index (`dssm_retrieval_index.pt`) | Default Query CSV |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`--domain criminal`** *(Default)* | Criminal Law | `HierKG2\edges.csv`<br>`HierKG2\nodes.csv` | `HierKG2\rl_policy_model.pt` | `HierKG2\dssm_retrieval_index.pt` | `my_bns_questions.csv` |
| **`--domain family`** | Family & Matrimonial Law | `..\..\family\actionable\edges.csv`<br>`..\..\family\actionable\nodes.csv` | `..\..\family\actionable\rl_policy_model.pt` | `..\..\family\actionable\dssm_retrieval_index.pt` | `..\..\family\actionable\my_questions.csv` |
| **`--domain labour`** | Employment & Labour Law | `..\..\Employment_Labour\actionable\edges.csv`<br>`nodes.csv` | Domain RL checkpoint / fallback | Domain DSSM index / fallback | `my_bns_questions.csv` |

### How File Resolution Works:
When `--domain family` is passed:
1. It automatically resolves the relative path to `family\actionable\edges.csv` (850 nodes, 9,362 edges containing Hindu Marriage Act, Domestic Violence Act, Special Marriage Act, plus matrimonial IPC sections like 498A, 406).
2. It loads `family\actionable\rl_policy_model.pt` into memory to guide the graph traversal along matrimonial legal principles.
3. It loads the family DSSM index to retrieve real-world family court case law.
4. It sets the prompt persona to: `"You are a legal expert advising a client on a Family & Matrimonial Law matter..."`

---

## 4. Statutory Translation Engine

### 1. Penal Law: IPC $\rightarrow$ BNS (2023)
* Loaded from `bns_ipc_mapping_clean.csv` (737 rows; rows whose `Old_IPC` starts with `IEA_` hold the Evidence Act mappings).
* Mappings are normalized to strip punctuation and sub-clauses (`498(A)` $\rightarrow$ `498A`).
* Examples:
  * Cheating: IPC 415 / 420 $\rightarrow$ **BNS 318(4)**
  * Domestic Cruelty: IPC 498A $\rightarrow$ **BNS 85 / 86**
  * Defamation: IPC 499 / 500 $\rightarrow$ **BNS 356(1)**
  * Bigamy: IPC 494 $\rightarrow$ **BNS 82**

### 2. Evidence Law: IEA $\rightarrow$ BSA (2023)
* Loaded from the `IEA_` rows of `bns_ipc_mapping_clean.csv`.
* Normalized with `IEA_` prefix (`IEA_27`, `IEA_65B`).
* Examples:
  * Discovery Statement: IEA 27 $\rightarrow$ **BSA 23(2)**
  * Expert Opinions: IEA 45 $\rightarrow$ **BSA 39**
  * Electronic Records Certificate: IEA 65B $\rightarrow$ **BSA 63**
  * Presumption of Abetment to Suicide: IEA 113A $\rightarrow$ **BSA 117**
  * Presumption of Dowry Death: IEA 113B $\rightarrow$ **BSA 118**

### 3. Domain Civil Statutes (Preserved Without BNS Translation)
When running on Family or Labour graphs, the 2-hop traversal encounters civil sections:
* `HMA_13B`: Divorce by Mutual Consent (*Hindu Marriage Act*)
* `DVA_12`: Application to Magistrate (*Protection of Women from Domestic Violence Act*)
* `IDA_25F`: Conditions precedent to retrenchment (*Industrial Disputes Act*)
* `PGA_4`: Payment of Gratuity (*Payment of Gratuity Act*)

The engine pulls human-readable descriptions directly from `nodes.csv` and provides them to the LLM under `DOMAIN STATUTORY PROVISIONS`, ensuring civil laws are accurately preserved and cited under their actual parent Acts.

### 4. Procedure: CrPC $\rightarrow$ BNSS (2023)
* Loaded from `crpc_bnss_mapping.csv` (67 commonly used sections). Every row was checked against the *Correspondence Table and Comparison Summary of BNSS 2023 and CrPC 1973* (CAPT Bhopal); `Source_Page` gives the page in that PDF.
* Sub-section aware where the table supports it (`156(3)` $\rightarrow$ **BNSS 175(3)**); otherwise falls back to the section (`167(2)` $\rightarrow$ **BNSS 187**).
* Examples: FIR 154 $\rightarrow$ **173** · Magistrate-ordered investigation 156(3) $\rightarrow$ **175(3)** · notice of appearance 41A $\rightarrow$ **35** · maintenance 125 $\rightarrow$ **144** · remand 167 $\rightarrow$ **187** · chargesheet 173 $\rightarrow$ **193** · anticipatory bail 438 $\rightarrow$ **482** · bail 439 $\rightarrow$ **483** · inherent powers / quashing 482 $\rightarrow$ **528**.

### 5. Accuracy safeguards
* **Domain awareness.** `detect_legal_domain()` classifies each question as Criminal, Family & Matrimonial, or Employment & Labour. Only criminal questions get BNS/BNSS/BSA framing, offence-keyword candidates, graph-discovered penal sections and the full-catalogue semantic search. Family and labour questions are answered under their civil statutes, and a penal section is listed only if the user raised it. (`auto_detect_domain=False` keeps a fixed domain, as `run_bns_queries.py` does.)
* **No forced citations.** The prompt no longer hard-codes a list of BNS sections or orders the model to cite BSA s.63 for all evidence; s.63 is offered only when electronic records are in the facts. The old "fallback" that inserted defamation/cheating/obscenity sections into every empty menu is removed.
* **Unlabelled section numbers.** "u/s 156(3) before the CJM" or "quashing u/s 482" are classified from context as CrPC; "FIR u/s 420" as IPC; ambiguous numbers are not converted (previously every unlabelled number was treated as IPC, e.g. 482 $\rightarrow$ BNS 345(3)).
* **Transition rule.** Offences committed, and proceedings started, before 1 July 2024 remain under the old codes, so the prompt asks for "Section 420 IPC (now Section 318(4) BNS)" in such cases, and `sanitize_answer()` keeps the old reference when converting.
* **Whole-BNS similarity search off by default.** Raw InLegalBERT [CLS] similarities do not separate relevant from irrelevant provisions (about 66% of all 686 provisions pass the 0.45 threshold and the top 15 lie within ~0.03), so this step was adding unrelated sections such as soldier abetment or bigamy to FIR and bail questions. Set `SEMANTIC_STATUTE_SEARCH=true` (config `semantic_statute_search`) to re-enable it for experiments.
* **Safe sanitizer.** An IPC/CrPC/IEA section that is not in the concordance is never relabelled with its old number as a BNS/BNSS/BSA section; already annotated citations and transition sentences are left untouched.
* Tests: `tests/test_bns_converter.py` (no model checkpoints or API key needed).

---

## 5. Backend Integration Guide

Backend developers (e.g. integrating into FastAPI, Flask, or microservices) can run this module in two ways:

### Option A: Command-Line Execution (CLI)

```bash
# 1. Run on default Criminal domain
python run_bns_queries.py --domain criminal --input my_bns_questions.csv --output bns_test_answers_v4.csv

# 2. Run on Family & Matrimonial domain
python run_bns_queries.py --domain family --output bns_family_answers.csv

# 3. Run on Labour domain
python run_bns_queries.py --domain labour --output bns_labour_answers.csv
```

#### CLI Flags:
* `--domain`: Target domain (`criminal`, `family`, `labour`). Default: `criminal`.
* `--input`: Path to input CSV containing legal queries. (Defaults to domain preset).
* `--output`: Path to write the output CSV containing answers. Default: `bns_test_answers_v4.csv`.

---

### Option B: Programmatic Python Import (Direct Service API)

For real-time API endpoints (e.g., FastAPI `/api/ask`), you can import `run_bns_pipeline` directly:

```python
from run_bns_queries import run_bns_pipeline

# Trigger inference programmatically for a specific domain and custom CSV:
run_bns_pipeline(
    domain="family",
    custom_config={
        "input_csv": "user_uploaded_queries.csv",
        "output_csv": "api_results.csv",
        "openai_model": "gpt-4o-mini"
    }
)
```

### Input CSV Schema:
The input CSV needs at least one column containing the text of the legal query:
* Accepted column headers: `question`, `query`, or `query_text` (case-insensitive).

### Output CSV Schema:
The generated output CSV contains full auditability for every turn:
1. `domain`: The active legal domain (`Criminal Law`, `Family Law`, etc.).
2. `question`: The user's original query.
3. `start_node`: The initial entry node selected by semantic similarity.
4. `rl_reasoning_chain`: The 4-hop chain path traversed by the RL agent (`Node A → Node B → Node C`).
5. `discovered_ipcs`: All legacy IPC sections harvested via graph and regex.
6. `discovered_ieas`: All legacy Evidence Act sections harvested.
7. `discovered_domain_sections`: Any civil/labour sections harvested (e.g. `HMA_13B`).
8. `statutory_candidates_provided`: The exact ranked BNS/BSA menu fed to the LLM.
9. `generated_modern_answer`: The final client-facing legal advice.

---

## 6. Environment & Performance Optimization

### 1. Environment Variables (`.env`)
Create a `.env` file in the working directory:
```ini
OPENAI_API_KEY=your_openai_api_key_here
```

### 2. Thread Settings (Windows / OpenBLAS)
To prevent thread contention and memory allocation errors on Windows, the script automatically sets single-threaded BLAS execution:
```python
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

### 3. Pre-computed Embedding Cache (`modern_statutory_cache.pt`)
The script automatically pre-computes dense embeddings for all 689 BNS and BSA descriptions using `law-ai/InLegalBERT` in fast batches ($<9$ seconds) and caches them to `modern_statutory_cache.pt`. On subsequent runs, startup latency drops to **0.01 seconds**.
