# A Knowledge-Infused Framework for Multi-dimensional Legal Question Answering using Hierarchical Graphs and Reinforcement Learning

A multi-dimensional Legal Question Answering framework that combines **Hierarchical Knowledge Graphs (HKG)**, **Reinforcement Learning (RL)**-based reasoning-path expansion, and **neural past-case retrieval** to generate structured, interpretable legal answers. Built and evaluated on real-world queries from the [Vidhikarya](https://www.vidhikarya.com/) platform across Employment & Labour, Criminal, and Family law domains.

## Overview

Legal QA is hard because platforms like Vidhikarya provide **multiple, sometimes conflicting** advocate responses per query, and legal reasoning requires multi-hop traversal across statutes, sections, and case law. This project addresses both problems:

1. **Ground-truth construction** — an NLI-based contradiction filter + pairwise semantic comparison + three-dimensional scoring (Informativeness, Readability, Actionability) to select a reliable best answer per query from noisy multi-advocate data.
2. **Hierarchical Knowledge Graph** — 8 node types (Constitutional, Act, Section, Regulatory, CaseLaw, Procedural, Concept, Query) connected via rule-based and semantic edges, capturing legal knowledge at multiple levels of abstraction.
3. **RL-based reasoning** — a PPO agent (with GAE, chain-attention pooling, residual policy network) that extends a query's fact-rule chain by traversing the graph toward gold reasoning paths.
4. **RAG generation** — a neural style retriever surfaces relevant past cases, which are combined with the RL-expanded chain to prompt an LLM for the final answer.
