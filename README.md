# MedDistractQA

MedDistractQA is a benchmark designed to assess the ability of large language models (LLMs) to filter relevant clinical information from extraneous data, a common challenge in real-world medical applications. In clinical scenarios—especially with the rise of ambient dictation and assistive technologies that automatically generate notes—LLMs are exposed to distractions that can degrade their performance.

## Overview

Large language models have the potential to transform medicine, but real-world clinical environments often include noise and irrelevant details that hinder accurate interpretation. MedDistractQA addresses this challenge by embedding USMLE-style questions with simulated distractions such as:
- **MedDistractQA-Nonliteral: Polysemous words:** Terms with clinical meanings used in non-clinical contexts.
- **MedDistractQA-Bystander: Unrelated health references to third parties:** Mentions of health conditions that do not pertain to the core clinical query, but instead applied to an irrelevant third party.

Our benchmark experiments reveal that these distractions can reduce LLM accuracy by up to **17.9%**. Notably, common approaches like retrieval-augmented generation (RAG) and medical fine-tuning, in some cases, introduced additional complexities that degraded performance.

## Key Features

- **Robust Benchmark:** USMLE-style questions enhanced with real-world simulated distractions.
- **Insightful Findings:** Empirical evidence showing LLM performance drops when faced with extraneous information.
- **Evaluation Framework:** Tools and scripts to assess LLM resilience and guide the development of mitigation strategies.

To access the benchmark itself, please visit `https://huggingface.co/datasets/KrithikV/MedDistractQA/`
