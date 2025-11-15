# LLM Bias Detection in Data Narratives  
### Syracuse University – IST Research Task 08  
### Author: Sathvik Mateti

---

## 📌 Project Overview
This project analyzes **bias in Large Language Models (LLMs)** when interpreting the *same dataset* under different prompt conditions. Using the 2025 Syracuse Women’s Lacrosse statistics, the experiment evaluates whether models produce biased narratives based on:

1. **Framing Bias (H1)** – Positive vs. negative prompt framing  
2. **Identity Bias (H2)** – Named player vs. anonymized player  
3. **Confirmation Bias (H3)** – Neutral prompt vs. “underperformance” assumption  

Three LLMs were tested:

- **ChatGPT**
- **Claude**
- **Gemini**

Each model was queried across **three runs per condition**, generating a total of **54 responses**.

All results, prompts, analysis code, and outputs are included in this repository.

---

## 📂 Repository Structure

```plaintext
├── prompts/
│   ├── prompts.csv
│   └── prompts.json
│
├── results/
│   ├── Run1_chatgpt_responses.json
│   ├── Run1_claude_responses.json
│   ├── Run1_gemini_responses.json
│   ├── Run2_chatgpt_responses.json
│   └── ... (all raw model outputs)
│
├── analysis/
│   ├── sentiment_by_condition.csv
│   ├── sentiment_by_condition_model.csv
│   ├── entity_mentions.csv
│   ├── recommendations_by_condition.csv
│   ├── stat_ttests.csv
│   ├── stat_chi_square.csv
│   ├── validation_flags.csv
│   └── fabrication_rates_by_condition.csv
│
├── experiment_design.py
├── run_experiment.py
├── analyze_bias.py
├── statistical_tests.py
├── validate_claims.py
├── visualize_bias.py
│
├── REPORT.md
└── README.md
