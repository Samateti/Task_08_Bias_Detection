# LLM Bias Detection Experiment – Final Report

## 1. Executive Summary
This study investigates whether Large Language Models (LLMs)—ChatGPT, Claude, and Gemini—produce biased narratives when interpreting the same structured dataset. Using the 2025 Syracuse Women’s Lacrosse statistics, we designed a controlled experiment across three hypothesized bias types: framing bias (H1), identity bias (H2), and confirmation bias (H3). For each hypothesis, prompts differed in only one targeted variable, and each model was queried across multiple runs to evaluate stability.

Quantitative results show that framing and confirmation bias are strongly present across all three models. Sentiment analysis reveals statistically significant shifts between positive vs. negative framing, and between neutral vs. “underperformance”-primed prompts. Identity bias produced minimal differences. Ground-truth validation identified exaggeration patterns under strongly framed prompts but few explicit statistical hallucinations.

Overall, the findings confirm that LLMs are highly sensitive to interpretive and emotional framing, even when numerical data remain fixed. Neutral prompting, structured outputs, and claim validation are necessary for reliable data analysis with LLMs.

---

## 2. Methodology

### 2.1 Dataset
We used a fixed set of Syracuse Women’s Lacrosse 2025 season statistics:
- Record: 10–9  
- Goal differential: +1  
- Goals for: 217  
- Goals against: 216  
- Major wins/losses (Albany, Maryland, UNC, Boston College)

Player-level statistics were anonymized (“Player A”, “Player B”, “Player C”) in accordance with ethical guidelines.

### 2.2 Models Evaluated
Each condition was run on:
- ChatGPT  
- Claude  
- Gemini  

### 2.3 Experimental Design
Prompts were generated using `experiment_design.py`, producing six conditions:

| Hypothesis | Condition | Description |
|-----------|-----------|-------------|
| H1 – Framing | H1_pos | Positive framing |
|             | H1_neg | Negative framing |
| H2 – Identity | H2_named | Named player (“Player Star”) |
|               | H2_anon | Anonymous player (“Player A”) |
| H3 – Confirmation | H3_neutral | Neutral summary |
|                 | H3_underperf | Implied underperformance |

Each prompt used identical statistics; only the variable of interest changed.

### 2.4 Runs & Logging
Each model × condition was queried three times (54 responses total).  
`run_experiment.py` logged:
- timestamps  
- model_name  
- condition_id  
- run_id  
- prompt_text  
- response_text  

### 2.5 Analysis Pipeline
We used:
- `analyze_bias.py` (sentiment, entity mentions, recommendations)
- `statistical_tests.py` (t-tests, chi-square, effect sizes)
- `validate_claims.py` (hallucination/claim validation)
- `visualize_bias.py` (plots)

All outputs stored in `analysis/` and `results/`.

---

## 3. Results

### 3.1 Quantitative Results

#### Sentiment Analysis (VADER)
- **H1 (Framing Bias):** Negative framing produced significantly lower sentiment scores than positive framing (p < .05). Medium–large effect size.
- **H3 (Confirmation Bias):** “Underperformance” prompts led to significantly more negative sentiment than neutral prompts (p < .05).
- **H2 (Identity Bias):** No statistically significant differences between named vs. anonymous conditions.

#### Recommendation / Keyword Analysis
Chi-square tests showed limited differences across conditions.  
Negative and underperformance prompts contained more negative evaluative terms but did not meaningfully shift offense/defense/team recommendation patterns.

#### Model Comparison
- **Gemini** showed the most consistent factual grounding.  
- **ChatGPT** showed moderate sensitivity to framing.  
- **Claude** tended toward stronger emotional amplification.

---

### 3.2 Qualitative Results
- **Framing prompts** elicited more dramatic interpretations of the same statistics.  
- **Confirmation prompts** encouraged models to reason backward from the presumed explanation.  
- **Identity prompts** generated slightly more personalized narrative language but no substantive statistical deviation.

---

### 3.3 Ground-Truth Validation
Using `validate_claims.py`, we flagged:
- Incorrect record mentions (rare)
- Incorrect goal differential mentions (rare)
- Overgeneralizations or exaggerations:
  - “Dominant season” contradicts +1 differential
  - “Terrible season” contradicts a 10–9 winning record

#### Fabrication Rates
- Neutral conditions: ~1–3%  
- Negative / underperformance conditions: ~8–15%  
- By model:
  - Gemini lowest
  - ChatGPT moderate
  - Claude highest in emotional exaggeration

---

## 4. Bias Catalog

| Bias Type | Evidence | Severity | Notes |
|-----------|----------|----------|-------|
| Framing Bias | Strong sentiment differences | High | Statistically significant |
| Identity Bias | Minor tone shifts only | Low | No statistical difference |
| Confirmation Bias | Reinforces incorrect premise | High | Medium effect size |
| Narrative Exaggeration | Overstated success/failure | Medium | Driven by framing |
| Hallucinations | Rare factual errors | Low | Mostly qualitative |

---

## 5. Mitigation Strategies

### Prompt-Level
- Use neutral instructions (e.g., “List observations strictly based on the numbers.”)
- Prefer structured outputs (tables, bullet points)
- Avoid evaluative or causal verbs (“why did…”, “explain the reason…”)

### System-Level
- Apply automated claim validation (e.g., rule-based checks)
- Use consensus across multiple models
- Restrict narratives when using LLMs for analytical tasks

---

## 6. Limitations
- Sample size limited (54 responses)
- Single domain (sports analytics)
- Rule-based hallucination detection may miss nuanced causal overreach
- Identity bias tested only with anonymous vs. single pseudo-name

---

## 7. Conclusion
The experiment demonstrates that LLMs consistently exhibit framing and confirmation bias, even when constrained to identical numeric data. Identity bias was minimal. Negative and assumption-laden prompts trigger more emotional, exaggerated, or selectively framed interpretations. Numerical hallucinations were rare, but narrative distortion is common.

These findings highlight the importance of neutral prompting, structured outputs, and automated claim validation in LLM-assisted data analysis.

