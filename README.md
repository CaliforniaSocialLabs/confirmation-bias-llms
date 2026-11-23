# Confirmation vs. Falsification Information-Seeking in Human-LLM Conversations

This repository contains the data pipeline and local LLM classification framework for studying **confirmation bias in human-AI information seeking** using the [`lmsys/lmsys-chat-1m`](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) dataset.

Specifically, we measure whether users who hold an explicit prior belief on a **political, social policy, or economic policy** issue query large language models using **confirming**, **falsifying**, or **neutral** language.

---

## 1. Theoretical Motivation & 3-Stage Hierarchical Rubric

A one-sided query (e.g., *"Why is government spending bad?"*) cannot be classified as confirmation-seeking unless we know the user's **prior belief**. Without a stated prior belief, the query could represent open curiosity, devil's advocacy, or a school assignment.

To ensure construct validity, every conversation is evaluated strictly on the **user's messages** (`user_only=True`, preventing contamination from the assistant's reply) across three sequential stages:

### Stage 1 — Domain / Topic Relevance (`is_policy_or_political`)
Is the user genuinely discussing or inquiring about **politics, social policy, economic policy, law/regulation, or a controversial social issue**?
- **Categories:** `politics`, `social_policy`, `economic_policy`, `other_controversial_social_issue`, or `non_relevant`.
- **Exclusion Rule (The "Pasted Context / NLP Task" Guard):** If a prompt mentions political terms solely within an external text pasted for summarization (e.g., *"Summarize this State of the Union speech"*), Named Entity Recognition, reading comprehension (`Context: ... Answer`), translation, or fictional roleplay, `is_policy_or_political` is set to `False` (`non_relevant`).

### Stage 2 — Explicit User Prior Belief (`user_states_prior_belief`)
If Stage 1 is `True`, does the user state or clearly imply their **own personal belief, stance, or hypothesis** in their own voice (e.g., *"I think government spending is bad"* or *"I believe universal basic income causes inflation"*)?
- Opinions appearing inside pasted quotes, articles, or transcripts are **not** attributed to the user.
- If `user_states_prior_belief` is `False`, Stage 3 is automatically gated to `"not_applicable"`.

### Stage 3 — Information-Seeking Direction (`seeking_type`)
Conditional on `is_policy_or_political == True` **and** `user_states_prior_belief == True`, how does the user seek information relative to their stated belief?

| `seeking_type` | Definition | Canonical Example |
| :--- | :--- | :--- |
| **`confirming`** | User asks for arguments, evidence, or validation that **supports/agrees with** their stated belief. | *"I think gov spending is bad. How is it bad?"* |
| **`falsifying`** | User asks for counter-arguments, counter-examples, or evidence that **challenges/contradicts** their stated belief. | *"I think gov spending is bad, how is it good?"* |
| **`neutral`** | User states a prior belief but asks for a **balanced, two-sided, or objective** overview. | *"I think gov spending is bad. Tell me the pros/cons?"* |
| **`not_applicable`** | Not a policy topic, no prior belief stated, or user states a belief without asking an information-seeking question. | *"How is government spending bad for the economy?"* (No prior belief stated) |

---

## 2. Dataset & Pre-Filtering Methodology (`src/load_data.py`)

The source dataset is [`lmsys/lmsys-chat-1m`](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) (1,000,000 real-world conversations from LMSYS Chatbot Arena). Because the majority of Chatbot Arena traffic consists of programming, math, creative writing, and NLP formatting tasks, classifying random rows wastes GPU compute on `non_relevant` conversations.

`src/load_data.py` streams from Hugging Face and applies a transparent, reproducible **pre-filter** on user turns:

1. **Language Filter:** Keeps conversations tagged `language == "English"` (configurable via `--language`).
2. **Noise & Copy-Paste Exclusion:**
   - Rejects prompts containing markdown code blocks (` ``` `).
   - Rejects prompts matching NLP/copy-paste task templates (`NOISE_TASK_PATTERNS`: e.g., `Context:`, `Passage:`, `Transcript:`, `Summarize`, `State of the Union`, `Paraphrase`, `Translate`, `Named Entity`, `Speaker 1:`, `Roleplay`, `Cover letter`, `Multiple-choice`, etc.).
   - Evaluates only conversational user turns between `20` and `1,500` characters (`--max-turn-chars 1500`) to exclude multi-page pasted articles.
3. **Same-Turn Topic + Belief Co-occurrence:**
   - Matches word-boundary political, economic, and social policy terms (`POLICY_TOPIC_TERMS`) and first-person belief markers (`BELIEF_MARKER_PATTERNS`, e.g., `"I think"`, `"I believe"`, `"in my opinion"`, `"I feel like"`, `"I support"`, `"I oppose"`, `"don't you think"`, etc.).
   - Requires the policy keyword and belief marker to co-occur within the **same user turn** so multi-turn conversations do not falsely trigger across unrelated turns.
   - *(Note: Belief-marker pre-filtering can be toggled off via `--no-belief-marker` to estimate unconditional base rates across all policy conversations.)*
4. **Chatbot Arena Side-by-Side Deduplication:**
   - In LMSYS Chatbot Arena, a single user prompt is frequently routed to two models simultaneously (creating duplicate user prompts with different `conversation_id` and `model` values).
   - `load_data.py` normalizes the first user turn (lowercase, whitespace-collapsed) and deduplicates on the fly (`--no-deduplicate` to disable).

---

## 3. Environment & Hardware Specifications

### Reference Hardware & Model
- **OS:** Windows (PowerShell)
- **GPU:** NVIDIA GeForce RTX 5070 Ti (16 GB VRAM)
- **Python:** `3.12.10`
- **Local LLM Runtime:** [Ollama](https://ollama.com/)
- **Classifier Model:** `gemma4:12b` (`6114515d63c1`, 8.0 GB quantized weights)
- **Decoding Parameters:**
  - `temperature = 0.0` (deterministic greedy decoding; `0.1` fallback on retry)
  - `num_ctx = 8192` tokens (with total user-text cap at `8,000` characters across multi-turn marathons)
  - `format = UserConfirmationClassification.model_json_schema()` (Pydantic v2 grammar-constrained JSON output)

### Repository Structure
```text
confirmationBias/
├── README.md                  # Research documentation & reproducibility guide
├── requirements.txt           # Python package dependencies
├── .env.example               # Template for HF_TOKEN and OLLAMA_MODEL
├── .gitignore                 # Excludes venv/, .env, and large data artifacts
├── data/
│   ├── raw/                   # Pre-filtered Parquet samples streamed from lmsys-chat-1m
│   └── processed/             # JSONL classification outputs from gemma4:12b
├── notebooks/
│   ├── 01_explore_data.ipynb          # Interactive exploration and rubric verification
│   └── 02_statistical_analysis.ipynb  # Interactive statistical tests, tables & robustness checks
└── src/
    ├── load_data.py           # Streaming/1M pre-filter, Arena deduplication, and full scanner
    ├── classify.py            # 3-stage Pydantic schema, system prompt, and Ollama checkpoint runner
    └── analyze.py             # Exact binomial, multinomial Chi2, cluster-bootstrap & fragility suite
```

---

## 4. Step-by-Step Reproducibility Guide

### Step 1: Set Up Python Virtual Environment & Ollama
```powershell
# 1. Activate or use the local Python 3.12 virtual environment
.\venv\Scripts\pip.exe install -r requirements.txt

# 2. Ensure Ollama is installed and pull Gemma 4 12B
ollama pull gemma4:12b
```

### Step 2: Configure Hugging Face Access
1. Accept the dataset license at [https://huggingface.co/datasets/lmsys/lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m).
2. Create a Read token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. Copy `.env.example` to `.env` and set your token:
   ```ini
   HF_TOKEN=hf_your_token_here
   OLLAMA_MODEL=gemma4:12b
   ```

### Step 3: Verify Classifier on Canonical Smoke Tests
Run the 6 built-in test cases (confirming, falsifying, neutral, no-prior-belief, non-political, and pasted-context trap):
```powershell
.\venv\Scripts\python.exe src\classify.py --smoke-test
```

### Step 4: Scan All 1,000,000 Conversations Locally (`--scan-all`)
You can extract candidates across all 1,000,000 rows using two complementary retrieval modes (or run the exact same pipeline on `allenai/WildChat-1M` via `--dataset wildchat`):

1. **Mode A — Explicit & Declarative Belief Markers (`candidates_full_1m_belief_crossturn.parquet`):**
   ```powershell
   .\venv\Scripts\python.exe src\load_data.py --scan-all --allow-cross-turn
   ```
2. **Mode B — GPU Sentence-Embedding Semantic Filter (`candidates_full_1m_all_policy_sem1000.parquet`):**
   Ranks all clean policy-question conversations using Ollama GPU embeddings (`nomic-embed-text`) without requiring literal `"I think"` / `"I believe"` regex phrases:
   ```powershell
   ollama pull nomic-embed-text
   .\venv\Scripts\python.exe src\load_data.py --scan-all --no-belief-marker --semantic-filter --semantic-top-k 1000
   ```
3. *(Optional)* **Cross-Dataset Replication on `allenai/WildChat-1M`:**
   ```powershell
   .\venv\Scripts\python.exe src\load_data.py --dataset wildchat --scan-all --allow-cross-turn
   ```

### Step 5: Run Batch Classification with `gemma4:12b` (Automatic Checkpoint/Resume)
`classify.py` automatically checkpoints every row to disk and resumes where it left off:
```powershell
# Classify Mode A candidates
.\venv\Scripts\python.exe src\classify.py --input data\raw\candidates_full_1m_belief_crossturn.parquet --no-think

# Classify Mode B (semantic-filtered) candidates
.\venv\Scripts\python.exe src\classify.py --input data\raw\candidates_full_1m_all_policy_sem1000.parquet --no-think
```

### Step 6: Run Statistical Tests, Export CSVs & Score Human Validation
`src/analyze.py` accepts one or multiple `.jsonl` files (automatically deduplicating by `conversation_id` when pooling runs):
```powershell
# 1. Analyze Mode A (Explicit Belief Markers) alone
.\venv\Scripts\python.exe src\analyze.py --input data\processed\classified_candidates_full_1m_belief_crossturn.jsonl --output-report data\processed\statistical_report.txt

# 2. Analyze Mode B (GPU Semantic Filter) alone
.\venv\Scripts\python.exe src\analyze.py --input data\processed\classified_candidates_full_1m_all_policy_sem1000.jsonl --output-report data\processed\statistical_report_sem1000.txt

# 3. Analyze Combined Deduplicated Pool (Mode A + Mode B) and export Stage-3 + Blind Validation CSVs
.\venv\Scripts\python.exe src\analyze.py --input data\processed\classified_candidates_full_1m_belief_crossturn.jsonl data\processed\classified_candidates_full_1m_all_policy_sem1000.jsonl --output-report data\processed\statistical_report_combined.txt --export-csv

# 4. Compute Cohen's kappa after filling out human_validation_blind_sample_60.csv
.\venv\Scripts\python.exe src\analyze.py --score-validation
```

---

## 6. Empirical Results (Full 1,000,000-Conversation Scan)

We evaluated `lmsys/lmsys-chat-1m` across all 1,000,000 conversations using both **Mode A (Explicit Belief Regex, $N = 1,021$)** and **Mode B (GPU Semantic-Embedding Filter without belief regex, $N = 1,000$)**, yielding **`1,922` unique classified candidates** and **`332` unique Stage-3 belief-bearing policy queries** (`data/processed/stage3_seeking_conversations_332.csv`).

### Comparison Across Retrieval Methods & Combined Pool

| Metric | Mode A: Explicit Belief Regex | Mode B: GPU Semantic Filter (`--no-belief-marker`) | **Combined Deduplicated Pool (A ∪ B)** |
| :--- | :---: | :---: | :---: |
| **Unique Candidates Classified** | `1,021` | `1,000` | **`1,922`** |
| **Stage 1: Policy / Political Topic** | `271` (`26.5%`) | `772` (`77.2%`) | **`966` (`50.3%`)** |
| **Stage 2: User States Prior Belief** | `216` (`79.7%`) | `241` (`31.2%`) | **`387` (`40.1%`)** |
| **Stage 3: Information-Seeking ($N$)** | `188` | `206` | **`332`** |
| **`confirming` Count & Share** | `121` (**64.4%**) `[57.3%, 70.9%]` | `141` (**68.4%**) `[61.8%, 74.4%]` | **`221` (66.6%) `[61.3%, 71.4%]`** |
| **`neutral` Count & Share** | `52` (**27.7%**) `[21.8%, 34.5%]` | `38` (**18.4%**) `[13.7%, 24.3%]` | **`80` (24.1%) `[19.8%, 29.0%]`** |
| **`falsifying` Count & Share** | `15` (**8.0%**) `[4.9%, 12.7%]` | `27` (**13.1%**) `[9.2%, 18.4%]` | **`31` (9.3%) `[6.7%, 12.9%]`** |
| **Directional Confirming Share ($p_{\text{confirm}}$)** | **89.0%** (`121` vs. `15`) | **83.9%** (`141` vs. `27`) | **87.7% (`221` vs. `31`)** |
| **Confirmation : Falsification Ratio** | **8.07×** | **5.22×** | **7.13×** |
| **Exact Binomial $p$-value (1-sided)** | $p = 4.52 \times 10^{-22}$ | $p = 4.02 \times 10^{-20}$ | **$p = 7.91 \times 10^{-37}$** |
| **Confirming vs. (Neutral + Falsifying)** | $p = 5.01 \times 10^{-5}$ | $p = 6.32 \times 10^{-8}$ | **$p = 7.87 \times 10^{-10}$** |
| **Independent Semantic Clusters** | `132 / 136` | `168 / 168` | **`248 / 252`** |
| **Cluster-Bootstrapped 95% CI** | `[83.6%, 94.0%]` | `[78.0%, 89.3%]` | **`[83.6%, 91.6%]`** |
| **Tipping-Point Fragility Index** | `94` conversations | `99` conversations | **`174` conversations** |
| **Max Tolerable Design Effect ($DEFF$)** | $DEFF \le 30.54$ | $DEFF \le 28.59$ | **$DEFF \le 52.95$** |

### Cross-Dataset Replication (`lmsys-chat-1m` vs. `WildChat-1M` — 2,000,000 Total Conversations)

To test external validity across platforms, we ran the exact same dual-funnel pipeline (Mode A Explicit Belief Regex + Mode B GPU Semantic Embeddings) on **`allenai/WildChat-1M`** (1,000,000 real-world user–ChatGPT conversations). The directional confirmation share replicated to within **0.1 percentage points** (`87.7%` vs. `87.8%`), and all `455` Stage-3 conversations across the 2,000,000-conversation corpus are exported to `data/processed/stage3_seeking_conversations_455.csv`:

| Metric | `lmsys/lmsys-chat-1m` (1M Raw) | `allenai/WildChat-1M` (1M Raw) | **Combined 2M Corpus (2M Raw)** |
| :--- | :---: | :---: | :---: |
| **Unique Candidates Classified** | `1,922` | `1,567` | **`3,489`** |
| **Stage 1: Policy / Political Topic** | `966` (`50.3%`) | `665` (`42.4%`) | **`1,631` (`46.7%`)** |
| **Stage 2: User States Prior Belief** | `387` (`40.1%`) | `150` (`22.6%`) | **`537` (`32.9%`)** |
| **Stage 3: Information-Seeking ($N$)** | `332` | `123` | **`455`** |
| **`confirming` Count & Share** | `221` (**66.6%**) `[61.3%, 71.4%]` | `65` (**52.8%**) `[44.1%, 61.4%]` | **`286` (62.9%) `[58.3%, 67.2%]`** |
| **`neutral` Count & Share** | `80` (**24.1%**) `[19.8%, 29.0%]` | `49` (**39.8%**) `[31.6%, 48.7%]` | **`129` (28.4%) `[24.4%, 32.7%]`** |
| **`falsifying` Count & Share** | `31` (**9.3%**) `[6.7%, 12.9%]` | `9` (**7.3%**) `[3.9%, 13.3%]` | **`40` (8.8%) `[6.5%, 11.8%]`** |
| **Directional Confirming Share ($p_{\text{confirm}}$)** | **87.7%** (`221` vs. `31`) | **87.8%** (`65` vs. `9`) | **87.7% (`286` vs. `40`)** |
| **Confirmation : Falsification Ratio** | **7.13×** | **7.22×** | **7.15×** |
| **Exact Binomial $p$-value (1-sided)** | $p = 7.91 \times 10^{-37}$ | $p = 6.76 \times 10^{-12}$ | **$p = 2.90 \times 10^{-47}$** |
| **Confirming vs. (Neutral + Falsifying)** | $p = 7.87 \times 10^{-10}$ | $p = 0.294$ | **$p = 2.29 \times 10^{-8}$** |
| **Independent Semantic Clusters** | `248 / 252` | `74 / 74` | **`322 / 326`** |
| **Cluster-Bootstrapped 95% CI** | `[83.6%, 91.6%]` | `[79.7%, 94.6%]` | **`[84.0%, 91.1%]`** |
| **Tipping-Point Fragility Index** | `174` conversations | `46` conversations | **`228` conversations** |
| **Max Tolerable Design Effect ($DEFF$)** | $DEFF \le 52.95$ | $DEFF \le 15.66$ | **$DEFF \le 68.61$** |
1. **Robustness to Retrieval Strategy (Explicit vs. Implicit Belief Framing):**
   - A potential critique of Mode A alone is that filtering for explicit phrases like `"I think"` or `"I believe"` might select for more opinionated users. Mode B directly refutes this: using GPU sentence embeddings *without* any belief-marker regex (`--no-belief-marker --semantic-filter`) yields `77.2%` Stage-1 policy precision and replicates the massive confirmation preference (`68.4%` confirming overall; `83.9%` directional confirming, $p = 4.02 \times 10^{-20}$).
2. **Pooled Statistical Power ($N = 332$, $p = 7.91 \times 10^{-37}$):**
   - Combining both retrieval funnels nearly doubles the unique Stage-3 sample to **`332` conversations** (`221` confirming, `80` neutral, `31` falsifying).
   - Confirming queries outnumber falsifying queries by **7.13×** (`87.7%` vs. `12.3%`, $p = 7.91 \times 10^{-37}$) and outnumber `neutral` and `falsifying` combined by **2.0×** (`66.6%` vs. `33.4%`, $p = 7.87 \times 10^{-10}$).
3. **Consistency Across Policy Domains ($N = 332$):**
   - Confirmation-seeking dominates across `social_policy` (`68%` confirming vs. `9%` falsifying), `politics` (`61%` vs. `18%`), and `economic_policy` (`60%` vs. `3%`), with no significant cross-domain heterogeneity ($\chi^2 = 6.63, p = 0.152$).
4. **Robustness to Non-Independence ($248 / 252$ Independent Clusters):**
   - Collapsing to 1 observation per semantic cluster yields `217` confirming vs. `31` falsifying (**87.5%**, $p = 7.48 \times 10^{-36}$). A single repeat user would have had to author **`174` of the `221` confirming conversations** (`78.7%`) to overturn significance at $\alpha = 0.05$.

---

## 5. Pilot Validation & Iterative Refinements

During pilot validation on 50-conversation batches:
1. **Run 1 (`sample_english_policy_500.parquet` — Topic keyword filter only, no belief marker requirement):**
   - Only 12 of 500 conversations contained a first-person belief marker, and most were inside long copy-pasted transcripts or reading-comprehension prompts (e.g., Enron email QA blocks starting with `Context: ... Answer`).
   - `gemma4:12b` accurately rejected 47/50 rows as `not_applicable`, but was tricked by 1 reading-comprehension paste where a question appeared inside `Context:`.
2. **Run 2 (`sample_english_policy_belief_100.parquet` — Same-turn Policy + Belief filter & Pasted-Context Guard):**
   - **100% of remaining NLP/summarization tasks** were caught and rejected by Stage 1 (`is_policy_or_political = False`).
   - Signal density increased from **0% to 36%** genuine prior-belief policy conversations (18/50 rows), yielding clear instances of `confirming` (8), `neutral` (3), and `falsifying` (2) information-seeking behavior, plus 5 rows where a belief was stated without an information-seeking question (`not_applicable`).
3. **Run 3 Refinements (Current Version):**
   - Added **Chatbot Arena first-turn deduplication** (`normalize_first_user_turn`) to prevent side-by-side Arena battles from double-counting identical user prompts.
   - Increased Ollama context window (`num_ctx = 8192`) and capped cumulative multi-turn text at `8,000` characters (`max_total_chars = 8000`) to prevent empty JSON responses on 35+ turn conversations.
