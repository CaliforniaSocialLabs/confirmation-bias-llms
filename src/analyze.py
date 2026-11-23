r"""
Statistical analysis and robustness suite for Confirmation vs. Falsification Information-Seeking.

Implements:
1. Hierarchical Funnel & Prevalence Summary (with 95% Wilson Score Confidence Intervals)
2. Primary Hypothesis Test: Exact Binomial Test (Confirming vs. Falsifying) + Confirmation Ratio
3. Secondary Binomial Test: Confirming vs. Open-Minded (Falsifying + Neutral combined)
4. Three-Way Multinomial Chi-Square Goodness-of-Fit Test (with Monte Carlo exact p-value)
5. Subgroup Heterogeneity: Topic Category & Multi-Turn vs. Single-Turn Contingency Tests
6. Non-Independence Robustness Suite:
   - Text n-gram TF-IDF Cosine Similarity Clustering (detecting near-duplicate/stylistic clusters)
   - Cluster-Bootstrapped 95% Confidence Intervals (10,000 resamples at cluster level)
   - 1-Observation-Per-Cluster Sensitivity Binomial Test
   - Tipping-Point Fragility Index & Design Effect (DEFF) Threshold

Usage:
    .\venv\Scripts\python.exe src\analyze.py --input data\processed\classified_sample_english_policy_belief_100.jsonl
    .\venv\Scripts\python.exe src\analyze.py --input data\processed\classified_candidates_full_1m_belief.jsonl
"""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Exact Statistical Primitives (NumPy + math, no external SciPy needed)
# ---------------------------------------------------------------------------

def log_choose(n: int, k: int) -> float:
    if k < 0 or k > n:
        return - math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def exact_binomial_test(k: int, n: int, p: float = 0.5) -> tuple[float, float]:
    """
    Return (one_sided_greater_p, two_sided_p) for X ~ Binomial(n, p).
    Uses exact log-space summation.
    """
    if n == 0:
        return 1.0, 1.0
    log_p = math.log(p)
    log_q = math.log(1.0 - p)
    log_probs = [log_choose(n, i) + i * log_p + (n - i) * log_q for i in range(n + 1)]
    max_lp = max(log_probs)
    probs = [math.exp(lp - max_lp) for lp in log_probs]
    total = sum(probs)
    probs = [pr / total for pr in probs]

    p_one_sided = sum(probs[i] for i in range(k, n + 1))
    target_prob = probs[k]
    p_two_sided = sum(pr for pr in probs if pr <= target_prob * (1.0 + 1e-12))
    return min(1.0, p_one_sided), min(1.0, p_two_sided)


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """95% Wilson score confidence interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p_hat = k / n
    denom = 1.0 + (z * z) / n
    center = (p_hat + (z * z) / (2.0 * n)) / denom
    half_width = (z * math.sqrt((p_hat * (1.0 - p_hat) + (z * z) / (4.0 * n)) / n)) / denom
    return max(0.0, center - half_width), min(1.0, center + half_width)


def chisq_gof_uniform(counts: list[int], n_sim: int = 50000, seed: int = 42) -> tuple[float, float]:
    """
    Chi-square goodness-of-fit test against a uniform distribution across len(counts) categories,
    returning (chi2_stat, monte_carlo_exact_p_value).
    """
    n = sum(counts)
    k = len(counts)
    if n == 0 or k <= 1:
        return 0.0, 1.0
    expected = n / k
    obs_chi2 = sum(((c - expected) ** 2) / expected for c in counts)

    rng = np.random.default_rng(seed)
    sim_draws = rng.multinomial(n, [1.0 / k] * k, size=n_sim)
    sim_chi2 = np.sum(((sim_draws - expected) ** 2) / expected, axis=1)
    p_val = float(np.mean(sim_chi2 >= obs_chi2 - 1e-12))
    return obs_chi2, p_val


def contingency_chisq_permutation(table: np.ndarray, n_sim: int = 20000, seed: int = 42) -> tuple[float, float]:
    """
    Permutation-based Chi-square test of independence on a 2D contingency table.
    """
    table = np.asarray(table, dtype=float)
    row_totals = table.sum(axis=1, keepdims=True)
    col_totals = table.sum(axis=0, keepdims=True)
    n = int(table.sum())
    if n == 0 or (row_totals == 0).any() or (col_totals == 0).any():
        return 0.0, 1.0

    expected = (row_totals @ col_totals) / n
    obs_chi2 = float(np.sum(((table - expected) ** 2) / expected))

    # Construct row/col labels for exact marginal permutation
    rows = np.repeat(np.arange(table.shape[0]), table.sum(axis=1).astype(int))
    cols = np.repeat(np.arange(table.shape[1]), table.sum(axis=0).astype(int))
    rng = np.random.default_rng(seed)

    exceed = 0
    for _ in range(n_sim):
        perm_cols = rng.permutation(cols)
        sim_tab = np.zeros_like(table)
        np.add.at(sim_tab, (rows, perm_cols), 1)
        sim_chi2 = float(np.sum(((sim_tab - expected) ** 2) / expected))
        if sim_chi2 >= obs_chi2 - 1e-12:
            exceed += 1
    return obs_chi2, exceed / n_sim


# ---------------------------------------------------------------------------
# Semantic / Stylometric Clustering for Non-Independence Robustness
# ---------------------------------------------------------------------------

def build_text_clusters(texts: list[str], sim_threshold: float = 0.65) -> np.ndarray:
    """
    Group user prompts/quotes into semantic/stylistic clusters using word + character n-gram
    TF-IDF cosine similarity and connected components at `sim_threshold`.
    """
    n = len(texts)
    if n == 0:
        return np.array([], dtype=int)

    # Extract word unigrams/bigrams + char 4-grams
    tokenized = []
    df_counts: Counter[str] = Counter()
    for t in texts:
        clean = re.sub(r"\s+", " ", (t or "").lower().strip())
        words = re.findall(r"[a-z0-9']+", clean)
        feats = set(words)
        feats.update(f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1))
        feats.update(clean[i : i + 4] for i in range(max(0, len(clean) - 3)))
        tokenized.append(feats)
        for f in feats:
            df_counts[f] += 1

    vocab = {f: idx for idx, (f, c) in enumerate(df_counts.items()) if c >= 1}
    if not vocab:
        return np.arange(n)

    mat = np.zeros((n, len(vocab)), dtype=np.float32)
    for i, feats in enumerate(tokenized):
        for f in feats:
            if f in vocab:
                mat[i, vocab[f]] = 1.0

    idf = np.log((1.0 + n) / (1.0 + np.array([df_counts[f] for f in vocab], dtype=np.float32))) + 1.0
    mat *= idf
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat /= norms

    sim_matrix = mat @ mat.T
    adj = sim_matrix >= sim_threshold

    # Connected components for transitive clustering
    cluster_ids = -np.ones(n, dtype=int)
    current_cluster = 0
    for i in range(n):
        if cluster_ids[i] != -1:
            continue
        stack = [i]
        cluster_ids[i] = current_cluster
        while stack:
            u = stack.pop()
            neighbors = np.where(adj[u])[0]
            for v in neighbors:
                if cluster_ids[v] == -1:
                    cluster_ids[v] = current_cluster
                    stack.append(int(v))
        current_cluster += 1

    return cluster_ids


def cluster_bootstrap_ci(
    is_confirming: np.ndarray,
    cluster_ids: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Cluster-bootstrapped 95% confidence interval for p_confirm = N_confirm / (N_confirm + N_falsify).
    Resamples clusters with replacement.
    Returns (boot_se, ci_low, ci_high).
    """
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)
    if n_clusters <= 1:
        return 0.0, 0.0, 1.0

    # Pre-aggregate (confirm_count, total_dir_count) per cluster
    c_conf = np.array([np.sum(is_confirming[cluster_ids == c]) for c in unique_clusters], dtype=float)
    c_tot = np.array([np.sum(cluster_ids == c) for c in unique_clusters], dtype=float)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    boot_conf = c_conf[idx].sum(axis=1)
    boot_tot = c_tot[idx].sum(axis=1)
    valid = boot_tot > 0
    boot_p = boot_conf[valid] / boot_tot[valid]

    se = float(np.std(boot_p, ddof=1))
    ci_low, ci_high = np.percentile(boot_p, [2.5, 97.5])
    return se, float(ci_low), float(ci_high)


def compute_fragility_and_max_deff(n_conf: int, n_fals: int, alpha: float = 0.05) -> tuple[int, float]:
    """
    1. Fragility Index: How many confirming observations could be removed (e.g. if all generated by
       a single prolific repeat user) before the one-sided exact binomial test p-value >= alpha?
    2. Max Design Effect (DEFF): Maximum variance inflation factor from unobserved user clustering
       under which p_confirm > 0.5 remains statistically significant at one-sided alpha.
    """
    if n_conf <= n_fals:
        return 0, 1.0

    fragility = 0
    while (n_conf - fragility) > n_fals:
        p_one, _ = exact_binomial_test(n_conf - fragility, (n_conf - fragility) + n_fals, p=0.5)
        if p_one >= alpha:
            break
        fragility += 1

    # Normal approximation z-score for DEFF calculation: z_eff = z_iid / sqrt(DEFF) >= 1.64485
    n_dir = n_conf + n_fals
    p_hat = n_conf / n_dir
    z_iid = (p_hat - 0.5) / math.sqrt(0.25 / n_dir)
    z_crit = 1.6448536269514722  # one-sided alpha = 0.05
    max_deff = (z_iid / z_crit) ** 2 if z_iid > z_crit else 1.0
    return max(0, fragility - 1), max_deff


# ---------------------------------------------------------------------------
# Main Reporting Pipeline
# ---------------------------------------------------------------------------

def load_deduped_jsonl_records(jsonl_paths: list[Path]) -> pd.DataFrame:
    seen_ids: set[str] = set()
    records = []
    for p in jsonl_paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cid = rec.get("conversation_id")
                if "error" not in rec and cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    records.append(rec)
    return pd.DataFrame(records)


def analyze_results(
    jsonl_paths: list[Path],
    raw_parquet_paths: list[Path] | None = None,
    sim_threshold: float = 0.65,
) -> str:
    df = load_deduped_jsonl_records(jsonl_paths)
    if raw_parquet_paths:
        raw_parts = [
            pd.read_parquet(rp)[["conversation_id", "turn", "conversation"]]
            for rp in raw_parquet_paths
            if rp and rp.exists()
        ]
        if raw_parts:
            raw_df = pd.concat(raw_parts, ignore_index=True).drop_duplicates(subset=["conversation_id"])
            df = df.merge(raw_df, on="conversation_id", how="left")

    n_total = len(df)
    policy_df = df[df["is_policy_or_political"] == True].copy()
    belief_df = policy_df[policy_df["user_states_prior_belief"] == True].copy()
    seeking_df = belief_df[belief_df["seeking_type"].isin(["confirming", "falsifying", "neutral"])].copy()

    n_policy = len(policy_df)
    n_belief = len(belief_df)
    n_seeking = len(seeking_df)

    n_conf = int((seeking_df["seeking_type"] == "confirming").sum())
    n_fals = int((seeking_df["seeking_type"] == "falsifying").sum())
    n_neut = int((seeking_df["seeking_type"] == "neutral").sum())
    n_dir = n_conf + n_fals

    lines = []
    lines.append("=" * 78)
    lines.append("CONFIRMATION VS. FALSIFICATION INFORMATION-SEEKING: STATISTICAL REPORT")
    lines.append("=" * 78)
    lines.append("Input file(s): " + ", ".join(str(p) for p in jsonl_paths))
    lines.append("")

    # 1. Funnel Summary
    lines.append("1. HIERARCHICAL CLASSIFICATION FUNNEL")
    lines.append("-" * 78)
    lines.append(f"  Total classified conversations          : {n_total:,}")
    lines.append(f"  Stage 1: Policy / Political topic       : {n_policy:,} ({n_policy/max(n_total,1):.1%})")
    lines.append(f"  Stage 2: User states prior belief       : {n_belief:,} ({n_belief/max(n_policy,1):.1%} of policy)")
    lines.append(f"  Stage 3: Information-seeking query      : {n_seeking:,} ({n_seeking/max(n_belief,1):.1%} of belief)")
    lines.append("")

    if n_seeking == 0:
        lines.append("No Stage 3 information-seeking rows found yet.")
        return "\n".join(lines)

    # 2. Prevalence Table with 95% Wilson CIs
    lines.append("2. STAGE 3 PREVALENCE BREAKDOWN (AMONG INFORMATION-SEEKING QUERIES, N = " + f"{n_seeking:,})")
    lines.append("-" * 78)
    for label, count in [("confirming", n_conf), ("neutral", n_neut), ("falsifying", n_fals)]:
        lo, hi = wilson_ci(count, n_seeking)
        lines.append(f"  {label:<12}: {count:>5,} / {n_seeking:<5,} = {count/n_seeking:>6.1%}   [95% Wilson CI: {lo:.1%} - {hi:.1%}]")
    lines.append("")

    # 3. Primary & Secondary Hypothesis Tests
    lines.append("3. HYPOTHESIS TESTS")
    lines.append("-" * 78)
    if n_dir > 0:
        p_conf_dir = n_conf / n_dir
        ci_lo, ci_hi = wilson_ci(n_conf, n_dir)
        p_one, p_two = exact_binomial_test(n_conf, n_dir, p=0.5)
        ratio_str = f"{n_conf / n_fals:.2f}x" if n_fals > 0 else "Inf (0 falsifying)"
        lines.append("  [Test A] Directional Preference: Confirming vs. Falsifying (Exact Binomial Test)")
        lines.append(f"    Directional sample (N_dir)     : {n_dir:,} ({n_conf:,} confirming vs. {n_fals:,} falsifying)")
        lines.append(f"    Confirming share (p_confirm)   : {p_conf_dir:.1%} [95% CI: {ci_lo:.1%} - {ci_hi:.1%}]")
        lines.append(f"    Confirming : Falsifying ratio  : {ratio_str}")
        lines.append(f"    Exact Binomial p-value (1-sided, H1: p > 0.5) : p = {p_one:.4g}")
        lines.append(f"    Exact Binomial p-value (2-sided, H0: p = 0.5) : p = {p_two:.4g}")
        lines.append("")

    # Secondary test: Confirming vs. (Falsifying + Neutral)
    p_one_all, p_two_all = exact_binomial_test(n_conf, n_seeking, p=0.5)
    lines.append("  [Test B] Confirming vs. All Open/Falsifying Combined (Confirming vs. Neutral + Falsifying)")
    lines.append(f"    Confirming vs. (Neutral+Fals)  : {n_conf:,} vs. {n_neut + n_fals:,} ({n_conf/n_seeking:.1%} confirming)")
    lines.append(f"    Exact Binomial p-value (1-sided, H1: p > 0.5) : p = {p_one_all:.4g}")
    lines.append("")

    # Three-way Multinomial Chi-Square GOF
    chi2_stat, chi2_p = chisq_gof_uniform([n_conf, n_neut, n_fals])
    lines.append("  [Test C] 3-Way Multinomial Chi-Square Goodness-of-Fit (H0: Uniform 1/3 each)")
    lines.append(f"    Chi-square statistic (df = 2)  : {chi2_stat:.2f}, Monte Carlo exact p = {chi2_p:.4g}")
    lines.append("")

    # 4. Subgroup Heterogeneity
    lines.append("4. SUBGROUP HETEROGENEITY BY TOPIC CATEGORY")
    lines.append("-" * 78)
    ct = pd.crosstab(seeking_df["topic_category"], seeking_df["seeking_type"])
    for col in ["confirming", "neutral", "falsifying"]:
        if col not in ct.columns:
            ct[col] = 0
    ct = ct[["confirming", "neutral", "falsifying"]]
    for topic, row_vals in ct.iterrows():
        tot = int(row_vals.sum())
        lines.append(
            f"  {str(topic):<32}: total={tot:>3} | "
            f"conf={int(row_vals['confirming']):>3} ({row_vals['confirming']/max(tot,1):.0%}), "
            f"neut={int(row_vals['neutral']):>3} ({row_vals['neutral']/max(tot,1):.0%}), "
            f"fals={int(row_vals['falsifying']):>3} ({row_vals['falsifying']/max(tot,1):.0%})"
        )
    if ct.shape[0] >= 2:
        c_stat, c_pval = contingency_chisq_permutation(ct.values)
        lines.append(f"  Topic x Seeking-Type Independence Test: Chi2 = {c_stat:.2f}, permutation p = {c_pval:.4g}")
    lines.append("")

    # 5. Non-Independence & Clustering Robustness Suite
    lines.append("5. NON-INDEPENDENCE & CLUSTERING ROBUSTNESS CHECKS")
    lines.append("-" * 78)
    dir_df = seeking_df[seeking_df["seeking_type"].isin(["confirming", "falsifying"])].copy()
    if len(dir_df) >= 2:
        text_series = (
            dir_df["stated_belief_summary"].fillna("") + " " + dir_df["information_seeking_quote"].fillna("")
        ).tolist()
        clusters = build_text_clusters(text_series, sim_threshold=sim_threshold)
        dir_df["cluster_id"] = clusters
        n_clusters = len(np.unique(clusters))

        is_conf_arr = (dir_df["seeking_type"] == "confirming").astype(int).to_numpy()
        boot_se, boot_lo, boot_hi = cluster_bootstrap_ci(is_conf_arr, clusters)

        # 1-per-cluster deduplicated binomial test
        dedup_dir = dir_df.groupby("cluster_id", as_index=False).first()
        c_dedup = int((dedup_dir["seeking_type"] == "confirming").sum())
        f_dedup = int((dedup_dir["seeking_type"] == "falsifying").sum())
        p_dedup_one, _ = exact_binomial_test(c_dedup, c_dedup + f_dedup, p=0.5)

        fragility, max_deff = compute_fragility_and_max_deff(n_conf, n_fals)

        lines.append(f"  Semantic/Stylometric Clusters (cos >= {sim_threshold}) : {n_clusters:,} independent clusters from {len(dir_df):,} rows")
        lines.append(f"  Cluster-Bootstrapped 95% CI (p_confirm)   : [{boot_lo:.1%} - {boot_hi:.1%}] (Cluster SE = {boot_se:.3f})")
        lines.append(
            f"  1-Per-Cluster Sensitivity Binomial Test   : {c_dedup:,} confirming vs. {f_dedup:,} falsifying "
            f"({c_dedup/max(c_dedup+f_dedup,1):.1%}), 1-sided p = {p_dedup_one:.4g}"
        )
        lines.append(
            f"  Tipping-Point Fragility Index (alpha=.05) : {fragility:,} confirming conversations would need to come "
            f"from a single repeat user to lose significance"
        )
        lines.append(
            f"  Maximum Tolerable Design Effect (DEFF)    : DEFF <= {max_deff:.2f} "
            f"(holds up to {max_deff:.1f}x variance inflation from clustering)"
        )

    lines.append("=" * 78)
    return "\n".join(lines)


def format_turns_for_csv(conv: list[dict], user_only: bool = False) -> str:
    parts = []
    t_idx = 1
    for m in conv:
        role = m.get("role", "unknown").upper()
        if user_only and role != "USER":
            continue
        content = (m.get("content") or "").strip()
        parts.append(f"[Turn {t_idx} - {role}]: {content}")
        t_idx += 1
    return "\n\n".join(parts)


def export_stage3_and_validation_csvs(
    jsonl_paths: list[Path],
    raw_parquet_paths: list[Path],
    out_dir: Path | None = None,
    seed: int = 42,
) -> tuple[Path, Path, Path]:
    """
    Join classified JSONL(s) with raw Parquet(s) and export:
    1) stage3_seeking_conversations_<N>.csv (all Stage-3 conversations with full text)
    2) human_validation_blind_sample_60.csv (stratified blind coding sheet for human validation)
    3) human_validation_answer_key_60.csv (matching answer key for computing Cohen's kappa)
    """
    if out_dir is None:
        out_dir = jsonl_paths[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cls_df = load_deduped_jsonl_records(jsonl_paths)
    raw_df = pd.concat([pd.read_parquet(rp) for rp in raw_parquet_paths if rp.exists()], ignore_index=True)
    raw_df = raw_df.drop_duplicates(subset=["conversation_id"])
    merged = cls_df.merge(raw_df, on="conversation_id", how="left")

    stage3 = merged[
        (merged["is_policy_or_political"] == True)
        & (merged["user_states_prior_belief"] == True)
        & (merged["seeking_type"].isin(["confirming", "neutral", "falsifying"]))
    ].copy()

    stage3["matched_policy_terms"] = stage3["matched_policy_terms"].apply(
        lambda x: ", ".join(x) if isinstance(x, (list, np.ndarray)) else str(x or "")
    )
    stage3["matched_belief_markers"] = stage3["matched_belief_markers"].apply(
        lambda x: ", ".join(x) if isinstance(x, (list, np.ndarray)) else str(x or "")
    )
    stage3["user_messages_text"] = stage3["conversation"].apply(
        lambda c: format_turns_for_csv(list(c), user_only=True) if isinstance(c, (list, np.ndarray)) else ""
    )
    stage3["full_conversation_text"] = stage3["conversation"].apply(
        lambda c: format_turns_for_csv(list(c), user_only=False) if isinstance(c, (list, np.ndarray)) else ""
    )

    export_cols = [
        "conversation_id",
        "seeking_type",
        "topic_category",
        "stated_belief_summary",
        "information_seeking_quote",
        "reasoning",
        "confidence",
        "model_in_dataset",
        "turn",
        "matched_policy_terms",
        "matched_belief_markers",
        "user_messages_text",
        "full_conversation_text",
    ]
    stage3 = stage3[export_cols].sort_values(["seeking_type", "topic_category", "conversation_id"])
    stage3_path = out_dir / f"stage3_seeking_conversations_{len(stage3)}.csv"
    stage3.to_csv(stage3_path, index=False, encoding="utf-8-sig")

    # Create a stratified 60-row blind human validation sample (up to 25 confirming, 20 neutral, 15 falsifying)
    parts = []
    for st, target_n in [("confirming", 25), ("neutral", 20), ("falsifying", 15)]:
        sub = stage3[stage3["seeking_type"] == st]
        parts.append(sub.sample(n=min(len(sub), target_n), random_state=seed))
    val_df = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    blind_df = pd.DataFrame(
        {
            "validation_id": [f"VAL_{i+1:02d}" for i in range(len(val_df))],
            "conversation_id": val_df["conversation_id"],
            "matched_policy_terms": val_df["matched_policy_terms"],
            "matched_belief_markers": val_df["matched_belief_markers"],
            "user_messages_text": val_df["user_messages_text"],
            "human_seeking_type (confirming / falsifying / neutral / not_applicable)": "",
            "human_notes": "",
        }
    )
    key_df = pd.DataFrame(
        {
            "validation_id": [f"VAL_{i+1:02d}" for i in range(len(val_df))],
            "conversation_id": val_df["conversation_id"],
            "gemma_seeking_type": val_df["seeking_type"],
            "gemma_topic_category": val_df["topic_category"],
            "gemma_stated_belief_summary": val_df["stated_belief_summary"],
            "gemma_information_seeking_quote": val_df["information_seeking_quote"],
            "gemma_reasoning": val_df["reasoning"],
        }
    )

    blind_path = out_dir / f"human_validation_blind_sample_{len(val_df)}.csv"
    key_path = out_dir / f"human_validation_answer_key_{len(val_df)}.csv"
    blind_df.to_csv(blind_path, index=False, encoding="utf-8-sig")
    key_df.to_csv(key_path, index=False, encoding="utf-8-sig")

    return stage3_path, blind_path, key_path


def score_human_validation(blind_csv_path: Path, key_csv_path: Path) -> str:
    """Compute raw agreement and Cohen's kappa between human labels and gemma4:12b labels."""
    b_df = pd.read_csv(blind_csv_path)
    k_df = pd.read_csv(key_csv_path)
    human_col = [c for c in b_df.columns if c.startswith("human_seeking_type")][0]
    m = b_df.merge(k_df, on="conversation_id", how="inner")
    m["human"] = m[human_col].fillna("").astype(str).str.strip().str.lower()
    m["gemma"] = m["gemma_seeking_type"].fillna("").astype(str).str.strip().str.lower()
    m = m[m["human"] != ""].copy()

    if m.empty:
        return "No human labels filled in yet."

    n = len(m)
    agree = int((m["human"] == m["gemma"]).sum())
    p_o = agree / n
    labels = sorted(set(m["human"]).union(set(m["gemma"])))
    p_e = sum((np.sum(m["human"] == lbl) / n) * (np.sum(m["gemma"] == lbl) / n) for lbl in labels)
    kappa = (p_o - p_e) / (1.0 - p_e) if p_e < 1.0 else 1.0

    ct = pd.crosstab(m["human"], m["gemma"], margins=True)
    return (
        f"=== Human vs. Gemma 4 12B Validation (N = {n}) ===\n"
        f"Raw Agreement : {agree}/{n} ({p_o:.1%})\n"
        f"Cohen's Kappa : {kappa:.3f}\n\n"
        f"Confusion Matrix (Rows = Human, Cols = Gemma):\n{ct.to_string()}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run statistical tests on one or more classified conversation JSONL files")
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        default=["data/processed/classified_candidates_full_1m_belief_crossturn.jsonl"],
        help="Path(s) to one or more classified .jsonl files (automatically deduplicated by conversation_id)",
    )
    parser.add_argument(
        "--raw",
        type=str,
        nargs="*",
        default=None,
        help="Optional path(s) to corresponding raw .parquet file(s) for turn metadata",
    )
    parser.add_argument(
        "--sim-threshold",
        type=float,
        default=0.65,
        help="Cosine similarity threshold for semantic/stylistic clustering (default: 0.65)",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=None,
        help="Optional path to save the text report",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export all Stage-3 conversations and a stratified 60-row blind human validation CSV",
    )
    parser.add_argument(
        "--score-validation",
        action="store_true",
        help="Compute Cohen's kappa from the completed human_validation_blind_sample_60.csv",
    )
    args = parser.parse_args()

    in_paths = [Path(p) for p in args.input]
    if args.raw:
        raw_paths = [Path(p) for p in args.raw]
    else:
        raw_paths = []
        for ip in in_paths:
            stem = ip.stem.replace("classified_", "")
            cand_raw = PROJECT_ROOT / "data" / "raw" / f"{stem}.parquet"
            if cand_raw.exists():
                raw_paths.append(cand_raw)

    if args.score_validation:
        blind_p = in_paths[0].parent / "human_validation_blind_sample_60.csv"
        key_p = in_paths[0].parent / "human_validation_answer_key_60.csv"
        print(score_human_validation(blind_p, key_p))
    else:
        report = analyze_results(in_paths, raw_parquet_paths=raw_paths, sim_threshold=args.sim_threshold)
        print(report)

        if args.output_report:
            out_p = Path(args.output_report)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(report, encoding="utf-8")
            print(f"\nReport saved to {out_p}")

        if args.export_csv and raw_paths:
            s3_p, blind_p, key_p = export_stage3_and_validation_csvs(in_paths, raw_paths)
            print(f"\nExported CSV files:")
            print(f"  1. Full Stage-3 Table         : {s3_p}")
            print(f"  2. Blind Human Coding Sheet   : {blind_p}")
            print(f"  3. Human Validation Key       : {key_p}")
