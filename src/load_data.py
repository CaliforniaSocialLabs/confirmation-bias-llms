r"""
Stream or scan all 1,000,000 conversations in lmsys/lmsys-chat-1m, pre-filtering for
genuine user conversations about politics/social policy where the user states a belief
and seeks information, while deduplicating side-by-side Chatbot Arena prompts.

Usage:
    # Scan all 1,000,000 cached rows into data/raw/candidates_full_1m_belief.parquet (~1,000+ unique candidates)
    .\venv\Scripts\python.exe src\load_data.py --scan-all

    # Also allow policy keyword and belief marker across adjacent turns in the same conversation (~1,500 candidates)
    .\venv\Scripts\python.exe src\load_data.py --scan-all --allow-cross-turn

    # Stream a quick 200-row sample
    .\venv\Scripts\python.exe src\load_data.py --sample-size 200
"""

import argparse
import os
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Comprehensive word-boundary terms covering Politics, Social Policy, Economic Policy, and Controversial Social Issues.
POLICY_TOPIC_TERMS = [
    # Government, Political Systems & Figures
    r"government\w*", r"politician\w*", r"political\w*", r"politics", r"democrat\w*", r"republican\w*",
    r"liberal\w*", r"conservative\w*", r"socialis\w+", r"capitalis\w+", r"communis\w+", r"marxis\w+",
    r"fascis\w+", r"libertarian\w*", r"authoritarian\w*", r"totalitarian\w*", r"anarch\w+",
    r"left-wing", r"right-wing", r"progressive\w*", r"ideolog\w+", r"populism", r"populist\w*",
    r"congress\w*", r"senate", r"parliament\w*", r"president\w*", r"election\w*", r"voting", r"voter\w*",
    r"bipartisan\w*", r"partisan\w*", r"policy", r"policies", r"legislation", r"lawmaker\w*",
    r"regulation\w*", r"deregulation", r"supreme court", r"constitutional\w*", r"constitution",
    r"amendment", r"civil rights", r"human rights", r"democracy", r"democratic",
    r"biden", r"trump", r"obama", r"putin", r"zelensky\w*", r"netanyahu",
    # Economic & Fiscal Policy
    r"gov\w* spending", r"public spending", r"deficit", r"national debt", r"inflation", r"recession",
    r"federal reserve", r"central bank\w*", r"interest rate\w*", r"minimum wage", r"minimum-wage",
    r"universal basic income", r"\bubi\b", r"welfare", r"food stamps", r"taxation", r"taxes",
    r"tax cut\w*", r"tax hike\w*", r"wealth tax", r"tariff\w*", r"free trade", r"protectionis\w+",
    r"free market\w*", r"subsid\w+", r"social security", r"medicare", r"medicaid",
    r"student loan\w*", r"student debt", r"affordable housing", r"housing crisis", r"homeless\w*",
    r"rent control", r"gentrification", r"labor union\w*", r"unions", r"inequality",
    r"wealth gap", r"income gap", r"billionaire\w*", r"working class", r"middle class",
    r"privatiz\w+", r"nationaliz\w+", r"monopol\w+", r"corporat\w+",
    # Social, Cultural & Foreign Policy
    r"healthcare", r"health care", r"obamacare", r"single-payer", r"abortion", r"pro-life", r"pro-choice",
    r"reproductive rights", r"reproductive freedom", r"gun control", r"firearm\w*", r"gun rights", r"guns", r"mass shooting\w*",
    r"immigration", r"immigrant\w*", r"border\w*", r"asylum", r"deport\w+", r"refugee\w*",
    r"affirmative action", r"diversity equity", r"reparations", r"segregation", r"racism", r"racist\w*",
    r"sexism", r"feminis\w+", r"patriarch\w+", r"woke culture", r"wokeism", r"wokeness", r"anti-woke", r"cancel culture", r"transgender\w*",
    r"lgbt\w*", r"same-sex", r"gay marriage", r"gender identity", r"pronouns",
    r"death penalty", r"capital punishment", r"criminal justice", r"police", r"policing",
    r"incarceration", r"prison\w*", r"drug legalization", r"marijuana", r"cannabis",
    r"free speech", r"censorship", r"hate speech", r"misinformation", r"propaganda", r"media bias",
    r"climate change", r"global warming", r"carbon tax", r"renewable energy", r"fossil fuel\w*",
    r"green energy", r"nuclear energy", r"nuclear power", r"emissions", r"electric vehicle\w*",
    r"public education", r"school voucher\w*", r"charter school\w*", r"homeschool\w*",
    r"foreign aid", r"military\w*", r"defense spending", r"\bnato\b", r"sanctions",
    r"ukraine", r"russia\w*", r"china", r"taiwan", r"israel\w*", r"palestin\w*", r"iran",
    r"vaccine\w*", r"vaccination\w*", r"lockdown\w*", r"covid\w*", r"pandemic",
    r"euthanasia", r"assisted suicide", r"prostitution", r"sex work\w*", r"animal rights", r"vegan\w*",
]

# Phrases indicating the user is stating a personal belief, stance, or hypothesis (including negations & modifiers)
BELIEF_MARKER_PATTERNS = [
    r"\bi (?:really |strongly |personally |honestly |truly |firmly |just )?(?:don't |do not |can't |cannot )?(?:think|believe|feel|suspect|doubt|agree|disagree|support|oppose)\b",
    r"\bin my (?:humble |personal |honest )?(?:opinion|view|estimation|experience|judgment|judgement|mind)\b",
    r"\bmy (?:view|opinion|stance|position|take|belief|theory|hypothesis|argument|point|concern) is\b",
    r"\bi(?:'m| am) (?:convinced|skeptical|against|in favor of|pro-|anti-|not sure why|tired of|worried that|concerned that)\b",
    r"\bi(?:'d| would) (?:argue|say|contend|suggest|think)\b",
    r"\b(?:it )?seems to me\b",
    r"\bthe way i see it\b",
    r"\bas i see it\b",
    r"\bfrom my (?:perspective|point of view)\b",
    r"\bi personally\b",
    r"\bdon't you (?:think|agree|feel)\b",
    r"\bisn't it (?:true|obvious|clear|unfair|wrong|bad|crazy)\b",
    r"\bwouldn't you (?:say|agree|think)\b",
    r"\bwhy do (?:people|some people|so many people|liberals|conservatives|democrats|republicans|politicians|governments) (?:still )?(?:think|believe|support|defend|oppose|ignore|pretend)\b",
    r"\bhow can anyone (?:justify|support|defend|believe)\b",
    r"\bi don't (?:understand|get|see) (?:why|how)\b",
]

# Question mark or explicit information-seeking imperative
QUESTION_INQUIRY_PATTERNS = [
    r"\?",
    r"\btell me\b",
    r"\bexplain\b",
    r"\bgive me\b",
    r"\bprovide\b",
    r"\bshare with me\b",
    r"\bname some\b",
    r"\blist some\b",
    r"\bwhat are\b",
    r"\bwhat is\b",
    r"\bwhy is\b",
    r"\bwhy do\b",
    r"\bhow is\b",
    r"\bhow do\b",
    r"\bpros and cons\b",
    r"\bpros/cons\b",
]

# Patterns that strongly indicate copy-pasted articles, NLP tasks, summarization, or roleplay/jailbreaks
NOISE_TASK_PATTERNS = [
    r"^\s*context\s*:",
    r"^\s*passage\s*:",
    r"^\s*article\s*:",
    r"^\s*transcript\s*:",
    r"^\s*instruction\s*:",
    r"###\s*(?:instruction|answer|explanation)",
    r"\bbased on the following passage\b",
    r"\bgiven the document below\b",
    r"\bfactually consistent\b",
    r"\byour task is to\b",
    r"\bin this hypothetical story\b",
    r"\btake this (?:essay|conclusion) and expand\b",
    r"\bsummari[sz]e\b",
    r"\bstate of the union\b",
    r"\bparaphrase\b",
    r"\btranslate\b",
    r"\bproofread\b",
    r"\babbreviate\b",
    r"\brewrite the following\b",
    r"\bedit the following\b",
    r"\bextract\b.*\bfrom\b",
    r"\bnamed entity\b",
    r"\bspeaker 1\s*:",
    r"\bspeaker 2\s*:",
    r"\bwrite a story\b",
    r"\bwrite an? (?:erotic|fictional|short) story\b",
    r"\berotica\b",
    r"\bin the style of\b",
    r"\bact as a\b",
    r"\bcounter[- ]narrative\b",
    r"\broleplay\b",
    r"\bignore all previous\b",
    r"\bdeveloper mode\b",
    r"\bcurriculum vitae\b",
    r"\bprofile summary\b",
    r"\bcover letter\b",
    r"\bmultiple[- ]choice\b",
    r"\bwhich of the following\b",
]

# Declarative / identity stance markers for users who state a belief without saying "I think" / "I believe"
DECLARATIVE_BELIEF_PATTERNS = [
    r"\b(?:speaking )?as a (?:conservative|liberal|libertarian|socialist|democrat|republican|progressive|leftist|right-winger|christian|muslim|atheist|feminist|capitalist|marxist)\b",
    r"\bi (?:really |strongly )?(?:hate|dislike|detest|resent|admire|favor|prefer|fear|worry about)\b",
    r"\b(?:should|must|ought to|need to|needs to) (?:not |never )?be (?:banned|legal\w*|illegal|abolished|regulated|deregulated|taxed|funded|defunded|mandatory|privatized|nationalized)\b",
    r"\b(?:is|are) (?:clearly |obviously |inherently |fundamentally |completely |totally |nothing more than )?(?:a scam|a failure|a disaster|a waste|a human right|unconstitutional|immoral|corrupt|evil|destructive)\b",
]

POLICY_REGEX = re.compile(r"\b(?:" + "|".join(POLICY_TOPIC_TERMS) + r")\b", re.IGNORECASE)
BELIEF_REGEX = re.compile("|".join(BELIEF_MARKER_PATTERNS), re.IGNORECASE)
DECLARATIVE_BELIEF_REGEX = re.compile("|".join(DECLARATIVE_BELIEF_PATTERNS), re.IGNORECASE)
QUESTION_REGEX = re.compile("|".join(QUESTION_INQUIRY_PATTERNS), re.IGNORECASE)
NOISE_REGEX = re.compile("|".join(NOISE_TASK_PATTERNS), re.IGNORECASE | re.MULTILINE)


def get_hf_token() -> str | None:
    load_dotenv(PROJECT_ROOT / ".env")
    token = os.getenv("HF_TOKEN", "").strip()
    return token if token and token != "hf_your_token_here" else None


def normalize_first_user_turn(conversation: list[dict]) -> str:
    """Return a whitespace-normalized lowercase string of the first user turn for Arena deduplication."""
    for msg in conversation:
        if msg.get("role", "").lower() == "user" and msg.get("content"):
            return re.sub(r"\s+", " ", msg["content"].strip().lower())
    return ""


def resolve_dataset_name(dataset_arg: str) -> tuple[str, str]:
    """Return (hf_dataset_id, short_tag) for lmsys or wildchat."""
    low = dataset_arg.strip().lower()
    if low in ("lmsys", "lmsys-chat-1m", "lmsys/lmsys-chat-1m"):
        return "lmsys/lmsys-chat-1m", "lmsys"
    if low in ("wildchat", "wildchat-1m", "allenai/wildchat-1m", "allenai/wildchat"):
        return "allenai/WildChat-1M", "wildchat"
    tag = re.sub(r"[^a-z0-9]+", "_", low).strip("_")
    return dataset_arg, tag


def check_prefilter(
    conversation: list[dict],
    max_turn_chars: int = 2000,
    require_question: bool = True,
    allow_cross_turn: bool = False,
    include_declarative: bool = True,
) -> tuple[list[str], list[str], bool]:
    """
    Inspect user turns for policy terms, belief markers, and information-seeking questions
    while rejecting copy-pasted walls of text, code blocks, and NLP/summarization templates.

    Returns:
        (matched_policy_terms, matched_belief_markers, is_clean_conversational)
    """
    user_turns = [
        msg.get("content", "").strip()
        for msg in conversation
        if msg.get("role", "").lower() == "user" and msg.get("content")
    ]
    if not user_turns:
        return [], [], False

    full_user_text = "\n".join(user_turns)

    # 1. Reject code blocks or obvious NLP/summarization/roleplay/jailbreak templates
    if "```" in full_user_text or NOISE_REGEX.search(full_user_text):
        return [], [], False

    # 2. Check conversational user turns (not 5,000-character pasted documents)
    short_turns = [t for t in user_turns if 20 <= len(t) <= max_turn_chars]
    if not short_turns:
        return [], [], False

    conversational_text = "\n".join(short_turns)
    if require_question and not QUESTION_REGEX.search(conversational_text):
        return [], [], False

    # 3. Match policy terms and belief markers (either within the same turn or across turns)
    cooccurring_policy = set()
    cooccurring_belief = set()
    all_policy = set()
    all_belief = set()

    for turn in short_turns:
        t_pol = {m.group(0).lower() for m in POLICY_REGEX.finditer(turn)}
        t_bel = {m.group(0).lower() for m in BELIEF_REGEX.finditer(turn)}
        if include_declarative and "." in turn and "?" in turn:
            t_bel.update(m.group(0).lower() for m in DECLARATIVE_BELIEF_REGEX.finditer(turn))
        all_policy.update(t_pol)
        all_belief.update(t_bel)
        if t_pol and t_bel:
            cooccurring_policy.update(t_pol)
            cooccurring_belief.update(t_bel)

    if allow_cross_turn:
        if all_policy and all_belief:
            return list(all_policy), list(all_belief), True
        return list(all_policy), [], True

    if cooccurring_belief:
        return list(cooccurring_policy), list(cooccurring_belief), True
    return list(all_policy), [], True


def _collect_from_iterable(
    dataset_iterable,
    total_rows: int | None,
    sample_size: int | None,
    language: str | None,
    min_turns: int,
    filter_policy: bool,
    require_belief_marker: bool,
    require_question: bool,
    allow_cross_turn: bool,
    max_turn_chars: int,
    deduplicate: bool,
    output_path: Path,
) -> Path:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    seen_first_turns: set[str] = set()
    stats = {
        "scanned": 0,
        "english": 0,
        "clean_with_question": 0,
        "policy_match": 0,
        "belief_match": 0,
        "duplicates_skipped": 0,
    }

    desc = "Scanning all 1M rows" if sample_size is None else f"Collecting {sample_size} conversations"
    pbar = tqdm(total=total_rows if sample_size is None else sample_size, desc=desc)

    for row in dataset_iterable:
        stats["scanned"] += 1
        if sample_size is None:
            pbar.update(1)

        if language and row.get("language") != language:
            continue
        if row.get("turn", 0) < min_turns:
            continue
        stats["english"] += 1

        conv = row["conversation"]
        policy_matches, belief_matches, is_clean = check_prefilter(
            conv,
            max_turn_chars=max_turn_chars,
            require_question=require_question,
            allow_cross_turn=allow_cross_turn,
        )
        if not is_clean:
            continue
        stats["clean_with_question"] += 1

        if filter_policy and not policy_matches:
            continue
        stats["policy_match"] += 1

        if require_belief_marker and not belief_matches:
            continue
        stats["belief_match"] += 1

        if deduplicate:
            first_turn_key = normalize_first_user_turn(conv)
            if first_turn_key in seen_first_turns:
                stats["duplicates_skipped"] += 1
                continue
            seen_first_turns.add(first_turn_key)

        cid = row.get("conversation_id") or row.get("conversation_hash") or f"row_{stats['scanned']}"
        clean_conv = [{"role": m.get("role", ""), "content": m.get("content", "")} for m in conv]

        records.append(
            {
                "conversation_id": cid,
                "model": row.get("model", "unknown"),
                "conversation": clean_conv,
                "turn": row.get("turn", len(clean_conv) // 2),
                "language": row.get("language", language or "unknown"),
                "redacted": bool(row.get("redacted", False)),
                "matched_policy_terms": policy_matches,
                "matched_belief_markers": belief_matches,
                "has_belief_marker": len(belief_matches) > 0,
            }
        )

        if sample_size is not None:
            pbar.update(1)

        if stats["scanned"] % 5000 == 0 or sample_size is not None:
            pbar.set_postfix(
                kept=len(records),
                scanned=stats["scanned"],
                dedup=stats["duplicates_skipped"],
                rate=f"{len(records)/max(stats['scanned'], 1):.2%}",
            )

        if sample_size is not None and len(records) >= sample_size:
            break

    pbar.close()
    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)

    print("\n=== Pre-Filter Funnel Summary ===")
    print(f"  Total rows scanned       : {stats['scanned']:,}")
    print(f"  Language & turn match    : {stats['english']:,}")
    print(f"  Clean + has question (?) : {stats['clean_with_question']:,}")
    print(f"  Policy keyword match     : {stats['policy_match']:,}")
    print(f"  Belief marker match      : {stats['belief_match']:,}")
    print(f"  Arena duplicates skipped : {stats['duplicates_skipped']:,}")
    print(f"  Final unique candidates  : {len(df):,} ({len(df)/max(stats['scanned'], 1):.3%})")
    print(f"Saved to: {output_path}")
    return output_path


def apply_ollama_semantic_filter(
    df: pd.DataFrame,
    embed_model: str = "nomic-embed-text",
    top_k: int = 1000,
    batch_size: int = 64,
) -> pd.DataFrame:
    """
    Score candidate policy conversations using GPU sentence embeddings via Ollama.
    Computes cosine similarity to positive political-belief prototypes minus negative
    NLP/math/roleplay prototypes, returning the top_k highest-scoring conversations.
    """
    import numpy as np
    import ollama

    pos_prototypes = [
        "I think this government policy is harmful and bad for the economy. Why do people still support it?",
        "In my opinion universal healthcare and social welfare are essential. What are the arguments for and against my view?",
        "I believe current immigration, tax, and regulation laws are unfair and broken. Don't you agree?",
        "Speaking as a voter, I strongly oppose this political ideology because it hurts working people. How can it be justified?",
    ]
    neg_prototypes = [
        "Please summarize the following passage, article, or document and answer the multiple-choice reading comprehension question.",
        "Solve this mathematical proof, programming code problem, or write a fictional roleplay story about characters.",
        "Who is the current president and when was the constitution signed? Give a factual historical overview.",
    ]

    print(f"Running GPU sentence-embedding filter on {len(df):,} rows using Ollama model '{embed_model}'...")
    pos_emb = np.asarray(ollama.embed(model=embed_model, input=pos_prototypes)["embeddings"], dtype=np.float32)
    neg_emb = np.asarray(ollama.embed(model=embed_model, input=neg_prototypes)["embeddings"], dtype=np.float32)

    texts = [
        " ".join(
            m.get("content", "")[:600]
            for m in conv
            if m.get("role", "").lower() == "user" and m.get("content")
        )[:1000]
        for conv in df["conversation"]
    ]

    scores = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[start : start + batch_size]
        emb = np.asarray(ollama.embed(model=embed_model, input=batch)["embeddings"], dtype=np.float32)
        pos_sim = np.max(emb @ pos_emb.T, axis=1)
        neg_sim = np.max(emb @ neg_emb.T, axis=1)
        scores.extend((pos_sim - 0.5 * neg_sim).tolist())

    df = df.copy()
    df["semantic_belief_score"] = scores
    df = df.sort_values("semantic_belief_score", ascending=False).head(top_k).reset_index(drop=True)
    print(f"Kept top {len(df):,} conversations by semantic belief score.")
    return df


def stream_sample(
    sample_size: int = 200,
    dataset_name: str = "lmsys/lmsys-chat-1m",
    language: str | None = "English",
    min_turns: int = 1,
    filter_policy: bool = True,
    require_belief_marker: bool = True,
    require_question: bool = True,
    allow_cross_turn: bool = False,
    max_turn_chars: int = 2000,
    deduplicate: bool = True,
    output_path: Path | None = None,
) -> Path:
    """Stream a filtered subset of lmsys-chat-1m or WildChat-1M without downloading the full dataset."""
    hf_id, ds_tag = resolve_dataset_name(dataset_name)
    if output_path is None:
        prefix = "" if ds_tag == "lmsys" else f"{ds_tag}_"
        lang_tag = f"_{language.lower()}" if language else ""
        policy_tag = "_policy" if filter_policy else ""
        belief_tag = "_belief" if require_belief_marker else ""
        output_path = RAW_DATA_DIR / f"{prefix}sample{lang_tag}{policy_tag}{belief_tag}_{sample_size}.parquet"

    token = get_hf_token()
    print(
        f"Streaming from '{hf_id}' "
        f"(target={sample_size}, language={language}, filter_policy={filter_policy}, "
        f"require_belief_marker={require_belief_marker}, require_question={require_question}, "
        f"allow_cross_turn={allow_cross_turn}, deduplicate={deduplicate})..."
    )
    ds = load_dataset(hf_id, split="train", streaming=True, token=token)
    return _collect_from_iterable(
        dataset_iterable=ds,
        total_rows=None,
        sample_size=sample_size,
        language=language,
        min_turns=min_turns,
        filter_policy=filter_policy,
        require_belief_marker=require_belief_marker,
        require_question=require_question,
        allow_cross_turn=allow_cross_turn,
        max_turn_chars=max_turn_chars,
        deduplicate=deduplicate,
        output_path=output_path,
    )


def scan_full_dataset(
    dataset_name: str = "lmsys/lmsys-chat-1m",
    language: str | None = "English",
    min_turns: int = 1,
    filter_policy: bool = True,
    require_belief_marker: bool = True,
    require_question: bool = True,
    allow_cross_turn: bool = False,
    max_turn_chars: int = 2000,
    deduplicate: bool = True,
    semantic_filter: bool = False,
    semantic_top_k: int = 1000,
    embed_model: str = "nomic-embed-text",
    output_path: Path | None = None,
) -> Path:
    """Download/cache the full 1M dataset locally and extract all matching conversations."""
    hf_id, ds_tag = resolve_dataset_name(dataset_name)
    if output_path is None:
        prefix = "" if ds_tag == "lmsys" else f"{ds_tag}_"
        belief_tag = "_belief" if require_belief_marker else "_all_policy"
        cross_tag = "_crossturn" if allow_cross_turn else ""
        sem_tag = f"_sem{semantic_top_k}" if semantic_filter else ""
        output_path = RAW_DATA_DIR / f"{prefix}candidates_full_1m{belief_tag}{cross_tag}{sem_tag}.parquet"

    token = get_hf_token()
    print(f"Loading full '{hf_id}' dataset from local Arrow cache...")
    ds = load_dataset(hf_id, split="train", token=token)
    print(f"Dataset ready ({len(ds):,} rows). Scanning all rows...")

    saved_path = _collect_from_iterable(
        dataset_iterable=ds,
        total_rows=len(ds),
        sample_size=None,
        language=language,
        min_turns=min_turns,
        filter_policy=filter_policy,
        require_belief_marker=require_belief_marker,
        require_question=require_question,
        allow_cross_turn=allow_cross_turn,
        max_turn_chars=max_turn_chars,
        deduplicate=deduplicate,
        output_path=output_path,
    )

    if semantic_filter:
        df = pd.read_parquet(saved_path)
        df = apply_ollama_semantic_filter(df, embed_model=embed_model, top_k=semantic_top_k)
        df.to_parquet(saved_path, index=False)
        print(f"Saved semantically filtered top-{len(df):,} candidates to {saved_path}")

    return saved_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load, stream, or full-scan lmsys-chat-1m or WildChat-1M")
    parser.add_argument(
        "--dataset",
        type=str,
        default="lmsys/lmsys-chat-1m",
        help="Dataset name: 'lmsys' ('lmsys/lmsys-chat-1m') or 'wildchat' ('allenai/WildChat-1M')",
    )
    parser.add_argument(
        "--scan-all",
        action="store_true",
        help="Download/cache the full 1M dataset and scan all 1,000,000 rows for candidate conversations",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Alias for --scan-all (downloads full 1M dataset and extracts all matching candidates)",
    )
    parser.add_argument("--sample-size", type=int, default=200, help="Number of conversations to collect when streaming")
    parser.add_argument("--language", type=str, default="English", help="Filter by language (or '' for all)")
    parser.add_argument("--min-turns", type=int, default=1, help="Minimum number of turns in conversation")
    parser.add_argument("--max-turn-chars", type=int, default=2000, help="Max chars per user turn to avoid pasted articles")
    parser.add_argument(
        "--filter-policy",
        action="store_true",
        default=True,
        help="Pre-filter for politics/social/economic policy keywords (default: True)",
    )
    parser.add_argument(
        "--no-filter-policy",
        action="store_false",
        dest="filter_policy",
        help="Disable policy keyword pre-filtering",
    )
    parser.add_argument(
        "--require-belief-marker",
        action="store_true",
        default=True,
        help="Require user belief phrases like 'I think', 'I believe', etc. (default: True)",
    )
    parser.add_argument(
        "--no-belief-marker",
        action="store_false",
        dest="require_belief_marker",
        help="Do not require explicit belief marker phrases",
    )
    parser.add_argument(
        "--allow-cross-turn",
        action="store_true",
        default=False,
        help="Allow policy keyword and belief marker to appear in different user turns of the same conversation",
    )
    parser.add_argument(
        "--no-question-filter",
        action="store_false",
        dest="require_question",
        default=True,
        help="Do not require a question mark or inquiry phrase in the user's turns",
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_false",
        dest="deduplicate",
        default=True,
        help="Disable deduplication of side-by-side Chatbot Arena prompts",
    )
    parser.add_argument(
        "--semantic-filter",
        action="store_true",
        help="Use GPU sentence embeddings (via Ollama) to rank policy conversations by belief similarity",
    )
    parser.add_argument(
        "--semantic-top-k",
        type=int,
        default=1000,
        help="Number of top-scoring conversations to keep when --semantic-filter is enabled (default: 1000)",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="nomic-embed-text",
        help="Ollama embedding model for --semantic-filter (default: nomic-embed-text)",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional custom output .parquet path")
    args = parser.parse_args()

    lang = args.language if args.language else None
    out_path = Path(args.output) if args.output else None

    if args.scan_all or args.full:
        scan_full_dataset(
            dataset_name=args.dataset,
            language=lang,
            min_turns=args.min_turns,
            filter_policy=args.filter_policy,
            require_belief_marker=args.require_belief_marker,
            require_question=args.require_question,
            allow_cross_turn=args.allow_cross_turn,
            max_turn_chars=args.max_turn_chars,
            deduplicate=args.deduplicate,
            semantic_filter=args.semantic_filter,
            semantic_top_k=args.semantic_top_k,
            embed_model=args.embed_model,
            output_path=out_path,
        )
    else:
        stream_sample(
            sample_size=args.sample_size,
            dataset_name=args.dataset,
            language=lang,
            min_turns=args.min_turns,
            filter_policy=args.filter_policy,
            require_belief_marker=args.require_belief_marker,
            require_question=args.require_question,
            allow_cross_turn=args.allow_cross_turn,
            max_turn_chars=args.max_turn_chars,
            deduplicate=args.deduplicate,
            output_path=out_path,
        )
