r"""
Classify user conversations from lmsys/lmsys-chat-1m using a 3-stage hierarchical rubric
with local Ollama (gemma4:12b) and structured JSON outputs, with automatic checkpoint/resume.

Stage 1 (Topic): Is the user genuinely discussing politics, social policy, economics, or controversial social issues?
Stage 2 (Prior Belief): Does the user state their OWN personal prior belief/stance in their own voice?
Stage 3 (Information-Seeking Language): Given that prior belief, is the user's request confirming, falsifying, or neutral?

Usage:
    # Run smoke test on canonical examples + pasted-context trap test
    .\venv\Scripts\python.exe src\classify.py --smoke-test

    # Classify all extracted candidates from the 1M dataset (automatically resumes if interrupted!)
    .\venv\Scripts\python.exe src\classify.py --input data\raw\candidates_full_1m_belief.parquet

    # Classify a specific slice (e.g. first 100 rows)
    .\venv\Scripts\python.exe src\classify.py --input data\raw\candidates_full_1m_belief.parquet --limit 100
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Literal

import ollama
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

load_dotenv(PROJECT_ROOT / ".env")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:12b")


class UserConfirmationClassification(BaseModel):
    # Stage 1: Domain / Topic check
    is_policy_or_political: bool = Field(
        description=(
            "True ONLY if the user is genuinely discussing or asking about politics, government, social policy, "
            "economic policy, law/regulation, or controversial social issues. "
            "False if the prompt is an NLP task (summarization, extraction, reading comprehension 'Context:... Answer'), "
            "fictional roleplay, coding, or non-political."
        )
    )
    topic_category: Literal[
        "politics",
        "social_policy",
        "economic_policy",
        "other_controversial_social_issue",
        "non_relevant",
    ] = Field(description="Primary topic category of the user's prompt(s).")

    # Stage 2: Prior belief check
    user_states_prior_belief: bool = Field(
        description=(
            "True ONLY if the user states their OWN personal belief, stance, or hypothesis in their own voice "
            "(e.g., 'I think government spending is bad'). "
            "MUST be False if the opinion/belief only appears inside a pasted article, email, transcript, "
            "reading-comprehension passage ('Context: ...'), or fictional story."
        )
    )
    stated_belief_summary: str | None = Field(
        description=(
            "REQUIRED string if user_states_prior_belief is True (quote or summary of the user's stated belief, "
            "e.g. 'I think gov spending is bad'); null if user_states_prior_belief is False."
        ),
    )

    # Stage 3: Information-seeking direction relative to prior belief
    seeking_type: Literal["confirming", "falsifying", "neutral", "not_applicable"] = Field(
        description=(
            "MUST be 'not_applicable' if is_policy_or_political is False OR user_states_prior_belief is False. "
            "Otherwise, relative to the user's stated belief: "
            "'confirming': User asks for arguments/evidence supporting their stated belief (e.g. 'I think gov spending is bad. How is it bad?'). "
            "'falsifying': User asks for arguments/evidence challenging or contradicting their stated belief (e.g. 'I think gov spending is bad. How is it good?'). "
            "'neutral': User asks for balanced/both-sides information despite stating a belief (e.g. 'I think gov spending is bad. Tell me the pros and cons.')."
        )
    )
    information_seeking_quote: str | None = Field(
        description=(
            "REQUIRED verbatim quote from the user showing how they asked for information when seeking_type is "
            "confirming/falsifying/neutral; null if not_applicable."
        ),
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")
    reasoning: str = Field(
        description="Brief 1-2 sentence explanation tracing Stage 1 (topic), Stage 2 (prior belief), and Stage 3 (seeking type)."
    )


SYSTEM_PROMPT = """You are an expert computational social science annotator analyzing how users seek information from LLMs.
Follow this strict 3-step decision tree focusing on the USER's messages:

STEP 1 — TOPIC RELEVANCE & GENUINE CONVERSATION CHECK:
Determine if the user is genuinely discussing or asking about politics, social policy, economic policy, or a controversial social issue.
CRITICAL TRAP TO AVOID: Users often paste external text (news articles, emails, transcripts, reading comprehension passages like "Context: ... Answer", or fictional roleplay) for summarization, extraction, or QA.
If the prompt is an NLP/reading-comprehension task, document summarization, roleplay, programming, or trivia, set:
- is_policy_or_political = false
- topic_category = "non_relevant"
- user_states_prior_belief = false
- seeking_type = "not_applicable"

STEP 2 — USER'S OWN PRIOR BELIEF CHECK:
If Step 1 is True, check whether the USER states their OWN personal prior belief, opinion, or hypothesis in their own voice (e.g., "I think government spending is bad", "I feel like universal basic income would cause inflation").
- Without a specific prior belief stated by the user, we cannot know if their question is confirming or falsifying.
- Do NOT attribute opinions inside pasted quotes, transcripts, or reading-comprehension passages to the user.
- If the user asks a question without stating their own belief (even a one-sided question like "Why is government spending bad?"), set:
  - user_states_prior_belief = false
  - stated_belief_summary = null
  - seeking_type = "not_applicable"

STEP 3 — INFORMATION-SEEKING LANGUAGE (only when Step 1 AND Step 2 are True):
Evaluate how the user seeks information relative to their stated prior belief:
- "confirming": The user asks for information, reasons, or evidence that AGREES with or supports their stated belief.
  Example: "I think gov spending is bad. How is it bad?" -> confirming
- "falsifying": The user asks for information, counter-arguments, or evidence that CHALLENGES or goes against their stated belief.
  Example: "I think gov spending is bad, how is it good?" -> falsifying
- "neutral": The user states a belief but asks for an objective, balanced, or two-sided overview.
  Example: "I think gov spending is bad. Tell me the pros/cons?" -> neutral
- "not_applicable": The user states a belief but is NOT asking a question to explore, test, confirm, or falsify it (e.g., asking to proofread an essay or just making a statement).

Return valid JSON matching the schema."""


def format_conversation(
    messages: list[dict],
    user_only: bool = True,
    max_turn_chars: int = 3000,
    max_total_chars: int = 8000,
) -> str:
    """Format conversation turns, capping per-turn and total length so context never overflows."""
    lines = []
    turn_idx = 1
    total_chars = 0
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        if user_only and role != "USER":
            continue
        content = msg.get("content", "").strip()
        if len(content) > max_turn_chars:
            content = content[:max_turn_chars] + "... [TURN TRUNCATED]"
        entry = f"[Turn {turn_idx} - {role}]: {content}"
        if total_chars + len(entry) > max_total_chars:
            remaining = max(0, max_total_chars - total_chars)
            if remaining > 100:
                lines.append(entry[:remaining] + "... [CONVERSATION TRUNCATED]")
            break
        lines.append(entry)
        total_chars += len(entry)
        turn_idx += 1
    return "\n\n".join(lines)


def classify_conversation(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    user_only: bool = True,
    think: bool = True,
    max_retries: int = 3,
) -> UserConfirmationClassification:
    formatted_text = format_conversation(messages, user_only=user_only)
    last_err = None

    for attempt in range(max_retries):
        try:
            temp = 0.0 if attempt == 0 else 0.1
            # On retry (attempt >= 1), disable thinking mode (`think=False`) so long multi-turn
            # conversations cannot exhaust the output token budget inside the `<think>` block.
            use_think = think if attempt == 0 else False
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Analyze the following user messages:\n\n{formatted_text}"},
                ],
                think=use_think,
                format=UserConfirmationClassification.model_json_schema(),
                options={"temperature": temp, "num_ctx": 8192, "num_predict": 2048},
            )
            raw_content = (response.message.content or "").strip()
            if not raw_content:
                raise ValueError("Empty response from Ollama")

            res = UserConfirmationClassification.model_validate_json(raw_content)

            # Enforce strict hierarchical consistency
            if not res.is_policy_or_political:
                res.topic_category = "non_relevant"
                res.user_states_prior_belief = False
                res.stated_belief_summary = None
                res.seeking_type = "not_applicable"
                res.information_seeking_quote = None
            elif not res.user_states_prior_belief:
                res.stated_belief_summary = None
                res.seeking_type = "not_applicable"
                res.information_seeking_quote = None

            return res
        except Exception as e:
            last_err = e
            time.sleep(0.5)

    raise RuntimeError(f"Failed after {max_retries} attempts: {last_err}")


def load_existing_checkpoint(output_path: Path) -> tuple[dict[str, dict], dict[str, int]]:
    """
    Load already-classified records from output_path (excluding errored rows so they can be retried).
    Rewrites the file without errored rows if any errors were present.
    """
    completed: dict[str, dict] = {}
    counts = {"confirming": 0, "falsifying": 0, "neutral": 0, "not_applicable": 0, "error": 0}
    if not output_path.exists():
        return completed, counts

    had_errors = False
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                cid = rec.get("conversation_id")
                if not cid or "error" in rec:
                    had_errors = True
                    continue
                completed[cid] = rec
                st = rec.get("seeking_type", "not_applicable")
                counts[st] = counts.get(st, 0) + 1
            except json.JSONDecodeError:
                had_errors = True

    if had_errors:
        with open(output_path, "w", encoding="utf-8") as f:
            for rec in completed.values():
                f.write(json.dumps(rec) + "\n")

    return completed, counts


def run_batch(
    input_path: Path,
    output_path: Path | None = None,
    limit: int | None = None,
    model: str = DEFAULT_MODEL,
    user_only: bool = True,
    think: bool = True,
    overwrite: bool = False,
):
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(input_path)
    if limit:
        df = df.head(limit).copy()

    if output_path is None:
        output_path = PROCESSED_DATA_DIR / f"classified_{input_path.stem}.jsonl"

    if overwrite and output_path.exists():
        output_path.unlink()

    completed, counts = load_existing_checkpoint(output_path)
    remaining_df = df[~df["conversation_id"].isin(completed.keys())].copy()

    if completed:
        print(
            f"Resuming from checkpoint: {len(completed):,} already completed, "
            f"{len(remaining_df):,} remaining -> {output_path}"
        )
    else:
        print(f"Classifying {len(remaining_df):,} conversations with {model} (user_only={user_only}, think={think}) -> {output_path}")

    if remaining_df.empty:
        print(f"All requested rows are already classified! Summary: {counts}")
        return

    with open(output_path, "a", encoding="utf-8") as f:
        pbar = tqdm(remaining_df.iterrows(), total=len(remaining_df), desc="Classifying")
        for _, row in pbar:
            conv = list(row["conversation"])
            try:
                result = classify_conversation(conv, model=model, user_only=user_only, think=think)
                counts[result.seeking_type] = counts.get(result.seeking_type, 0) + 1
                record = {
                    "conversation_id": row["conversation_id"],
                    "model_in_dataset": row["model"],
                    "classifier_model": model,
                    **result.model_dump(),
                }
            except Exception as e:
                counts["error"] += 1
                record = {
                    "conversation_id": row["conversation_id"],
                    "error": str(e),
                }
            f.write(json.dumps(record) + "\n")
            f.flush()
            pbar.set_postfix(
                conf=counts["confirming"],
                fals=counts["falsifying"],
                neut=counts["neutral"],
                na=counts["not_applicable"],
                err=counts["error"],
            )

    print(f"Done! Summary: {counts}\nResults saved to {output_path}")


SMOKE_TEST_CASES = [
    ("Confirming example", [{"role": "user", "content": "I think gov spending is bad. How is it bad?"}]),
    ("Falsifying example", [{"role": "user", "content": "I think gov spending is bad, how is it good?"}]),
    ("Neutral example", [{"role": "user", "content": "I think gov spending is bad. Tell me the pros/cons?"}]),
    ("No prior belief stated", [{"role": "user", "content": "How is government spending bad for the economy?"}]),
    ("Non-political topic", [{"role": "user", "content": "I think Python is slow. How can I vectorize this loop in NumPy?"}]),
    (
        "Pasted context trap (Row 44)",
        [
            {
                "role": "user",
                "content": (
                    "Context:\nHow can NAME_1 look at himself in the mirror knowing how many employees he harmed?\n"
                    "Is it true conservatives are mad at NAME_1 because of his position on CO2?\n"
                    "Page Two December 5, 2001 I personally can't blame the fall of the company on NAME_1.\n"
                    "But that old scoundrel had played too many scams.\n\nAnswer"
                ),
            }
        ],
    ),
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify conversations with Gemma 4 12B via Ollama")
    parser.add_argument("--smoke-test", action="store_true", help="Run test classifications on canonical examples")
    parser.add_argument("--input", type=str, help="Path to input .parquet sample file")
    parser.add_argument("--output", type=str, default=None, help="Path to output .jsonl file")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to classify")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model name (default: gemma4:12b)")
    parser.add_argument(
        "--include-assistant",
        action="store_true",
        help="Include assistant replies in prompt (default: evaluate user turns only)",
    )
    parser.add_argument(
        "--no-think",
        action="store_false",
        dest="think",
        default=True,
        help="Disable internal <think> tokens for much faster classification",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore existing checkpoint and overwrite output file from scratch",
    )
    args = parser.parse_args()

    if args.smoke_test:
        print(f"Running 6-case smoke test with local Ollama model '{args.model}' (think={args.think})...\n")
        for label, messages in SMOKE_TEST_CASES:
            print(f"=== {label} ===")
            res = classify_conversation(
                messages, model=args.model, user_only=not args.include_assistant, think=args.think
            )
            print(json.dumps(res.model_dump(), indent=2))
            print()
    elif args.input:
        out = Path(args.output) if args.output else None
        run_batch(
            Path(args.input),
            output_path=out,
            limit=args.limit,
            model=args.model,
            user_only=not args.include_assistant,
            think=args.think,
            overwrite=args.overwrite,
        )
    else:
        parser.print_help()
