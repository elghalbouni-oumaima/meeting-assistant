"""
action_extractor.py
Extracts action items and responsible people from a meeting transcript.
Uses flan-t5-base with a carefully crafted instruction prompt.
"""

import re
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = CONFIG["models"]["action_extractor"]
PARAMS     = CONFIG["action_extractor"]

_tokenizer = None
_model     = None


def _load():
    global _tokenizer, _model
    if _tokenizer is None:
        print(f"DEBUG — loading action extractor: {MODEL_NAME}")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)


def _parse_output(raw: str) -> list[dict]:
    """
    Parses model output into structured action items.
    Expected format per line: "Person: task description"
    Falls back gracefully if format is not followed.
    """
    actions = []
    lines   = raw.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # remove list markers like "1.", "-", "•"
        line = re.sub(r"^[\d\.\-\•\*]+\s*", "", line).strip()
        if not line:
            continue

        # try to split "Owner: task" or "task (Owner)"
        if ":" in line:
            parts = line.split(":", 1)
            owner = parts[0].strip()
            task  = parts[1].strip()

            # sanity check — owner should be a short name, not a sentence
            if len(owner.split()) <= 3 and task:
                actions.append({"task": task, "owner": owner})
                continue

        # fallback — no clear owner found
        actions.append({"task": line, "owner": "Unassigned"})

    return actions


def extract_actions(transcript: str) -> list[dict]:
    """
    Returns a list of action items extracted from the transcript.
    Each item: {"task": str, "owner": str}
    """
    _load()

    # remove metadata lines before sending to model
    clean_lines = []
    for line in transcript.splitlines():
        line = line.strip()
        if not line:
            continue
        if any(line.lower().startswith(k) for k in ["meeting:", "attendees:", "date:", "location:"]):
            continue
        clean_lines.append(line)
    clean_transcript = "\n".join(clean_lines)

    prompt = f"""Extract all action items from this meeting transcript.
For each action item, write who is responsible and what they need to do.
Format: Person: task description
Only include concrete tasks with a clear owner.

Meeting transcript:
{clean_transcript[:1000]}

Action items:"""

    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=PARAMS["max_input_tokens"],
    ).to(DEVICE)

    output = _model.generate(
        **inputs,
        max_new_tokens=PARAMS["max_new_tokens"],
        num_beams=PARAMS["num_beams"],
        early_stopping=True,
        no_repeat_ngram_size=2,
    )

    raw = _tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"DEBUG — raw model output:\n{raw}")

    actions = _parse_output(raw)

    # fallback — if parsing found nothing meaningful
    if not actions:
        actions = [{"task": raw, "owner": "Unassigned"}]

    return actions