"""
sentiment.py
Analyzes sentiment per speaker turn and computes overall meeting tone.
Uses distilbert-base-uncased-finetuned-sst-2-english.
"""

import re
import yaml
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = CONFIG["models"]["sentiment"]

_tokenizer = None
_model     = None


def _load():
    global _tokenizer, _model
    if _tokenizer is None:
        print(f"DEBUG — loading sentiment model: {MODEL_NAME}")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model     = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME
        ).to(DEVICE)



def _parse_turns(transcript: str) -> list[dict]:
    """
    Splits transcript into speaker turns.
    Skips metadata lines.
    Returns: [{"speaker": str, "text": str}, ...]
    """
    turns = []
    for line in transcript.splitlines():
        line = line.strip()
        if not line:
            continue
        if any(line.lower().startswith(k) for k in
               ["meeting:", "attendees:", "date:", "location:"]):
            continue
        match = re.match(r"^([A-Za-z]+):\s*(.+)$", line)
        if match:
            turns.append({
                "speaker": match.group(1),
                "text"   : match.group(2).strip(),
            })
    return turns


def _predict(text: str) -> dict:
    """
    Runs model on a single text.
    Returns: {"label": "POSITIVE"/"NEUTRAL"/"NEGATIVE", "score": float}
    """
    _load()

    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = _model(**inputs)

    probs     = torch.softmax(outputs.logits, dim=1)[0]
    label_idx = torch.argmax(probs).item()

    # cardiffnlp uses: 0=negative, 1=neutral, 2=positive
    label_map = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
    label     = label_map.get(label_idx, _model.config.id2label[label_idx].upper())
    score     = round(probs[label_idx].item() * 100, 1)

    return {"label": label, "score": score}


def analyze(transcript: str) -> dict:
    """
    Analyzes full transcript — returns overall tone, per-speaker counts, turns.
    """
    turns = _parse_turns(transcript)

    if not turns:
        result = _predict(transcript[:512])
        return {"overall": result, "by_speaker": {}, "turns": []}

    analyzed_turns = []
    by_speaker     = {}

    for turn in turns:
        result = _predict(turn["text"])
        label  = result["label"]
        score  = result["score"]

        analyzed_turns.append({
            "speaker": turn["speaker"],
            "text"   : turn["text"],
            "label"  : label,
            "score"  : score,
        })

        if turn["speaker"] not in by_speaker:
            by_speaker[turn["speaker"]] = {
                "POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0
            }
        by_speaker[turn["speaker"]][label] += 1

    # overall = most frequent label
    from collections import Counter
    label_counts  = Counter(t["label"] for t in analyzed_turns)
    overall_label = label_counts.most_common(1)[0][0]
    overall_score = round(
        label_counts[overall_label] / len(analyzed_turns) * 100, 1
    )

    return {
        "overall"   : {"label": overall_label, "score": overall_score},
        "by_speaker": by_speaker,
        "turns"     : analyzed_turns,
    }