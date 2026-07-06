"""
summarizer.py
Loads BART from config and summarizes a meeting transcript.
"""

import re
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = CONFIG["models"]["summarizer"]
PARAMS     = CONFIG["summarizer"]

_tokenizer = None
_model     = None


def _load():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)


def _preprocess(transcript: str) -> str:
    """
    Converts speaker-turn format into clean paragraph text.
    Removes speaker names and joins lines into flowing sentences.

    Example:
      "Sarah: Let's start." → "Let's start."
    """
    lines = transcript.splitlines()
    cleaned = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # remove "Speaker: " prefix
        line = re.sub(r"^[A-Za-z ]+:\s*", "", line)
        if line:
            cleaned.append(line)

    # join into one paragraph — BART works better with flowing text
    return " ".join(cleaned)


def summarize(transcript: str) -> str:
    """
    Returns a concise summary of the meeting transcript.
    Preprocesses the transcript before sending to BART.
    """
    _load()
    logger.info(f"Loading model: {MODEL_NAME}")
    # clean the transcript first
    clean_text = _preprocess(transcript)
    prompt = f"""
    You are an AI meeting assistant.

    Summarize the following meeting.

    Focus on:
    - the main topics discussed
    - important decisions
    - action items
    - deadlines
    - problems or blockers

    Write the summary in 4-6 concise bullet points.

    Meeting Transcript:
    {clean_text}
    """
    inputs = _tokenizer(
        clean_text,
        return_tensors="pt",
        truncation=True,
        max_length=PARAMS["max_input_tokens"],
    ).to(DEVICE)

    output = _model.generate(
        **inputs,
        max_length=PARAMS["max_output_tokens"],
        min_length=PARAMS["min_output_tokens"],
        length_penalty=PARAMS["length_penalty"],
        no_repeat_ngram_size=PARAMS["no_repeat_ngram_size"],
        num_beams=PARAMS["num_beams"],
        early_stopping=True,

    )

    return _tokenizer.decode(output[0], skip_special_tokens=True)