"""
action_extractor.py
Extracts action items from meeting transcripts using a hybrid approach:
1. Rule-based extraction — finds lines where someone commits to doing something
2. Light model cleanup — makes the task description cleaner
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

# keywords that signal a task or commitment
ACTION_KEYWORDS = [
    "i'll", "i will", "i can", "will fix", "will send",
    "will write", "will have", "will do", "will prepare",
    "will finish", "will complete", "will handle", "will reach",
    "will contact", "will submit", "will draft", "will review",
    "can draft", "can reach", "can contact", "can prepare",
    "i'll take care", "let me", "i'll make sure",
]


def _load():
    global _tokenizer, _model
    if _tokenizer is None:
        print(f"DEBUG — loading: {MODEL_NAME}")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)


def _extract_speakers(transcript: str) -> list[str]:
    """Find all speaker names in the transcript."""
    matches = re.findall(r"^([A-Za-z]+):", transcript, re.MULTILINE)
    return list(dict.fromkeys(matches))  # deduplicated, order preserved

def _parse_conversation(transcript):
    conversation = []

    for line in transcript.splitlines():

        line = line.strip()

        if not line:
            continue

        if any(line.lower().startswith(k) for k in
               ["meeting:", "attendees:", "date:", "location:"]):
            continue

        m = re.match(r"^([A-Za-z]+):\s*(.+)$", line)

        if m:
            conversation.append({
                "speaker": m.group(1),
                "text": m.group(2)
            })

    return conversation
def _rule_based_extract(transcript: str) -> list[dict]:
    """
    Finds lines where a speaker makes a commitment.
    """
    actions     = []
    speakers    = _extract_speakers(transcript)
    seen_lines  = set()

    lines = transcript.splitlines()
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # skip metadata
        if any(line.lower().startswith(k) for k in
               ["meeting:", "attendees:", "date:", "location:"]):
            continue

        speaker_match = re.match(r"^([A-Za-z]+):\s*(.+)$", line)
        if not speaker_match:
            continue

        current_speaker = speaker_match.group(1)
        text            = speaker_match.group(2).strip()
        text_lower      = text.lower()

        # skip short filler lines
        if len(text.split()) < 4:
            continue

        # check for action keyword
        if any(kw in text_lower for kw in ACTION_KEYWORDS):
            if text not in seen_lines:
                seen_lines.add(text)

                # get previous speaker line for context if text is vague
                context = ""
                if i > 0:
                    prev = lines[i - 1].strip()
                    prev_match = re.match(r"^([A-Za-z]+):\s*(.+)$", prev)
                    if prev_match:
                        context = prev_match.group(2).strip()

                actions.append({
                    "speaker" : current_speaker,
                    "raw_text": text,
                    "context" : context,
                })

        # "please, Name, do X" pattern — task assigned TO someone else
        please_match = re.search(
            r"(?:please[,\s]+)?([A-Za-z]+)[,\s]+please\s+(.+)|"
            r"please\s+([A-Za-z]+)[,\s]+(.+)",
            text, re.IGNORECASE
        )
        if please_match:
            # figure out which group matched
            if please_match.group(1) and please_match.group(2):
                assigned_to = please_match.group(1)
                task_text   = please_match.group(2).strip()
            else:
                assigned_to = please_match.group(3)
                task_text   = please_match.group(4).strip()

            # only add if assigned_to is a real speaker
            if assigned_to in speakers and task_text not in seen_lines:
                seen_lines.add(task_text)
                actions.append({
                    "speaker" : assigned_to,
                    "raw_text": task_text,
                    "context" : "",
                })

    return actions


def _clean_task(raw_text: str, context: str = "") -> str:
    """
    Converts first-person commitment to a clean task description.
    Uses context from the previous line to fill in vague pronouns.
    """
    text = raw_text.strip()

    # remove leading filler phrases before the actual commitment
    filler_patterns = [
        r"^(that'?s?\s+\w+\s+\w+\.?\s*)",   # "That's a known bug."
        r"^(great\.?\s*)",                    # "Great."
        r"^(sure,?\s*)",                      # "Sure,"
        r"^(okay\.?\s*)",                     # "Okay."
        r"^(yes,?\s*)",                       # "Yes,"
        r"^(also,?\s*[A-Za-z]+,?\s*)",        # "Also, John,"
    ]
    for pattern in filler_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    # replace vague pronoun "them" with context if available
    if context and re.search(r"\bthem\b|\bit\b", text, re.IGNORECASE):
        # extract the object from context (what was asked)
        obj_match = re.search(
            r"(?:have|get|send|prepare|finish|complete)\s+(?:the\s+)?(.+?)(?:\s+ready|\s+by|\?|$)",
            context, re.IGNORECASE
        )
        if obj_match:
            obj = obj_match.group(1).strip()
            text = re.sub(r"\bthem\b", obj, text, flags=re.IGNORECASE)
            text = re.sub(r"\bit\b",   obj, text, flags=re.IGNORECASE)

    # convert first-person → imperative
    replacements = [
        (r"^I'll\s+",            "",      re.IGNORECASE),
        (r"^I will\s+",          "",      re.IGNORECASE),
        (r"^I can\s+",           "",      re.IGNORECASE),
        (r"^Sure,?\s+I'll\s+",   "",      re.IGNORECASE),
        (r"^Yes,?\s+I will\s+",  "",      re.IGNORECASE),
        (r"\bI'll\b",            "will",  re.IGNORECASE),
        (r"\bI will\b",          "will",  re.IGNORECASE),
        (r"\bmy\b",              "their", re.IGNORECASE),
    ]
    for pattern, replacement, flags in replacements:
        text = re.sub(pattern, replacement, text, flags=flags)

    # capitalize first letter
    text = text.strip(" .,")
    if text:
        text = text[0].upper() + text[1:]

    return text

def extract_actions(transcript: str) -> list[dict]:
    """
    Main function — returns list of {"task": str, "owner": str}
    """
    raw_actions = _rule_based_extract(transcript)
    print(f"DEBUG — found {len(raw_actions)} raw actions")
    for a in raw_actions:
        print(f"  {a['speaker']}: {a['raw_text']}")

    if not raw_actions:
        return [{"task": "No clear action items found in transcript.", "owner": "—"}]

    # clean each task description
    actions = []
    for a in raw_actions:
        clean = _clean_task(a["raw_text"], a.get("context", ""))
        if clean:
            actions.append({
                "task" : clean,
                "owner": a["speaker"],
            })

    # deduplicate by task text
    seen  = set()
    final = []
    for a in actions:
        if a["task"] not in seen:
            seen.add(a["task"])
            final.append(a)

    return final