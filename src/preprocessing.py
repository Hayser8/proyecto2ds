
import re, math, unicodedata
import pandas as pd

__all__ = ["clean_text", "truncate_head_tail", "clean_and_truncate_row"]

def _strip_control_chars(s: str) -> str:
    return "".join(ch for ch in s if (unicodedata.category(ch)[0] != "C") or ch in ("\n", "\t"))

def clean_text(x) -> str:
    s = "" if pd.isna(x) else str(x)
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _strip_control_chars(s)
    s = re.sub(r"[^\S\n]+", " ", s)
    return s.strip()

def truncate_head_tail(s: str, max_chars: int, tail_frac: float = 0.25) -> str:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    if len(s) <= max_chars:
        return s
    head_len = int(math.ceil((1.0 - tail_frac) * max_chars))
    tail_len = max_chars - head_len
    return s[:head_len].rstrip() + "\n...\n" + s[-tail_len:].lstrip()

def clean_and_truncate_row(row: dict, max_char_prompt: int, max_char_response: int) -> dict:
    pr = clean_text(row.get("prompt", ""))
    ra = clean_text(row.get("response_a", ""))
    rb = clean_text(row.get("response_b", ""))
    pr = truncate_head_tail(pr, max_char_prompt)
    ra = truncate_head_tail(ra, max_char_response)
    rb = truncate_head_tail(rb, max_char_response)
    row = dict(row)
    row["prompt"] = pr
    row["response_a"] = ra
    row["response_b"] = rb
    return row
