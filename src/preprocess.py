"""Text cleaning shared by training and the Streamlit app."""
import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_HTML = re.compile(r"<[^>]+>")
# Real Reuters articles in the ISOT dataset start with "CITY (Reuters) - ".
# Leaving this in lets a model "cheat" by spotting the source, so we strip it.
_REUTERS = re.compile(r"^.{0,80}?\(reuters\)\s*-?\s*")
_NON_ALPHA = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, drop URLs/HTML/source tags/punctuation/digits, collapse spaces."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _REUTERS.sub("", text)
    text = _URL.sub(" ", text)
    text = _HTML.sub(" ", text)
    text = _NON_ALPHA.sub(" ", text)
    return _SPACES.sub(" ", text).strip()
