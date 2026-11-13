from src.preprocess import clean_text


def test_clean_removes_urls_html_and_reuters_tag():
    raw = "WASHINGTON (Reuters) - Visit <b>https://x.com/a</b> NOW!!! 123"
    assert clean_text(raw) == "visit now"


def test_clean_handles_non_string():
    assert clean_text(None) == ""
