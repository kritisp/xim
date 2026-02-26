import re

def clean_text(text: str) -> str:
    """Removes special characters and standardizes whitespace."""
    text = re.sub(r'[^\w\s]', '', text)
    return ' '.join(text.split()).lower()
