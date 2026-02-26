import pandas as pd
import re
import jellyfish
from langdetect import detect

STOPWORDS = {"today", "news", "india", "samachar", "daily", "the"}
PERIODICITY_MAP = {
    "daily": "D", "weekly": "W", "monthly": "M", "fortnightly": "F",
    "annual": "A", "bi-weekly": "BW", "quarterly": "Q"
}

def clean_text(text):
    if not isinstance(text, str): return ""
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords
    words = text.split()
    filtered_words = [w for w in words if w not in STOPWORDS]
    return " ".join(filtered_words)

def normalize_periodicity(p_string):
    if not isinstance(p_string, str): return "U" # Unknown
    p = p_string.lower().strip()
    return PERIODICITY_MAP.get(p, "U")

def get_language(text):
    if not isinstance(text, str) or not text.strip():
        return "unknown"
    try:
        lang = detect(text)
        return "hi" if lang == "hi" else "en"
    except:
        return "unknown"

def preprocess_titles(input_csv="combined_raw.csv", output_csv="combined_preprocessed.csv"):
    """
    Step 3: Cleans & Preprocesses raw merged titles.
    Generates: title_en_clean, title_hi_clean, phonetics, language.
    """
    print(f"Loading {input_csv} for preprocessing...")
    df = pd.read_csv(input_csv)
    
    # 1. Clean Titles
    print("Cleaning English Titles...")
    df["title_en_clean"] = df["Title Name (English)"].apply(clean_text)
    
    print("Cleaning Hindi Titles...")
    df["title_hi_clean"] = df["Hindi Title"].apply(clean_text)
    
    # 2. Extract normalized periodicity tag
    print("Normalizing periodicity tags...")
    if "Periodicity" in df.columns:
        df["periodicity_tag"] = df["Periodicity"].apply(normalize_periodicity)
    else:
        df["periodicity_tag"] = "U"
        
    # 3. Detect Language on English / Hindi Fields just in case
    print("Detecting mixed script languages...")
    df["detected_lang_en"] = df["title_en_clean"].apply(get_language)
    
    # 4. Phonetic Codes (Only on English clean strings for accuracy)
    print("Generating Metaphone and Soundex codes...")
    def get_phonetics(t):
        if not t: return "", ""
        return jellyfish.metaphone(t), jellyfish.soundex(t)

    phonetics = df["title_en_clean"].apply(get_phonetics)
    df["metaphone_code"] = [p[0] for p in phonetics]
    df["soundex_code"] = [p[1] for p in phonetics]
    
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Preprocessing Complete. Saved '{output_csv}' with shape {df.shape}")

if __name__ == "__main__":
    preprocess_titles()
