import json
import pandas as pd
import spacy
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

# -------------------------------------------------------------------------
# 1) Initial Setup
# -------------------------------------------------------------------------
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

nlp = spacy.load("en_core_web_sm")
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()

# Regex for valuations like $5m, $5 million, $1.2B, etc.
valuation_pattern = re.compile(r"\$\d+(?:\.\d+)?(?:[mbMB]| million| billion)?")

# -------------------------------------------------------------------------
# 2) Helper Functions
# -------------------------------------------------------------------------
def merge_adjacent_entities_by_label(ents, text, target_label):
    """
    Merge adjacent entities of the same target_label if there's no comma between them.
    Example: "Asya Bradley" recognized as (Asya) + (Bradley) => merge them.
    """
    filtered = [ent for ent in ents if ent.label_ == target_label]
    filtered.sort(key=lambda x: x.start)
    merged = []
    i = 0
    while i < len(filtered):
        current_text = filtered[i].text
        j = i + 1
        while j < len(filtered):
            inter = text[filtered[j - 1].end : filtered[j].start]
            if "," not in inter:
                current_text += " " + filtered[j].text
                j += 1
            else:
                break
        merged.append(current_text)
        i = j
    return merged

def deduplicate_list(seq):
    """
    Deduplicate items in a list (case-insensitive) while preserving order.
    """
    seen = set()
    result = []
    for item in seq:
        norm = item.lower()
        if norm not in seen:
            seen.add(norm)
            result.append(item)
    return result

def filter_founder_names(founders, startups):
    def normalize_name(name):
        return name.strip().lower()
    filtered = []
    for founder in founders:
        skip = False
        for startup in startups:
            if startup != "N/A" and normalize_name(founder) in normalize_name(startup):
                skip = True
                break
        if not skip:
            filtered.append(founder)
    return filtered

def deduplicate_phrase(phrase):
    """
    Remove duplicate words within a phrase, preserving the first occurrence.
    E.g. "Digital Mercury Digital Mercury" -> "Digital Mercury"
    """
    words = phrase.split()
    seen = set()
    result = []
    for w in words:
        lw = w.lower()
        if lw not in seen:
            seen.add(lw)
            result.append(w)
    return " ".join(result)

def preprocess_text(text):
    """
    Preprocess excerpt by removing stop words, punctuation,
    lemmatizing and then stemming each token.
    """
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        lemma = token.lemma_.lower()
        stem = ps.stem(lemma)
        tokens.append(stem)
    return " ".join(tokens)

def extract_valuation_strings(text):
    """
    Return all valuation-like strings found in 'text',
    e.g. ["$21M", "$21 million"].
    """
    return valuation_pattern.findall(text)

def tokenize_text(text):
    doc = nlp(text)
    return [token.text for token in doc if not token.is_space]

def get_sentiment_label(text):
    """
    Perform sentiment analysis using NLTK VADER on 'text'.
    Returns 'positive', 'negative', or 'neutral'.
    """
    scores = sia.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"

# -------------------------------------------------------------------------
# 3) Main Processing
# -------------------------------------------------------------------------
def process_articles(articles_list):
    """
    Takes a list of article dictionaries,
    performs NER + merging + filtering + excerpt preprocessing + sentiment,
    returns a list of processed results.
    """
    results = []
    for article in articles_list:
        title = article.get("title", "")
        excerpt_orig = article.get("excerpt", "")

        # Combine title + excerpt for entity extraction
        combined_text = f"{title} {excerpt_orig}"
        doc = nlp(combined_text)

        # 1) Extract and merge PERSON, ORG entities
        merged_persons = merge_adjacent_entities_by_label(doc.ents, combined_text, "PERSON")
        merged_orgs = merge_adjacent_entities_by_label(doc.ents, combined_text, "ORG")

        merged_persons = deduplicate_list(merged_persons)
        merged_orgs = deduplicate_list(merged_orgs)

        # Filter out founder names that appear in startup names
        filtered_founders = filter_founder_names(merged_persons, merged_orgs)
        merged_orgs = [deduplicate_phrase(org) for org in merged_orgs]
        filtered_founders = [deduplicate_phrase(f) for f in filtered_founders]

        # 2) Extract valuations and remove duplicates
        raw_valuations = extract_valuation_strings(combined_text)  # e.g. ["$21M", "$21 million"]
        raw_valuations = deduplicate_list(raw_valuations)  # keep unique phrases

        # Convert each valuation phrase to tokens, then combine them
        valuation_tokens_combined = []
        for val_str in raw_valuations:
            valuation_tokens_combined += tokenize_text(val_str)

        # Deduplicate final list of tokens
        valuation_tokens_combined = deduplicate_list(valuation_tokens_combined)

        # 3) Preprocess the excerpt
        preprocessed_excerpt = preprocess_text(excerpt_orig) if excerpt_orig else "N/A"

        # 4) Sentiment analysis on the original excerpt
        sentiment_label = "N/A"
        if excerpt_orig.strip():
            sentiment_label = get_sentiment_label(excerpt_orig)

        results.append({
            "excerpt": preprocessed_excerpt if preprocessed_excerpt else "N/A",
            "startup_name": ", ".join(merged_orgs) if merged_orgs else "N/A",
            "founder_name": ", ".join(filtered_founders) if filtered_founders else "N/A",
            "valuation": ", ".join(valuation_tokens_combined) if valuation_tokens_combined else "N/A",
            "sentiment": sentiment_label
        })
    return results

def main():
    # Load JSON data
    with open("techcrunch_articles.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    all_processed = []
    for key in ["indians", "startups", "founders"]:
        articles_list = data.get(key, [])
        processed = process_articles(articles_list)
        all_processed.extend(processed)

    # Convert to DataFrame
    processed_df = pd.DataFrame(all_processed)

    # Write to JSON (you can also do CSV if desired)
    processed_df.to_json("techcrunch_extracted_nlp_fixed.json", orient="records", indent=2)

    # Print first few rows
    print(processed_df.head(5))

if __name__ == "__main__":
    main()
