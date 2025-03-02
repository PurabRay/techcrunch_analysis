import json
import pandas as pd
import spacy
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

nlp = spacy.load("en_core_web_sm")
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()

valuation_pattern = re.compile(r"\$\d+(?:\.\d+)?(?:[mbMB]| million| billion)?")

def merge_adjacent_entities_by_label(ents, text, target_label):
    """
    Merge adjacent entities of the same target_label if there's no comma between them.
    Example: "Asya Bradley" recognized separately => merge them if no comma.
    """
    filtered = [ent for ent in ents if ent.label_ == target_label]
    filtered.sort(key=lambda x: x.start)
    merged = []
    i = 0
    while i < len(filtered):
        current_text = filtered[i].text
        j = i + 1
        while j < len(filtered):
            inter = text[filtered[j-1].end : filtered[j].start]
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
    """
    Remove founder names that appear (or are contained) in any startup name.
    e.g., if 'Asya Bradley' is in both, remove from founder.
    """
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
    E.g. "Digital Mercury Digital Mercury" => "Digital Mercury"
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

def parse_and_unify_valuation(val_str):
    """
    1) Ensure val_str has:
        - A currency marker: '$'
        - A numeric portion
        - A magnitude (M/m/million or B/b/billion)
    2) Normalize to a standard phrase: '$X million' or '$X billion'.

    Return the unified string, or None if it doesn't meet the 3-element rule.
    """
    pattern = re.compile(r"^\$(?P<num>\d+(?:\.\d+)?)(?P<magnitude>[mMbB]| million| billion)$")
    match = pattern.match(val_str.strip())
    if not match:
        
        return None

    num = match.group("num")  
    magnitude = match.group("magnitude").strip().lower() 

    if magnitude in ["m", "million"]:
        return f"${num} million"
    elif magnitude in ["b", "billion"]:
        return f"${num} billion"
    else:
        return None

def extract_valuation_strings(text):
    """
    Return a deduplicated list of unified valuations from the text.
    We only keep if they have all 3: $ + numeric + magnitude
    """
    raw_matches = valuation_pattern.findall(text) 
    raw_matches = deduplicate_list(raw_matches)  
    unified_vals = []

    for val_str in raw_matches:
        unified = parse_and_unify_valuation(val_str)
        if unified:
            unified_vals.append(unified)
    unified_vals = deduplicate_list(unified_vals)
    return unified_vals

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

        combined_text = f"{title} {excerpt_orig}"
        doc = nlp(combined_text)

        merged_persons = merge_adjacent_entities_by_label(doc.ents, combined_text, "PERSON")
        merged_orgs = merge_adjacent_entities_by_label(doc.ents, combined_text, "ORG")
        merged_persons = deduplicate_list(merged_persons)
        merged_orgs = deduplicate_list(merged_orgs)

        filtered_founders = filter_founder_names(merged_persons, merged_orgs)
        merged_orgs = [deduplicate_phrase(org) for org in merged_orgs]
        filtered_founders = [deduplicate_phrase(f) for f in filtered_founders]

        valuations = extract_valuation_strings(combined_text)
        valuation_str = ", ".join(valuations) if valuations else "N/A"

        preprocessed_excerpt = preprocess_text(excerpt_orig) if excerpt_orig else "N/A"

        sentiment_label = "N/A"
        if excerpt_orig.strip():
            sentiment_label = get_sentiment_label(excerpt_orig)

        results.append({
            "excerpt": preprocessed_excerpt if preprocessed_excerpt else "N/A",
            "startup_name": ", ".join(merged_orgs) if merged_orgs else "N/A",
            "founder_name": ", ".join(filtered_founders) if filtered_founders else "N/A",
            "valuation": valuation_str,
            "sentiment": sentiment_label
        })
    return results

def main():
    
    with open("techcrunch_articles.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    all_processed = []
    for key in ["indians", "startups", "founders"]:
        articles_list = data.get(key, [])
        processed = process_articles(articles_list)
        all_processed.extend(processed)

    processed_df = pd.DataFrame(all_processed)

    processed_df.to_json("techcrunch_extracted_nlp_fixed.json", orient="records", indent=2)


if __name__ == "__main__":
    main()
