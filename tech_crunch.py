import json
import re
import torch
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    pipeline
)


ner_model_name = "dslim/bert-base-NER"
ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)

valuation_pattern = re.compile(
    r"\$\d+(?:\.\d+)?(?:\s?(?:million|billion|M|B))?",
    re.IGNORECASE
)

def extract_entities(text):
    """
    Tokenizes the text with offsets, runs the NER model, and merges contiguous tokens.
    Returns lists of founders (PER), startups (ORG), and valuations (via regex).
    """
    encoded = ner_tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
    offsets = encoded["offset_mapping"].squeeze(0).tolist()
    tokens = ner_tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
    
    with torch.no_grad():
        outputs = ner_model(**{k: v for k, v in encoded.items() if k != "offset_mapping"})
    logits = outputs.logits
    pred_ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()
    id2label = ner_model.config.id2label

    entities = []
    current_entity = None

    for idx, (token, offset, label_id) in enumerate(zip(tokens, offsets, pred_ids)):
        if offset[0] == 0 and offset[1] == 0:
            continue  

        label = id2label[label_id]
        if label == "O":
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None
            continue

        parts = label.split("-")
        prefix = parts[0] if len(parts) > 1 else ""
        entity_type = parts[1] if len(parts) > 1 else label

        if prefix == "B" or current_entity is None:
            if current_entity is not None:
                entities.append(current_entity)
            current_entity = {"label": entity_type, "start": offset[0], "end": offset[1]}
        else:
            if (entity_type == current_entity["label"]) and (offset[0] <= current_entity["end"] + 3):
                current_entity["end"] = offset[1]
            else:
                entities.append(current_entity)
                current_entity = {"label": entity_type, "start": offset[0], "end": offset[1]}
    if current_entity is not None:
        entities.append(current_entity)

    
    for ent in entities:
        ent["text"] = text[ent["start"]:ent["end"]].strip()

    merged_entities = merge_adjacent_entities(entities, text)
    
    founders = set()
    startups = set()
    for ent in merged_entities:
        if ent["label"] == "PER":
            
            parts = ent["text"].split()
            if len(parts) >= 2 and all(len(part) >= 2 for part in parts):
                founders.add(ent["text"])
        elif ent["label"] == "ORG":
            if len(ent["text"]) >= 3:
                startups.add(ent["text"])
    
    valuations = valuation_pattern.findall(text)
    return list(founders), list(startups), valuations

def merge_adjacent_entities(entities, text):
    """
    Merge entities of the same type that are close together.
    """
    if not entities:
        return []
    sorted_entities = sorted(entities, key=lambda e: e["start"])
    merged = [sorted_entities[0]]
    for current in sorted_entities[1:]:
        previous = merged[-1]
        if previous["label"] == current["label"]:
            if current["start"] - previous["end"] <= 5:
                merged[-1]["end"] = current["end"]
                merged[-1]["text"] = text[merged[-1]["start"]:merged[-1]["end"]].strip()
                continue
            between_text = text[previous["end"]:current["start"]].strip().lower()
            connecting_words = [" and ", " & ", " of ", " for ", "'s ", " at ", " in ", " ipo "]
            if any(conn in f" {between_text} " for conn in connecting_words) and len(between_text) < 10:
                merged[-1]["end"] = current["end"]
                merged[-1]["text"] = text[merged[-1]["start"]:merged[-1]["end"]].strip()
                continue
        merged.append(current)
    return merged

def normalize_entity(entity):
    """
    Clean and standardize an entity string.
    """
    entity = entity.replace("’", "'").strip()
    entity = entity.replace("–", "-").replace("—", "-")
    entity = re.sub(r'\s+', ' ', entity).strip()
    tokens = entity.split()
    if len(tokens) > 1:
        tokens = [tok for tok in tokens if (len(tok) > 1 or (len(tok) == 1 and tok.isalpha()) or '.' in tok)]
    normalized = " ".join(tokens)
    normalized = re.sub(r"'(?!\s*s)", "", normalized)
    normalized = re.sub(r"'\s+s", "'s", normalized)
    
    if len(normalized) < 3:
        return ""
    filter_words = {"she", "he", "they", "i", "we", "you", "this", "that", "the", "a", "an"}
    if normalized.lower() in filter_words:
        return ""
    return normalized

def filter_substrings(entity_list):
    """
    Remove any entity that is a substring (whole-word) of another.
    """
    
    sorted_entities = sorted(entity_list, key=len, reverse=True)
    filtered = []
    for item in sorted_entities:
        if not item or len(item) < 3:
            continue
        
        pattern = r'\b' + re.escape(item.lower()) + r'\b'
        if any(re.search(pattern, other.lower()) and item.lower() != other.lower() for other in filtered):
            continue
        filtered.append(item)
    return filtered

def filter_entities_by_excerpt(entity_list, excerpt):
    """
    Keep only those entities that appear as whole words in the excerpt.
    """
    filtered = []
    excerpt_lower = excerpt.lower()
    for entity in entity_list:
        pattern = r'\b' + re.escape(entity.lower()) + r'\b'
        if re.search(pattern, excerpt_lower):
            filtered.append(entity)
    return filtered

def normalize_valuation(val):
    """
    Normalize a valuation string.
    """
    v = val.lower().replace(" ", "")
    v = v.replace("million", "m").replace("billion", "b")
    return v

def parse_valuation(val):
    """
    Parse a normalized valuation string into a numeric value.
    """
    v = val.replace("$", "")
    match = re.match(r"(\d+(?:\.\d+)?)([mb]?)", v)
    if not match:
        return None
    num = float(match.group(1))
    unit = match.group(2)
    if unit == "m":
        return num * 1e6
    elif unit == "b":
        return num * 1e9
    else:
        return num

def deduplicate_valuations(valuation_list):
    """
    Deduplicate valuations by normalizing and comparing their numeric values.
    If two valuations differ by less than 10%, choose the longer string.
    """
    parsed = []
    for val in valuation_list:
        norm = normalize_valuation(val)
        num = parse_valuation(norm)
        if num is not None:
            parsed.append((val, norm, num))
    if not parsed:
        return []
    
    unique = {}
    for orig, norm, num in parsed:
        if norm in unique:
            existing_num = parse_valuation(normalize_valuation(unique[norm]))
            if abs(num - existing_num) / ((num + existing_num) / 2) < 0.1:
                if len(orig) > len(unique[norm]):
                    unique[norm] = orig
            elif num > existing_num:
                unique[norm] = orig
        else:
            unique[norm] = orig
    return list(unique.values())


sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer)
sentiment_mapping = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}

def get_sentiment(text):
    """
    Uses the sentiment pipeline to analyze text and return its sentiment.
    """
    if not text.strip():
        return "neutral"
    results = sentiment_pipeline(text)
    label = results[0]["label"]
    return sentiment_mapping.get(label, label)


def main():
    parser = argparse.ArgumentParser(description="Extract entities and sentiment from articles.")
    parser.add_argument("--input", type=str, default="techcrunch_articles.json", help="Path to input JSON file")
    parser.add_argument("--output", type=str, default="filtered.json", help="Path to output JSON file")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    articles = []
    if isinstance(data, dict):
        for section, arts in data.items():
            for article in arts:
                if "section" not in article:
                    article["section"] = section
                articles.append(article)
    elif isinstance(data, list):
        articles = data
    else:
        raise ValueError("Unsupported data format. Expecting a list or dict.")

    filtered_data = []
    for article in articles:
        title = article.get("title", "")
        excerpt = article.get("excerpt", "")
        author = article.get("author", "")
        category = article.get("category", article.get("section", ""))
        
       
        extraction_text = excerpt.strip() if excerpt.strip() else f"{title} {author} {category}"
        
        founders, startups, valuations = extract_entities(extraction_text)
        
        founders = [normalize_entity(x) for x in founders]
        startups = [normalize_entity(x) for x in startups]
        founders = filter_substrings(founders)
        startups = filter_substrings(startups)
        
        
        if excerpt.strip():
            founders = filter_entities_by_excerpt(founders, excerpt)
            startups = filter_entities_by_excerpt(startups, excerpt)
        
        valuations = deduplicate_valuations(valuations)
        sentiment = get_sentiment(excerpt)
        
        filtered_data.append({
            "title": title,
            "excerpt": excerpt,
            "section": category,
            "founders": founders,
            "startups": startups,
            "valuations": valuations,
            "sentiment": sentiment
        })
    
    with open(args.output, "w", encoding="utf-8") as out_f:
        json.dump(filtered_data, out_f, indent=4)
    
    print(f"Extraction complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()
