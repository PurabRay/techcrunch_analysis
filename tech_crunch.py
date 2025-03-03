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
        
        entity_type = label.split("-")[-1]
        if current_entity is None:
            current_entity = {"label": entity_type, "start": offset[0], "end": offset[1]}
        else:
            
            if (entity_type == current_entity["label"]) and (offset[0] <= current_entity["end"] + 1):
                current_entity["end"] = offset[1]
            else:
                entities.append(current_entity)
                current_entity = {"label": entity_type, "start": offset[0], "end": offset[1]}
    if current_entity is not None:
        entities.append(current_entity)
    for ent in entities:
        ent["text"] = text[ent["start"]:ent["end"]].strip()
    
    founders = set()
    startups = set()
    for ent in entities:
        if ent["label"] == "PER":
            founders.add(ent["text"])
        elif ent["label"] == "ORG":
            startups.add(ent["text"])
    
    valuations = valuation_pattern.findall(text)
    return list(founders), list(startups), valuations


def normalize_entity(entity):
    """
    Normalizes an extracted entity by replacing fancy quotes,
    splitting into tokens, and removing tokens that are too short
    (unless the entire entity is very short).
    """
    
    entity = entity.replace("’", "'").strip()
    tokens = entity.split()
    
    if len(tokens) > 1:
        tokens = [tok for tok in tokens if len(tok) >= 3]
    return " ".join(tokens)

def filter_substrings(entity_list):
    """
    Remove any string that is a substring of another (case-insensitive).
    """
    filtered = []
    for item in entity_list:
        if not any((item.lower() != other.lower() and item.lower() in other.lower()) for other in entity_list):
            filtered.append(item)
    return filtered


def normalize_valuation(val):
    """
    Normalize valuation string to a canonical form:
      - Lowercase,
      - Remove spaces,
      - Replace "million" with "m" and "billion" with "b".
    """
    v = val.lower().replace(" ", "")
    v = v.replace("million", "m").replace("billion", "b")
    return v

def parse_valuation(val):
    """
    Parses a normalized valuation string into a numeric value.
    For example, "$210m" becomes 210e6 and "$3b" becomes 3e9.
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
    Deduplicate valuations by normalizing them.
    If there are multiple unique valuations and the ratio of the maximum to minimum
    exceeds 10, return only the highest numeric valuation.
    """
    parsed = []
    for val in valuation_list:
        norm = normalize_valuation(val)
        num = parse_valuation(norm)
        if num is not None:
            parsed.append((val, norm, num))
    if not parsed:
        return []
    
    unique_norm = {}
    for orig, norm, num in parsed:
        if norm in unique_norm:
            current_num = parse_valuation(normalize_valuation(unique_norm[norm]))
            if num > current_num:
                unique_norm[norm] = orig
        else:
            unique_norm[norm] = orig
    unique_vals = list(unique_norm.values())
    nums = [parse_valuation(normalize_valuation(val)) for val in unique_vals]
    if len(nums) > 1 and max(nums) / min(nums) > 10:
        max_val = unique_vals[nums.index(max(nums))]
        return [max_val]
    return unique_vals


sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer)
sentiment_mapping = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}

def get_sentiment(text):
    """
    Uses the sentiment pipeline to analyze text and returns "negative",
    "neutral", or "positive".
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
        raise ValueError("Unsupported data format in input file. Expecting a list or dict.")

    filtered_data = []
    for article in articles:
        
        text_content = " ".join(str(article.get(field, "")) for field in ["title", "excerpt", "author"])
        founders, startups, valuations = extract_entities(text_content)
        
        
        founders = [normalize_entity(x) for x in founders]
        startups = [normalize_entity(x) for x in startups]
        founders = filter_substrings(founders)
        startups = filter_substrings(startups)
        valuations = deduplicate_valuations(valuations)
        
        excerpt = article.get("excerpt", "")
        sentiment = get_sentiment(excerpt)
        
        filtered_data.append({
            "title": article.get("title", ""),
            "excerpt": excerpt,
            "section": article.get("section", ""),
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
