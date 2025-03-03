import json
import re
import torch
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
    Tokenizes the text (with offsets), runs the NER model, and merges tokens
    that appear to be part of the same entity based on their offset continuity.
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
            current_entity = {
                "label": entity_type,
                "start": offset[0],
                "end": offset[1]
            }
        else:
            if (entity_type == current_entity["label"]) and (offset[0] <= current_entity["end"] + 1):
                current_entity["end"] = offset[1]
            else:
                entities.append(current_entity)
                current_entity = {
                    "label": entity_type,
                    "start": offset[0],
                    "end": offset[1]
                }
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

    valuations = sorted(set(valuation_pattern.findall(text)))
    return list(founders), list(startups), valuations


sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer)


sentiment_mapping = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}

def get_sentiment(text):
    """
    Analyzes the sentiment of the text using the sentiment pipeline and returns
    one of "negative", "neutral", or "positive".
    """
    if not text.strip():
        return "neutral"
    results = sentiment_pipeline(text)
    label = results[0]["label"]
    return sentiment_mapping.get(label, label)


def main():
    
    with open("techcrunch_articles.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered_data = []
    
    for section, articles in data.items():
        for article in articles:
            
            text_content = " ".join(str(article.get(field, "")) for field in ["title", "excerpt", "author"])
            founders, startups, valuations = extract_entities(text_content)
            
            excerpt = article.get("excerpt", "")
            sentiment = get_sentiment(excerpt)
            
            filtered_data.append({
                "title": article.get("title", ""),
                "excerpt": excerpt,
                "section": section,
                "founders": founders,
                "startups": startups,
                "valuations": valuations,
                "sentiment": sentiment
            })

    
    with open("filtered.json", "w", encoding="utf-8") as out_f:
        json.dump(filtered_data, out_f, indent=4)

    print("Extraction complete. Results saved to filtered.json")

if __name__ == "__main__":
    main()
