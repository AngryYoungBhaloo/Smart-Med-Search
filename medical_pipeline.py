#!/usr/bin/env python3
"""
medical_pipeline.py

This script does the following:

1. Connects to Elasticsearch ("localhost:9200", basic_auth: "elastic"/"changeme").
2. (Re)creates an index named "medical-terms-index" to store terms, embeddings, context, classification.
3. Loads terms from 'wordlist.txt' (one term per line).
4. Merges these terms with the sanitized GPT cache (from gpt4omini_cache_sanitized.json).
5. If RUN_GPT=True, calls GPT 4o-mini for any terms missing from the cache. 
   (But if you already have them all in the cache, it won't re-call GPT.)
6. Creates text+context embeddings. 
7. Deduplicates the embeddings at a high threshold (e.g., 0.995) so near-identical items collapse to one record.
8. Exports all final docs to a CSV.

You can skip GPT classification entirely (RUN_GPT=False) if the cache is already complete.
"""

import os
import time
import json
import re
import math
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers
from openai import OpenAI

# -------------------- CONFIG ----------------------------------

RUN_GPT = False  # Set to False if you do NOT want to call GPT at all. 
                 # (Use the sanitized cache only.)

ES_HOST = "http://localhost:9200"
ES_USER = "elastic"
ES_PASS = "changeme"
INDEX_NAME = "medical-terms-index"

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

EMBED_MODEL = "text-embedding-3-large"  # 3072 dimensions
CLASSIFICATION_MODEL = "gpt-4o-mini"    # We won't call it if RUN_GPT=False

EMBED_DIM = 3072
BATCH_SIZE = 50

# If you want to make the dedup threshold stricter (so that almost identical items are removed),
# raise it closer to 1.0. 
DEDUP_THRESHOLD = 0.995  

WORDLIST_FILE = "wordlist.txt"
OUTPUT_CSV = "final_medical_terms.csv"
SANITIZED_CACHE_FILE = "gpt4omini_cache_sanitized.json"

# -------------------- Categories ----------------------------------
CATEGORIES = [
    "Disease or Syndrome",
    "Symptom or Sign",
    "Lab Test or Result",
    "Medication or Drug",
    "Medical Device",
    "Procedure or Therapy",
    "Anatomical Structure",
    "Organ System",
    "Cell or Tissue",
    "Organism (Bacteria, Virus, etc.)",
    "Medical Specialty or Department",
    "Risk Factor",
    "Preventive Measure",
    "Patient Demographic",
    "Genetic Factor",
    "Nutritional or Dietary Factor",
    "Healthcare Setting or Facility",
    "Healthcare Occupation or Role",
    "Healthcare Program or Policy",
    "Medical Research or Study",
    "Other"
]

# -------------------- ES Setup ----------------------------------

es = Elasticsearch(ES_HOST, basic_auth=(ES_USER, ES_PASS))

def recreate_index():
    """
    Deletes the old index if it exists, and creates a fresh one with the desired mapping.
    """
    if es.indices.exists(index=INDEX_NAME):
        print(f"Deleting existing index '{INDEX_NAME}'...")
        es.indices.delete(index=INDEX_NAME)
    print(f"Creating new index '{INDEX_NAME}'...")
    index_body = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "term": {"type": "keyword"},
                "context": {"type": "text"},
                "embedding": {"type": "dense_vector", "dims": EMBED_DIM},
                "classification": {"type": "keyword"}
            }
        }
    }
    es.indices.create(index=INDEX_NAME, body=index_body)
    print(f"Index '{INDEX_NAME}' created.")

def fetch_all_docs(chunk_size=1000):
    """
    Scroll through all docs in the index, returning a list of dicts.
    """
    data = []
    resp = es.search(
        index=INDEX_NAME,
        body={"query": {"match_all": {}}, "size": chunk_size},
        scroll='2m'
    )
    scroll_id = resp["_scroll_id"]
    hits = resp["hits"]["hits"]

    while hits:
        for h in hits:
            src = h["_source"]
            data.append({
                "id": h["_id"],
                "term": src["term"],
                "context": src["context"],
                "embedding": src["embedding"],
                "classification": src["classification"]
            })
        resp = es.scroll(scroll_id=scroll_id, scroll='2m')
        scroll_id = resp["_scroll_id"]
        hits = resp["hits"]["hits"]

    return data

def export_csv():
    """
    Fetch all docs from ES and export to OUTPUT_CSV.
    """
    print("Fetching all documents for export...")
    all_docs = fetch_all_docs()
    df = pd.DataFrame(all_docs)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Exported {len(df)} documents to '{OUTPUT_CSV}'.")

# -------------------- Load Wordlist & Cache ---------------------

def load_wordlist():
    """
    Load unique terms from wordlist.txt. If not found, return empty.
    """
    if not os.path.exists(WORDLIST_FILE):
        print(f"Warning: {WORDLIST_FILE} not found, proceeding without it.")
        return []
    with open(WORDLIST_FILE, "r", encoding="utf-8") as f:
        terms = [line.strip() for line in f if line.strip()]
    terms = list(set(terms))
    print(f"Loaded {len(terms)} unique terms from '{WORDLIST_FILE}'.")
    return terms

def load_sanitized_cache():
    """
    Load the sanitized cache file (gpt4omini_cache_sanitized.json).
    """
    if not os.path.exists(SANITIZED_CACHE_FILE):
        print(f"Error: {SANITIZED_CACHE_FILE} not found. Run sanitize_cache.py first.")
        return {}
    with open(SANITIZED_CACHE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_gpt_cache(cache_dict):
    """
    (Optional) Save the updated GPT data if we do new classification.
    """
    with open(SANITIZED_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache_dict, f, indent=2, ensure_ascii=False)

# -------------------- GPT classification (optional) ------------

def classify_terms_batch_with_context(terms_batch):
    """
    Calls GPT 4o-mini to get context+classification for each term.
    If GPT fails, fallback to {context:"", classification:"Other"} for that term.
    """
    cats_str = ", ".join(CATEGORIES)
    prompt = (
        "For the given medical term, provide the following information using only key words:\n"
        "1) Brief definition\n2) Synonyms\n3) Typical use-case info (depending on category)\n"
        "Return JSON with these keys exactly: \"term\", \"context\", \"classification\".\n"
        f"Classification must be one of [{cats_str}].\n\n"
        "Format: {\"term\": \"...\", \"context\": \"...\", \"classification\": \"...\"}\n"
        "Here are the terms:\n"
    )
    for t in terms_batch:
        prompt += f"Term: {t}\n"

    try:
        response = client.chat.completions.create(
            model=CLASSIFICATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling GPT 4o-mini: {e}")
        # fallback
        return {t: {"context": "", "classification": "Other"} for t in terms_batch}

    # parse the lines of JSON
    results = {}
    lines = content.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # remove code fences if any
        line = line.replace("```json", "").replace("```", "").strip()
        try:
            j = json.loads(line)
            term_val = j.get("term", "").strip()
            ctx_val = j.get("context", "").strip()
            cls_val = j.get("classification", "").strip()
            if term_val:
                results[term_val] = {
                    "context": ctx_val,
                    "classification": cls_val
                }
        except Exception as parse_ex:
            print(f"Failed to parse line: {line} | Error: {parse_ex}")

    return results

def classify_missing_terms(cache_data, all_terms):
    """
    If RUN_GPT=True, call GPT for any terms not in cache_data. 
    Otherwise, do nothing.
    """
    if not RUN_GPT:
        print("RUN_GPT=False, skipping GPT calls.")
        return cache_data

    # find which terms are missing from the cache
    missing_terms = [t for t in all_terms if t not in cache_data]
    if not missing_terms:
        print("No missing terms in the cache. Skipping GPT calls.")
        return cache_data

    print(f"Classifying {len(missing_terms)} missing terms via GPT 4o-mini...")
    results = {}
    for i in tqdm(range(0, len(missing_terms), BATCH_SIZE)):
        batch = missing_terms[i : i + BATCH_SIZE]
        batch_res = classify_terms_batch_with_context(batch)
        results.update(batch_res)
        # merge into cache_data
        for k, v in batch_res.items():
            cache_data[k] = v
        save_gpt_cache(cache_data)
        time.sleep(1)  # prevent rate-limit issues

    print("Done classifying missing terms.")
    return cache_data

# -------------------- Embedding + Dedup on text+context --------

def embed_batch(texts_batch):
    """
    Get embeddings for a batch of texts from the OpenAI embedding API.
    Returns a list of float[] vectors in the same order.
    """
    response = client.embeddings.create(
        input=texts_batch,
        model=EMBED_MODEL
    )
    return [d.embedding for d in response.data]

def is_near_duplicate(embedding):
    """
    Query Elasticsearch for the top 3 similar docs using text+context embeddings,
    and see if any has cosine similarity >= DEDUP_THRESHOLD.
    """
    query_body = {
        "size": 3,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": embedding}
                }
            }
        }
    }
    resp = es.search(index=INDEX_NAME, body=query_body)
    for hit in resp["hits"]["hits"]:
        cos_sim = hit["_score"] - 1.0
        if cos_sim >= DEDUP_THRESHOLD:
            return True
    return False

def doc_exists(term):
    """
    Returns True if Elasticsearch already has a doc with _id == term,
    otherwise False.
    """
    try:
        doc = es.get(index=INDEX_NAME, id=term)
        return doc["found"]
    except Exception:
        return False


def embed_and_deduplicate_terms(final_data):
    """
    final_data is a list of dicts:
      [
        { "term": <str>, "context": <str>, "classification": <str> },
        ...
      ]

    We embed "term + context", deduplicate at DEDUP_THRESHOLD,
    and only insert new docs (skip any doc that already exists in ES).
    """
    print(f"Embedding {len(final_data)} items and deduplicating with threshold={DEDUP_THRESHOLD}...")
    processed_count = 0

    for i in tqdm(range(0, len(final_data), BATCH_SIZE)):
        batch = final_data[i : i + BATCH_SIZE]

        # Separate docs that are already in ES (we skip these)
        new_docs = []
        for d_item in batch:
            if not doc_exists(d_item["term"]):
                new_docs.append(d_item)

        # If entire batch is already in ES, just mark them processed
        if not new_docs:
            processed_count += len(batch)
            continue

        # Build the texts for embedding
        texts_batch = [(x["term"] + " " + x["context"]).strip() for x in new_docs]
        embeddings = embed_batch(texts_batch)  # your existing embed_batch()

        to_upload = []
        for (d_item, emb) in zip(new_docs, embeddings):
            # Check near-duplicate
            if not is_near_duplicate(emb):
                op = {
                    "_index": INDEX_NAME,
                    "_id": d_item["term"],  # or you could pick a unique ID
                    "_source": {
                        "term": d_item["term"],
                        "context": d_item["context"],
                        "classification": d_item["classification"],
                        "embedding": emb
                    }
                }
                to_upload.append(op)

        if to_upload:
            helpers.bulk(es, to_upload)

        processed_count += len(batch)

    print(f"Done. Processed {processed_count} items in total batches (skipped & deduped included).")

    
# -------------------- MAIN FLOW -------------------------

def main():
    # 1) Re-create the ES index from scratch (delete old)
    recreate_index()

    # 2) Load wordlist
    wordlist_terms = load_wordlist()

    # 3) Load the sanitized cache
    cache_data = load_sanitized_cache()

    # 4) Combine wordlist + cache keys, so we embed everything
    all_terms = set(wordlist_terms).union(set(cache_data.keys()))
    all_terms = list(all_terms)
    print(f"Total terms after merging wordlist & cache: {len(all_terms)}")

    # 5) If RUN_GPT=True, classify missing terms 
    #    (But if RUN_GPT=False, skip GPT calls)
    cache_data = classify_missing_terms(cache_data, all_terms)

    # 6) Build the final list of term-data from the cache
    #    If something is not in the cache, fallback classification = "Other"
    final_data = []
    for t in all_terms:
        info = cache_data.get(t, {"context":"", "classification":"Other"})
        final_data.append({
            "term": t,
            "context": info["context"],
            "classification": info["classification"]
        })

    # 7) Embed + Deduplicate on text+context
    embed_and_deduplicate_terms(final_data)

    # 8) Export to CSV
    export_csv()

if __name__ == "__main__":
    main()
