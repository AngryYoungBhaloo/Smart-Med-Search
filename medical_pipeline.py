#medical_pipeline.py
#!/usr/bin/env python3
"""
medical_pipeline.py

This script does the following:
1. Connects to Elasticsearch (which must be running on localhost:9200 with basic_auth "elastic"/"changeme").
2. Creates (or uses) an index named "medical-terms-index" to store terms, embeddings, and classification.
3. Reads terms from 'wordlist.txt' (one term per line).
4. Fetches embeddings for each new (not-yet-indexed) term from OpenAI model "text-embedding-3-large".
5. Checks if a new term is a near-duplicate (cosine similarity >= 0.98). If so, it is skipped.
6. Indexes all unique terms with their embeddings.
7. Classifies all unclassified terms in small batches using OpenAI chat model "gpt-4o-mini".
   The categories are more granular, e.g. "Disease or Syndrome", "Symptom or Sign", etc.
8. Exports the final set of documents (term, embedding, classification) to a CSV named "final_medical_terms.csv".

The script is designed to pick up where it left off:
- If the index already has some terms, it won't embed them again.
- If a term is already classified, it won't re-classify it.
Hence you can safely re-run the script if it ever fails mid-way.
"""

import os
import time
import json
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from elasticsearch import Elasticsearch, helpers

###############################################################################
# PART 1: CONFIG & CONSTANTS
###############################################################################

# ========== DOCKER/ELASTICSEARCH SETTINGS ==========
ES_HOST = "http://localhost:9200"     # Where Elasticsearch is listening
ES_USER = "elastic"                   # Basic auth user
ES_PASS = "changeme"                  # Basic auth password
INDEX_NAME = "medical-terms-index"

# ========== OPENAI SETTINGS ==========
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=OPENAI_API_KEY)
EMBED_MODEL = "text-embedding-3-large"   # 3072 dims
CLASSIFICATION_MODEL = "gpt-4o-mini"     # do not rename

# ========== EMBEDDINGS/DEDUP OPTIONS ==========
EMBED_DIM = 3072
BATCH_SIZE = 50
DEDUP_THRESHOLD = 0.98  # Cosine similarity threshold to consider near-duplicate

# ========== FILE INPUT/OUTPUT ==========
WORDLIST_FILE = "wordlist.txt"
OUTPUT_CSV = "final_medical_terms.csv"

# ========== CATEGORIES FOR CLASSIFICATION (GRANULAR) ==========
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

###############################################################################
# PART 2: ELASTICSEARCH SETUP
###############################################################################

# Create Elasticsearch client
es = Elasticsearch(ES_HOST, basic_auth=(ES_USER, ES_PASS))

def create_index_if_not_exists():
    """Creates the medical-terms-index if it doesn't already exist,
    mapping for 'term', 'embedding' (dense_vector), and 'classification'."""
    if es.indices.exists(index=INDEX_NAME):
        print(f"Index '{INDEX_NAME}' already exists; will use it as-is.")
        return
    print(f"Creating new index '{INDEX_NAME}'...")
    index_body = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "term": {"type": "keyword"},
                "embedding": {"type": "dense_vector", "dims": EMBED_DIM},
                "classification": {"type": "keyword"}
            }
        }
    }
    es.indices.create(index=INDEX_NAME, body=index_body)
    print(f"Index '{INDEX_NAME}' created.")

###############################################################################
# PART 3: READ WORDLIST & PREP
###############################################################################

def load_wordlist():
    """Loads unique terms from wordlist.txt, stripping empty lines."""
    if not os.path.exists(WORDLIST_FILE):
        print(f"Error: Could not find {WORDLIST_FILE} in current directory.")
        return []
    with open(WORDLIST_FILE, "r", encoding="utf-8") as f:
        terms = [line.strip() for line in f if line.strip()]
    terms = list(set(terms))  # remove duplicates in the file
    print(f"Loaded {len(terms)} unique terms from '{WORDLIST_FILE}'.")
    return terms

def already_in_index(term):
    """Check if a term doc exists in the index, by ID or a direct match."""
    try:
        doc = es.get(index=INDEX_NAME, id=term)
        return doc["found"]
    except:
        return False

###############################################################################
# PART 4: EMBEDDING + DEDUP
###############################################################################

def embed_batch(terms_batch):
    """Get embeddings for a batch of terms from OpenAI Embeddings API."""
    response = client.embeddings.create(
        input=terms_batch,
        model=EMBED_MODEL
    )
    # The embeddings come back in the same order as input
    return [item.embedding for item in response.data]

def is_near_duplicate(embedding):
    """
    Query Elasticsearch for top 3 similar docs.
    If any doc has cosSim >= DEDUP_THRESHOLD, it's a near-duplicate.
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
    for h in resp["hits"]["hits"]:
        # _score = cosSim + 1
        cos_sim = h["_score"] - 1.0
        if cos_sim >= DEDUP_THRESHOLD:
            return True
    return False

def embed_and_index_terms(all_terms):
    """
    For all terms in wordlist, if not already in index, embed them in small batches,
    check near-duplicate, and index if unique. Classification is left blank initially.
    """
    to_process = [t for t in all_terms if not already_in_index(t)]
    if not to_process:
        print("No new terms to embed. Skipping embedding step.")
        return
    
    print(f"Embedding and indexing {len(to_process)} new terms with dedup check...")
    for i in tqdm(range(0, len(to_process), BATCH_SIZE)):
        batch_terms = to_process[i : i + BATCH_SIZE]
        embeddings = embed_batch(batch_terms)
        
        # We'll store each doc if it isn't a near-duplicate
        ops = []
        for term_str, emb in zip(batch_terms, embeddings):
            if not is_near_duplicate(emb):
                ops.append({
                    "_index": INDEX_NAME,
                    "_id": term_str,
                    "_source": {
                        "term": term_str,
                        "embedding": emb,
                        "classification": ""  # blank for now
                    }
                })
        if ops:
            helpers.bulk(es, ops)
    print("Done embedding and storing to Elasticsearch with dedup.")

###############################################################################
# PART 5: CLASSIFICATION
###############################################################################

def fetch_unclassified_docs(batch_size=50):
    """Returns a batch of docs with classification = '' (still unclassified)."""
    body = {
        "query": {
            "term": {"classification": ""}
        },
        "size": batch_size
    }
    return es.search(index=INDEX_NAME, body=body)

def classify_terms_batch(terms_batch):
    """Call OpenAI ChatCompletion to classify each term into exactly one of the categories."""
    # Build the prompt with granular categories
    cats_str = ", ".join(CATEGORIES)
    prompt = (f"Classify each term into EXACTLY one of the following categories:\n"
              f"[{cats_str}].\n\n")
    for t in terms_batch:
        prompt += f"Term: {t}\n"
    prompt += (
        "\nOutput JSON lines, each line => {\"term\": \"<term>\", \"category\": \"<one_of_the_list>\"}.\n"
    )

    try:
        response = client.chat.completions.create(
        model=CLASSIFICATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    except Exception as e:
        print(f"Error calling OpenAI classification: {str(e)}")
        # If there's an API error, fallback to all "Other" classification
        return {term: "Other" for term in terms_batch}

    content = response.choices[0].message.content.strip()
    # In case the response has code fences
    content = content.replace("```json", "").replace("```", "").strip()

    class_map = {}
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            j = json.loads(line)
            class_map[j["term"]] = j["category"]
        except:
            # If parse fails for a line, fallback
            pass

    # Default to "Other" if not parsed
    for t in terms_batch:
        if t not in class_map:
            class_map[t] = "Other"

    return class_map

def classify_unclassified():
    """
    Loops over unclassified docs in small batches.
    Classifies them and updates them in Elasticsearch.
    Respects progress so if re-run, it won't re-classify anything that already has classification.
    """
    print("Classifying terms in batches. This may take a while...")

    total_unclassified = es.count(index=INDEX_NAME, body={
        "query": {"term": {"classification": ""}}
    })["count"]
    if total_unclassified == 0:
        print("No unclassified terms found. Skipping classification.")
        return

    pbar = tqdm(total=total_unclassified, desc="Classifying", unit="term")
    while True:
        res = fetch_unclassified_docs()
        hits = res["hits"]["hits"]
        if not hits:
            break  # no more unclassified docs

        chunk_terms = [h["_source"]["term"] for h in hits]
        c_map = classify_terms_batch(chunk_terms)

        update_ops = []
        for h in hits:
            term_str = h["_source"]["term"]
            cat = c_map.get(term_str, "Other")
            update_ops.append({
                "_op_type": "update",
                "_index": INDEX_NAME,
                "_id": term_str,
                "doc": {"classification": cat}
            })
        if update_ops:
            helpers.bulk(es, update_ops)
            pbar.update(len(update_ops))

        # small pause to avoid hitting rate limits
        time.sleep(1)
    pbar.close()
    print("Classification complete.")

###############################################################################
# PART 6: EXPORT TO CSV
###############################################################################

def fetch_all_docs(chunk_size=1000):
    """Scroll through all docs in the index, returning a list of dicts."""
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
                "embedding": src["embedding"],
                "classification": src["classification"]
            })
        resp = es.scroll(scroll_id=scroll_id, scroll='2m')
        scroll_id = resp["_scroll_id"]
        hits = resp["hits"]["hits"]

    return data

def export_csv():
    """Fetch all docs from ES and export to 'final_medical_terms.csv'."""
    print("Fetching all documents for export...")
    all_docs = fetch_all_docs()
    df = pd.DataFrame(all_docs)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Exported {len(df)} documents to '{OUTPUT_CSV}'.")

###############################################################################
# MAIN
###############################################################################

def main():
    # 1. Create index if needed
    create_index_if_not_exists()

    # 2. Load wordlist
    all_terms = load_wordlist()
    if not all_terms:
        print("No terms found. Exiting.")
        return

    # 3. Embed + Index Terms (skip those already in index)
    embed_and_index_terms(all_terms)

    # 4. Classify any unclassified docs
    classify_unclassified()

    # 5. Export to CSV
    export_csv()

if __name__ == "__main__":
    main()
