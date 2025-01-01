import openai
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from elasticsearch import Elasticsearch, helpers

##############################
# PART 1: CONFIG
##############################

# OpenAI API key
openai.api_key = "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" 


es = Elasticsearch("http://localhost:9200", basic_auth=("elastic", "changeme"))

INDEX_NAME = "medical-terms-index"
EMBED_MODEL = "text-embedding-3-large"  # 3072 dims
EMBED_DIM = 3072

# Terms with cosine similarity >= this are considered duplicates
DEDUP_THRESHOLD = 0.98

##############################
# PART 2: CREATE/RESET INDEX
##############################
if es.indices.exists(index=INDEX_NAME):
    es.indices.delete(index=INDEX_NAME)

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
print(f"Created (or reset) index '{INDEX_NAME}'")

##############################
# PART 3: READ WORDLIST
##############################
FILE_PATH = "wordlist.txt"  # Must be in the same directory
try:
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        all_terms = [line.strip() for line in f if line.strip()]
    print("Total terms:", len(all_terms))
except FileNotFoundError:
    print(f"Error: Cannot find '{FILE_PATH}'. Make sure it's in this directory.")
    exit(1)

##############################
# PART 4: EMBEDDING + DEDUP
##############################
def embed_batch(terms_batch):
    """Get embeddings for a batch of terms from OpenAI."""
    response = openai.Embedding.create(
        input=terms_batch,
        model=EMBED_MODEL
    )
    # The embeddings come back in the same order as input
    return [item["embedding"] for item in response["data"]]

def is_near_duplicate(embedding):
    """
    Query Elasticsearch for top 3 similar docs.
    If any doc has cosSim >= DEDUP_THRESHOLD, it’s a near-duplicate.
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

BATCH_SIZE = 50

print("Embedding and indexing with dedup check...")
for i in tqdm(range(0, len(all_terms), BATCH_SIZE)):
    batch_terms = all_terms[i : i + BATCH_SIZE]
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
                    "classification": ""
                }
            })
    if ops:
        helpers.bulk(es, ops)

print("Done embedding and storing to Elasticsearch with dedup.")

##############################
# PART 5: CLASSIFICATION
##############################
# If you really want to use "gpt-4o-mini," note that it's not an official OpenAI model.

CLASSIFICATION_MODEL = "gpt-4o-mini"

def classify_terms_batch(terms_batch):
    prompt = (
        "Classify each term into exactly one category: "
        "[Symptom, Disease, Lab Test, Organ, Organism, Specialty, Other].\n\n"
    )
    for t in terms_batch:
        prompt += f"Term: {t}\n"
    prompt += (
        "\nOutput JSON lines, each line => {\"term\": \"<term>\", \"category\": \"<category>\"}.\n"
    )

    response = openai.ChatCompletion.create(
        model=CLASSIFICATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    content = response.choices[0].message["content"].strip()

    class_map = {}
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            j = json.loads(line)
            class_map[j["term"]] = j["category"]
        except:
            pass
    return class_map

def fetch_unclassified_docs(batch_size=100):
    """
    Returns a batch of docs with classification = "".
    """
    body = {
        "query": {
            "term": {"classification": ""}
        },
        "size": batch_size
    }
    return es.search(index=INDEX_NAME, body=body)

print("Classifying terms in small batches...")
while True:
    res = fetch_unclassified_docs(100)
    hits = res["hits"]["hits"]
    if not hits:
        break

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

print("Classification complete.")

##############################
# PART 6: EXPORT CSV LOCALLY
##############################
def fetch_all_docs(index_name, chunk_size=1000):
    data = []
    resp = es.search(
        index=index_name,
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

all_docs = fetch_all_docs(INDEX_NAME)
df = pd.DataFrame(all_docs)
csv_file = "final_medical_terms.csv"
df.to_csv(csv_file, index=False)
print(f"Saved {len(df)} docs to '{csv_file}' in this directory.")
