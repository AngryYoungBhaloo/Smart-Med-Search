#!/usr/bin/env python3
import os
import json
from flask import Flask, render_template_string, request
from elasticsearch import Elasticsearch
from openai import OpenAI
from dotenv import load_dotenv

# ----------------- CONFIGURATION -----------------

ES_HOST = "http://localhost:9200"       # Elasticsearch host
ES_USER = "elastic"                     # Basic auth user
ES_PASS = "changeme"                    # Basic auth password
INDEX_NAME = "medical-terms-index"

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)
EMBED_MODEL = "text-embedding-3-large"   # Using the same embedding model

# Granular categories (same as used in medical_pipeline.py)
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

# Connect to Elasticsearch
es = Elasticsearch(ES_HOST, basic_auth=(ES_USER, ES_PASS))

# ----------------- FLASK APP SETUP -----------------

app = Flask(__name__)

HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
    <title>Medical Terms Search</title>
</head>
<body>
    <h1>Medical Terms Search</h1>
    <form method="post">
        <label for="query">Search Query:</label><br>
        <input type="text" id="query" name="query" size="80" required><br><br>
        <label for="categories">Select Categories:</label><br>
        {% for cat in categories %}
            <input type="checkbox" name="categories" value="{{ cat }}"> {{ cat }}<br>
        {% endfor %}
        <br>
        <input type="submit" value="Search">
    </form>
    <hr>
    {% if results %}
        <h2>Search Results:</h2>
        <ul>
        {% for res in results %}
            <li>
                <strong>Term:</strong> {{ res.term }}<br>
                <strong>Classification:</strong> {{ res.classification }}<br>
                <strong>Context:</strong> {{ res.context }}<br>
                <strong>Score:</strong> {{ res.score }}
            </li>
        {% endfor %}
        </ul>
    {% endif %}
</body>
</html>
"""

def search_medical_terms(query_text, selected_categories, top_n=10):
    """
    Generates a query embedding using the specified embedding model,
    then searches Elasticsearch for documents with high cosine similarity.
    Optionally filters results by the provided classification categories.
    """
    # Generate query embedding (expects a list input)
    response = client.embeddings.create(
        input=[query_text],
        model=EMBED_MODEL
    )
    query_embedding = response.data[0].embedding

    # Build the Elasticsearch query: use script_score for cosine similarity,
    # and apply a filter if one or more categories are selected.
    es_query = {
        "size": top_n,
        "query": {
            "bool": {
                "must": [
                    {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                "params": {"query_vector": query_embedding}
                            }
                        }
                    }
                ],
                "filter": []
            }
        }
    }
    if selected_categories:
        es_query["query"]["bool"]["filter"].append({
            "terms": {"classification": selected_categories}
        })

    resp = es.search(index=INDEX_NAME, body=es_query)
    results = []
    for hit in resp["hits"]["hits"]:
        source = hit["_source"]
        results.append({
            "term": source["term"],
            "classification": source["classification"],
            "context": source["context"],
            "score": hit["_score"]
        })
    return results

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    if request.method == "POST":
        query_text = request.form.get("query")
        selected_categories = request.form.getlist("categories")
        results = search_medical_terms(query_text, selected_categories)
    return render_template_string(HTML_TEMPLATE, results=results, categories=CATEGORIES)

if __name__ == "__main__":
    app.run(debug=True)
