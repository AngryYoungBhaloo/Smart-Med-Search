#!/usr/bin/env python3
"""
sanitize_cache.py

Reads your existing gpt4omini_cache.json, removes or fixes malformed data,
and writes gpt4omini_cache_sanitized.json. This helps avoid infinite classification loops,
and ensures you have valid JSON for each term in the cache.
"""

import json
import re
import sys

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

def sanitize_text(s: str) -> str:
    """
    Remove/escape characters that might break JSON parsing (unescaped quotes, newlines, etc.).
    """
    s = s.replace("\n", " ").replace("\r", " ")
    s = s.replace('"', "'")  # swap double quotes for single quotes
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def main():
    input_file = "gpt4omini_cache.json"
    output_file = "gpt4omini_cache_sanitized.json"

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
    except Exception as e:
        print(f"Failed to load {input_file}. Error: {e}")
        sys.exit(1)

    new_cache = {}
    for term, info in cache_data.items():
        # If context or classification missing, default them
        context = info.get("context", "")
        classification = info.get("classification", "Other")

        # Sanitize text
        context = sanitize_text(context)
        if classification not in CATEGORIES:
            classification = "Other"

        new_cache[term] = {
            "context": context,
            "classification": classification
        }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(new_cache, f, indent=2, ensure_ascii=False)

    print(f"Sanitized cache written to {output_file}")

if __name__ == "__main__":
    main()
