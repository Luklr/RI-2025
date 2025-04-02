import pandas as pd
import numpy as np
import os
import re
import json

def terms(data: list) -> dict:
    data.sort()
    unique = []
    count = 0
    last_word = ""
    for word in data:
        if word != last_word:
            if last_word != "":
                unique.append({"term": last_word, "tf": count})
            count = 1
            last_word = word
        else:
            count += 1

    if last_word != "":
        unique.append({"term": last_word, "tf": count})
    return unique

def tokenize(text: str) -> list:
    text = text.lower()
    text = re.sub(r'[^a-z]', " ", text)
    return text.split()

def accounting(data: list):
    sorted_terms = sorted(data, key=lambda x: x["term"])
    with open('collection_test/collection_data.json', 'r') as file:
        collection_data = json.load(file)
    print("term                    df")
    for term in sorted_terms:
        print(f"{term['term']:<18}      {term['df']}")

    print("Stadistics: ")
    print(f"Total terms: {len(sorted_terms)}")
    print(f"Total terms (debug): {collection_data["statistics"]["num_terms"]}")
    total_tokens = 0
    for term in sorted_terms:
        total_tokens += term["tf"]
    print(f"Total tokens: {total_tokens}")
    print(f"Total tokens (debug): {collection_data["statistics"]["num_tokens"]}")

def term_in_data(data: list, term: str) -> bool:
    for element in data:
        if element["term"] == term:
            return True
    return False

def set_term(data: list, term: str, tf: int, doc: int) -> list:
    for element in data:
        if element["term"] == term:
            element["df"] += 1
            element["tf"] += tf
            element["docs"].append(doc)
            return data

def main():
    data = []
    for i in range (10000):
        url = f"TestCollection/doc{i}.txt"
        with open(url, 'r') as file:
            text = file.read()
            tokens_list = tokenize(text)
        document_terms = terms(tokens_list)
        for term in document_terms:
            if not term_in_data(data,term["term"]):
                data.append({
                    "term": term["term"],
                    "df": 1,
                    "tf": term["tf"],
                    "docs": [i]
                })
            else:
                set_term(data, term["term"], term["tf"], i)

    accounting(data)


if __name__ == "__main__":
    main()
