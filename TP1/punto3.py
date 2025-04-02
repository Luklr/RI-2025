import pandas as pd
import numpy as np
import os
import re
import sys
import unicodedata
import json

class Tokenizer:
    def __init__(self):
        self.patterns = [
            r'\b(?:Dr|Lic|Ing|Sr|Sra|S\.A|etc)\.\b',  # Abreviaturas comunes
            r'\b[A-Z](?:\.?[A-Z]+){1,7}\b',  # Siglas (NASA, ONU, etc.)
            r'\b(?:\d{1,4}(?:[.-]?\d{1,4}){0,3})\b',  # Números, incluyendo teléfonos
            r'\b(?:ftp|https|http)?://[^\s/$.?#].[^\s]*\b',  # URLs
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b',  # Correos electrónicos
            r'\b(?:[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b(?!\.)',  # Nombres propios
            r'\b[0-9]{2}[-/.][0-9]{2}[-/.][0-9]{4}\b',  # Fechas (12/12/2023, etc.)
            r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+(?:-[a-zA-Z]+)*',  # Palabras normales
        ]
        
        self.regex = re.compile('|'.join(self.patterns))

    def remove_accents(self, text_list):
        replacement = str.maketrans("áéíóúüÁÉÍÓÚÜ", "aeiouuAEIOUU")
        text_without_accents = []
        for word in text_list:
            word = word.translate(replacement)
            text_without_accents.append(word)
        return text_without_accents

    def remove_stopwords(self, text_list, stopwords_path):
        with open(stopwords_path, 'r', encoding="utf-8") as file:
            stopwords = file.read()
        stopwords = stopwords.lower()
        stopwords = re.split(r'[, \n]+', stopwords)
        stopwords = self.remove_accents(stopwords)
        text_without_stopwords = []
        
        for word in text_list:
            if word not in stopwords:
                text_without_stopwords.append(word)
        return text_without_stopwords

    def tokenize(self, text, stopwords: bool = False, stopwords_path: str = None):
        text_list = self.regex.findall(text)
        text_list = self.remove_accents(text_list)
        processed_tokens = []
        for token in text_list:
            # Aplicar minúscula solo a palabras normales
            if re.fullmatch(r'[a-zA-Z]+(?:-[a-zA-Z]+)*', token):
                token = token.lower()
            processed_tokens.append(token)
        if stopwords:
            processed_tokens = self.remove_stopwords(processed_tokens, stopwords_path)
        return processed_tokens

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

def file_reader1()->list:
    data = []
    tokenizer = Tokenizer()
    for i in range(1000):
        url = f"RE_collection_test/doc{i}.txt"
        with open(url, "r", encoding="utf-8") as file:
            text = file.read()
            tokens = tokenizer.tokenize(text)
        document_terms = terms(tokens)
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
    return data

def file_reader2(path: str, stopwords: bool, stopwords_path: str = None) -> list:
    files = os.listdir(path)
    data = []
    doc_data = []
    tokenizer = Tokenizer()
    for i, file in enumerate(files):
        with open(f"{path}/{file}", 'r', encoding="utf-8") as file:
            text = file.read()
            text_tokens = tokenizer.tokenize(text, stopwords, stopwords_path)
            terms_data = terms(text_tokens)
        for term in terms_data:
            doc_data.append({
                "doc": i,
                "term": term["term"],
                "tf": term["tf"]
            })
            if not term_in_data(data,term["term"]):
                data.append({
                    "term": term["term"],
                    "df": 1,
                    "tf": term["tf"],
                    "docs": [i]
                })
            else:
                set_term(data, term["term"], term["tf"], i)
    return data, doc_data


def docs(doc_data: list) -> list:
    doc_data = sorted(doc_data, key=lambda x: x["doc"])
    docs = []
    last_doc = -1
    term_count = 0
    tf_total = 0
    for doc in doc_data:
        if doc["doc"] != last_doc:
            term_count = 0
            tf_total = 0
            docs.append({
                "doc": doc["doc"],
                "terms": term_count,
                "tf": tf_total
            })
            last_doc = doc["doc"]
        else:
            term_count += 1
            tf_total += doc["tf"]
            docs[-1]["terms"] = term_count
            docs[-1]["tf"] = tf_total
    
    return docs

def accounting1(data: list):
    sorted_terms = sorted(data, key=lambda x: x["term"])
    with open('RE_collection_test/collection_data.json', 'r', encoding="utf-8") as file:
        collection_data = json.load(file)
    print(f"term {" "*96} df")
    for term in sorted_terms:
        print(f"{term['term']:<102}{term['df']}")

    print("Stadistics: ")
    print(f"Total terms: {len(sorted_terms)}")
    print(f"Total terms (debug): {collection_data["statistics"]["num_terms"]}")
    total_tokens = 0
    for term in sorted_terms:
        total_tokens += term["tf"]
    print(f"Total tokens: {total_tokens}")
    print(f"Total tokens (debug): {collection_data["statistics"]["num_tokens"]}")

def accounting2(path: str, data: list, doc_data: list):
    # terminos.txt
    sorted_terms = sorted(data, key=lambda x: x["term"])
    with open('punto3/terminos.txt', 'w', encoding="utf-8") as file:
        for term in sorted_terms:
            file.write(f"{term['term']} {term["tf"]} {term['df']}\n")
    
    # estadisticas.txt
    files = os.listdir(path)
    total_tokens = 0
    for term in data:
        total_tokens += term["tf"]
    
    term_large = 0
    for term in data:
        term_large += len(term["term"])
    avg_term_large = term_large/len(data)

    
    documents = docs(doc_data)
    min_doc_length = min(documents, key=lambda x: x["tf"])
    max_doc_length = max(documents, key=lambda x: x["tf"])

    one_time_terms = [term for term in data if term["df"] == 1]
    
    with open('punto3/estadisticas.txt', 'w', encoding="utf-8") as file:
        file.write(f"{len(files)}\n")
        file.write(f"{total_tokens} {len(sorted_terms)}\n")
        file.write(f"{total_tokens/len(files)} {len(sorted_terms)/len(files)}\n")
        file.write(f"{avg_term_large}\n")
        file.write(f"{min_doc_length['tf']} {min_doc_length['terms']} {max_doc_length['tf']} {max_doc_length['terms']}\n")
        file.write(f"{len(one_time_terms)}\n")

    # frecuencias.txt
    major_to_minor_terms = sorted(doc_data, key=lambda x: x["tf"], reverse=True)
    minor_to_major_terms = sorted(doc_data, key=lambda x: x["tf"], reverse=False)
    with open('punto3/frecuencias.txt', 'w', encoding="utf-8") as file:
        for i, term in enumerate(major_to_minor_terms):
            if i < 10:
                file.write(f"{term['term']} {term['tf']}\n")
            else:
                break
        for i, term in enumerate(minor_to_major_terms):
            if i < 10:
                file.write(f"{term['term']} {term['tf']}\n")
            else:
                break

def main():
    # RE_collection_test
    data = file_reader1()
    accounting1(data)

    path="RI-tknz-data"
    # RI-tknz-data
    data, doc_data = file_reader2(path, True, "stopwords.txt")
    accounting2(path, data, doc_data)
    # tokenizer = Tokenizer()
    # text = "Hola, ¿cómo estás? de que tal?"
    # print(tokenizer.tokenize(text, stopwords=True, stopwords_path="stopwords.txt"))


if __name__ == "__main__":
    main()