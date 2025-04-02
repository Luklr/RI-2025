import pandas as pd
import numpy as np
import os
import re
import sys
import unicodedata

def remove_accents(text):
        replacement = str.maketrans("áéíóúüÁÉÍÓÚÜ", "aeiouuAEIOUU")
        if isinstance(text, list):  # Si es una lista, procesar cada palabra
            return [word.translate(replacement) for word in text]
        elif isinstance(text, str):  # Si es una cadena, procesarla directamente
            return text.translate(replacement)
        else:
            raise TypeError("El argumento debe ser una lista o una cadena.")

def tokenize(text: str, stopwords: bool, stopwords_path: str = None) -> list:
    text = text.lower()
    
    text = remove_accents(text)
    text = re.sub(r'[^a-z]', " ", text)
    text = text.split()
    if stopwords:
        with open(stopwords_path, 'r') as file:
            stopwords = file.read()
        stopwords = stopwords.lower()
        stopwords = remove_accents(stopwords)
        stopwords = re.sub(r'[^a-z]', " ", stopwords)
        stopwords = stopwords.split()
        # print(stopwords)
        text = [word for word in text if word not in stopwords]
    return text

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

def read_files(path: str, stopwords: bool, stopwords_path: str = None) -> list:
    files = os.listdir(path)
    data = []
    doc_data = []
    for i, file in enumerate(files):
        with open(f"{path}/{file}", 'r', encoding="utf-8") as file:
            text = file.read()
            text_tokens = tokenize(text, stopwords, stopwords_path)
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

def accounting(path: str, data: list, doc_data: list) -> None:
    # terminos.txt
    sorted_terms = sorted(data, key=lambda x: x["term"])
    with open('terminos.txt', 'w') as file:
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
    
    with open('estadisticas.txt', 'w') as file:
        file.write(f"{len(files)}\n")
        file.write(f"{total_tokens} {len(sorted_terms)}\n")
        file.write(f"{total_tokens/len(files)} {len(sorted_terms)/len(files)}\n")
        file.write(f"{avg_term_large}\n")
        file.write(f"{min_doc_length['tf']} {min_doc_length['terms']} {max_doc_length['tf']} {max_doc_length['terms']}\n")
        file.write(f"{len(one_time_terms)}\n")

    # frecuencias.txt
    major_to_minor_terms = sorted(doc_data, key=lambda x: x["tf"], reverse=True)
    minor_to_major_terms = sorted(doc_data, key=lambda x: x["tf"], reverse=False)
    with open('frecuencias.txt', 'w') as file:
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
    if len(sys.argv) < 3 or (sys.argv[2] == "1" and len(sys.argv) != 4) or (sys.argv[2] == "0" and len(sys.argv) != 3):
        print("Uso:")
        print("python punto2.py [directorio/de/documentos] [palabras_vacias: 1|0] [directorio/de/palabras_vacias (opcional si 1)]")
        sys.exit(1)

    if not os.path.isdir(sys.argv[1]):
        print("El directorio de documentos no existe.")
        sys.exit(1)

    if sys.argv[2] not in ["1", "0"]:
        print("El segundo argumento debe ser 1 o 0.")
        sys.exit(1)
    if sys.argv[2] == "1" and not os.path.isfile(sys.argv[3]):
        print("El archivo de palabras vacías no existe.")
        sys.exit(1)
    
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    arg3 = sys.argv[3] if arg2 == "1" else None
    arg2 = True if arg2 == "1" else False

    data, doc_data = read_files(arg1, arg2, arg3)
    accounting(arg1, data, doc_data)

if __name__ == "__main__":
    main()
