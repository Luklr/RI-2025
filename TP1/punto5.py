import pandas as pd
import numpy as np
import os
import re
import sys
import unicodedata
import json
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
import time

class Tokenizer:
    PATTERNS = {
        "abbreviations": r'\b(?:Dr|Lic|Ing|Sr|Sra|S\.A|etc)\.\b',
        "acronyms": r'\b[A-Z](?:\.?[A-Z]+){1,7}\b',
        "numbers": r'\b\d{1,4}(?:[.-]?\d{1,4}){0,3}\b',
        "urls": r'\b(?:ftp|https|http)?://[^\s/$.?#].[^\s]*\b',
        "emails": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b',
        "names": r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b(?!\.)',
        "dates": r'\b\d{2}[-/.]\d{2}[-/.]\d{4}\b',
        "words": r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+(?:-[a-zA-Z]+)*'
    }

    def __init__(self, **kwargs):
        patterns = [regex for key, regex in self.PATTERNS.items() if kwargs.get(key, False)]
        self.regex = re.compile('|'.join(patterns) if patterns else '|'.join(self.PATTERNS.values()))
        self.porter = PorterStemmer()
        self.lancaster = LancasterStemmer()
        self.snowball = SnowballStemmer("spanish")

    def remove_accents(self, text_list):
        replacement = str.maketrans("áéíóúüÁÉÍÓÚÜ", "aeiouuAEIOUU")
        text_without_accents = []
        for word in text_list:
            word = word.translate(replacement)
            text_without_accents.append(word)
        return text_without_accents
    
    def stemming(self, text_list, method = "porter"):
        if method == "porter":
            stemmed_list = [self.porter.stem(word) for word in text_list]
        elif method == "lancaster":
            stemmed_list = [self.lancaster.stem(word) for word in text_list]
        elif method == "snowball":
            stemmed_list = [self.snowball.stem(word) for word in text_list]
        else:
            raise ValueError("Invalid method. Use 'porter', 'lancaster' or 'snowball'.")
        return stemmed_list

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

    def tokenize(self, text, stopwords: bool = False, stopwords_path: str = None, stemming: bool = False, stemming_method: str = "porter"):
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
        if stemming:
            processed_tokens = self.stemming(processed_tokens, stemming_method)
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



def parse_trec_file(url: str, stemming_method: str = "porter")->list:
    data = []
    tokenizer = Tokenizer(abreviations=True, acronyms=True, numbers=True, urls=True, emails=True, names=True, dates=True, words=True)
    with open(url, "r", encoding="utf-8") as file:
        doc_text = ""
        doc_no = 0
        for line in file:
            line = line.lower()
            line = line.strip()

            if re.match(r"<doc>", line):
                doc_text = ""
                continue

            if re.match(r"<docno>([0-9]+)</docno>", line):
                doc_no = re.search(r"<docno>([0-9]+)</docno>", line).group(1)
                continue

            doc_text += line

            if re.match(r"</doc>", line):
                # proceso el documento encontrado
                tokens = tokenizer.tokenize(doc_text, False, None, True, stemming_method)

                document_terms = terms(tokens)

                for term in document_terms:
                    if not term_in_data(data,term["term"]):
                        data.append({
                            "term": term["term"],
                            "df": 1,
                            "tf": term["tf"],
                            "docs": [doc_no]
                        })
                    else:
                        set_term(data, term["term"], term["tf"], doc_no)

    return data

def accounting(data_porter: list, data_lancaster: list, time_porter: float, time_lancaster: float):
    sorted_terms_porter = sorted(data_porter, key=lambda x: x["term"])
    sorted_terms_lancaster = sorted(data_lancaster, key=lambda x: x["term"])
    print("Stadistics: ")
    print(f"Total terms porter: {len(sorted_terms_porter)}")
    print(f"Total terms lancaster: {len(sorted_terms_lancaster)}")
    total_tokens_porter = 0
    for term in sorted_terms_porter:
        total_tokens_porter += term["tf"]
    total_tokens_lancaster = 0
    for term in sorted_terms_lancaster:
        total_tokens_lancaster += term["tf"]
    print(f"Total tokens porter: {total_tokens_porter}")
    print(f"Total tokens lancaster: {total_tokens_lancaster}")
    print(f"Time porter: {time_porter}")
    print(f"Time lancaster: {time_lancaster}")

def main():
    data_porter = []
    data_lancaster = []
    url = "vaswani/corpus/doc-text.trec"
    stemming_method = "porter"
    inicio = time.perf_counter()
    data_porter = parse_trec_file(url, stemming_method)
    fin = time.perf_counter()
    time_porter = fin - inicio
    
    stemming_method = "lancaster"
    inicio = time.perf_counter()
    data_lancaster = parse_trec_file(url, stemming_method)
    fin = time.perf_counter()
    time_lancaster = fin - inicio

    accounting(data_porter, data_lancaster, time_porter, time_lancaster)

if __name__ == "__main__":
    main()