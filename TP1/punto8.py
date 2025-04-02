import numpy as np
import matplotlib.pyplot as plt
import collections
import re
from punto5 import Tokenizer
import nltk
from nltk.corpus import stopwords

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

def set_term(data: list, term: str, tf: int) -> list:
    for element in data:
        if element["term"] == term:
            element["df"] += 1
            element["tf"] += tf
            return data

def read_file(url):
    data = []
    tokenizer = Tokenizer(words=True, names=True, abbreviations=True, numbers=True)
    with open(url, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            tokens = tokenizer.tokenize(line)
            terms_data = terms(tokens)
            for term in terms_data:
                if not term_in_data(data,term["term"]):
                    data.append({
                        "term": term["term"],
                        "df": 1,
                        "tf": term["tf"]
                    })
                else:
                    set_term(data, term["term"], term["tf"])
    return data

def stopwords_analysis(data):
    data.sort(key=lambda x: x["tf"], reverse=True)
    n_terms = len(data)

    # podo el 10, 20 y 30% de los datos mas frecuentes
    pruned_data_10pct = data[int(n_terms*0.1):]
    pruned_data_20pct = data[int(n_terms*0.2):]
    pruned_data_30pct = data[int(n_terms*0.3):]

    nltk.download('stopwords')
    stop_words = set(stopwords.words('spanish'))

    # verifico la cantidad de palabras podadas que son stopwords
    # y guardo aquellas que no lo son
    data_10pct = set([term["term"] for term in data]) - set([term["term"] for term in pruned_data_10pct])
    data_20pct = set([term["term"] for term in data]) - set([term["term"] for term in pruned_data_20pct])
    data_30pct = set([term["term"] for term in data]) - set([term["term"] for term in pruned_data_30pct])
    
    total_stopwords_10pct = 0
    non_stopwords_terms_10pct = []
    for term in data_10pct:
        if term in stop_words:
            total_stopwords_10pct+=1
        else:
            non_stopwords_terms_10pct.append(term)
    
    total_stopwords_20pct = 0
    non_stopwords_terms_20pct = []
    for term in data_20pct:
        if term in stop_words:
            total_stopwords_20pct+=1
        else:
            non_stopwords_terms_20pct.append(term)
    
    total_stopwords_30pct = 0
    non_stopwords_terms_30pct = []
    for term in data_30pct:
        if term in stop_words:
            total_stopwords_30pct+=1
        else:
            non_stopwords_terms_30pct.append(term)
    
    print(f"Poda de palabras mas frecuentes: ")
    print(f"Porcentaje de terminos podados que eran stopwords (10%): {(total_stopwords_10pct / len(data_10pct)):.2f}")
    print(f"Porcentaje de terminos podados que eran stopwords (20%): {(total_stopwords_20pct / len(data_20pct)):.2f}")
    print(f"Porcentaje de terminos podados que eran stopwords (30%): {(total_stopwords_30pct / len(data_30pct)):.2f}")

    with open("punto8_terminos_podados_10pct.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(non_stopwords_terms_10pct))
    with open("punto8_terminos_podados_20pct.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(non_stopwords_terms_20pct))
    with open("punto8_terminos_podados_30pct.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(non_stopwords_terms_30pct))


def zipf_analysis(data):
    data.sort(key=lambda x: x["tf"], reverse=True)
    ranks = np.arange(1, len(data) + 1)
    freqs = np.array([entry["tf"] for entry in data])
    
    # Ajuste con Zipf usando regresión lineal en escala logarítmica
    log_ranks = np.log(ranks)
    log_freqs = np.log(freqs)
    coeffs = np.polyfit(log_ranks, log_freqs, 1)  # Ajuste lineal en log-log
    
    alfa = coeffs[0]
    c = np.exp(coeffs[1])
    
    n_terms = len(data)

    n_tokens_10pct = 0
    n_tokens_20pct = 0
    n_tokens_30pct = 0
    for i in range(1, int(n_terms * 0.1) + 1):
        n_tokens_10pct += c * i ** alfa
    for i in range(1, int(n_terms * 0.2) + 1):
        n_tokens_20pct += c * i ** alfa
    for i in range(1, int(n_terms * 0.3) + 1):
        n_tokens_30pct += c * i ** alfa

    # frecuencias reales
    n_tokens_10pct_real = 0
    n_tokens_20pct_real = 0
    n_tokens_30pct_real = 0
    for i in range(1, int(n_terms * 0.1) + 1):
        n_tokens_10pct_real += freqs[i]
    for i in range(1, int(n_terms * 0.2) + 1):
        n_tokens_20pct_real += freqs[i]
    for i in range(1, int(n_terms * 0.3) + 1):
        n_tokens_30pct_real += freqs[i]

    print(f"Cantidad de palabras calculadas en el 10% del vocabulario: {n_tokens_10pct}")
    print(f"Cantidad de palabras calculadas en el 20% del vocabulario: {n_tokens_20pct}")
    print(f"Cantidad de palabras calculadas en el 30% del vocabulario: {n_tokens_30pct}")
    print(f"Cantidad de palabras reales en el 10% del vocabulario: {n_tokens_10pct_real}")
    print(f"Cantidad de palabras reales en el 20% del vocabulario: {n_tokens_20pct_real}")
    print(f"Cantidad de palabras reales en el 30% del vocabulario: {n_tokens_30pct_real}")

def main():
    url = "pg2000.txt"
    data = read_file(url)
    zipf_analysis(data)
    stopwords_analysis(data)

if __name__ == "__main__":
    main()
