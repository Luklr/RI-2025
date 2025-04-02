import pandas as pd
import numpy as np
import os
import re
import sys
import unicodedata
import json
import time
from sklearn.metrics import confusion_matrix
from langdetect import detect

def remove_accents(text):
    replacement = str.maketrans("áéíóúüÁÉÍÓÚÜ", "aeiouuAEIOUU")
    if isinstance(text, list):  # Si es una lista, procesar cada palabra
        return [word.translate(replacement) for word in text]
    elif isinstance(text, str):  # Si es una cadena, procesarla directamente
        return text.translate(replacement)
    else:
        raise TypeError("El argumento debe ser una lista o una cadena.")

def training_first_method(url):
    data = {}
    documents = ["English", "French", "Italian"]
    for doc in documents:
        url_doc = url+doc
        data[doc] = {}
        with open(url_doc, 'r', encoding='ISO-8859-1') as file:
            for line in file:
                line = line.strip()
                line = line.lower()
                chars = remove_accents(line)
                chars = re.sub(r'\W', '', chars) # quito caracteres especiales
                chars = re.sub(r'\d', '', chars) # quito numeros
                for char in chars:
                    data[doc][char] = data[doc].get(char, 0) + 1
    
    return data

def training_second_method(url):
    data = {}
    documents = ["English", "French", "Italian"]
    for doc in documents:
        url_doc = url+doc
        data[doc] = {}
        with open(url_doc, 'r', encoding='ISO-8859-1') as file:
            for line in file:
                line = line.strip()
                line = line.lower()
                chars = remove_accents(line)
                chars = re.sub(r'\W', '', chars)
                chars = re.sub(r'\d', '', chars)
                for i in range(len(chars)-1):
                    char1, char2 = line[i], line[i + 1]
                    bigram = char1 + char2
                    data[doc][bigram] = data[doc].get(bigram, 0) + 1

    return data

def test_first_method(training_data, url_test):
    results = []
    documents = ["English", "French", "Italian"]
    with open(url_test, 'r', encoding='ISO-8859-1') as file:
        for line in file:
            line = line.strip()
            line = line.lower()
            chars = remove_accents(line)
            chars = re.sub(r'\W', '', chars)
            chars = re.sub(r'\d', '', chars)

            test_data = {}
            for char in chars:
                test_data[char] = test_data.get(char, 0) + 1
            
            best_lang = None
            best_score = -1

            for lang in documents:
                train_data = training_data.get(lang, {})
                all_chars = set(test_data.keys()).union(set(train_data.keys()))

                test_vector = np.array([test_data.get(c, 0) for c in all_chars])
                train_vector = np.array([train_data.get(c, 0) for c in all_chars])

                if len(test_vector) > 1:  
                    correlation = np.corrcoef(test_vector, train_vector)[0, 1]
                else:
                    correlation = 0
                
                if correlation > best_score:
                    best_score = correlation
                    best_lang = lang
            
            results.append((line, best_lang, best_score))
    return results

def test_second_method(training_data, url_test):
    results = []
    documents = ["English", "French", "Italian"]
    with open(url_test, 'r', encoding='ISO-8859-1') as file:
        for line in file:
            line = line.strip()
            line = line.lower()
            chars = remove_accents(line)
            chars = re.sub(r'\W', '', chars)
            chars = re.sub(r'\d', '', chars)

            test_data = {}
            for i in range(len(chars)-1):
                char1, char2 = line[i], line[i + 1]
                bigram = char1 + char2
                test_data[bigram] = test_data.get(bigram, 0) + 1
            
            best_lang = None
            best_score = -1

            for lang in documents:
                train_data = training_data.get(lang, {})
                all_chars = set(test_data.keys()).union(set(train_data.keys()))

                test_vector = np.array([test_data.get(c, 0) for c in all_chars])
                train_vector = np.array([train_data.get(c, 0) for c in all_chars])

                if len(test_vector) > 1:  
                    correlation = np.corrcoef(test_vector, train_vector)[0, 1]
                else:
                    correlation = 0
                
                if correlation > best_score:
                    best_score = correlation
                    best_lang = lang
            
            results.append((line, best_lang, best_score))
    return results

def test_langdetect_method(url_test):
    results = []
    with open(url_test, 'r', encoding='ISO-8859-1') as file:
        for line in file:
            line = line.strip()
            line = line.lower()
            chars = remove_accents(line)
            chars = re.sub(r'\W', '', chars)
            chars = re.sub(r'\d', '', chars)
            
            try:
                detected_lang = detect(line)
                if detected_lang == "en":
                    best_lang = "English"
                elif detected_lang == "fr":
                    best_lang = "French"
                elif detected_lang == "it":
                    best_lang = "Italian"
                else:
                    best_lang = "Unknown"
            except:
                best_lang = "Unknown"
            
            results.append((line, best_lang))
    return results

def comparing_with_solution(results: list, url_solution: str):
    documentos = ["English", "French", "Italian"]
    with open(url_solution, 'r', encoding='ISO-8859-1') as file:
        true_labels = [line.strip().split(" ", 1)[1] for line in file]

        predicted_labels = [pred[1] for pred in results]
        cm = confusion_matrix(true_labels, predicted_labels, labels=documentos)
        df_cm = pd.DataFrame(cm, index=documentos, columns=documentos)

    return df_cm

def main():
    url = "languageIdentificationData/training/"
    url_test = "languageIdentificationData/test"
    url_solution = "languageIdentificationData/solution"

    training_data_first_method = training_first_method(url)
    training_data_second_method = training_second_method(url)
    results_first_method = test_first_method(training_data_first_method, url_test)
    results_second_method = test_second_method(training_data_second_method, url_test)
    results_langdetect = test_langdetect_method(url_test)

    cm_first = comparing_with_solution(results_first_method, url_solution)
    cm_second = comparing_with_solution(results_second_method, url_solution)
    cm_langdetect = comparing_with_solution(results_langdetect, url_solution)
    print("First solution: ")
    print(cm_first)
    print("Second solution: ")
    print(cm_second)
    print("LangDetect solution: ")
    print(cm_langdetect)


if __name__ == "__main__":
    main()