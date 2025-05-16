import bisect
import struct
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from Tokenizer import Tokenizer
import pickle
import logging
from time import perf_counter
import heapq

def get_vocabulary(vocabulary_path: str) -> dict:
    """
    Load the vocabulary from a .txt file.
    :param vocabulary_path: Path to the vocabulary file.
    :return: A dictionary representing the vocabulary.
    """
    vocabulary = {}
    with open(vocabulary_path, "r") as file:
        for line in file:
            term, df, posting_list = line.strip().split(":")
            posting_list = posting_list.split(",")
            posting_list.remove("")
            posting_list = [int(doc_id) for doc_id in posting_list]
            posting_list = sorted(posting_list, key=lambda x: x)
            vocabulary[term] = [int(df), posting_list]

    return vocabulary

def daat(vocabulary: dict, query: list[str], top_k: int = 10):
    """
    Perform a DAAT (Document-at-a-Time) query on the vocabulary.
    :param vocabulary: The vocabulary dictionary.
    :param query: The query list of terms.
    :param top_k: The number of top documents to return.
    :return: A list of tuples containing the document ID.
    """
    list_of_posting_lists = []
    for term in query:
        list_of_posting_lists.append((term, vocabulary[term][1]))
    
    heap = []
    candidates = []
    for term, posting_list in list_of_posting_lists: 
        for doc_id in posting_list:
            if doc_id not in candidates:
                candidates.append(doc_id) 
    candidates.sort()

    for doc_id in candidates:
        score = 0
        for term, posting_list in list_of_posting_lists:
            for docid in posting_list:
                if doc_id == docid:
                    score += 1
                    break
        if len(heap) < top_k:
            heapq.heappush(heap, (score, doc_id))
        else:
            heapq.heappushpop(heap, (score, doc_id))

    result = sorted(heap, key=lambda x: -x[0])
    return result

def taat(vocabulary: dict, query: list[str], top_k: int = 10):
    """
    Perform a TAAT (Term-at-a-Time) query on the vocabulary.
    :param vocabulary: The vocabulary dictionary.
    :param query: The query list of terms.
    :param top-k: The number of top documents to return.
    :return: A list of tuples containing the document ID and score.
    """
    list_of_posting_lists = []
    for term in query:
        list_of_posting_lists.append((term, vocabulary[term][1]))
    
    acumulated_score = []

    for term, posting_list in list_of_posting_lists:
        for doc_id in posting_list:
            doc_finded = False
            for i in range(len(acumulated_score)):
                if acumulated_score[i][1] == doc_id:
                    acumulated_score[i][0] += 1
                    doc_finded = True
                    break
            if not doc_finded:
                acumulated_score.append([1, doc_id])
    sorted_acumulated_score = sorted(acumulated_score, key=lambda x: x[0], reverse=True)
    return sorted_acumulated_score[:top_k]

def main():
    args = sys.argv
    if len(args) != 3:
        print("Usage: python punto6.py <query> <top_k>")
        return
    
    query = args[1]
    if args[2].isalpha():
        print("El valor de k debe ser un número entero.")
        return
    k = int(args[2])
    if k <= 0:
        print("El valor de k debe ser mayor que 0.")
        return
    
    query = query.strip().split()

    # Load the vocabulary
    vocabulary_path = "../data/dump10k/dump10k.txt"
    vocabulary = get_vocabulary(vocabulary_path)

    query = [term for term in query if term in vocabulary]
    print(query)

    # Perform DAAT and TAAT queries
    start_daat_time = perf_counter()
    daat_result = daat(vocabulary, query, k)
    end_daat_time = perf_counter()
    start_taat_time = perf_counter()
    taat_result = taat(vocabulary, query, k)
    end_taat_time = perf_counter()

    print(f"Tiempo de búsqueda con DAAT: {end_daat_time - start_daat_time:.4f}s")
    print("DAAT Result:")
    for score, doc_id in daat_result:
        print(f"{doc_id}:{score}")

    print(f"Tiempo de búsqueda con TAAT: {end_taat_time - start_taat_time:.4f}s")
    print("\nTAAT Result:")
    for score, doc_id in taat_result:
        print(f"{doc_id}:{score}")


if __name__ == "__main__":
    main()