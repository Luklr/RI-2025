import pickle
import struct
import os
import sys
import heapq
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from Tokenizer import Tokenizer

def load_vocabulary(vocabulary_path: str) -> dict:
    """
    Load the vocabulary from a pkl file.
    :param vocabulary_path: Path to the vocabulary file.
    :return: A dictionary representing the vocabulary.
    """
    with open(vocabulary_path, "rb") as f:
        vocabulary = pickle.load(f)
    return vocabulary

def load_id_to_docs(id_to_docs_path: dict) -> str:
    """
    Load the id_to_docs from a pkl file.
    :param id_to_docs_path: Path to the id_to_docs file.
    :return: A dictionary representing the id_to_docs.
    """
    with open(id_to_docs_path, "rb") as f:
        id_to_docs = pickle.load(f)
    return id_to_docs

def document_at_a_time(query, k, vocabulary: dict, id_to_docs: dict) -> list[tuple]:
    """
    Evaluate a query and return the top-k documents.
    :param query: The query as a dictionary of term frequencies.
    :param k: The number of top documents to return.
    :param vocabulary: The vocabulary dictionary.
    :param id_to_docs: The mapping from doc_id to document info.
    :return: A list of tuples (doc_id, score) for the top-k documents.
    """
    query_norm = 0.0
    heap = []
    list_of_posting_list = []

    for term, freq in query.items():
        query_norm += freq ** 2
    query_norm = query_norm ** 0.5

    for term in query:
        posting_list = get_posting_list(term, vocabulary)  # [(doc_id, freq), ...]
        list_of_posting_list.append((term, posting_list)) # [(term, [(doc_id, freq), ...]), ...]

    candidates = []
    for term, posting_list in list_of_posting_list: 
        for doc_id, freq in posting_list:
            if doc_id not in candidates:
                candidates.append(doc_id) # {doc_id: {term: freq, ...}, ...}
    candidates.sort()

    for doc_id in candidates: # d in U
        numerador = 0
        for term_posting_list in list_of_posting_list: # li in L
            for docid, freq in term_posting_list[1]:
                if doc_id == docid:
                    numerador += query[term_posting_list[0]] * freq

        doc_norm = id_to_docs[doc_id][1]
        if doc_norm == 0 or query_norm == 0:
            continue

        score = numerador / (query_norm * doc_norm)

        # Guardar en heap como Top-K
        if len(heap) < k:
            heapq.heappush(heap, (score, doc_id))
        else:
            heapq.heappushpop(heap, (score, doc_id))

    # Ordenar resultados de mayor a menor score
    result = sorted(heap, key=lambda x: -x[0])
    return result


def get_posting_list(term: str, vocabulary:dict) -> list[tuple]:
    """
    Get the posting list for a term from the vocabulary.
    :param term: The term to search for.
    :param vocabulary: The vocabulary dictionary.
    :return: A list that contain document IDs.
    """
    if not term in vocabulary:
        return []

    pointer = vocabulary[term][0]
    postings_count = vocabulary[term][1]
    with open("index.bin", "rb") as f:
        f.seek(pointer)
        posting_list = []
        offset = pointer + postings_count * 8
        while f.tell() < offset:
            doc_id, freq = struct.unpack('II', f.read(8))
            posting_list.append((doc_id, freq))
    
    return posting_list


def parse_query(query: str, vocabulary: dict) -> dict:
    """
    Parse query into a vectorial expression.
    :param query: The query to parse.
    :return: The parsed expression.
    """
    term_freq = {}
    tokenizer = Tokenizer(names=True, dates=True, urls=True, emails=True, 
                         words=True, numbers=True, abbreviations=True)
    terms: list = tokenizer.tokenize(query, html_tags=True)
    for term in terms:
        if term not in vocabulary:
            continue
        if term in term_freq:
            term_freq[term] += 1
        else:
            term_freq[term] = 1
    return term_freq


def main():
    args = os.sys.argv
    if len(args) != 3:
        print("Usage: python punto4.py <query> <top-k>")
        return

    query = args[1]
    if args[2].isalpha():
        print("El valor de k debe ser un número entero.")
        return
    k = int(args[2])
    if k <= 0:
        print("El valor de k debe ser mayor que 0.")
        return

    # Load the vocabulary
    vocabulary_path = "vocabulary.pkl"
    vocabulary = load_vocabulary(vocabulary_path)

    # Load the id_to_docs
    id_to_docs_path = "id_to_docs.pkl"
    id_to_docs = load_id_to_docs(id_to_docs_path)

    parsed_query = parse_query(query, vocabulary)

    # Parse the expression and evaluate it
    result = document_at_a_time(parsed_query, k, vocabulary, id_to_docs)
    
    # Print the result
    if not result:
        print("No se encontraron documentos que coincidan con la consulta.")
        return
    
    for score, doc_id in result:
        print(f"{id_to_docs[doc_id][0]}:{doc_id}:{score:.4f}")

if __name__ == "__main__":
    main()