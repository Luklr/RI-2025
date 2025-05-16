import pickle
import struct
import os
import sys
import heapq
from time import perf_counter
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

def get_skip_list(term: str, vocabulary: dict) -> list[tuple]:
    """
    Get the skip list for a term from the vocabulary.
    :param term: The term to search for.
    :param vocabulary: The vocabulary dictionary.
    :return: A list of tuples containing the document ID and frequency.
    """
    if not term in vocabulary:
        return None

    pointer = vocabulary[term][3][0]
    skip_count = vocabulary[term][3][1]
    with open("skip_list.bin", "rb") as f:
        f.seek(pointer)
        skip_list = []
        end = pointer + skip_count * 12  # Cada entrada son 12 bytes (III)
        while f.tell() < end:
            data = f.read(12)
            if len(data) < 12:
                break
            doc_id, skip_to_doc_id, offset = struct.unpack('III', data)
            skip_list.append((doc_id, skip_to_doc_id, offset))
    
    return skip_list

def main():
    args = os.sys.argv
    if len(args) != 2:
        print("Usage: python punto5_1.py <term>")
        return
    
    term = args[1]

    # Load the vocabulary
    vocabulary_path = "vocabulary_with_sl.pkl"
    vocabulary = load_vocabulary(vocabulary_path)

    # Load the id_to_docs
    id_to_docs_path = "id_to_docs.pkl"
    id_to_docs = load_id_to_docs(id_to_docs_path)

    skip_list = get_skip_list(term, vocabulary)
    posting_list = get_posting_list(term, vocabulary)

    posting_list_ordered = []
    for doc_id, freq in posting_list:
        posting_list_ordered.append((doc_id, id_to_docs[doc_id][0], freq))
    posting_list_ordered.sort(key=lambda x: x[1])

    for doc_id, name, freq in posting_list_ordered:
        print(f"{doc_id}:{name}:{freq}")


if __name__ == "__main__":
    main()