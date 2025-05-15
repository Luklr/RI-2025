import struct
import os
import sys
import pickle
import time

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

def get_posting(term: str, vocabulary: dict) -> list[tuple]:
    """
    Get the posting list for a term from the vocabulary.
    :param term: The term to search for.
    :param vocabulary: The vocabulary dictionary.
    :return: A list of tuples containing the document ID and frequency.
    """
    if not term in vocabulary:
        return None

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

def main():
    args = os.sys.argv
    if len(args) != 2:
        print("Usage: python punto1_1.py <term>")
        return
    
    term = args[1]

    # Load the vocabulary
    vocabulary_path = "vocabulary.pkl"
    vocabulary = load_vocabulary(vocabulary_path)
    # print(vocabulary)

    # Load the id_to_docs
    id_to_docs_path = "id_to_docs.pkl"
    id_to_docs = load_vocabulary(id_to_docs_path)

    # Get the posting list for the term
    posting_list = get_posting(term, vocabulary)

    if posting_list is None or len(posting_list) == 0:
        print(f"The term '{term}' was not found in the vocabulary.")
        return

    # Print the posting list
    for doc_id, freq in posting_list:
        print(f"{id_to_docs[doc_id][0]}:{doc_id}:{freq}")

if __name__ == "__main__":
    main()