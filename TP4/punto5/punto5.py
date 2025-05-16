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

def add_skip_lists(vocabulary: dict[tuple]) -> dict[tuple]:
    """
    Add a skip list to all posting lists.
    :param vocabulary: The original vocabulary.
    :return: The vocabulary modified with a pointer to skip_list.
    """

    open("skip_list.bin", "wb").close()
    vocabulary_with_sl = {}
    for term, term_data in vocabulary.items():
        posting_list_pointer = term_data[0]
        posting_list = get_posting_list(term, vocabulary)
        skip_list = []
        skip_size = len(posting_list) ** 0.5
        if skip_size < 1:
            skip_size = 1
        else:
            skip_size = int(skip_size)
        for i in range(0, len(posting_list) - skip_size, skip_size):
            doc_id = posting_list[i][0]
            skip_to_doc_id = posting_list[i + skip_size][0]
            offset_actual = posting_list_pointer + i * 8
            skip_list.append((doc_id, skip_to_doc_id, offset_actual))
        with open(f"skip_list.bin", "ab") as index_file:
            skip_list_pointer = index_file.tell()
            for doc_id, skip_to_doc_id, offset in skip_list:
                index_file.write(struct.pack('III', doc_id, skip_to_doc_id, offset))
        vocabulary_with_sl[term] = [term_data[0], term_data[1], term_data[2], [skip_list_pointer, len(skip_list)]]
    
    pickle.dump(vocabulary_with_sl, open("vocabulary_with_sl.pkl", "wb"))

    return vocabulary_with_sl

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

def search_docid_with_skip_list(term: str, doc_id: int, posting_list: list[tuple], skip_list: list[tuple]) -> bool:
    """
    Search for a document ID in the posting_list of a term using the skip list.
    Uses skip list offsets to efficiently locate doc_id in the postings.
    """
    start_index = 0

    for i in range(len(skip_list)):
        sl_doc_id, sl_next_doc_id, offset = skip_list[i]

        if sl_doc_id == doc_id:
            return True
        if sl_doc_id > doc_id:
            start_doc_id = skip_list[i - 1][0] if i > 0 else posting_list[0][0]
            start_index = next((j for j, (d, _) in enumerate(posting_list) if d == start_doc_id), 0)
            break
        if i == len(skip_list) - 1:
            start_doc_id = sl_doc_id
            start_index = next((j for j, (d, _) in enumerate(posting_list) if d == start_doc_id), 0)

    for i in range(start_index, len(posting_list)):
        current_doc_id = posting_list[i][0]
        if current_doc_id == doc_id:
            return True
        if current_doc_id > doc_id:
            return False

    return False
# def search_docid_with_skip_list(term: str, doc_id: int, posting_list: list[tuple], skip_list: list[tuple]) -> bool:
#     """
#     Búsqueda en memoria con skip list.
#     """
#     if not skip_list:
#         return any(doc == doc_id for doc, _ in posting_list)

#     i = 0
#     while i < len(posting_list):
#         current_doc = posting_list[i][0]
#         if current_doc == doc_id:
#             return True
#         elif current_doc > doc_id:
#             return False

#         # Si hay salto, usamos skip_list
#         skip_entry = next((s for s in skip_list if s[0] == current_doc), None)
#         if skip_entry and skip_entry[1] <= doc_id:
#             i = skip_entry[2] + int(len(posting_list) ** 0.5)  # ir al próximo salto
#         else:
#             i += 1

#     return False

def main():
    args = os.sys.argv
    if len(args) != 4:
        print("Usage: python punto5.py <term1> AND <term2>")
        return
    
    term1 = args[1]
    term2 = args[3]
    if args[2] != "AND":
        print("El operador debe ser 'AND'.")
        return

    # Load the vocabulary
    vocabulary_path = "vocabulary.pkl"
    vocabulary = load_vocabulary(vocabulary_path)

    vocabulary = add_skip_lists(vocabulary)

    # Load the id_to_docs
    id_to_docs_path = "id_to_docs.pkl"
    id_to_docs = load_id_to_docs(id_to_docs_path)

    start_time = perf_counter()
    # Check if the term exists in the vocabulary
    term1_postings = get_posting_list(term1, vocabulary)
    term2_postings = get_posting_list(term2, vocabulary)
    result = []
    for posting in term1_postings:
        if search_docid_with_skip_list(term2, posting[0], term2_postings, get_skip_list(term2, vocabulary)):
            result.append(posting[0])
    end_time = perf_counter()
    print(f"Tiempo de búsqueda con skip list: {end_time - start_time:.4f} segundos")
    
    for doc_id in result:
        print(f"{id_to_docs[doc_id][0]}:{doc_id}")

    print("\n\n======================\n\n")
    
    start_time = perf_counter()
    term1_postings = get_posting_list(term1, vocabulary)
    term2_postings = get_posting_list(term2, vocabulary)
    result = []
    term2_doc_ids = {doc_id for doc_id, _ in term2_postings}
    for posting in term1_postings:
        if posting[0] in term2_doc_ids:
            result.append(posting[0])
    
    end_time = perf_counter()
    print(f"Tiempo de búsqueda sin skip list: {end_time - start_time:.4f} segundos")
    for doc_id in result:
        print(f"{id_to_docs[doc_id][0]}:{doc_id}")

# def main():
#     args = os.sys.argv
#     if len(args) != 4:
#         print("Usage: python punto5.py <term1> AND <term2>")
#         return

#     term1, term2 = args[1], args[3]
#     if args[2] != "AND":
#         print("El operador debe ser 'AND'.")
#         return

#     # Cargar datos
#     vocabulary = load_vocabulary("vocabulary.pkl")
#     id_to_docs = load_id_to_docs("id_to_docs.pkl")

#     # Cargar todas las posting lists en memoria
#     all_postings = {term: get_posting_list(term, vocabulary) for term in [term1, term2]}

#     # Agregar skip lists en memoria
#     vocabulary, skip_lists = add_skip_lists(vocabulary, all_postings)

#     # Búsqueda con skip list en memoria
#     start_time = perf_counter()
#     term1_postings = all_postings.get(term1, [])
#     result = []
#     for doc_id, _ in term1_postings:
#         if search_docid_with_skip_list(term2, doc_id, all_postings.get(term2, []), skip_lists.get(term2, [])):
#             result.append(doc_id)
#     end_time = perf_counter()
#     print(f"Tiempo de búsqueda con skip list (en memoria): {end_time - start_time:.4f} segundos")
#     for doc_id in result:
#         print(f"{id_to_docs[doc_id][0]}:{doc_id}")

#     print("\n\n======================\n\n")

#     # Búsqueda tradicional en memoria
#     start_time = perf_counter()
#     term2_doc_ids = {doc_id for doc_id, _ in all_postings.get(term2, [])}
#     result = [doc_id for doc_id, _ in term1_postings if doc_id in term2_doc_ids]
#     end_time = perf_counter()
#     print(f"Tiempo de búsqueda sin skip list (en memoria): {end_time - start_time:.4f} segundos")
#     for doc_id in result:
#         print(f"{id_to_docs[doc_id][0]}:{doc_id}")


if __name__ == "__main__":
    main()