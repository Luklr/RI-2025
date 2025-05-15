import boolean
import pickle
import struct
import os
from time import perf_counter

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

def load_full_index(vocabulary: dict, index_path="index.bin") -> dict:
    """
    Load the full index into memory from disk.
    :param vocabulary: The vocabulary with pointers and posting counts.
    :param index_path: Path to the binary index file.
    :return: A dictionary {term: set(doc_ids)}
    """
    full_index = {}
    with open(index_path, "rb") as f:
        for term, data in vocabulary.items():
            f.seek(data[0])
            posting_list = set()
            for _ in range(data[1]):
                bytes_read = f.read(8)
                if len(bytes_read) != 8:
                    print(f"Error leyendo postings para término '{term}' en offset {f.tell()}. Se esperaban 8 bytes pero se leyeron {len(bytes_read)}.")
                    break  # o raise si querés cortar la ejecución
                doc_id, freq = struct.unpack('II', bytes_read)
                posting_list.add(doc_id)
            full_index[term] = posting_list
    return full_index

def evaluar_expr(expr, all_docs: set, full_index: dict) -> set:
    if expr.__class__.__name__ == 'NOT':
        return all_docs - evaluar_expr(expr.args[0], all_docs, full_index)

    if expr.__class__.__name__ == 'Symbol':
        term = str(expr)
        return full_index.get(term, set())

    if expr.operator == "&":
        return evaluar_expr(expr.args[0], all_docs, full_index) & evaluar_expr(expr.args[1], all_docs, full_index)
    elif expr.operator == "|":
        return evaluar_expr(expr.args[0], all_docs, full_index) | evaluar_expr(expr.args[1], all_docs, full_index)

    return set()

def get_posting_set(term: str, vocabulary:dict) -> set:
    """
    Get the posting list for a term from the vocabulary.
    :param term: The term to search for.
    :param vocabulary: The vocabulary dictionary.
    :return: A set that contain document IDs.
    """
    if not term in vocabulary:
        return set()

    pointer = vocabulary[term][0]
    postings_count = vocabulary[term][1]
    with open("index.bin", "rb") as f:
        f.seek(pointer)
        posting_list = set()
        offset = pointer + postings_count * 8
        while f.tell() < offset:
            doc_id, freq = struct.unpack('II', f.read(8))
            posting_list.add(doc_id)
    
    return posting_list

def id_to_docs_set(id_to_docs: dict) -> set:
    """
    Convert the dictionary "id_to_docs" into a set.
    :param id_to_docs: The id_to_docs dictionary.
    :return: A set of document IDs.
    """
    return {key for key, value in id_to_docs.items()}

def main():
    args = os.sys.argv
    if len(args) != 2:
        print("Usage: python punto3_memory.py <boolean_query>")
        return

    # start_time = perf_counter()

    expression = args[1]
    algebra = boolean.BooleanAlgebra()

    # Cargar datos
    vocabulary = load_vocabulary("vocabulary.pkl")
    id_to_docs = load_id_to_docs("id_to_docs.pkl")

    # end_time = perf_counter()

    # Cargar índice entero en memoria
    full_index = load_full_index(vocabulary, "index.bin")

    start_time = perf_counter()

    parsed_expression = algebra.parse(expression)

    all_docs = id_to_docs_set(id_to_docs)

    # Evaluar expresión
    result_set = evaluar_expr(parsed_expression, all_docs, full_index)

    end_time = perf_counter()
    # No cuento el tiempo de ejecución de la carga del índice
    print(f"Tiempo de ejecución: { (end_time - start_time):.4f}s")

    if not result_set:
        print("No se encontraron documentos que coincidan con la consulta.")
        return

    for doc_id in result_set:
        print(f"{id_to_docs[doc_id][0]}:{doc_id}")

if __name__ == "__main__":
    main()