import boolean
import pickle
import struct
import os

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

def evaluar_expr(expr, all_docs: set, vocabulary: dict) -> set:
    """
    Evaluate a boolean expression and return the set of document IDs that match the expression.
    :param expr: The boolean expression to evaluate.
    :return: A set of document IDs that match the expression.
    """
    if expr.__class__.__name__ == 'NOT':
        return all_docs - evaluar_expr(expr.args[0], all_docs, vocabulary)

    if expr.__class__.__name__ == 'Symbol':
        term = str(expr)
        return get_posting_set(term, vocabulary)

    if expr.operator == "&":
        return evaluar_expr(expr.args[0], all_docs, vocabulary) & evaluar_expr(expr.args[1], all_docs, vocabulary)
    elif expr.operator == "|":
        return evaluar_expr(expr.args[0], all_docs, vocabulary) | evaluar_expr(expr.args[1], all_docs, vocabulary)

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
        print("Usage: python punto2.py <boolean_query>")
        return

    expression = args[1]
    
    algebra = boolean.BooleanAlgebra()

    # Load the vocabulary
    vocabulary_path = "vocabulary.pkl"
    vocabulary = load_vocabulary(vocabulary_path)

    # Load the id_to_docs
    id_to_docs_path = "id_to_docs.pkl"
    id_to_docs = load_id_to_docs(id_to_docs_path)

    parsed_expression = algebra.parse(expression)

    # Create the set of all documents
    all_docs = id_to_docs_set(id_to_docs)

    # Parse the expression and evaluate it
    result_set = evaluar_expr(parsed_expression, all_docs, vocabulary)
    
    # Print the result
    if not result_set:
        print("No se encontraron documentos que coincidan con la consulta.")
        return
    
    for doc_id in result_set:
        print(f"{id_to_docs[doc_id][0]}:{doc_id}")

if __name__ == "__main__":
    main()