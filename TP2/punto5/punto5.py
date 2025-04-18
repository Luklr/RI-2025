import os
import sys
import re
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from Tokenizer import Tokenizer
from Matrix import DynamicMatrix
import math

def add_terms_to_matrix(matrix, terms, doc_name):
    doc_col = matrix.ensure_doc_exists(doc_name)
    
    for term in terms:
        term_row = matrix.ensure_term_exists(term)
        current_value = matrix[term_row, doc_col]
        matrix[term_row, doc_col] = current_value + 1
    return matrix



def term_frequency_matrix(path: str) -> DynamicMatrix:
    tokenizer = Tokenizer(words=True, names=True)
    data = DynamicMatrix()

    # Cambiar esto de ["terms"] a "terms"
    data.add_row("terms")  # Fila 0, clave "terms"
    for root, _, filenames in os.walk(path):
        for f in filenames:
            if f.endswith(".txt") or f.endswith(".html"):
                filepath = os.path.join(root, f)

                with open(filepath, "r", encoding="utf-8", errors="ignore") as file:
                    for line in file:
                        terms = tokenizer.tokenize(line, html_tags=True)
                        data = add_terms_to_matrix(data, terms, filepath)
    return data

def tf_idf(tf_matrix: DynamicMatrix) -> DynamicMatrix:
    print("Calculo de TF/IDF")
    num_docs = tf_matrix.columns() - 1
    rows = tf_matrix.rows()
    cols = tf_matrix.columns()
    
    print(f"Cantidad de terminos: {rows - 1}")
    
    df = [0] * rows
    for i in range(1, rows):
        df[i] = sum(1 for j in range(1, cols) if tf_matrix[i, j] > 0)
    
    idf = [0] * rows
    for i in range(1, rows):
        idf[i] = math.log(num_docs / (df[i] + 1e-10)) if df[i] > 0 else 0
    
    block_size = 1000  
    for i_start in range(1, rows, block_size):
        i_end = min(i_start + block_size, rows)
        for j in range(1, cols):
            for i in range(i_start, i_end):
                tf = tf_matrix[i, j]
                if tf > 0:
                    tf_matrix[i, j] = (1 + math.log(tf)) * idf[i]
    
    print("Termina de calcular el TF/IDF")
    return tf_matrix

def search(tf_idf_matrix: DynamicMatrix, query: str) -> list:
    print("Buscando:", query)
    tokenizer = Tokenizer(words=True, names=True)
    query_terms = tokenizer.tokenize(query, html_tags=True)
    
    query_term_counts = {}
    for term in query_terms:
        query_term_counts[term] = query_term_counts.get(term, 0) + 1
    
    doc_scores = {}
    query_norm = 0.0
    processed_terms = set()
    
    # Precalcular IDFs y términos existentes
    num_docs = tf_idf_matrix.columns() - 1
    existing_terms = {tf_idf_matrix[i, 0]: i for i in range(1, tf_idf_matrix.rows())}
    
    # Calcular componentes de la query una sola vez
    query_weights = {}
    for term, count in query_term_counts.items():
        if term in existing_terms:
            row_idx = existing_terms[term]
            df = sum(1 for j in range(1, tf_idf_matrix.columns()) if tf_idf_matrix[row_idx, j] > 0)
            idf = math.log(num_docs / (df + 1))
            tf = 1 + math.log(count) if count > 0 else 0
            weight = tf * idf
            query_weights[term] = weight
            query_norm += weight ** 2
            processed_terms.add(term)
    
    query_norm = math.sqrt(query_norm) if query_norm > 0 else 1.0
    
    # Calcular scores para cada documento
    for doc_col in range(1, tf_idf_matrix.columns()):
        doc_name = tf_idf_matrix[0, doc_col]
        score = 0.0
        doc_norm = 0.0
        
        for term in processed_terms:
            row_idx = existing_terms[term]
            doc_weight = tf_idf_matrix[row_idx, doc_col]
            if doc_weight > 0:
                score += query_weights[term] * doc_weight
                doc_norm += doc_weight ** 2
        
        if doc_norm > 0:
            doc_norm = math.sqrt(doc_norm)
            normalized_score = score / (query_norm * doc_norm)
            doc_scores[doc_name] = normalized_score
    
    # Ordenar resultados
    sorted_results = sorted(
        ({"doc_name": name, "score": score} for name, score in doc_scores.items()),
        key=lambda x: x["score"],
        reverse=True
    )
    
    return sorted_results

def main():
    if len(sys.argv) != 3:
        print("Uso:")
        print("python punto2.py [directorio/de/documentos] [consulta]")
        sys.exit(1)

    if not os.path.isdir(sys.argv[1]):
        print("El directorio de documentos no existe.")
        sys.exit(1)
    
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]

    tf_matrix = term_frequency_matrix(arg1)

    tf_idf_matrix = tf_idf(tf_matrix)
    
    results = search(tf_idf_matrix, arg2)

    print("Resultados de la búsqueda:")
    for result in results:
        print(result)

if __name__ == "__main__":
    main()