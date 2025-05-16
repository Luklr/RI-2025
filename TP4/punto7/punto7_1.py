import pickle
import struct
import os
import sys
import heapq
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from Tokenizer import Tokenizer
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

def vbyte_decompress(compressed):
    docIDs = []
    num = 0
    shift = 0
    for byte in compressed:
        num |= (byte & 0x7F) << shift  # Añade los 7 bits al número
        if (byte & 0x80) == 0:         # Si MSB = 0, terminamos
            docIDs.append(num)
            num = 0
            shift = 0
        else:
            shift += 7
    return docIDs

def elias_gamma_decompress(compressed):
    freqs = []
    bitstream = ''.join(f'{byte:08b}' for byte in compressed)
    idx = 0
    n = len(bitstream)
    
    while idx < n:
        # Parte 1: Leer los '1's hasta encontrar un '0' (unario)
        zero_pos = bitstream.find('0', idx)
        if zero_pos == -1:
            break  # No hay más números válidos
        
        length = zero_pos - idx + 1
        if zero_pos + length > n:
            break  # No hay suficientes bits para leer el número
        
        # Parte 2: Leer los siguientes (length - 1) bits
        binary_part = bitstream[zero_pos + 1 : zero_pos + length]
        if not binary_part:
            num = 1
        else:
            num = (1 << (length - 1)) + int(binary_part, 2)
        
        freqs.append(num)
        idx = zero_pos + length
    
    return freqs

def get_compressed_posting_list(vocabulary: dict, term: str, path: str) -> tuple:
    """
    Extrae una posting list comprimida desde los archivos binarios.
    :param vocabulary: Diccionario del vocabulario (con o sin DGaps).
    :param term: Término a buscar.
    :param path: Ruta de la carpeta con los índices comprimidos.
    :return: (docIDs, freqs) descomprimidos.
    """
    if term not in vocabulary:
        return [], []
    
    doc_id_pointer, _, freq_pointer, _ = vocabulary[term]
    
    # Leer docIDs comprimidos
    with open(f"{path}/doc_id_index.bin", "rb") as f:
        f.seek(doc_id_pointer)
        # Asumimos que el siguiente pointer es el del próximo término (o EOF)
        next_term_pointer = min(
            [val[0] for val in vocabulary.values() if val[0] > doc_id_pointer],
            default=os.path.getsize(f"{path}/doc_id_index.bin")
        )
        compressed_docIDs = f.read(next_term_pointer - doc_id_pointer)
    
    # Leer freqs comprimidas
    with open(f"{path}/freq_index.bin", "rb") as f:
        f.seek(freq_pointer)
        next_term_pointer = min(
            [val[2] for val in vocabulary.values() if val[2] > freq_pointer],
            default=os.path.getsize(f"{path}/freq_index.bin")
        )
        compressed_freqs = f.read(next_term_pointer - freq_pointer)
    
    # Descomprimir
    docIDs = vbyte_decompress(compressed_docIDs)
    freqs = elias_gamma_decompress(compressed_freqs)
    
    return docIDs, freqs


def main():
    """
    Main function to load the vocabulary and id_to_docs, and process the query.
    """
    args = os.sys.argv
    if len(args) != 2:
        print("Usage: python punto1_1.py <term>")
        return
    
    term = args[1]
    
    path_with_dgaps = "index_with_dgaps"
    if not os.path.exists(path_with_dgaps):
        os.makedirs(path_with_dgaps)
    
    path_without_dgaps = "index_without_dgaps"
    if not os.path.exists(path_without_dgaps):
        os.makedirs(path_without_dgaps)
    
    vocabulary_path = "vocabulary.pkl"
    id_to_docs_path = "id_to_docs.pkl"
    
    vocabulary_without_dgaps = load_vocabulary(f"{path_without_dgaps}/{vocabulary_path}")
    vocabulary_with_dgaps = load_vocabulary(f"{path_with_dgaps}/{vocabulary_path}")
    id_to_docs = load_id_to_docs(id_to_docs_path)
    
    start_time = perf_counter()
    docIDs_no_dgaps, freqs_no_dgaps = get_compressed_posting_list(
            vocabulary_without_dgaps, term, path_without_dgaps
        )
    end_time = perf_counter()
    print(f"\nSin DGaps (tiempo: {end_time - start_time:.6f} seg):")
    print(f"DocIDs: {docIDs_no_dgaps}")
    print(f"Frecuencias: {freqs_no_dgaps}")



    start_time = perf_counter()
    docIDs_dgaps, freqs_dgaps = get_compressed_posting_list(
        vocabulary_with_dgaps, term, path_with_dgaps
    )
    if docIDs_dgaps:
        original_docIDs = [docIDs_dgaps[0]]
        for delta in docIDs_dgaps[1:]:
            original_docIDs.append(original_docIDs[-1] + delta)
    end_time = perf_counter()
    print(f"\nCon DGaps (tiempo: {end_time - start_time:.6f} seg):")
    print(f"DocIDs (delta): {docIDs_dgaps}")
    # Reconstruir docIDs originales desde DGaps
    print(f"DocIDs (original): {original_docIDs}")
    print(f"Frecuencias: {freqs_dgaps}")


if __name__ == "__main__":
    main()