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

def vbyte_compress(docIDs):
    compressed = bytearray()
    for num in docIDs:
        while True:
            byte = num & 0x7F  # Obtiene los 7 bits menos significativos
            num >>= 7          # Desplaza el número 7 bits a la derecha
            if num == 0:
                compressed.append(byte)  # Último byte, MSB = 0
                break
            else:
                compressed.append(byte | 0x80)  # Byte intermedio, MSB = 1
    return compressed

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

def elias_gamma_compress(freqs):
    compressed = bytearray()
    buffer = 0
    buffer_length = 0

    for num in freqs:
        if num == 0:
            raise ValueError("Elias-Gamma no soporta ceros")
        
        # Parte 1: Longitud del número en binario (en unario)
        length = num.bit_length()
        unary = (1 << (length - 1))  # 'length' unos seguidos de un 0
        
        # Parte 2: El número sin el bit más significativo
        binary = num - (1 << (length - 1)) if length > 1 else 0
        
        # Combinamos unario + binary en bits
        combined = (unary << (length - 1)) | binary
        
        # Convertimos a bytes y escribimos
        total_bits = 2 * length - 1
        combined_bin = bin(combined)[2:].zfill(total_bits)
        
        # Procesamos bit por bit para empaquetar en bytes
        for bit in combined_bin:
            buffer = (buffer << 1) | int(bit)
            buffer_length += 1
            if buffer_length == 8:
                compressed.append(buffer)
                buffer = 0
                buffer_length = 0
    
    # Añadir bits restantes si el buffer no está vacío
    if buffer_length > 0:
        compressed.append(buffer << (8 - buffer_length))
    
    return compressed

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

def process_index(index_path: str, id_to_docs: dict, vocabulary: dict, dgaps: bool = False, dgaps_path: str = None):
    """
    Process the index and write the compressed posting lists to a binary file.
    :param index_path: Path to the index file.
    :param id_to_docs: Dictionary mapping document IDs to their corresponding documents.
    :param vocabulary: Dictionary representing the vocabulary.
    """
    open(f"{dgaps_path}/doc_id_index.bin", "wb").close()
    open(f"{dgaps_path}/freq_index.bin", "wb").close()
    vocabulary_compressed = {} #{term: (doc_id_pointer, document_frequency, frequency_pointer, term_id)}
    with open(index_path, "rb") as f:
        for term in vocabulary:
            pointer = vocabulary[term][0]
            postings_count = vocabulary[term][1]
            f.seek(pointer)
            posting_list = []
            offset = pointer + postings_count * 8
            while f.tell() < offset:
                doc_id, freq = struct.unpack('II', f.read(8))
                posting_list.append((doc_id, freq))
            
            # Apply compression
            docIDs = [doc_id for doc_id, _ in posting_list]
            freqs = [freq for _, freq in posting_list]

            if dgaps:
                docIDs = [docIDs[0]] + [docIDs[i] - docIDs[i - 1] for i in range(1, len(docIDs))]
            
            compressed_docIDs = vbyte_compress(docIDs)
            compressed_freqs = elias_gamma_compress(freqs)
            
            # Write compressed data to file
            with open(f"{dgaps_path}/doc_id_index.bin", "ab") as out_f:
                doc_id_pointer = out_f.tell()
                out_f.write(compressed_docIDs)
            with open(f"{dgaps_path}/freq_index.bin", "ab") as out_f:
                frequency_pointer = out_f.tell()
                out_f.write(compressed_freqs)
            
            vocabulary_compressed[term] = (doc_id_pointer, len(posting_list), frequency_pointer, vocabulary[term][2])
    
    return vocabulary_compressed

def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size  # Tamaño en bytes

def main():
    """
    Main function to load the vocabulary and id_to_docs, and process the query.
    """
    vocabulary_path = "vocabulary.pkl"
    id_to_docs_path = "id_to_docs.pkl"
    
    vocabulary = load_vocabulary(vocabulary_path)
    id_to_docs = load_id_to_docs(id_to_docs_path)
    
    path_with_dgaps = "index_with_dgaps"
    if not os.path.exists(path_with_dgaps):
        os.makedirs(path_with_dgaps)
    
    path_without_dgaps = "index_without_dgaps"
    if not os.path.exists(path_without_dgaps):
        os.makedirs(path_without_dgaps)

    start_no_dgaps_time = perf_counter()
    vocabulary_without_dgaps = process_index("index.bin", id_to_docs, vocabulary, dgaps=False, dgaps_path=path_without_dgaps)
    end_no_dgaps_time = perf_counter()

    print(f"Tiempo de compresión sin dgaps: {end_no_dgaps_time - start_no_dgaps_time:.2f} segundos")
    size_without_dgaps = get_folder_size(path_without_dgaps)
    print(f"Tamaño sin DGaps: {size_without_dgaps} bytes ({size_without_dgaps / 1024:.2f} KB)")

    start_dgaps_time = perf_counter()
    vocabulary_with_dgaps = process_index("index.bin", id_to_docs, vocabulary, dgaps=True, dgaps_path=path_with_dgaps)
    end_dgaps_time = perf_counter()

    print(f"Tiempo de compresión con dgaps: {end_dgaps_time - start_dgaps_time:.2f} segundos")
    size_with_dgaps = get_folder_size(path_with_dgaps)
    print(f"Tamaño con DGaps: {size_with_dgaps} bytes ({size_with_dgaps / 1024:.2f} KB)")

    print(f"Reducción de tamaño: {100 * (1 - size_with_dgaps / size_without_dgaps):.2f}%")

    pickle.dump(vocabulary_without_dgaps, open(f"{path_without_dgaps}/vocabulary.pkl", "wb"))
    pickle.dump(vocabulary_with_dgaps, open(f"{path_with_dgaps}/vocabulary.pkl", "wb"))


if __name__ == "__main__":
    main()

