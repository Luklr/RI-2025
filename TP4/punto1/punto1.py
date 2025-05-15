import bisect
import struct
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from Tokenizer import Tokenizer
import pickle
import logging
from time import perf_counter
import math

def setup_logger():
    """Configura el logger para mostrar mensajes en consola y guardarlos en un archivo"""
    logger = logging.getLogger('BSBI_Logger')
    logger.setLevel(logging.INFO)
    
    # Formato de los mensajes
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                datefmt='%Y-%m-%d %H:%M:%S')
    
    # Handler para consola
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    
    # Handler para archivo
    fh = logging.FileHandler('bsbi_indexing.log')
    fh.setFormatter(formatter)
    
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def process_file(file, docid, tokenizer: Tokenizer, terms_to_id: dict, id_to_docs: dict) -> list[tuple]:
    term_freq = {}
    for line in file:
        # Read the file line by line
        if not line.strip():
            continue

        terms: list = tokenizer.tokenize(line, html_tags=True)
        
        for term in terms:
            if term in term_freq:
                term_freq[term] += 1
            else:
                term_freq[term] = 1
            if term not in terms_to_id:
                terms_to_id[term] = len(terms_to_id) + 1
    
    norm = 0.0
    for term, freq in term_freq.items():
        norm += freq ** 2
    norm = math.sqrt(norm)
    id_to_docs[docid].append(norm)

    return [(terms_to_id[term], docid, freq) for term, freq in term_freq.items()], id_to_docs


def process_chunk(chunk_docs: list[tuple], chunk_number: int):
    """
    Process a chunk of documents and write the term frequencies to a binary file.
    :param chunk_docs: List of tuples containing term IDs and their frequencies.
    :param chunk_number: The chunk number for naming the output file.
    """
    chunk_docs.sort(key=lambda x: x[0])  # Sort by term ID
    if not os.path.exists("chunks"):
        os.makedirs("chunks")
    with open(f"chunks/chunk_{chunk_number}.bin", "wb") as f:
        for tup in chunk_docs:
            f.write(struct.pack('III', *tup))


def merge_chunks(terms_to_id: dict, chunk_number: int)-> dict:
    """
    Merge the chunks into a single vocabulary.
    :param terms_to_id: Dictionary mapping terms to their IDs.
    :param docs_to_id: Dictionary mapping document IDs to their file paths.
    :param chunk_number: The number of chunks processed.
    :return: A dictionary representing the vocabulary.
    """
    with open("index.bin", "wb"):
        pass
    vocabulary = {}
    chunks_pointers = [open(f"chunks/chunk_{i}.bin", "rb") for i in range(chunk_number)]
    
    for pointer in chunks_pointers:
        pointer.seek(0)
    for term, term_id in terms_to_id.items():
        
        posting_list = []
        for pointer in chunks_pointers:
            same_term = True
            while same_term:
                pointer_position = pointer.tell()
                data = pointer.read(12)
                if len(data) < 12:
                    same_term = False
                    break  # llegó al final del archivo
                term_tuple = struct.unpack('III', data)
                if term_tuple[0] != term_id:
                    same_term = False
                    pointer.seek(pointer_position)
                    continue
                posting_list.append((term_tuple[1], term_tuple[2]))
        posting_list.sort(key=lambda x: x[0])
        with open(f"index.bin", "ab") as index_file:
            index_file_position = index_file.tell()
            for doc_id, freq in posting_list:
                index_file.write(struct.pack('II', doc_id, freq))
            
        vocabulary[term] = [index_file_position,len(posting_list),term_id]
    
    return vocabulary


def bsbi(input_dir, chunk_limit):
    """
    BSBI algorithm with improved logging and timing.
    """
    logger = setup_logger()
    logger.info("Starting BSBI algorithm...")
    
    # Initialize variables
    chunk_number = 0
    chunk_docs = []
    docs_read_for_chunk = 0
    total_docs = 0
    tokenizer = Tokenizer(names=True, dates=True, urls=True, emails=True, 
                         words=True, numbers=True, abbreviations=True)
    terms_to_id = {}
    id_to_docs = {}
    
    # Stats variables
    total_terms = 0
    chunk_times = []
    chunk_sizes = []
    
    logger.info(f"Processing directory: {input_dir}")
    logger.info(f"Chunk limit: {chunk_limit} documents per chunk")
    
    # Start total timer
    total_start = perf_counter()
    
    # Chunk generation phase
    logger.info("=== Chunk Generation Phase ===")
    for root, _, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            total_docs += 1
            
            with open(file_path, 'r', encoding='utf-8') as f:
                doc_id = len(id_to_docs) + 1
                id_to_docs[doc_id] = [file]
                
                # Process file and measure time
                start_file = perf_counter()
                terms_tuples, id_to_docs = process_file(f, doc_id, tokenizer, terms_to_id, id_to_docs)
                file_time = perf_counter() - start_file
                
                chunk_docs += terms_tuples
                docs_read_for_chunk += 1
                total_terms += len(terms_tuples)
                
                # Log file processing (every 50 files for less verbosity)
                if total_docs % 50 == 0:
                    logger.debug(f"Processed {total_docs} files | Last file: {file} ({file_time:.4f}s) | "
                                f"Current chunk size: {docs_read_for_chunk}/{chunk_limit}")
                
                # Process chunk when limit reached
                if docs_read_for_chunk >= chunk_limit:
                    start_chunk = perf_counter()
                    process_chunk(chunk_docs, chunk_number)
                    chunk_time = perf_counter() - start_chunk
                    
                    chunk_times.append(chunk_time)
                    chunk_sizes.append(docs_read_for_chunk)
                    
                    logger.info(f"Chunk {chunk_number} completed - "
                               f"Docs: {docs_read_for_chunk} | "
                               f"Terms: {len(chunk_docs)} | "
                               f"Time: {chunk_time:.4f}s")
                    
                    docs_read_for_chunk = 0
                    chunk_number += 1
                    chunk_docs = []

    # Process last chunk if any remaining
    if chunk_docs:
        start_chunk = perf_counter()
        process_chunk(chunk_docs, chunk_number)
        chunk_time = perf_counter() - start_chunk
        
        chunk_times.append(chunk_time)
        chunk_sizes.append(docs_read_for_chunk)
        
        logger.info(f"Final Chunk {chunk_number} completed - "
                   f"Docs: {docs_read_for_chunk} | "
                   f"Terms: {len(chunk_docs)} | "
                   f"Time: {chunk_time:.4f}s")
        chunk_number += 1

    # Chunk generation summary
    chunk_gen_time = sum(chunk_times)
    avg_chunk_time = sum(chunk_times) / len(chunk_times) if chunk_times else 0
    logger.info("=== Chunk Generation Summary ===")
    logger.info(f"Total chunks: {chunk_number}")
    logger.info(f"Total documents: {total_docs}")
    logger.info(f"Total terms processed: {total_terms}")
    logger.info(f"Total chunk processing time: {chunk_gen_time:.4f}s")
    logger.info(f"Average chunk time: {avg_chunk_time:.4f}s")
    logger.info(f"Median chunk time: {sorted(chunk_times)[len(chunk_times)//2]:.4f}s")
    logger.info(f"Chunk size range: {min(chunk_sizes)}-{max(chunk_sizes)} docs")

    # Merging phase
    logger.info("=== Merging Phase ===")
    start_merge = perf_counter()
    vocabulary = merge_chunks(terms_to_id, chunk_number)
    merge_time = perf_counter() - start_merge
    
    logger.info(f"Merged {chunk_number} chunks in {merge_time:.4f}s")
    logger.info(f"Vocabulary size: {len(vocabulary)} terms")

    # Final summary
    total_time = perf_counter() - total_start
    logger.info("=== Final Summary ===")
    logger.info(f"Total indexing time: {total_time:.4f}s")
    logger.info(f"Documents per second: {total_docs/total_time:.2f}")
    logger.info(f"Terms per second: {total_terms/total_time:.2f}")

    # Save results
    pickle.dump(id_to_docs, open("id_to_docs.pkl", "wb"))
    pickle.dump(vocabulary, open("vocabulary.pkl", "wb"))
    logger.info("Results saved to id_to_docs.pkl and vocabulary.pkl")


def main():
    args = os.sys.argv
    if len(args) != 3:
        print("Usage: python punto1.py <input_dir> <docs_read_for_chunk>")
        return

    input_dir = args[1]
    docs_read_for_chunk = int(args[2])
    if docs_read_for_chunk <= 0:
        print("Error: docs_read_for_chunk must be greater than 0")
        return
    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a directory")
        return
    
    bsbi(input_dir, docs_read_for_chunk)


if __name__ == "__main__":
    main()