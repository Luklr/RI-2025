import pyterrier as pt
import os
from scipy.stats import spearmanr

pt.init()

def index_files(path, index_path):
    index_path = os.path.abspath(index_path)
    print("Ruta de índice:", index_path)

    if not os.path.exists(index_path):
        os.makedirs(index_path)

    docs = []
    for root, _, filenames in os.walk(path):
        for f in filenames:
            if f.endswith(".html"):
                filepath = os.path.join(root, f)
                with open(filepath, "r", encoding="utf-8", errors="ignore") as file:
                    content = ""
                    for line in file:
                        line = line.strip()
                        if line:
                            content += line + " "
                    docs.append({"docno": f, "text": content})

    print(f"Total archivos: {len(docs)}")

    indexer = pt.IterDictIndexer(index_path, overwrite=True, meta={'docno': 50})
    index_ref = indexer.index(docs)

    print("Indexing completo")
    return index_ref

def models(index_ref):
    # Crea un modelo de recuperación de información
    bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25")
    tfidf = pt.BatchRetrieve(index_ref, wmodel="TF_IDF")
    return bm25, tfidf

def main():
    index_ref = index_files("../data/wiki-small/en/articles", "index")
    bm25, tfidf = models(index_ref)

    # Realiza una consulta de ejemplo
    queries = ["military base", "football players", "computer science", "dogs pets", "wood houses"]
    results_bm25 = [bm25.search(query) for query in queries]
    results_tfidf = [tfidf.search(query) for query in queries]

    print("Resultados BM25:")
    for i, query in enumerate(queries):
        print(f"Consulta: {query}")
        print("Resultados BM25:")
        print(results_bm25[i].head(10))
        print("Resultados TF-IDF:")
        print(results_tfidf[i].head(10))

    print("""-------------------------------------------------------------------------------------

Correlación entre rankings BM25 y TF-IDF:

""")
    print("""-------------------------------------------------------------------------------------
Para los 10 primeros resultados: 
""")
    for i in range(len(queries)):
        rho, _ = spearmanr(results_bm25[i]['docid'].head(10), results_tfidf[i]['docid'].head(10))
        print(f"Consulta: {queries[i]}")
        print(f"Correlación de Spearman: {rho:.4f}")
    
    print("""-------------------------------------------------------------------------------------
Para los 25 primeros resultados: 
""")
    for i in range(len(queries)):
        rho, _ = spearmanr(results_bm25[i]['docid'].head(25), results_tfidf[i]['docid'].head(25))
        print(f"Consulta: {queries[i]}")
        print(f"Correlación de Spearman: {rho:.4f}")
    
    print("""-------------------------------------------------------------------------------------
Para los 50 primeros resultados: 
""")
    for i in range(len(queries)):
        rho, _ = spearmanr(results_bm25[i]['docid'].head(50), results_tfidf[i]['docid'].head(50))
        print(f"Consulta: {queries[i]}")
        print(f"Correlación de Spearman: {rho:.4f}")

if __name__ == "__main__":
    main()