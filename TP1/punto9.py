from matplotlib import pyplot as plt
import numpy as np
import csv
from punto5 import Tokenizer
import sys

def terms(data: list) -> dict:
    data.sort()
    unique = []
    count = 0
    last_word = ""
    for word in data:
        if word != last_word:
            if last_word != "":
                unique.append({"term": last_word, "tf": count})
            count = 1
            last_word = word
        else:
            count += 1

    if last_word != "":
        unique.append({"term": last_word, "tf": count})
    return unique

def read_file(url):
    data = []
    tokenizer = Tokenizer(words=True, names=True, abbreviations=True, numbers=True)
    terms_processed = 0
    unique_terms = set()
    output_data = []

    with open(url, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            tokens = tokenizer.tokenize(line)
            terms_data = terms(tokens)
            terms_processed += len(tokens)
            for term in terms_data:
                unique_terms.add(term["term"])
            output_data.append((terms_processed, len(unique_terms)))
    
    return output_data

def write_output(output_data, output_file):
    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(output_data)

def heaps_law_fit(output_data):
    N = np.array([x[0] for x in output_data])
    V = np.array([x[1] for x in output_data])
    log_N = np.log(N)
    log_V = np.log(V)
    beta, log_k = np.polyfit(log_N, log_V, 1)
    k = np.exp(log_k)
    return k, beta

def plot_heaps_law(output_data, k, beta):
    N = np.array([x[0] for x in output_data])
    V = np.array([x[1] for x in output_data])
    fitted_V = k * N ** beta
    
    plt.figure(figsize=(8, 5))
    plt.scatter(N, V, label="Datos Reales", alpha=0.6)
    plt.plot(N, fitted_V, label=f"Ajuste Heaps (k={k:.2f}, beta={beta:.2f})", color='red')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("# Términos Totales Procesados")
    plt.ylabel("# Términos Únicos")
    plt.legend()
    plt.title("Ley de Heaps")
    plt.show()

def main():
    if len(sys.argv) < 3:
        print("Uso: python script.py <archivo_entrada> <archivo_salida>")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    data = read_file(input_file)
    write_output(data, output_file)
    k, beta = heaps_law_fit(data)
    print(f"Valores ajustados: k = {k}, beta = {beta}")
    plot_heaps_law(data, k, beta)

if __name__ == "__main__":
    main()