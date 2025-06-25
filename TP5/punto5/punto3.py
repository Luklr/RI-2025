import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, parse_qs
import plotly.graph_objects as go
import networkx as nx
import time
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

class WebCrawler:
    def __init__(self):
        """Inicializa el crawler con detección de páginas estáticas/dinámicas"""
        self.todo_list = []  # (url, logical_depth, physical_depth, is_seed)
        self.done_list = set()
        self.graph = nx.DiGraph()
        self.domain_counts = defaultdict(int)
        self.seed_domains = set()
        self.MAX_PAGES_SEED = 30
        self.MAX_LOGICAL_DEPTH = 3
        self.MAX_PHYSICAL_DEPTH = 3
        self.failed_requests = set()
        self.static_pages = set()
        self.dynamic_pages = set()
        self.MAX_TOTAL_PAGES = 500  # Límite total de páginas
        self.crawled_pages = []  # Lista ordenada de páginas crawleadas
        
        # Extensiones consideradas estáticas
        self.STATIC_EXTENSIONS = {
            '.html', '.htm', '.xhtml', '.shtml',
            '.css', '.js', '.json',
            '.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp', '.ico',
            '.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx',
            '.txt', '.csv', '.xml', '.woff', '.woff2', '.ttf', '.eot'
        }

    def is_static_page(self, url):
        """Determina si una URL es estática basado en extensión y parámetros"""
        parsed = urlparse(url)
        
        # Verificar extensión de archivo
        path = parsed.path.lower()
        if any(path.endswith(ext) for ext in self.STATIC_EXTENSIONS):
            return True
            
        # Verificar query string
        if parsed.query:
            return False
            
        # Verificar si termina con slash (probable página estática)
        if path.endswith('/'):
            return True
            
        # Si no tiene extensión ni parámetros, asumir estática
        if '.' not in path.split('/')[-1]:
            return True
            
        return False

    def get_domain(self, url):
        """Extrae el dominio principal de una URL"""
        try:
            return urlparse(url).netloc
        except:
            return None

    def fetch_page(self, url: str):
        """Descarga el contenido de una página con manejo robusto de errores"""
        if url in self.failed_requests:
            return None
            
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept-Language': 'en-US,en;q=0.9'
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Clasificar la página
            if self.is_static_page(url):
                self.static_pages.add(url)
            else:
                self.dynamic_pages.add(url)
                
            return response.text
        except Exception as e:
            print(f"Error fetching {url}: {str(e)[:100]}...")
            self.failed_requests.add(url)
            return None

    def parse_page(self, page_content: str, base_url: str):
        """Extrae enlaces válidos de una página con filtrado robusto"""
        if not page_content:
            return []
        
        try:
            soup = BeautifulSoup(page_content, 'html.parser')
            links = set()
            
            for link in soup.find_all('a', href=True):
                try:
                    full_url = urljoin(base_url, link['href'])
                    if full_url.startswith(('http://', 'https://')):
                        links.add(full_url)
                except:
                    continue
                    
            return list(links)
        except Exception as e:
            print(f"Error parsing page: {str(e)[:100]}...")
            return []

    def crawl(self, seed_sites):
        """Ejecuta el crawling con límite de 500 páginas"""
        self.seed_domains = {self.get_domain(url) for url in seed_sites if self.get_domain(url)}
        self.todo_list = [(url, 0, 0, self.get_domain(url)) for url in seed_sites if self.get_domain(url)]
        
        while self.todo_list and len(self.crawled_pages) < self.MAX_TOTAL_PAGES:
            current_url, l_depth, p_depth, origin_domain = self.todo_list.pop(0)
            current_domain = self.get_domain(current_url)
            
            if not current_domain:
                continue
                
            if p_depth > self.MAX_PHYSICAL_DEPTH:
                continue
                
            if l_depth > self.MAX_LOGICAL_DEPTH:
                continue
                
            if current_url in self.done_list or current_url in self.failed_requests:
                continue
                
            print(f"Crawling {len(self.crawled_pages)+1}/500: {current_url} (Logical: {l_depth}, Physical: {p_depth})")
            page_content = self.fetch_page(current_url)
            
            if page_content:
                self.done_list.add(current_url)
                self.crawled_pages.append(current_url)
                
                # Añadir nodo al grafo con tipo de página
                page_type = 'static' if self.is_static_page(current_url) else 'dynamic'
                self.graph.add_node(current_url, 
                                  domain=current_domain,
                                  origin_domain=origin_domain,
                                  depth=l_depth,
                                  page_type=page_type)
                
                for link in self.parse_page(page_content, current_url):
                    target_domain = self.get_domain(link)
                    if not target_domain:
                        continue
                    
                    new_origin = origin_domain
                    new_l_depth = l_depth + 1
                    new_p_depth = p_depth + (0 if target_domain == current_domain else 1)
                    
                    if new_p_depth > self.MAX_PHYSICAL_DEPTH:
                        continue
                        
                    # Añadir el nodo destino con su profundidad antes del edge
                    if link not in self.graph.nodes:
                        self.graph.add_node(link, 
                                          domain=target_domain,
                                          origin_domain=new_origin,
                                          depth=new_l_depth,
                                          page_type='static' if self.is_static_page(link) else 'dynamic')
                    
                    self.graph.add_edge(current_url, link)
                    
                    if (link not in self.done_list and 
                        link not in self.failed_requests and 
                        not any(link == u for u, *_ in self.todo_list)):
                        self.todo_list.append((link, new_l_depth, new_p_depth, new_origin))
                
                time.sleep(0.5)  # Reducir delay para llegar a 500 más rápido

    def calculate_metrics(self):
        """Calcula PageRank y HITS para las páginas crawleadas"""
        print(f"\nCalculando métricas para {len(self.crawled_pages)} páginas crawleadas...")
        
        # Filtrar grafo solo con páginas crawleadas
        crawled_graph = self.graph.subgraph(self.crawled_pages).copy()
        
        # Calcular PageRank
        try:
            pagerank_scores = nx.pagerank(crawled_graph, alpha=0.85, max_iter=100)
        except:
            # Si falla, usar valores uniformes
            pagerank_scores = {url: 1.0/len(self.crawled_pages) for url in self.crawled_pages}
        
        # Calcular HITS
        try:
            hits_scores = nx.hits(crawled_graph, max_iter=100)
            authorities = hits_scores[1]  # authorities
        except:
            # Si falla, usar valores uniformes
            authorities = {url: 1.0/len(self.crawled_pages) for url in self.crawled_pages}
        
        return pagerank_scores, authorities

    def simulate_pagerank_crawling(self, pagerank_scores):
        """Simula crawling siguiendo orden de PageRank"""
        # Ordenar páginas por PageRank (mayor a menor)
        pagerank_order = sorted(self.crawled_pages, 
                               key=lambda x: pagerank_scores.get(x, 0), 
                               reverse=True)
        return pagerank_order

    def calculate_overlap(self, pagerank_order, authorities_order):
        """Calcula evolución del overlap entre dos ordenamientos"""
        overlaps = []
        
        for i in range(1, len(pagerank_order) + 1):
            # Top-i páginas de cada ranking
            top_pagerank = set(pagerank_order[:i])
            top_authorities = set(authorities_order[:i])
            
            # Calcular overlap
            intersection = len(top_pagerank.intersection(top_authorities))
            overlap_percentage = (intersection / i) * 100
            overlaps.append(overlap_percentage)
        
        return overlaps

    def calculate_correlations(self, pr_values, auth_values):
        """Calcula correlaciones de Pearson y Spearman"""
        try:
            # Correlación de Pearson (lineal)
            pearson_corr, pearson_p = pearsonr(pr_values, auth_values)
        except:
            pearson_corr, pearson_p = 0.0, 1.0
            
        try:
            # Correlación de Spearman (ranking/monotónica)
            spearman_corr, spearman_p = spearmanr(pr_values, auth_values)
        except:
            spearman_corr, spearman_p = 0.0, 1.0
        
        return {
            'pearson': {'correlation': pearson_corr, 'p_value': pearson_p},
            'spearman': {'correlation': spearman_corr, 'p_value': spearman_p}
        }

    def analyze_and_plot(self):
        """Realiza el análisis completo y genera gráficos"""
        if len(self.crawled_pages) == 0:
            print("No hay páginas crawleadas para analizar")
            return
        
        print(f"\n=== ANÁLISIS DE {len(self.crawled_pages)} PÁGINAS CRAWLEADAS ===")
        
        # Calcular métricas
        pagerank_scores, authorities = self.calculate_metrics()
        
        # Ordenar páginas por cada métrica
        pagerank_order = sorted(self.crawled_pages, 
                               key=lambda x: pagerank_scores.get(x, 0), 
                               reverse=True)
        
        authorities_order = sorted(self.crawled_pages, 
                                  key=lambda x: authorities.get(x, 0), 
                                  reverse=True)
        
        # Simular crawling por PageRank
        print("\nSimulando crawling siguiendo orden PageRank...")
        
        # Calcular overlap entre orden real de crawling y orden por Authorities
        real_vs_auth_overlap = self.calculate_overlap(self.crawled_pages, authorities_order)
        
        # Calcular overlap entre orden por PageRank y orden por Authorities  
        pagerank_vs_auth_overlap = self.calculate_overlap(pagerank_order, authorities_order)
        
        # Preparar datos para correlación
        pr_values = [pagerank_scores[url] for url in self.crawled_pages]
        auth_values = [authorities[url] for url in self.crawled_pages]
        
        # Calcular correlaciones
        correlations = self.calculate_correlations(pr_values, auth_values)
        
        # Crear gráfico con 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))
        
        # Gráfico 1: Evolución del overlap
        x = range(1, len(self.crawled_pages) + 1)
        ax1.plot(x, real_vs_auth_overlap, 'b-', linewidth=2, label='Crawling Real vs Authorities')
        ax1.plot(x, pagerank_vs_auth_overlap, 'r-', linewidth=2, label='Crawling PageRank vs Authorities')
        ax1.set_xlabel('Número de páginas (Top-k)')
        ax1.set_ylabel('Porcentaje de Overlap (%)')
        ax1.set_title('Evolución del Overlap respecto al ranking por Authorities')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(1, len(self.crawled_pages))
        ax1.set_ylim(0, 100)
        
        # Gráfico 2: Correlación (scatter plot)
        ax2.scatter(pr_values, auth_values, alpha=0.6, s=30)
        ax2.set_xlabel('PageRank Score')
        ax2.set_ylabel('Authority Score')
        ax2.set_title(f'Correlación PageRank vs Authority\n'
                     f'Pearson: {correlations["pearson"]["correlation"]:.3f} | '
                     f'Spearman: {correlations["spearman"]["correlation"]:.3f}')
        ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: Comparación de rankings (top 50)
        top_n = min(50, len(self.crawled_pages))
        
        # Crear rankings para top_n páginas
        pagerank_ranks = {url: i+1 for i, url in enumerate(pagerank_order[:top_n])}
        auth_ranks = {url: i+1 for i, url in enumerate(authorities_order[:top_n])}
        
        # Páginas que aparecen en ambos top_n
        common_pages = set(pagerank_order[:top_n]) & set(authorities_order[:top_n])
        
        pr_positions = [pagerank_ranks[url] for url in common_pages]
        auth_positions = [auth_ranks[url] for url in common_pages]
        
        ax3.scatter(pr_positions, auth_positions, alpha=0.7, s=40)
        ax3.plot([1, top_n], [1, top_n], 'r--', alpha=0.5, label='Ranking perfecto')
        ax3.set_xlabel(f'Posición en ranking PageRank (Top-{top_n})')
        ax3.set_ylabel(f'Posición en ranking Authorities (Top-{top_n})')
        ax3.set_title(f'Comparación de posiciones en rankings (Top-{top_n})\n'
                     f'Páginas en común: {len(common_pages)}/{top_n}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, top_n+1)
        ax3.set_ylim(0, top_n+1)
        
        plt.tight_layout()
        plt.savefig('pagerank_analysis_spearman.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Estadísticas finales
        print(f"\n=== RESULTADOS ===")
        print(f"Total páginas analizadas: {len(self.crawled_pages)}")
        print(f"Overlap final (Real vs Auth): {real_vs_auth_overlap[-1]:.1f}%")
        print(f"Overlap final (PageRank vs Auth): {pagerank_vs_auth_overlap[-1]:.1f}%")
        
        # Correlaciones
        print(f"\n=== CORRELACIONES ===")
        print(f"Pearson (lineal):     {correlations['pearson']['correlation']:.3f} (p={correlations['pearson']['p_value']:.3e})")
        print(f"Spearman (ranking):   {correlations['spearman']['correlation']:.3f} (p={correlations['spearman']['p_value']:.3e})")
        
        # Interpretación de Spearman
        spearman_val = correlations['spearman']['correlation']
        if abs(spearman_val) >= 0.7:
            strength = "fuerte"
        elif abs(spearman_val) >= 0.5:
            strength = "moderada"
        elif abs(spearman_val) >= 0.3:
            strength = "débil"
        else:
            strength = "muy débil o inexistente"
        
        direction = "positiva" if spearman_val > 0 else "negativa"
        print(f"Interpretación Spearman: Correlación {direction} {strength}")
        
        # Top 10 por cada métrica
        print(f"\nTop 10 PageRank:")
        for i, url in enumerate(pagerank_order[:10]):
            score = pagerank_scores[url]
            print(f"{i+1:2d}. {score:.6f} - {url[:80]}...")
            
        print(f"\nTop 10 Authorities:")
        for i, url in enumerate(authorities_order[:10]):
            score = authorities[url]
            print(f"{i+1:2d}. {score:.6f} - {url[:80]}...")
        
        return {
            'pagerank_scores': pagerank_scores,
            'authorities': authorities,
            'real_vs_auth_overlap': real_vs_auth_overlap,
            'pagerank_vs_auth_overlap': pagerank_vs_auth_overlap,
            'correlations': correlations
        }

    def visualize(self, output_file="web_graph.html"):
        """Visualización básica del grafo"""
        if len(self.graph.nodes) == 0:
            print("No hay datos para visualizar")
            return
            
        try:
            import plotly.io as pio
            
            # Análisis de distribución de profundidades
            depth_distribution = defaultdict(int)
            for node in self.crawled_pages:
                depth = self.graph.nodes[node].get('depth', 0)
                depth_distribution[depth] += 1
            
            print("\nDistribución de profundidades lógicas:")
            max_depth = max(depth_distribution.keys()) if depth_distribution else 0
            for depth in range(0, max_depth + 1):
                count = depth_distribution.get(depth, 0)
                print(f"Profundidad {depth}: {count} páginas ({count/len(self.crawled_pages):.1%})")
            
            print(f"\nResumen del crawling:")
            print(f"- Total páginas crawleadas: {len(self.crawled_pages)}")
            print(f"- Páginas estáticas: {len(self.static_pages)}")
            print(f"- Páginas dinámicas: {len(self.dynamic_pages)}")
            print(f"- Profundidad máxima: {max_depth}")
            
        except Exception as e:
            print(f"Error al visualizar: {str(e)}")


if __name__ == "__main__":
    crawler = WebCrawler()

    # Configuración para llegar a 500 páginas
    crawler.MAX_LOGICAL_DEPTH = 3
    crawler.MAX_PHYSICAL_DEPTH = 3
    sitios_iniciales = [
        "https://www.google.com",
        "https://www.youtube.com",
        "https://mail.google.com",
        "https://outlook.office.com",
        "https://www.facebook.com",
        "https://docs.google.com",
        "https://chatgpt.com",
        "https://login.microsoftonline.com",
        "https://www.linkedin.com",
        "https://accounts.google.com",
        "https://x.com",
        "https://www.bing.com",
        "https://www.instagram.com",
        "https://drive.google.com",
        "https://campus-1001.ammon.cloud",
        "https://github.com",
        "https://web.whatsapp.com",
        "https://duckduckgo.com",
        "https://www.reddit.com",
        "https://calendar.google.com"
    ]
    
    print("Iniciando crawling de 500 páginas...")
    crawler.crawl(seed_sites=sitios_iniciales)
    
    print(f"\nCrawling completado: {len(crawler.crawled_pages)} páginas")
    
    # Análisis de métricas y visualización
    results = crawler.analyze_and_plot()
    
    # Visualización básica del grafo
    crawler.visualize()