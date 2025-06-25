import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import plotly.graph_objects as go
import networkx as nx
import time
from collections import defaultdict

class WebCrawler:
    def __init__(self):
        """Inicializa el crawler con todos los controles necesarios"""
        self.todo_list = []  # (url, logical_depth, physical_depth, is_seed)
        self.done_list = set()
        self.graph = nx.DiGraph()
        self.domain_counts = defaultdict(int)
        self.seed_domains = set()
        self.MAX_PAGES_SEED = 30  # Límite estricto 20-50 páginas por sitio semilla
        self.MAX_LOGICAL_DEPTH = 3
        self.MAX_PHYSICAL_DEPTH = 3
        self.failed_requests = set()  # Para evitar reintentar URLs fallidas

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
        """Ejecuta el crawling considerando URLs externas como parte del sitio semilla"""
        self.seed_domains = {self.get_domain(url) for url in seed_sites if self.get_domain(url)}
        self.todo_list = [(url, 1, 1, self.get_domain(url)) for url in seed_sites if self.get_domain(url)]
        
        while self.todo_list:
            current_url, l_depth, p_depth, origin_domain = self.todo_list.pop(0)
            current_domain = self.get_domain(current_url)
            
            if not current_domain:
                continue
                
            # Verificación estricta de límites (basado en el dominio de origen)
            if origin_domain in self.seed_domains and self.domain_counts[origin_domain] >= self.MAX_PAGES_SEED:
                continue
                
            if p_depth > self.MAX_PHYSICAL_DEPTH:
                continue
                
            if l_depth > self.MAX_LOGICAL_DEPTH:
                continue
                
            if current_url in self.done_list or current_url in self.failed_requests:
                continue
                
            print(f"Crawling: {current_url} (Logical: {l_depth}, Physical: {p_depth}, Origin: {origin_domain}, Pages: {self.domain_counts.get(origin_domain, 0)}/{self.MAX_PAGES_SEED})")
            page_content = self.fetch_page(current_url)
            
            if page_content:
                self.done_list.add(current_url)
                if origin_domain in self.seed_domains:
                    self.domain_counts[origin_domain] += 1
                
                # Añadir nodo al grafo
                self.graph.add_node(current_url, 
                                domain=current_domain,
                                origin_domain=origin_domain,
                                depth=l_depth)
                
                for link in self.parse_page(page_content, current_url):
                    target_domain = self.get_domain(link)
                    if not target_domain:
                        continue
                    
                    # Mantener el mismo dominio de origen para todas las URLs encontradas
                    new_origin = origin_domain
                    new_l_depth = l_depth + 1
                    new_p_depth = p_depth + (0 if target_domain == current_domain else 1)
                    
                    # Verificar límites basados en el dominio de origen
                    if new_origin in self.seed_domains and self.domain_counts.get(new_origin, 0) >= self.MAX_PAGES_SEED:
                        continue
                        
                    if new_p_depth > self.MAX_PHYSICAL_DEPTH:
                        continue
                        
                    self.graph.add_edge(current_url, link)
                    
                    if (link not in self.done_list and 
                        link not in self.failed_requests and 
                        not any(link == u for u, *_ in self.todo_list)):
                        self.todo_list.append((link, new_l_depth, new_p_depth, new_origin))
                
                time.sleep(1)

    def visualize(self, output_file="web_graph.html"):
        """Visualización sin labels en los nodos"""
        if len(self.graph.nodes) == 0:
            print("No hay datos para visualizar")
            return
            
        try:
            import plotly.io as pio
            
            # Configuración inicial (igual que antes)
            for node in self.graph.nodes():
                if 'domain' not in self.graph.nodes[node]:
                    self.graph.nodes[node]['domain'] = self.get_domain(node) or 'unknown'
                if 'is_seed' not in self.graph.nodes[node]:
                    self.graph.nodes[node]['is_seed'] = self.graph.nodes[node]['domain'] in self.seed_domains
            
            pos = nx.spring_layout(self.graph, k=0.7, iterations=100)  # Más espaciado
            
            # Preparar aristas (igual que antes)
            edge_x = []
            edge_y = []
            for edge in self.graph.edges():
                try:
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                except:
                    continue
                    
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.7, color='rgba(136, 136, 136, 0.3)'),  # Líneas más sutiles
                hoverinfo='none',
                mode='lines')
            
            # Preparar nodos (MODIFICADO - sin texto)
            node_x = []
            node_y = []
            hover_text = []
            node_colors = []
            
            for node in self.graph.nodes():
                try:
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    domain = self.graph.nodes[node]['domain']
                    is_seed = self.graph.nodes[node]['is_seed']
                    hover_text.append(f"<b>{domain}</b><br>{node[:100]}...")  # Solo hover
                    node_colors.append('#1f77b4' if is_seed else '#ff7f0e')  # Azul/Naranja
                except:
                    continue
                    
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',  # Solo marcadores, sin texto
                hovertext=hover_text,
                hoverinfo='text',
                marker=dict(
                    showscale=False,  # Sin escala de color
                    color=node_colors,
                    size=12,  # Tamaño uniforme
                    line=dict(width=1, color='DarkSlateGrey'))
            )
            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='Mapa de Relaciones',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=0, l=0, r=0, t=30),  # Márgenes mínimos
                    xaxis=dict(showgrid=False, zeroline=False, visible=False),  # Ejes ocultos
                    yaxis=dict(showgrid=False, zeroline=False, visible=False),
                    plot_bgcolor='white'  # Fondo blanco
                )
            )
            
            # Exportar HTML
            pio.write_html(
                fig, 
                file=output_file,
                auto_open=True,
                config={'displayModeBar': False}  # Oculta barra de herramientas
            )
            
            print(f"Gráfico generado en {output_file} (sin labels)")
            
        except Exception as e:
            print(f"Error al visualizar: {str(e)}")


if __name__ == "__main__":
    crawler = WebCrawler()

    crawler.MAX_PAGES_SEED = 30
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
    
    crawler.crawl(
        seed_sites=sitios_iniciales,
    )
    
    crawler.visualize()