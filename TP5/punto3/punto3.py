import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, parse_qs
import plotly.graph_objects as go
import networkx as nx
import time
from collections import defaultdict

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
        """Ejecuta el crawling con clasificación de páginas"""
        self.seed_domains = {self.get_domain(url) for url in seed_sites if self.get_domain(url)}
        self.todo_list = [(url, 0, 0, self.get_domain(url)) for url in seed_sites if self.get_domain(url)]
        
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
                    
                    # Mantener el mismo dominio de origen para todas las URLs encontradas
                    new_origin = origin_domain
                    new_l_depth = l_depth + 1
                    new_p_depth = p_depth + (0 if target_domain == current_domain else 1)
                    
                    # Verificar límites basados en el dominio de origen
                    if new_origin in self.seed_domains and self.domain_counts.get(new_origin, 0) >= self.MAX_PAGES_SEED:
                        continue
                        
                    if new_p_depth > self.MAX_PHYSICAL_DEPTH:
                        continue
                        
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
                
                time.sleep(1)

    def visualize(self, output_file="web_graph.html"):
        """Visualización con análisis de profundidades lógicas"""
        if len(self.graph.nodes) == 0:
            print("No hay datos para visualizar")
            return
            
        try:
            import plotly.io as pio
            
            # 1. Análisis de distribución de profundidades
            depth_distribution = defaultdict(int)
            for node in self.graph.nodes:
                depth = self.graph.nodes[node].get('depth', 0)
                depth_distribution[depth] += 1
            
            print("\nDistribución de profundidades lógicas:")
            max_depth = max(depth_distribution.keys()) if depth_distribution else 0
            for depth in range(0, max_depth + 1):
                count = depth_distribution.get(depth, 0)
                print(f"Profundidad {depth}: {count} páginas ({count/len(self.graph.nodes):.1%})")
            
            # 2. Configuración de visualización
            for node in self.graph.nodes():
                if 'domain' not in self.graph.nodes[node]:
                    self.graph.nodes[node]['domain'] = self.get_domain(node) or 'unknown'
                if 'page_type' not in self.graph.nodes[node]:
                    self.graph.nodes[node]['page_type'] = 'static' if self.is_static_page(node) else 'dynamic'
                if 'depth' not in self.graph.nodes[node]:
                    self.graph.nodes[node]['depth'] = 0
            
            pos = nx.spring_layout(self.graph, k=0.7, iterations=100)
            
            # 3. Preparar datos del gráfico
            edge_x, edge_y = [], []
            for edge in self.graph.edges():
                try:
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                except:
                    continue
                    
            node_x, node_y, hover_text, node_colors, node_sizes = [], [], [], [], []
            for node in self.graph.nodes():
                try:
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    
                    # Info para tooltip
                    domain = self.graph.nodes[node]['domain']
                    page_type = self.graph.nodes[node]['page_type']
                    depth = self.graph.nodes[node]['depth']
                    hover_text.append(
                        f"<b>Domain:</b> {domain}<br>"
                        f"<b>Type:</b> {page_type}<br>"
                        f"<b>Depth:</b> {depth}<br>"
                        f"<b>URL:</b> {node[:100]}..."
                    )
                    
                    # Color por tipo de página
                    node_colors.append('#1f77b4' if page_type == 'static' else '#ff7f0e')
                    
                    # Tamaño por profundidad (las más profundas más pequeñas)
                    node_sizes.append(15 - (depth * 2))
                except:
                    continue
            
            # 4. Crear figura con subplots
            from plotly.subplots import make_subplots
            
            fig = make_subplots(
                rows=1, cols=2,
                column_widths=[0.7, 0.3],
                specs=[[{"type": "scatter"}, {"type": "bar"}]]
            )
            
            # Gráfico principal
            fig.add_trace(
                go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color='rgba(136, 136, 136, 0.3)'),
                    hoverinfo='none',
                    mode='lines'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers',
                    hovertext=hover_text,
                    hoverinfo='text',
                    marker=dict(
                        color=node_colors,
                        size=node_sizes,
                        line=dict(width=1, color='DarkSlateGrey')
                    )
                ),
                row=1, col=1
            )
            
            # Gráfico de distribución de profundidades
            depths = sorted(depth_distribution.keys())
            counts = [depth_distribution[d] for d in depths]
            
            fig.add_trace(
                go.Bar(
                    x=depths,
                    y=counts,
                    name="Profundidad",
                    marker_color='#2ca02c',
                    text=counts,
                    textposition='auto'
                ),
                row=1, col=2
            )
            
            # 5. Configuración de layout
            fig.update_layout(
                title_text=f'Análisis de Profundidad (Total: {len(self.graph.nodes)} páginas)',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=60),
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False),
                xaxis2=dict(title='Profundidad Lógica'),
                yaxis2=dict(title='Cantidad de Páginas'),
                plot_bgcolor='white'
            )
            
            # 6. Exportar HTML
            pio.write_html(
                fig, 
                file=output_file,
                auto_open=True,
                config={'displayModeBar': True}
            )
            
            print(f"\nResumen final:")
            print(f"- Páginas estáticas: {len(self.static_pages)}")
            print(f"- Páginas dinámicas: {len(self.dynamic_pages)}")
            print(f"- Profundidad máxima alcanzada: {max_depth}")
            print(f"Gráfico generado en {output_file}")
            
        except Exception as e:
            print(f"Error al visualizar: {str(e)}")


if __name__ == "__main__":
    crawler = WebCrawler()

    crawler.MAX_PAGES_SEED = 300
    crawler.MAX_LOGICAL_DEPTH = 5
    crawler.MAX_PHYSICAL_DEPTH = 1
    sitios_iniciales = [
        "https://www.amazon.com"
    ]
    
    crawler.crawl(
        seed_sites=sitios_iniciales,
    )
    
    crawler.visualize()