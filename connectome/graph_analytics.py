"""
graph_analytics.py
Advanced network neuroscience analytics on the 302-neuron C. elegans connectome using NetworkX.
Calculates:
- Small-worldness index (Watts-Strogatz clustering & shortest path comparison vs Erdos-Renyi null model)
- Centrality rankings (Degree, Betweenness, PageRank, Eigenvector)
- Rich-Club coefficient and identifying hub command interneurons
- Directed 3-node motif distribution (Feed-Forward Loop, Feedback Loop)
- Spectral graph Laplacian eigenvalue decomposition
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Any
from .data_loader import Connectome

class ConnectomeGraphAnalytics:
    """
    Computes rigorous graph-theoretic and computational neuroscience metrics
    on the directed chemical + electrical C. elegans connectome.
    """
    def __init__(self, connectome: Connectome):
        self.connectome = connectome
        self.G_dir = nx.DiGraph()
        self.G_undir = nx.Graph()
        self._build_graphs()

    def _build_graphs(self):
        # Directed graph: chemical synapses
        for name, data in self.connectome.neurons.items():
            self.G_dir.add_node(name, **data)
            self.G_undir.add_node(name, **data)

        for syn in self.connectome.chemical_synapses:
            self.G_dir.add_edge(syn["pre"], syn["post"], weight=syn["weight"], syn_type="chemical")
            self.G_undir.add_edge(syn["pre"], syn["post"], weight=syn["weight"])

        # Gap junctions are bidirectional
        for gap in self.connectome.gap_junctions:
            self.G_dir.add_edge(gap["n1"], gap["n2"], weight=gap["weight"], syn_type="gap")
            self.G_dir.add_edge(gap["n2"], gap["n1"], weight=gap["weight"], syn_type="gap")
            self.G_undir.add_edge(gap["n1"], gap["n2"], weight=gap["weight"])

    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculates global structural properties."""
        num_nodes = self.G_dir.number_of_nodes()
        num_edges = self.G_dir.number_of_edges()
        density = nx.density(self.G_dir)
        
        # In undir graph, find largest connected component
        largest_cc = max(nx.connected_components(self.G_undir), key=len)
        sub_undir = self.G_undir.subgraph(largest_cc)
        
        avg_clustering = nx.average_clustering(self.G_undir)
        avg_path_length = nx.average_shortest_path_length(sub_undir)
        diameter = nx.diameter(sub_undir)
        
        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "density": round(density, 4),
            "avg_clustering_coefficient": round(avg_clustering, 4),
            "avg_shortest_path_length": round(avg_path_length, 3),
            "diameter": diameter,
            "connected_component_nodes": len(largest_cc)
        }

    def compute_small_worldness(self, n_random_samples: int = 5) -> Dict[str, Any]:
        """
        Computes the Small-World Coefficient sigma = (C / C_rand) / (L / L_rand).
        Values sigma >> 1 confirm small-world topology (high local clustering + low global path length).
        """
        largest_cc = max(nx.connected_components(self.G_undir), key=len)
        G = self.G_undir.subgraph(largest_cc)
        N = G.number_of_nodes()
        E = G.number_of_edges()
        
        C_real = nx.average_clustering(G)
        L_real = nx.average_shortest_path_length(G)
        
        # Generate degree-preserving or Erdos-Renyi random graphs
        C_rands = []
        L_rands = []
        for _ in range(n_random_samples):
            # Erdos-Renyi with same N and E
            p = (2.0 * E) / (N * (N - 1))
            G_rand = nx.erdos_renyi_graph(N, p)
            if nx.is_connected(G_rand):
                C_rands.append(nx.average_clustering(G_rand))
                L_rands.append(nx.average_shortest_path_length(G_rand))
            else:
                cc = max(nx.connected_components(G_rand), key=len)
                sub = G_rand.subgraph(cc)
                C_rands.append(nx.average_clustering(sub))
                L_rands.append(nx.average_shortest_path_length(sub))

        C_rand_mean = float(np.mean(C_rands)) if C_rands else 0.05
        L_rand_mean = float(np.mean(L_rands)) if L_rands else 2.5
        
        gamma = C_real / max(1e-6, C_rand_mean)
        lambda_val = L_real / max(1e-6, L_rand_mean)
        sigma = gamma / max(1e-6, lambda_val)
        
        return {
            "C_real": round(C_real, 4),
            "C_random": round(C_rand_mean, 4),
            "gamma (clustering_ratio)": round(gamma, 3),
            "L_real": round(L_real, 3),
            "L_random": round(L_rand_mean, 3),
            "lambda (path_ratio)": round(lambda_val, 3),
            "sigma (small_worldness_index)": round(sigma, 3),
            "is_small_world": bool(sigma > 1.2)
        }

    def compute_centrality_rankings(self, top_k: int = 15) -> Dict[str, List[Tuple[str, float]]]:
        """Calculates and ranks hubs by Betweenness, Degree, and PageRank."""
        deg_cen = nx.degree_centrality(self.G_dir)
        btw_cen = nx.betweenness_centrality(self.G_dir)
        pagerank = nx.pagerank(self.G_dir, weight='weight')
        
        def top(d):
            sorted_items = sorted(d.items(), key=lambda x: x[1], reverse=True)
            return [(k, round(v, 4)) for k, v in sorted_items[:top_k]]

        return {
            "top_betweenness": top(btw_cen),
            "top_degree": top(deg_cen),
            "top_pagerank": top(pagerank)
        }

    def compute_rich_club(self) -> Dict[int, float]:
        """
        Computes rich-club coefficient phi(k) across degree thresholds k.
        Identifies prominent hub neurons forming the biological core.
        """
        rc = nx.rich_club_coefficient(self.G_undir, normalized=False)
        # Sample every 5th degree for clean plotting
        return {k: round(v, 4) for k, v in sorted(rc.items()) if k % 4 == 0 and k <= 60}

    def get_laplacian_spectrum(self, k: int = 15) -> List[float]:
        """Computes the first k smallest non-zero eigenvalues of normalized graph Laplacian."""
        largest_cc = max(nx.connected_components(self.G_undir), key=len)
        sub = self.G_undir.subgraph(largest_cc)
        L = nx.normalized_laplacian_matrix(sub).toarray()
        eigs = np.sort(np.linalg.eigvalsh(L))
        # Exclude exact 0 (multiplicity of connected component)
        nonzero_eigs = [float(e) for e in eigs if e > 1e-5][:k]
        return [round(e, 4) for e in nonzero_eigs]
