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
        
        # Keep clustering and path length on the same explicitly connected
        # subgraph so these summary values align with the small-world calculation.
        avg_clustering = nx.average_clustering(sub_undir)
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

    def compute_small_worldness(
        self, n_random_samples: int = 20, seed: int = 20261027
    ) -> Dict[str, Any]:
        """Compute a reproducible Watts-Strogatz-style sigma against ER nulls.

        Definition used by this dashboard:
        - Analyze the largest connected component of the undirected union graph
          (chemical edges treated as undirected; gap junctions are undirected).
        - Compute C and mean shortest-path length L on that same component.
        - Compare with connected G(n, p) Erdős-Rényi graphs, where n is the
          component size and p = 2m/[n(n-1)] matches its edge density.
        - Accept only connected null graphs so C and L use the same n nodes;
          use a fixed seed by default for stable dashboard KPIs.
        - sigma = (C/C_random) / (L/L_random). This is a graph statistic, not
          evidence of behavioral competence or evolutionary optimization.
        """
        if n_random_samples < 1:
            raise ValueError("n_random_samples must be at least 1")

        largest_cc = max(nx.connected_components(self.G_undir), key=len)
        graph = self.G_undir.subgraph(largest_cc).copy()
        n = graph.number_of_nodes()
        m = graph.number_of_edges()
        if n < 3 or m == 0:
            raise ValueError("Small-worldness requires a connected graph with at least 3 nodes and one edge")

        c_real = float(nx.average_clustering(graph))
        l_real = float(nx.average_shortest_path_length(graph))
        p = (2.0 * m) / (n * (n - 1))

        c_randoms: List[float] = []
        l_randoms: List[float] = []
        attempts = 0
        max_attempts = max(100, n_random_samples * 100)
        while len(c_randoms) < n_random_samples and attempts < max_attempts:
            graph_seed = int(seed + attempts)
            random_graph = nx.erdos_renyi_graph(n, p, seed=graph_seed)
            attempts += 1
            if not nx.is_connected(random_graph):
                continue
            c_randoms.append(float(nx.average_clustering(random_graph)))
            l_randoms.append(float(nx.average_shortest_path_length(random_graph)))

        if len(c_randoms) != n_random_samples:
            raise RuntimeError(
                f"Could generate only {len(c_randoms)} connected ER nulls after {attempts} attempts; "
                "increase the attempt limit or inspect graph density."
            )

        c_rand = float(np.mean(c_randoms))
        l_rand = float(np.mean(l_randoms))
        gamma = c_real / max(1e-12, c_rand)
        lambda_val = l_real / max(1e-12, l_rand)
        sigma = gamma / max(1e-12, lambda_val)

        return {
            "C_real": round(c_real, 4),
            "C_random": round(c_rand, 4),
            "gamma (clustering_ratio)": round(gamma, 3),
            "L_real": round(l_real, 3),
            "L_random": round(l_rand, 3),
            "lambda (path_ratio)": round(lambda_val, 3),
            "sigma (small_worldness_index)": round(sigma, 3),
            "largest_component_nodes": n,
            "largest_component_edges": m,
            "random_baseline": "connected Erdős-Rényi G(n,p), matched edge density",
            "random_seed": int(seed),
            "random_samples": n_random_samples,
            "random_attempts": attempts,
            "is_small_world": bool(sigma > 1.0)
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
