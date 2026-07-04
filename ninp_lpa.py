import collections
import copy
from itertools import groupby
from operator import itemgetter


class NINP_LPA:
    """NINP-LPA: Node-Importance and Node-Group-Proximity enhanced Label Propagation.

    The implementation follows the three-stage pipeline described in the paper:
      1) Importance-guided initialization with structure enhancement (edge add/remove for key nodes).
      2) Label propagation guided by Node-Group Proximity (NGP) with a deterministic update order.
      3) Boundary refinement via neighbor-frequency voting and affiliation-degree tie-breaking.

    Notes on determinism:
      - NetworkX preserves node/neighbor insertion order. This code intentionally iterates over
        adjacency views (e.g., G[u]) rather than Python sets, to keep tie-breaking consistent.
      - Several caches are pre-built in __init__ to avoid repeated expensive set operations.
    """

    def __init__(self, G):
        self.G = G

        # Keep the original node order to make traversal/tie-breaking reproducible.
        self.nodes_list = list(G.nodes())

        # 1) Pre-build neighbor sets for fast intersection and O(1) membership checks.
        self.N_set = {v: set(G.neighbors(v)) for v in self.nodes_list}
        self.deg = {v: len(self.N_set[v]) for v in self.nodes_list}

        # 2) One-pass edge scan to cache:
        #    - Jaccard similarity (used in structure enhancement)
        #    - SNN similarity (used in later stages)
        # edge_cache key: tuple(sorted((u, v))) -> (jaccard, snn)
        self.edge_cache = {}

        # Triangle participation counts (for fast NI computation).
        self.triangle_counts = collections.defaultdict(int)

        # Node strength weight (sum of Jaccard over incident edges), used to select key nodes.
        self.node_strength_weight = collections.defaultdict(float)

        # --- One-pass scan over adjacency (preserve accumulation order as much as possible) ---
        for u in self.nodes_list:
            if self.deg[u] == 0:
                continue

            w_u = 0.0
            # Iterate over G[u] to preserve neighbor order.
            for v in self.G[u]:
                edge_key = (u, v) if u < v else (v, u)

                if edge_key not in self.edge_cache:
                    Nu = self.N_set[u]
                    Nv = self.N_set[v]

                    # Intersection size |Nu ∩ Nv|
                    intersection_size = len(Nu & Nv) if len(Nu) <= len(Nv) else len(Nv & Nu)

                    # Triangle contributions to endpoints.
                    self.triangle_counts[u] += intersection_size
                    self.triangle_counts[v] += intersection_size

                    # Jaccard: |Nu ∩ Nv| / |Nu ∪ Nv|
                    union_size = self.deg[u] + self.deg[v] - intersection_size
                    jaccard = intersection_size / union_size if union_size > 0 else 0.0

                    # SNN (rounded to match the original implementation):
                    # round(((c/deg_u + c/deg_v)/2), 3)
                    if self.deg[u] > 0 and self.deg[v] > 0:
                        snn = (intersection_size / self.deg[u] + intersection_size / self.deg[v]) / 2
                        snn = round(snn, 3)
                    else:
                        snn = 0.0

                    self.edge_cache[edge_key] = (jaccard, snn)

                # Accumulate node strength using cached Jaccard.
                w_u += self.edge_cache[edge_key][0]

            self.node_strength_weight[u] = w_u

        # Label structure: label[node] = [current_label, placeholder, history...]
        self.label = {node: [node, 0] for node in self.nodes_list}

        # Enhanced graph and its caches
        self.G_ = None
        self.N_set_ = None
        self.deg_ = None

    # ------------------  Structure enhancement (Strength Network) ------------------

    def _get_jaccard(self, u, v):
        """Fetch cached Jaccard similarity for an (undirected) node pair."""
        if u == v:
            return 0.0
        key = (u, v) if u < v else (v, u)
        return self.edge_cache.get(key, (0.0, 0.0))[0]

    def _compute_core_link_fast(self, vi, vj):
        """Compute Core_link(vi, vj) following the paper: Common - Single."""
        Ni = self.N_set[vi]
        Nj = self.N_set[vj]

        common = Ni & Nj
        single = Nj - Ni

        inv_deg_i = 1.0 / self.deg[vi] if self.deg[vi] else 0.0
        inv_deg_j = 1.0 / self.deg[vj] if self.deg[vj] else 0.0

        val_common = 0.0
        # Set iteration order is not guaranteed; the original code also iterates sets.
        for vu in common:
            val_common += self._get_jaccard(vi, vu) * inv_deg_i
            val_common += self._get_jaccard(vj, vu) * inv_deg_j

        val_single = 0.0
        for vw in single:
            val_single += self._get_jaccard(vj, vw) * inv_deg_j

        return val_common - val_single

    def run_strength_network(self):
        """Enhance the network structure by editing edges around top-k key nodes."""
        nodes_sorted = sorted(self.nodes_list, key=lambda v: self.node_strength_weight[v], reverse=True)

        k = max(1, int(0.25 * len(nodes_sorted)))
        key_nodes = nodes_sorted[:k]

        self.G_ = copy.deepcopy(self.G)

        for vi in key_nodes:
            # Snapshot current neighbors (preserve iteration order).
            current_neighbors = list(self.G_.neighbors(vi))

            # --- Edge removal ---
            for vj in current_neighbors:
                if self._compute_core_link_fast(vi, vj) < 0 and self.G_.has_edge(vi, vj):
                    self.G_.remove_edge(vi, vj)

            # --- Candidate edge additions (2-hop) ---
            # Build 2-hop candidates using neighbor sets. The final graph is insensitive to
            # the insertion order of edges, but we keep logic faithful to the original.
            com_nodes = set()
            for u in self.N_set[vi]:
                com_nodes.update(self.N_set[u])
            com_nodes.discard(vi)
            com_nodes -= self.N_set[vi]

            for vj in com_nodes:
                if self.G_.degree(vj) == 1:
                    continue
                if self._compute_core_link_fast(vi, vj) > 0 and not self.G_.has_edge(vi, vj):
                    self.G_.add_edge(vi, vj)

        # Build caches for the enhanced graph (used by select_label).
        self.N_set_ = {v: set(self.G_[v].keys()) for v in self.G_}
        self.deg_ = {v: len(self.N_set_[v]) for v in self.G_}
        return self.G_

    # ------------------  Importance & proximity computation ------------------

    def compute_NI_fast(self):
        """Compute normalized node importance NI."""
        NI_raw = {}
        max_ni = 0
        min_ni = float('inf')

        for v in self.nodes_list:
            dv = self.deg[v]
            # triangle_counts sums |N(u) ∩ N(v)| over incident edges;
            # each triangle is counted twice.
            tri_num = self.triangle_counts[v] // 2
            ni = dv + tri_num

            NI_raw[v] = ni
            if ni > max_ni:
                max_ni = ni
            if ni < min_ni:
                min_ni = ni

        denom = max_ni - min_ni
        if denom == 0:
            denom = 1.0

        return {v: (NI_raw[v] - min_ni) / denom for v in self.nodes_list}

    def compute_NNI_fast(self, NI):
        """Compute NNI weights used in NGP-guided propagation."""
        NNI = collections.defaultdict(dict)

        for u in self.nodes_list:
            if self.deg[u] == 1:
                # Keep insertion order by iterating adjacency.
                for v in self.G[u]:
                    NNI[u][v] = 0.0
                continue

            sims = {}
            max_sim = 0.0

            for v in self.G[u]:
                if self.deg[v] == 1:
                    s = 0.0
                else:
                    key = (u, v) if u < v else (v, u)
                    s = self.edge_cache.get(key, (0.0, 0.0))[1]  # cached SNN

                sims[v] = s
                if s > max_sim:
                    max_sim = s

            if max_sim > 0:
                inv_max = 1.0 / max_sim
                for v, s in sims.items():
                    # sqrt( NI[v] * (s / max_s) )
                    NNI[u][v] = (NI[v] * (s * inv_max)) ** 0.5
            else:
                for v in sims:
                    NNI[u][v] = 0.0

        return NNI

    # ------------------ Label initialization & propagation ------------------

    def _snn_dynamic(self, ni_set, nj_set):
        """Dynamic SNN computed on the enhanced graph G_."""
        if not ni_set or not nj_set:
            return 0.0
        common = len(ni_set & nj_set)
        val = (common / len(ni_set) + common / len(nj_set)) / 2
        return round(val, 3)

    def select_label(self, ni):
        """Importance-guided initial label assignment on the enhanced graph."""
        for node in self.nodes_list:
            if self.deg_.get(node, 0) > 1:
                best_nbr = None
                best_sim = -1.0

                node_neighbors_set = self.N_set_[node]

                # Iterate adjacency to preserve stable tie-breaking.
                for neighbor in self.G_[node]:
                    if neighbor == node:
                        continue

                    if self.deg_.get(neighbor, 0) == 1:
                        sim = 0.0
                    else:
                        sim = self._snn_dynamic(node_neighbors_set, self.N_set_[neighbor])

                    if sim > best_sim:
                        best_sim = sim
                        best_nbr = neighbor

                # If all similarities are zero, fall back to the neighbor with maximum degree.
                if best_sim == 0:
                    max_degree_neighbor = max(self.G_[node], key=lambda x: self.deg_.get(x, 0))
                    self.label[node][0] = max_degree_neighbor
                    self.label[node].append([max_degree_neighbor, -1])
                else:
                    if ni[node] > ni[best_nbr]:
                        self.label[node][0] = node
                        self.label[node].append((node, best_nbr))
                    else:
                        self.label[node][0] = best_nbr
                        self.label[node].append((best_nbr, best_nbr))
            else:
                self.label[node][0] = node
                self.label[node].append((node, node))

        # Propagate labels in descending NI order (stable sort).
        seq = sorted(ni, key=ni.get, reverse=True)
        for node in seq:
            try:
                target_node = self.label[node][2][0]
                self.label[node][0] = self.label[target_node][0]
            except IndexError:
                pass

        # Refine labels by neighbor frequency on the enhanced graph.
        for node in seq:
            if node not in self.N_set_:
                continue
            freq = collections.defaultdict(int)
            for neighbor in self.G_[node]:
                lab = self.label[neighbor][0]
                freq[lab] += 1
            if freq:
                self.label[node][0] = max(freq, key=freq.get)

    def label_propagation(self, NNI, v_queue):
        """NGP-guided label propagation on the original graph with a fixed update order."""
        while True:
            change = False

            for node in v_queue:
                if self.deg[node] == 1:
                    continue

                L_Ng = collections.defaultdict(float)
                # Iterate adjacency to preserve insertion order.
                for neighbor in self.G[node]:
                    c = self.label[neighbor][0]
                    w = NNI[node][neighbor]
                    L_Ng[c] += w

                if not L_Ng:
                    continue

                maxv = max(L_Ng.values())
                cands = {c for c, v in L_Ng.items() if v == maxv}

                if maxv == 0:
                    max_degree_neighbor = max(self.G[node], key=lambda x: self.deg.get(x, 0))
                    self.label[node][0] = self.label[max_degree_neighbor][0]
                else:
                    cur = self.label[node][0]

                    # Lazy rule: keep current label if it is among the tied best labels.
                    if cur in cands:
                        continue

                    # Otherwise, break ties by choosing the largest label id.
                    selected = max(cands)
                    self.label[node][0] = selected
                    self.label[node].append((selected, maxv))
                    change = True

            if not change:
                break

        # Post-process degree-1 nodes: inherit label from the only neighbor.
        for node in self.nodes_list:
            if self.deg[node] == 1:
                neighbor = next(iter(self.G[node]))
                self.label[node][0] = self.label[neighbor][0]

    def label_selection(self, node_influence, v_queue):
        """Boundary refinement with frequency voting + influence-based tie-breaking."""
        while True:
            change = False

            for node in v_queue:
                if self.G.degree(node) > 1:
                    # Preserve neighbor order for deterministic grouping.
                    current_node_neighbor = [(j, self.label[j][0]) for j in self.G[node]]

                    sorted_input = sorted(current_node_neighbor, key=itemgetter(1))
                    groups = groupby(sorted_input, key=itemgetter(1))

                    neighbors_frequency = [(lab, sum(1 for _ in grp)) for lab, grp in groups]
                    if not neighbors_frequency:
                        continue

                    temp_max = max(neighbors_frequency, key=itemgetter(1))
                    indices = [i for i, x in enumerate(neighbors_frequency) if x[1] == temp_max[1]]

                    if len(indices) == 1:
                        selected_label = temp_max[0]
                    else:
                        final_labels = [neighbors_frequency[i][0] for i in indices]
                        max_influence = []
                        for lab in final_labels:
                            temp_influence = 1.0
                            for nbr, nbr_lab in current_node_neighbor:
                                if nbr_lab == lab:
                                    temp_influence += node_influence[nbr]
                            max_influence.append((lab, temp_influence))
                        selected_label = max(max_influence, key=itemgetter(1))[0]

                    if selected_label != self.label[node][0]:
                        self.label[node][0] = selected_label
                        change = True

            if not change:
                break

    def get_communities(self):
        """Return communities as a list of node lists."""
        label_groups = collections.defaultdict(list)
        for node in self.nodes_list:
            lab = self.label[node][0]
            label_groups[lab].append(node)
        return list(label_groups.values())

    def execute(self):
        """Run the full NINP-LPA pipeline."""
        self.run_strength_network()
        NI = self.compute_NI_fast()
        NNI = self.compute_NNI_fast(NI)

        self.select_label(NI)
        v_queue = sorted(NI, key=NI.get, reverse=True)

        self.label_propagation(NNI, v_queue)
        self.label_selection(NI, v_queue)

        return self.get_communities()
