import sys

import pandas as pd
import numpy as np
import itertools
import networkx as nx
import random
import math
from math import log


def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))




def compute(infile, outfile):
    # ---------------- Load Data ----------------
    data = pd.read_csv(infile)
    nodes = list(data.columns)
    np.random.seed(0)

    # ---------------- Mutual Information ----------------
    def mutual_information(x, y):
        joint = data.groupby([x, y]).size().div(len(data))
        px = data[x].value_counts(normalize=True)
        py = data[y].value_counts(normalize=True)
        mi = 0.0
        for (xi, yi), pxy in joint.items():
            mi += pxy * log((pxy + 1e-9) / ((px[xi] * py[yi]) + 1e-9), 2)
        return mi

    # Compute pairwise MI
    edges = []
    for a, b in itertools.combinations(nodes, 2):
        w = mutual_information(a, b)
        edges.append((a, b, w))

    # ---------------- Chow–Liu Tree ----------------
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    T = nx.maximum_spanning_tree(G, weight="weight")

    # Orient edges from arbitrary root
    root = nodes[0]
    dag = nx.DiGraph()
    dag.add_nodes_from(nodes)
    for parent, child in nx.bfs_edges(T, source=root):
        dag.add_edge(parent, child)

    print(f"Initialized Chow–Liu tree with {len(dag.edges())} edges.")

    # ---------------- Bayesian Score Function ----------------
    def bayesian_score(graph):
        score = 0.0
        for node in graph.nodes:
            parents = list(graph.predecessors(node))
            if not parents:
                probs = data[node].value_counts(normalize=True)
                score += np.sum(np.log(probs + 1e-6))
            else:
                joint = data.groupby(parents + [node]).size()
                parent_counts = data.groupby(parents).size()
                for idx, count in joint.items():
                    if len(parents) == 1:
                        parent_total = parent_counts[idx[0]]
                    else:
                        parent_total = parent_counts[idx[:-1]]
                    p = count / parent_total
                    score += np.log(p + 1e-6)
        # Adaptive penalty: increases slightly with node count
        penalty = 0.05 * len(graph.edges()) * np.log(len(data) + len(nodes))
        return score - penalty

    # ---------------- Refinement with Patience ----------------
    best_graph = dag.copy()
    best_score = bayesian_score(best_graph)
    patience = 5  # stop after 5 iterations with no improvement
    patience_counter = 0
    iteration = 0

    print(f"Initial Bayesian score = {best_score:.3f}")

    while patience_counter < patience:
        improved = False
        best_candidate_score = best_score
        best_candidate_graph = None

        for a, b in itertools.permutations(nodes, 2):
            g = best_graph.copy()

            # --- Option 1: Add or remove edge ---
            if g.has_edge(a, b):
                g.remove_edge(a, b)
            else:
                g.add_edge(a, b)
            if nx.is_directed_acyclic_graph(g):
                s = bayesian_score(g)
                if s > best_candidate_score + 1e-6:
                    best_candidate_score = s
                    best_candidate_graph = g

            # --- Option 2: Reverse edge ---
            g = best_graph.copy()
            if g.has_edge(a, b):
                g.remove_edge(a, b)
                g.add_edge(b, a)
                if nx.is_directed_acyclic_graph(g):
                    s = bayesian_score(g)
                    if s > best_candidate_score + 1e-6:
                        best_candidate_score = s
                        best_candidate_graph = g

        if best_candidate_graph is not None:
            best_graph = best_candidate_graph
            best_score = best_candidate_score
            improved = True
            patience_counter = 0
            iteration += 1
            print(f"Refine iter {iteration}: score={best_score:.3f}, edges={len(best_graph.edges())}")
        else:
            patience_counter += 1

    print(f"Refinement complete after {iteration} iterations (no improvement for {patience} rounds).")
    print(f"Final edges = {len(best_graph.edges())}, Final score = {best_score:.3f}")

    # ---------------- Output ----------------
    with open(outfile, "w") as f:
        if not best_graph.edges():
            f.write("# No edges found (empty DAG)\n")
        else:
            for u, v in best_graph.edges():
                f.write(f"{u},{v}\n")

    print(f"✅ Done! Output written to: {outfile}")

def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()
