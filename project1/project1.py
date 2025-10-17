import sys

import pandas as pd
import numpy as np
import itertools
import networkx as nx
import random
from math import log
from functools import lru_cache

def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))


def compute(infile, outfile):
    data = pd.read_csv(infile)
    nodes = list(data.columns)
    np.random.seed(0)
    random.seed(0)
    n = len(nodes)

    def mutual_information(x, y):
        joint = data.groupby([x, y]).size().div(len(data))
        px = data[x].value_counts(normalize=True)
        py = data[y].value_counts(normalize=True)
        mi = 0.0
        for (xi, yi), pxy in joint.items():
            mi += pxy * log((pxy + 1e-9) / ((px[xi] * py[yi]) + 1e-9), 2)
        return mi

    edges = []
    for a, b in itertools.combinations(nodes, 2):
        w = mutual_information(a, b)
        edges.append((a, b, w))

    edges.sort(key=lambda x: x[2], reverse=True)
    top_pairs = [(a, b) for a, b, _ in edges[: max(5, len(edges)//4)]]

    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    T = nx.maximum_spanning_tree(G, weight="weight")

    root = nodes[0]
    dag = nx.DiGraph()
    dag.add_nodes_from(nodes)
    for parent, child in nx.bfs_edges(T, source=root):
        dag.add_edge(parent, child)

    print(f"Chow–Liu tree with {len(dag.edges())} edges (connected).")


    @lru_cache(maxsize=None)
    def local_score(node, parents_tuple):
        parents = list(parents_tuple)
        if not parents:
            probs = data[node].value_counts(normalize=True)
            return np.sum(np.log(probs + 1e-6))
        joint = data.groupby(parents + [node]).size()
        parent_counts = data.groupby(parents).size()
        s = 0.0
        for idx, count in joint.items():
            parent_total = parent_counts[idx[0]] if len(parents)==1 else parent_counts[idx[:-1]]
            p = count / parent_total
            s += np.log(p + 1e-6)
        penalty = 0.05 * len(parents) * np.log(len(data) + len(nodes))
        return s - penalty

    def full_score(graph):
        return sum(local_score(node, tuple(graph.predecessors(node))) for node in graph.nodes)

    best_graph = dag.copy()
    best_score = full_score(best_graph)
    print(f"Initial score: {best_score:.3f}")

    patience = 4
    patience_counter = 0
    iteration = 0
    max_moves = 100  

    while patience_counter < patience:
        improved = False
        best_candidate_graph = None
        best_candidate_score = best_score

        move_candidates = random.sample(top_pairs, min(len(top_pairs), max_moves))

        for a, b in move_candidates:
            for action in ["add", "remove", "reverse"]:
                g = best_graph.copy()

                if action == "add" and not g.has_edge(a, b):
                    g.add_edge(a, b)
                elif action == "remove" and g.has_edge(a, b):
                    g.remove_edge(a, b)
                elif action == "reverse" and g.has_edge(a, b):
                    g.remove_edge(a, b)
                    g.add_edge(b, a)
                else:
                    continue

                if not nx.is_directed_acyclic_graph(g):
                    continue

                if not nx.is_weakly_connected(g):
                    continue

                delta = 0.0
                if g.has_node(b):
                    new_parents = tuple(g.predecessors(b))
                    old_parents = tuple(best_graph.predecessors(b))
                    delta += local_score(b, new_parents) - local_score(b, old_parents)
                if action == "reverse" and g.has_node(a):
                    new_parents = tuple(g.predecessors(a))
                    old_parents = tuple(best_graph.predecessors(a))
                    delta += local_score(a, new_parents) - local_score(a, old_parents)

                candidate_score = best_score + delta
                if candidate_score > best_candidate_score + 1e-6:
                    best_candidate_score = candidate_score
                    best_candidate_graph = g

        if best_candidate_graph is not None:
            best_graph = best_candidate_graph
            best_score = best_candidate_score
            iteration += 1
            patience_counter = 0
            improved = True
            print(f"{iteration}: score={best_score:.3f}, edges={len(best_graph.edges())}")
        else:
            patience_counter += 1

    if not nx.is_weakly_connected(best_graph):
        comps = list(nx.weakly_connected_components(best_graph))
        for i in range(len(comps) - 1):
            src = random.choice(list(comps[i]))
            dst = random.choice(list(comps[i + 1]))
            best_graph.add_edge(src, dst)
            if not nx.is_directed_acyclic_graph(best_graph):
                best_graph.remove_edge(src, dst)
                best_graph.add_edge(dst, src)
        print("Reconnected all components.")

    print(f"Final edges = {len(best_graph.edges())}, Final score = {best_score:.3f}")
    write_gph(best_graph, {n: n for n in nodes}, outfile)


def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")
    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()
