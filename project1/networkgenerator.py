import sys
import networkx as nx
import matplotlib.pyplot as plt


def visualize_gph(filename):
    edges = []
    with open(filename, "r") as f:
        for line in f:
            parts = [x.strip() for x in line.strip().split(",")]
            if len(parts) == 2:
                edges.append(tuple(parts))

    G = nx.DiGraph(edges)

    pos = nx.spring_layout(G, seed=42, k=1.5, scale=3.0) 

    plt.figure(figsize=(10, 8))
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=(0.6, 0, 0, 0.2),
        edgecolors="black",
        node_size=1800,
        arrows=True,
        arrowsize=25,
        width=2,
        arrowstyle='-|>',
        font_weight="bold",
        font_size=10
    )

    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)
    visualize_gph(sys.argv[1])
