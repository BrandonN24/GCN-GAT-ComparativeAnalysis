import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
from torch_geometric.utils import to_networkx
from fa2 import ForceAtlas2

def forceatlas2_layout(G, iterations=500, scaling=10.0, gravity=1.0):
    """
    Compute ForceAtlas2 layout for a NetworkX graph.
    Returns a dict: {node: (x, y)}
    """

    forceatlas2 = ForceAtlas2(
        outboundAttractionDistribution=False,  # default
        linLogMode=False,
        adjustSizes=False,
        edgeWeightInfluence=1.0,

        jitterTolerance=1.0,
        barnesHutOptimize=True,
        barnesHutTheta=1.2,

        scalingRatio=scaling,
        strongGravityMode=False,
        gravity=gravity,

        verbose=False
    )

    # Convert graph to adjacency matrix
    A = nx.to_scipy_sparse_array(G, dtype=float)

    # Run FA2
    positions = forceatlas2.forceatlas2(
        A,
        pos=None,
        iterations=iterations
    )

    # Convert numpy array → dict keyed by node
    pos_dict = {node: positions[i] for i, node in enumerate(G.nodes())}
    return pos_dict

def visualize_graph_labels(data, pred, model_name, dataset_name):
    G = to_networkx(data, to_undirected=True)

    fig, ax1 = plt.subplots(1, 1, figsize=(7, 4))

    # Use fixed layout for consistent visualization across drawings.
    # Use dataset specific layout parameters to improve visualization for different datasets.
    if dataset_name.lower() == "cora":
        # pos = nx.spring_layout(G, seed=42, iterations=400, k=0.30) # k value adjusts the spread of the layout
        pos = forceatlas2_layout(G, iterations=800, scaling=8.0, gravity=1.0)
    elif dataset_name.lower() == "citeseer":
        # pos = nx.spring_layout(G, seed=42, iterations=450, k=0.40) # citeseer needs more repulsion
        pos = forceatlas2_layout(G, iterations=900, scaling=10.0, gravity=1.2)

    else:
        pos = nx.spring_layout(G, seed=42, k=0.35, iterations=400) # default layout

    num_classes = data.y.max().item() + 1
    cmap = plt.cm.Set1.colors[:num_classes]   # slice first N colors
    cmap = matplotlib.colors.ListedColormap(cmap)

    norm = matplotlib.colors.BoundaryNorm(
        boundaries=range(num_classes + 1),
        ncolors=num_classes
    )

    # ---- TRUE LABELS ----
    nx.draw(
        G,
        pos=pos,
        ax=ax1,
        node_color=norm(data.y.cpu()),
        with_labels=False,
        cmap=cmap,
        node_size=150,
        alpha=0.7
    )

    # ---- PREDICTED LABELS ----
    nx.draw(
        G,
        pos=pos,
        ax=ax1,
        node_color=norm(pred.cpu()),
        with_labels=False,
        cmap=cmap,
        node_size=50,
        alpha=0.7
    )

    ax1.set_title(f'{dataset_name} True Labels and Predicted Labels ({model_name})')

    # ---- COLORBAR (attach to ax2) ----

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig.colorbar(sm, ax=ax1, ticks=range(num_classes), label='Predicted Class')

    plt.tight_layout()
    plt.savefig(f'figures/{model_name.replace(" ", "_")}_{dataset_name}_label_visualization.png')
    plt.close()