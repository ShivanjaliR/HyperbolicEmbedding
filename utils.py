import pickle
from os.path import exists

import networkx as nx
from sklearn.manifold import TSNE
from node2vec import Node2Vec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from resource.constants import sign_bitcoin_positive_rating_bitcoin_node_embedding, \
    sign_bitcoin_negative_rating_bitcoin_node_embedding, sign_bitcoin_source_node_bitcoin_node_embedding, \
    sign_bitcoin_isomap, sign_bitcoin_source_node_isomap, sign_bitcoin_negative_rating_isomap, \
    sign_bitcoin_positive_rating_isomap, sign_bitcoin_isomap_title, sign_bitcoin_isomap_2D, sign_bitcoin_hope_2D, \
    sign_bitcoin_hope_title, sign_bitcoin_source_node_hope, sign_bitcoin_negative_rating_hope, \
    sign_bitcoin_positive_rating_hope, sign_bitcoin_hope, sign_bitcoin_poincare, sign_bitcoin_poincare_title, \
    sign_bitcoin_positive_rating_poincare, sign_bitcoin_negative_rating_poincare, sign_bitcoin_source_node_poincare, \
    sign_bitcoin_poincare_pkl, sign_bitcoin_poincare_1, sign_bitcoin_poincare_2
from sklearn.manifold import Isomap
from gensim.models.poincare import PoincareModel
import seaborn as sns


def read_pickel(filename):
    """
    Read pickel file
    :param filename: name of pickel file name
    :return: fileContent
    """
    fileContent = pickle.load(open(filename, "rb"))

    return fileContent

def save_as_pickle(filename, data):
    """
    Save Graph, Dataset, edges in pickle file
    :param filename: File name where you want to save graph/dataset/edges
    :param data: Data of graph/dataset/edges
    :return: None
    """
    with open(filename, 'wb') as output:
        pickle.dump(data, output)

def create_original_graph(source_nd, target_nd, rating):
    G = nx.DiGraph()
    edges = []
    nodes = []
    for i in range(len(source_nd)):
        edge = []
        edge.append(source_nd[i])
        edge.append(target_nd[i])
        edge.append(int(rating[i]))
        edges.append(edge)
        if nodes.__contains__(source_nd[i]) == False:
            nodes.append(source_nd[i])
        if nodes.__contains__(target_nd[i]) == False:
            nodes.append(target_nd[i])
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(edges)
    A = nx.to_numpy_array(G, weight="weight")
    return G, A

def createTripartiteGraph(graph, plot_title, image_name):
    nodes = graph.nodes()
    G = nx.Graph()
    G.add_nodes_from([("P" + str(list(nodes)[i]), {"color": "blue"}) for i in range(len(nodes))], bipartite=0)
    G.add_nodes_from([("S" + str(list(nodes)[i]), {"color": "green"}) for i in range(len(nodes))], bipartite=1)
    G.add_nodes_from([("N" + str(list(nodes)[i]), {"color": "red"}) for i in range(len(nodes))], bipartite=2)
    color_map = [G.nodes()._nodes[list(G.nodes())[i]]['color'] for i in range(len(G.nodes()))]

    pos_edges = 0
    neg_edges = 0
    for src in graph.adj:
        new_edges = []
        for trg in graph.adj[src]:
            weight = graph.adj[src][trg]['weight']
            # print(str(src) + " ->" + str(trg) + ":" + str(weight))
            if weight > 0:
                new_edges.append(("S" + str(src), "P" + str(trg)))
                pos_edges += 1
            elif weight < 0:
                new_edges.append(("S" + str(src), "N" + str(trg)))
                neg_edges += 1
        G.add_edges_from(new_edges)
    # Convert time difference to minutes
    # set figure size
    plt.figure()

    # Split nodes by bipartite_graph_degree_distribution set for plotting
    nodes1 = {n for n, d in G.nodes(data=True) if d['bipartite'] == 0}
    nodes2 = {n for n, d in G.nodes(data=True) if d['bipartite'] == 1}
    nodes3 = {n for n, d in G.nodes(data=True) if d['bipartite'] == 2}

    pos = {}
    pos.update((node, (1, index)) for index, node in enumerate(nodes1))  # Partition 1
    pos.update((node, (2, index)) for index, node in enumerate(nodes2))  # Partition 2
    pos.update((node, (3, index)) for index, node in enumerate(nodes3))  # Partition 3

    # Draw the graph
    nx.draw_networkx(G, pos, node_color=color_map)
    plt.savefig(image_name, format="PNG")
    # Show the plot
    # plt.show()
    return G, pos_edges, neg_edges


def featureExtraction_tripartite(G, title, plot_file_name):
    model = Node2Vec(G, dimensions=20, walk_length=10, num_walks=50, p=1, q=0.5, workers=1)
    model = model.fit()
    nodes = list(G.nodes())

    node_embeddings = []
    node_labels = []
    keys = []
    key_embedding = []
    colors = []
    indexes = []
    for node, key in zip(G.nodes(), model.wv.index_to_key):
        '''print("node", node)
        print("node vector", model.wv.get_vector(node))
        print("key", key)
        print("key vector", model.wv.get_vector(key)) '''
        node_embeddings.append(model.wv.get_vector(node))
        key_embedding.append(model.wv.get_vector(key))
        colors.append(G.nodes()[key]['color'])
        node_labels.append(node)
        keys.append(key)
        indexes.append(model.wv.key_to_index[key])

    tsne = TSNE(n_components=2, perplexity=25)
    embeddings_2d = tsne.fit_transform(np.array(key_embedding))
    figure = plt.figure(figsize=(15, 15))
    ax = figure.add_subplot(111)

    positive_x_embedding = []
    positive_y_embedding = []
    negative_x_embedding = []
    negative_y_embedding = []
    normal_x_embedding = []
    normal_y_embedding = []
    positive_labels = []
    negative_labels = []
    normal_labels = []
    positive_colors = []
    negative_colors = []
    normal_colors = []
    for i in range(len(embeddings_2d[:, 0])):
        if colors[i] == "blue":
            positive_x_embedding.append(embeddings_2d[i, 0])
            positive_y_embedding.append(embeddings_2d[i, 1])
            positive_labels.append(keys[i])
            positive_colors.append("blue")
        elif colors[i] == "red":
            negative_x_embedding.append(embeddings_2d[i, 0])
            negative_y_embedding.append(embeddings_2d[i, 1])
            negative_labels.append(keys[i])
            negative_colors.append("red")
        elif colors[i] == "green":
            normal_x_embedding.append(embeddings_2d[i, 0])
            normal_y_embedding.append(embeddings_2d[i, 1])
            normal_labels.append(keys[i])
            normal_colors.append("green")

    postive_scatter = ax.scatter(positive_x_embedding, positive_y_embedding, cmap="Set2", c=positive_colors,
                                 label="Positive Ratings")
    negative_scatter = ax.scatter(negative_x_embedding, negative_y_embedding, cmap="Set2", c=negative_colors,
                                  label="Source Ratings")
    normal_scatter = ax.scatter(normal_x_embedding, normal_y_embedding, cmap="Set2", c=normal_colors,
                                label="Source Nodes")
    plt.title(title)
    plt.legend()
    plt.savefig(plot_file_name)
    # plt.show()

    figure1 = plt.figure(figsize=(15, 15))
    ax1 = figure1.add_subplot(111)
    source_scatter1 = ax1.scatter(positive_x_embedding, positive_y_embedding, cmap="Set2", c=positive_colors,
                                  label="Positive_5_Closure Ratings")
    plt.title("Positive_5_Closure Rating Node Embedding")
    plt.legend()
    plt.savefig(sign_bitcoin_positive_rating_bitcoin_node_embedding)
    # plt.show()

    figure2 = plt.figure(figsize=(15, 15))
    ax2 = figure2.add_subplot(111)
    target_scatter1 = ax2.scatter(negative_x_embedding, negative_y_embedding, cmap="Set2", c=negative_colors,
                                  label="Source Ratings")
    plt.title("Source Rating Node Embedding")
    plt.legend()
    plt.savefig(sign_bitcoin_negative_rating_bitcoin_node_embedding)
    # plt.show()

    figure3 = plt.figure(figsize=(15, 15))
    ax2 = figure3.add_subplot(111)
    target_scatter1 = ax2.scatter(normal_x_embedding, normal_y_embedding, cmap="Set2", c=normal_colors,
                                  label="Source Nodes")
    plt.title("Source Nodes Node Embedding")
    plt.legend()
    plt.savefig(sign_bitcoin_source_node_bitcoin_node_embedding)
    # plt.show()

    key_embedding = pd.DataFrame(key_embedding, index=keys, columns=range(0, 20))

    return key_embedding, indexes, model.wv


def key_func(s):
    # extract the numeric part from the string
    num = int(s[1:])
    # separate the prefix (S or T) and the numeric part
    prefix = s[0]
    return (prefix, num)

def IsomapEmbedding(A, G):
    # Apply ISOMAP

    n_neighbors = 8
    n_components = 2
    isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
    X_transformed = isomap.fit_transform(A)
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], marker='o')
    plt.title('ISOMAP Dimensionality Reduction')
    plt.savefig(sign_bitcoin_isomap_2D)

    n_neighbors = 8
    n_components = 20
    isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
    X_transformed = isomap.fit_transform(A)

    tsne = TSNE(n_components=2, perplexity=25)
    embeddings_2d = tsne.fit_transform(np.array(X_transformed))
    figure = plt.figure(figsize=(15, 15))
    ax = figure.add_subplot(111)

    isomap_embedding = pd.DataFrame(embeddings_2d, index=list(G.nodes))
    node_embeddings = []
    node_labels = []
    keys = []
    colors = []
    for node in G.nodes():
        node_embeddings.append(isomap_embedding.loc[node])
        colors.append(G.nodes()[node]['color'])
        node_labels.append(node)
        keys.append(node)

    positive_x_embedding = []
    positive_y_embedding = []
    negative_x_embedding = []
    negative_y_embedding = []
    normal_x_embedding = []
    normal_y_embedding = []
    positive_labels = []
    negative_labels = []
    normal_labels = []
    positive_colors = []
    negative_colors = []
    normal_colors = []
    for i in range(len(embeddings_2d[:, 0])):
        if colors[i] == "blue":
            positive_x_embedding.append(embeddings_2d[i, 0])
            positive_y_embedding.append(embeddings_2d[i, 1])
            positive_labels.append(keys[i])
            positive_colors.append("blue")
        elif colors[i] == "red":
            negative_x_embedding.append(embeddings_2d[i, 0])
            negative_y_embedding.append(embeddings_2d[i, 1])
            negative_labels.append(keys[i])
            negative_colors.append("red")
        elif colors[i] == "green":
            normal_x_embedding.append(embeddings_2d[i, 0])
            normal_y_embedding.append(embeddings_2d[i, 1])
            normal_labels.append(keys[i])
            normal_colors.append("green")

    figure = plt.figure(figsize=(15, 15))
    ax = figure.add_subplot(111)
    postive_scatter = ax.scatter(positive_x_embedding, positive_y_embedding, cmap="Set2", c=positive_colors, label="Positive Ratings")
    negative_scatter = ax.scatter(negative_x_embedding, negative_y_embedding, cmap="Set2", c=negative_colors, label="Negative Ratings")
    normal_scatter = ax.scatter(normal_x_embedding, normal_y_embedding, cmap="Set2", c=normal_colors, label="Source Nodes")
    plt.title(sign_bitcoin_isomap_title)
    plt.legend()
    plt.savefig(sign_bitcoin_isomap)
    # plt.show()

    figure1 = plt.figure(figsize=(15, 15))
    ax1 = figure1.add_subplot(111)
    source_scatter1 = ax1.scatter(positive_x_embedding, positive_y_embedding, cmap="Set2", c=positive_colors,label="Positive Ratings")
    plt.title("Positive Rating Node Embedding")
    plt.legend()
    plt.savefig(sign_bitcoin_positive_rating_isomap)
    # plt.show()

    figure2 = plt.figure(figsize=(15, 15))
    ax2 = figure2.add_subplot(111)
    target_scatter1 = ax2.scatter(negative_x_embedding, negative_y_embedding, cmap="Set2", c=negative_colors, label="Negative Ratings")
    plt.title("Negative Rating Node Embedding")
    plt.legend()
    plt.savefig(sign_bitcoin_negative_rating_isomap)
    # plt.show()

    figure3 = plt.figure(figsize=(15, 15))
    ax2 = figure3.add_subplot(111)
    target_scatter1 = ax2.scatter(normal_x_embedding, normal_y_embedding, cmap="Set2", c=normal_colors, label="Source Nodes")
    plt.title("Source Nodes Node Embedding")
    plt.legend()
    plt.savefig(sign_bitcoin_source_node_isomap)
    # plt.show()

    return X_transformed

def hope_display(hope_embedding, G):
    tsne = TSNE(n_components=2, perplexity=25)
    embeddings = pd.DataFrame(hope_embedding).transpose()
    embeddings_2d = tsne.fit_transform(np.array(embeddings))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], marker='o')
    plt.title('HOPE Dimensionality Reduction')
    plt.savefig(sign_bitcoin_hope_2D)

    figure = plt.figure(figsize=(15, 15))
    ax = figure.add_subplot(111)

    hope_embedding = pd.DataFrame(embeddings_2d, index=list(G.nodes))
    node_embeddings = []
    node_labels = []
    keys = []
    colors = []
    for node in G.nodes():
        node_embeddings.append(hope_embedding.loc[node])
        colors.append(G.nodes()[node]['color'])
        node_labels.append(node)
        keys.append(node)

    positive_x_embedding = []
    positive_y_embedding = []
    negative_x_embedding = []
    negative_y_embedding = []
    normal_x_embedding = []
    normal_y_embedding = []
    positive_labels = []
    negative_labels = []
    normal_labels = []
    positive_colors = []
    negative_colors = []
    normal_colors = []
    for i in range(len(embeddings_2d[:, 0])):
        if colors[i] == "blue":
            positive_x_embedding.append(embeddings_2d[i, 0])
            positive_y_embedding.append(embeddings_2d[i, 1])
            positive_labels.append(keys[i])
            positive_colors.append("blue")
        elif colors[i] == "red":
            negative_x_embedding.append(embeddings_2d[i, 0])
            negative_y_embedding.append(embeddings_2d[i, 1])
            negative_labels.append(keys[i])
            negative_colors.append("red")
        elif colors[i] == "green":
            normal_x_embedding.append(embeddings_2d[i, 0])
            normal_y_embedding.append(embeddings_2d[i, 1])
            normal_labels.append(keys[i])
            normal_colors.append("green")

    figure = plt.figure(figsize=(15, 15))
    ax = figure.add_subplot(111)
    postive_scatter = ax.scatter(positive_x_embedding, positive_y_embedding, cmap="Set2", c=positive_colors,
                                 label="Positive Ratings")
    negative_scatter = ax.scatter(negative_x_embedding, negative_y_embedding, cmap="Set2", c=negative_colors,
                                  label="Negative Ratings")
    normal_scatter = ax.scatter(normal_x_embedding, normal_y_embedding, cmap="Set2", c=normal_colors,
                                label="Source Nodes")
    plt.title(sign_bitcoin_hope_title)
    plt.legend()
    plt.savefig(sign_bitcoin_hope)
    # plt.show()

    figure1 = plt.figure(figsize=(15, 15))
    ax1 = figure1.add_subplot(111)
    source_scatter1 = ax1.scatter(positive_x_embedding, positive_y_embedding, cmap="Set2", c=positive_colors,
                                  label="Positive Ratings")
    plt.title("Positive Rating Node Embedding")
    plt.legend()
    plt.savefig(sign_bitcoin_positive_rating_hope)
    # plt.show()

    figure2 = plt.figure(figsize=(15, 15))
    ax2 = figure2.add_subplot(111)
    target_scatter1 = ax2.scatter(negative_x_embedding, negative_y_embedding, cmap="Set2", c=negative_colors,
                                  label="Negative Ratings")
    plt.title("Negative Rating Node Embedding")
    plt.legend()
    plt.savefig(sign_bitcoin_negative_rating_hope)
    # plt.show()

    figure3 = plt.figure(figsize=(15, 15))
    ax2 = figure3.add_subplot(111)
    target_scatter1 = ax2.scatter(normal_x_embedding, normal_y_embedding, cmap="Set2", c=normal_colors,
                                  label="Source Nodes")
    plt.title("Source Nodes Node Embedding")
    plt.legend()
    plt.savefig(sign_bitcoin_source_node_hope)
    # plt.show()

def poincareEmbedding1(G):
    '''edges = pd.DataFrame(G.edges()).to_csv("edges.csv", index=False)
    relations = pd.read_csv("edges.csv")
    relations = PoincareRelations(relations.iloc[:, :2])
    model = PoincareModel(relations, negative=1)
    model.train(epochs=50)'''
    # train a new model from initial data
    initial_relations = [('kangaroo', 'marsupial'), ('kangaroo', 'mammal')]
    model = PoincareModel(initial_relations, negative=1)
    model.train(epochs=50)
    # online training: update the vocabulary and continue training
    online_relations = [('striped_skunk', 'mammal')]
    model.build_vocab(online_relations, update=True)
    model.train(epochs=50)

    embeddings = model.kv
    tsne = TSNE(n_components=2, perplexity=3)
    embeddings_2d = tsne.fit_transform(np.array(embeddings.vectors))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], marker='o')
    plt.title('Poincare Dimensionality Reduction')
    plt.savefig(sign_bitcoin_poincare)

    return embeddings

c_elegans_palette = {
    'ABarpaaa_lineage': '#91003f',  # embryonic lineage
    'Germline': '#7f2704',
    # Somatic gonad precursor cell
    'Z1_Z4': '#800026',

    # Two embryonic hypodermal cells that may provide a scaffold for the early organization of ventral bodywall muscles
    'XXX': '#fb8072',

    'Ciliated_amphid_neuron': '#c51b8a', 'Ciliated_non_amphid_neuron': '#fa9fb5',

    # immune
    'Coelomocyte': '#ffff33', 'T': '#54278f',

    # Exceratory
    'Excretory_cell': '#004529',
    'Excretory_cell_parent': '#006837',
    'Excretory_duct_and_pore': '#238443',
    'Parent_of_exc_duct_pore_DB_1_3': '#41ab5d',
    'Excretory_gland': '#78c679',
    'Parent_of_exc_gland_AVK': '#addd8e',
    'Rectal_cell': '#d9f0a3',
    'Rectal_gland': '#f7fcb9',
    'Intestine': '#7fcdbb',

    # esophagus, crop, gizzard (usually) and intestine
    'Pharyngeal_gland': '#fed976',
    'Pharyngeal_intestinal_valve': '#feb24c',
    'Pharyngeal_marginal_cell': '#fd8d3c',
    'Pharyngeal_muscle': '#fc4e2a',
    'Pharyngeal_neuron': '#e31a1c',

    # hypodermis (epithelial)
    'Parent_of_hyp1V_and_ant_arc_V': '#a8ddb5',
    'hyp1V_and_ant_arc_V': '#ccebc5',
    'Hypodermis': '#253494',
    'Seam_cell': '#225ea8',
    'Arcade_cell': '#1d91c0',

    # set of six cells that form a thin cylindrical sheet between pharynx and ring neuropile
    'GLR': '#1f78b4',

    # Glia, also called glial cells or neuroglia, are non-neuronal cells in the central nervous system
    'Glia': '#377eb8',

    # head mesodermal cell: the middle layer of cells or tissues of an embryo
    'Body_wall_muscle': '#9e9ac8',
    'hmc': '#54278f',
    'hmc_and_homolog': '#02818a',
    'hmc_homolog': '#bcbddc',
    'Intestinal_and_rectal_muscle': '#41b6c4',
    # Postembryonic mesoblast: the mesoderm of an embryo in its earliest stages.
    'M_cell': '#3f007d',

    # pharyngeal gland cel
    'G2_and_W_blasts': '#abdda4',

    'red': 'red',
    'blue': 'blue',
    'green': 'green',
    'unannotated': '#969696',
    'not provided': '#969696'
}

def plot_embedding(points, labels, ra):
    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1],
               c=labels,
               marker=".",
               cmap="tab10")
    ax.add_patch(plt.Circle((0, 0), radius=1, edgecolor="b", facecolor="None"))
    ax.axis("square")
    plt.savefig(sign_bitcoin_poincare_2)

def plot_embedding2(points, labels, ax):
    df = pd.DataFrame({"x": points[:, 0], "y": points[:, 1], "labels": labels})
    sns.scatterplot(data=df, x="x", y="y", hue="labels", s=20, palette="Set2")
    plt.gca().add_patch(plt.Circle((0, 0), 1, color='black', fill=False))
    plt.axis('equal')
    plt.axis('off')
    plt.title("Labeled Poincaré Embedding")
    plt.savefig(sign_bitcoin_poincare_1)


def plot_embedding1(points, labels, ax):

    df = pd.DataFrame({"x": points[:, 0], "y": points[:, 1], "labels":labels})

    point_size = 2
    font_size = 5
    alpha = 1.0

    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="labels",
        hue_order=np.unique(labels),
        palette=c_elegans_palette,
        alpha=alpha,
        edgecolor="none",
        ax=ax,
        s=point_size,
        legend=False
    )

    circle = plt.Circle((0, 0), radius=1, fc='none', color='black', zorder=0)
    ax.add_patch(circle)
    ax.plot(0, 0, '.', c=(0, 0, 0), ms=4)

    # fig.tight_layout()
    ax.axis('off')
    ax.axis('equal')

    ax.set_ylim([-0.94, 0.94])
    ax.set_xlim([-0.94, 0.94])
    plt.savefig("poincare_hyperbolic.png")



def poincareEmbedding(G):
    if exists(sign_bitcoin_poincare_pkl):
        model = read_pickel(sign_bitcoin_poincare_pkl)
    else:
        edges = pd.DataFrame(G.edges()).to_csv("edges.csv", index=False)
        relations_csv = pd.read_csv("edges.csv")
        #relations = PoincareRelations(relations_csv.iloc[:, :2])
        relations = [tuple(x) for x in relations_csv.iloc[:, :2].values.tolist()]
        model = PoincareModel(relations, negative=1)
        model.train(epochs=50)
        save_as_pickle(sign_bitcoin_poincare_pkl, model)
        # train a new model from initial data
        #initial_relations = [('kangaroo', 'marsupial'), ('kangaroo', 'mammal')]
        #model = PoincareModel(initial_relations, negative=1)
        #model.train(epochs=50)

        # online training: update the vocabulary and continue training
        #online_relations = [('striped_skunk', 'mammal')]
        #model.build_vocab(online_relations, update=True)
        #model.train(epochs=50)

    embeddings = model.kv
    tsne = TSNE(n_components=2, perplexity=3)
    embeddings_2d = tsne.fit_transform(np.array(embeddings.vectors))

    figure = plt.figure(figsize=(15, 15))
    ax = figure.add_subplot(111)

    hope_embedding = pd.DataFrame(embeddings_2d, index=list(G.nodes))
    node_embeddings = []
    node_labels = []
    keys = []
    colors = []
    for node in G.nodes():
        node_embeddings.append(hope_embedding.loc[node])
        colors.append(G.nodes()[node]['color'])
        node_labels.append(node)
        keys.append(node)

    positive_x_embedding = []
    positive_y_embedding = []
    negative_x_embedding = []
    negative_y_embedding = []
    normal_x_embedding = []
    normal_y_embedding = []
    positive_labels = []
    negative_labels = []
    normal_labels = []
    positive_colors = []
    negative_colors = []
    normal_colors = []
    for i in range(len(embeddings_2d[:, 0])):
        if colors[i] == "blue":
            positive_x_embedding.append(embeddings_2d[i, 0])
            positive_y_embedding.append(embeddings_2d[i, 1])
            positive_labels.append(keys[i])
            positive_colors.append("blue")
        elif colors[i] == "red":
            negative_x_embedding.append(embeddings_2d[i, 0])
            negative_y_embedding.append(embeddings_2d[i, 1])
            negative_labels.append(keys[i])
            negative_colors.append("red")
        elif colors[i] == "green":
            normal_x_embedding.append(embeddings_2d[i, 0])
            normal_y_embedding.append(embeddings_2d[i, 1])
            normal_labels.append(keys[i])
            normal_colors.append("green")

    fig, ax2 = plt.subplots(figsize=(6, 6))
    plot_embedding(embeddings_2d, colors, ax2)

    figure = plt.figure(figsize=(15, 15))
    ax = figure.add_subplot(111)
    postive_scatter = ax.scatter(positive_x_embedding, positive_y_embedding, cmap="Set2", c=positive_colors,
                                 label="Positive Ratings")
    negative_scatter = ax.scatter(negative_x_embedding, negative_y_embedding, cmap="Set2", c=negative_colors,
                                  label="Negative Ratings")
    normal_scatter = ax.scatter(normal_x_embedding, normal_y_embedding, cmap="Set2", c=normal_colors,
                                label="Source Nodes")
    plt.title(sign_bitcoin_poincare_title)
    plt.legend()
    plt.savefig(sign_bitcoin_poincare)
    # plt.show()

    figure1 = plt.figure(figsize=(15, 15))
    ax1 = figure1.add_subplot(111)
    source_scatter1 = ax1.scatter(positive_x_embedding, positive_y_embedding, cmap="Set2", c=positive_colors,
                                  label="Positive Ratings")
    plt.title("Positive Rating Node Embedding")
    plt.legend()
    plt.savefig(sign_bitcoin_positive_rating_poincare)
    # plt.show()

    figure2 = plt.figure(figsize=(15, 15))
    ax2 = figure2.add_subplot(111)
    target_scatter1 = ax2.scatter(negative_x_embedding, negative_y_embedding, cmap="Set2", c=negative_colors,
                                  label="Negative Ratings")
    plt.title("Negative Rating Node Embedding")
    plt.legend()
    plt.savefig(sign_bitcoin_negative_rating_poincare)
    # plt.show()

    figure3 = plt.figure(figsize=(15, 15))
    ax2 = figure3.add_subplot(111)
    target_scatter1 = ax2.scatter(normal_x_embedding, normal_y_embedding, cmap="Set2", c=normal_colors,
                                  label="Source Nodes")
    plt.title("Source Nodes Node Embedding")
    plt.legend()
    plt.savefig(sign_bitcoin_source_node_poincare)
    # plt.show()

    #plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], marker='o')
    #plt.title('Poincare Dimensionality Reduction')
    #plt.savefig(sign_bitcoin_poincare)

    return embeddings



