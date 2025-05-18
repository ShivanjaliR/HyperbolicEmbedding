import copy
import pickle
import sys
from os.path import exists
from gensim.viz.poincare import poincare_2d_visualization
from gensim.models.poincare import PoincareModel
import pandas as pd
from sklearn.manifold import TSNE

from resource.twitter_constants import twitter_poincare_2D_Visualization, twitter_2D_poincare_pkl, \
    twitter_poincare_soucre_2D_Visualization, twitter_2D_poincare_label_pkl, \
    twitter_2D_poincare_node_colors_pkl, twitter_2D_poincare_spectral_clustering_pickel, no_of_clusters, \
    twitter_2D_poincare_spectral_colors_pickel, \
    twitter_poincare_2D_spectral_Visualization, twitter_2D_poincare_spectral_clustering_precomputed_pickel, \
    twitter_2D_poincare_spectral_colors_precomputed_pickel, \
    twitter_poincare_2D_spectral_precomputed_Visualization, \
    twitter_2D_poincare_spectral_colors_precomputed_graph_based_pickel, \
    twitter_2D_poincare_spectral_clustering_precomputed_graph_based_pickel, \
    twitter_poincare_2D_spectral_precomputed_graph_based_Visualization, twitter_2D_poincare_DBSCAN_pickel, \
    twitter_2D_poincare_DBSCAN_colors_pickel, twitter_poincare_2D_DBSCAN_Visualization, inputfile, \
    twitter_output_file, twitter_bipartite_graph_pickel_file, twitter_bipartite_graph_name, \
    twitter_bipartite_graph_title, twitter_edges_csv, twitter_graph_info_csv, twitter_original_graph_pickel_file, \
    twitter_poincare_title, twitter_tsne_poincare, twitter_positive_rating_poincare, \
    twitter_poincare_2, twitter_source_node_poincare, twitter_poincare_target_2D_Visualization
from sklearn.cluster import SpectralClustering, DBSCAN
import json
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import lognorm
import networkx as nx



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

def plot_embedding(points, labels, ra):
    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1],
               c=labels,
               marker=".",
               cmap="tab10")
    ax.add_patch(plt.Circle((0, 0), radius=1, edgecolor="b", facecolor="None"))
    ax.axis("square")
    plt.savefig(twitter_poincare_2)

def plot_poincare_2d_visualization(pairs):
    if exists(twitter_2D_poincare_pkl) == False:
        model = PoincareModel(pairs, size=2, negative=4)
        model.train(epochs=100)
        save_as_pickle(twitter_2D_poincare_pkl, model)
        print("Finished Training")
    else:
        model = read_pickel(twitter_2D_poincare_pkl)
    fig = poincare_2d_visualization(model, pairs, "Poincare Hierarchy", show_node_labels=pairs)
    # Extract node labels and assign colors based on prefix
    labels = list(model.kv.index2word)
    save_as_pickle(twitter_2D_poincare_label_pkl, labels)

    def get_color(label):
        if label.startswith("T"):
            return 'red'
        elif label.startswith("S"):
            return 'green'
        else:
            return 'gray'  # fallback color

    fig_source = copy.deepcopy(fig)
    fig_target = copy.deepcopy(fig)

    node_colors = [get_color(label) for label in labels]
    save_as_pickle(twitter_2D_poincare_node_colors_pkl, node_colors)
    # Update the marker color in the trace that has both markers and text (the node labels)
    for trace in fig.data:
        trace.marker.color = node_colors
        trace.marker.colorbar = None  # Hide colorbar

    print("Show 2D Image")
    fig.show()
    fig.write_image(twitter_poincare_2D_Visualization)

    source_colors = ["green" if get_color(label) == "green" else "gray" for label in labels]
    target_colors = ["red" if get_color(label) == "red" else "gray" for label in labels]

    # Update the marker color in the trace that has both markers and text (the node labels)
    for trace in fig.data:
        trace.marker.color = source_colors
        trace.marker.colorbar = None  # Hide colorbar

    print("Show Source Nodes 2D Image")
    fig.show()
    fig.write_image(twitter_poincare_soucre_2D_Visualization)

    # Update the marker color in the trace that has both markers and text (the node labels)
    for trace in fig.data:
        trace.marker.color = target_colors
        trace.marker.colorbar = None  # Hide colorbar

    print("Show Target Nodes 2D Image")
    fig.show()
    fig.write_image(twitter_poincare_target_2D_Visualization)

    return model


def spectral_twitter(middleDimentions, no_of_clusters, pairs, tripartite_G):
    if exists(twitter_2D_poincare_spectral_clustering_pickel):
        clusters = read_pickel(twitter_2D_poincare_spectral_clustering_pickel)
    else:
        spectral = SpectralClustering(n_clusters=no_of_clusters, affinity='nearest_neighbors', n_neighbors=50,
                                      random_state=0)
        clusters = spectral.fit_predict(middleDimentions)
        save_as_pickle(twitter_2D_poincare_spectral_clustering_pickel, clusters)

        colors = [
            "#FF0000",  # Red
            "#00FF00",  # Green
            "#0000FF",  # Blue
            "#FFFF00",  # Yellow
            "#FF00FF",  # Magenta
            "#00FFFF",  # Cyan
            "#800000",  # Maroon
            "#008000",  # Green
            "#000080",  # Navy
            "#808000",  # Olive
            "#800080",  # Purple
            "#008080",  # Teal
            "#FFA500",  # Orange
            "#808080",  # Gray
            "#FFC0CB",  # Pink
            "#008000",  # Green
            "#800000",  # Maroon
            "#000080",  # Navy
            "#008080",  # Teal
            "#800080",  # Purple
            "#FF0000",  # Red
            "#00FF00",  # Green
            "#0000FF",  # Blue
            "#FFFF00",  # Yellow
            "#FF00FF",  # Magenta
        ]

        selected_colors = [colors[i] for i in clusters]
        save_as_pickle(twitter_2D_poincare_spectral_colors_pickel, selected_colors)

        model = read_pickel(twitter_2D_poincare_pkl)
        fig = poincare_2d_visualization(model, pairs, "Poincare Hierarchy", show_node_labels=pairs)
        for trace in fig.data:
            trace.marker.color = selected_colors
            trace.marker.colorbar = None  # Hide colorbar

        fig.show()
        fig.write_image(twitter_poincare_2D_spectral_Visualization)


def twitter_spectral_poincare(embeddings, pairs):
    def poincare_distance(u, v):
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        euclidean = np.linalg.norm(u - v)
        return np.arccosh(1 + 2 * (euclidean ** 2) / ((1 - norm_u ** 2) * (1 - norm_v ** 2)))

    # Example: embeddings is an (n_samples, dim) array of Poincaré embeddings
    def compute_poincare_distance_matrix(embeddings):
        n = embeddings.shape[0]
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = poincare_distance(embeddings[i], embeddings[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
        return dist_matrix

    # 1. Compute distance matrix
    distances = compute_poincare_distance_matrix(embeddings)

    # 2. Convert distances to affinities (e.g., Gaussian kernel)
    sigma = np.mean(distances)
    affinity = np.exp(-distances ** 2 / (2 * sigma ** 2))

    # 3. Perform spectral clustering
    clustering = SpectralClustering(n_clusters=no_of_clusters, affinity='precomputed')
    clusters = clustering.fit_predict(affinity)
    save_as_pickle(twitter_2D_poincare_spectral_clustering_precomputed_pickel, clusters)
    colors = [
        "#FF0000",  # Red
        "#00FF00",  # Green
        "#0000FF",  # Blue
        "#FFFF00",  # Yellow
        "#FF00FF",  # Magenta
        "#00FFFF",  # Cyan
        "#800000",  # Maroon
        "#008000",  # Green
        "#000080",  # Navy
        "#808000",  # Olive
        "#800080",  # Purple
        "#008080",  # Teal
        "#FFA500",  # Orange
        "#808080",  # Gray
        "#FFC0CB",  # Pink
        "#008000",  # Green
        "#800000",  # Maroon
        "#000080",  # Navy
        "#008080",  # Teal
        "#800080",  # Purple
        "#FF0000",  # Red
        "#00FF00",  # Green
        "#0000FF",  # Blue
        "#FFFF00",  # Yellow
        "#FF00FF",  # Magenta
    ]

    selected_colors = [colors[i] for i in clusters]
    save_as_pickle(twitter_2D_poincare_spectral_colors_precomputed_pickel, selected_colors)

    model = read_pickel(twitter_2D_poincare_pkl)
    fig = poincare_2d_visualization(model, pairs, "Poincare Hierarchy", show_node_labels=pairs)
    for trace in fig.data:
        trace.marker.color = selected_colors
        trace.marker.colorbar = None  # Hide colorbar

    fig.show()
    fig.write_image(twitter_poincare_2D_spectral_precomputed_Visualization)


def twitter_spectral_graph_based(adjacency_matrix, pairs):
    clustering = SpectralClustering(n_clusters=no_of_clusters, affinity='precomputed')
    clusters = clustering.fit_predict(adjacency_matrix)

    save_as_pickle(twitter_2D_poincare_spectral_clustering_precomputed_graph_based_pickel, clusters)
    colors = [
        "#FF0000",  # Red
        "#00FF00",  # Green
        "#0000FF",  # Blue
        "#FFFF00",  # Yellow
        "#FF00FF",  # Magenta
        "#00FFFF",  # Cyan
        "#800000",  # Maroon
        "#008000",  # Green
        "#000080",  # Navy
        "#808000",  # Olive
        "#800080",  # Purple
        "#008080",  # Teal
        "#FFA500",  # Orange
        "#808080",  # Gray
        "#FFC0CB",  # Pink
        "#008000",  # Green
        "#800000",  # Maroon
        "#000080",  # Navy
        "#008080",  # Teal
        "#800080",  # Purple
        "#FF0000",  # Red
        "#00FF00",  # Green
        "#0000FF",  # Blue
        "#FFFF00",  # Yellow
        "#FF00FF",  # Magenta
    ]

    selected_colors = [colors[i] for i in clusters]
    save_as_pickle(twitter_2D_poincare_spectral_colors_precomputed_graph_based_pickel, selected_colors)

    model = read_pickel(twitter_2D_poincare_pkl)
    fig = poincare_2d_visualization(model, pairs, "Poincare Hierarchy", show_node_labels=pairs)
    for trace in fig.data:
        trace.marker.color = selected_colors
        trace.marker.colorbar = None  # Hide colorbar

    fig.show()
    fig.write_image(twitter_poincare_2D_spectral_precomputed_graph_based_Visualization)


def DBSCAN_Clustering(embedding_vectors, pairs):
    def poincare_distance(u, v):
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        euclidean = np.linalg.norm(u - v)
        return np.arccosh(1 + 2 * (euclidean ** 2) / ((1 - norm_u ** 2) * (1 - norm_v ** 2)))

    dbscan = DBSCAN(eps=0.3, min_samples=5, metric=poincare_distance)
    clusters = dbscan.fit_predict(embedding_vectors)

    save_as_pickle(twitter_2D_poincare_DBSCAN_pickel, clusters)
    colors = [
        "#FF0000",  # Red
        "#00FF00",  # Green
        "#0000FF",  # Blue
        "#FFFF00",  # Yellow
        "#FF00FF",  # Magenta
        "#00FFFF",  # Cyan
        "#800000",  # Maroon
        "#008000",  # Green
        "#000080",  # Navy
        "#808000",  # Olive
        "#800080",  # Purple
        "#008080",  # Teal
        "#FFA500",  # Orange
        "#808080",  # Gray
        "#FFC0CB",  # Pink
        "#008000",  # Green
        "#800000",  # Maroon
        "#000080",  # Navy
        "#008080",  # Teal
        "#800080",  # Purple
        "#FF0000",  # Red
        "#00FF00",  # Green
        "#0000FF",  # Blue
        "#FFFF00",  # Yellow
        "#FF00FF",  # Magenta
    ]

    selected_colors = [colors[i] for i in clusters]
    save_as_pickle(twitter_2D_poincare_DBSCAN_colors_pickel, selected_colors)

    model = read_pickel(twitter_2D_poincare_pkl)
    fig = poincare_2d_visualization(model, pairs, "Poincare Hierarchy", show_node_labels=pairs)
    for trace in fig.data:
        trace.marker.color = selected_colors
        trace.marker.colorbar = None  # Hide colorbar

    fig.show()
    fig.write_image(twitter_poincare_2D_DBSCAN_Visualization)

def create_original_graph(inList, inWeight, outList, outWeight, usernameList):
    G = nx.DiGraph()
    for trg, in_edges in enumerate(inList):
        for src, wt in zip(in_edges,inWeight[trg]):
            G.add_edge(src,trg,weight=wt)

    for src, out_edges in enumerate(outList):
        for trg, wt in zip(out_edges, outWeight[src]):
            G.add_edge(src, trg, weight=wt)

    return G

def createBipartiteGraph(graph, plot_title, image_name):
    nodes = graph.nodes()
    G = nx.Graph()
    G.add_nodes_from([("S" + str(list(nodes)[i]), {"color": "green"}) for i in range(len(nodes))], bipartite=0)
    G.add_nodes_from([("T" + str(list(nodes)[i]), {"color": "red"}) for i in range(len(nodes))], bipartite=2)
    color_map = [G.nodes()._nodes[list(G.nodes())[i]]['color'] for i in range(len(G.nodes()))]

    for edge in graph.edges():
        src, trg = edge
        G.add_edge("S" + str(trg), "T" + str(src))

    # set figure size
    plt.figure()

    pos = nx.drawing.layout.bipartite_layout(G, list(G.nodes())[0:len(graph.nodes())], scale=10, aspect_ratio=5)
    nx.draw_networkx(G, pos=pos, node_color=color_map, node_size=50)
    plt.savefig(image_name, format="PNG")
    # Show the plot
    # plt.show()
    return G

def tsne_visualization(model, G):
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

    target_x_embedding = []
    target_y_embedding = []
    normal_x_embedding = []
    normal_y_embedding = []
    target_labels = []
    target_colors = []
    normal_labels = []
    normal_colors = []
    for i in range(len(embeddings_2d[:, 0])):
        if colors[i] == "red":
            target_x_embedding.append(embeddings_2d[i, 0])
            target_y_embedding.append(embeddings_2d[i, 1])
            target_labels.append(keys[i])
            target_colors.append("red")
        elif colors[i] == "green":
            normal_x_embedding.append(embeddings_2d[i, 0])
            normal_y_embedding.append(embeddings_2d[i, 1])
            normal_labels.append(keys[i])
            normal_colors.append("green")

    fig, ax2 = plt.subplots(figsize=(6, 6))
    plot_embedding(embeddings_2d, colors, ax2)

    figure = plt.figure(figsize=(15, 15))
    ax = figure.add_subplot(111)
    postive_scatter = ax.scatter(target_x_embedding, target_y_embedding, cmap="Set2", c=target_colors, label="Targets")
    normal_scatter = ax.scatter(normal_x_embedding, normal_y_embedding, cmap="Set2", c=normal_colors, label="Source Nodes")
    plt.title(twitter_poincare_title)
    plt.legend()
    plt.savefig(twitter_tsne_poincare)
    # plt.show()

    figure1 = plt.figure(figsize=(15, 15))
    ax1 = figure1.add_subplot(111)
    source_scatter1 = ax1.scatter(target_x_embedding, target_y_embedding, cmap="Set2", c=target_colors, label="Target Nodes")
    plt.title("Target Nodes Embedding")
    plt.legend()
    plt.savefig(twitter_positive_rating_poincare)
    # plt.show()

    figure3 = plt.figure(figsize=(15, 15))
    ax2 = figure3.add_subplot(111)
    target_scatter1 = ax2.scatter(normal_x_embedding, normal_y_embedding, cmap="Set2", c=normal_colors, label="Source Nodes")
    plt.title("Source Nodes Node Embedding")
    plt.legend()
    plt.savefig(twitter_source_node_poincare)
    # plt.show()

if __name__ == '__main__':
    sys.stdout = open(twitter_output_file, "w+")
    f = open(inputfile)
    data = json.load(f)

    inList = data[0]['inList']
    inWeight = data[0]['inWeight']
    outList = data[0]['outList']
    outWeight = data[0]['outWeight']
    usernameList = data[0]['usernameList']
    if exists(twitter_original_graph_pickel_file) == False:
        G = create_original_graph(inList, inWeight, outList, outWeight, usernameList)
        save_as_pickle(twitter_original_graph_pickel_file,G)
    else:
        G = read_pickel(twitter_original_graph_pickel_file)

    if exists(twitter_bipartite_graph_pickel_file):
        twitter_bipartite_G = read_pickel(twitter_bipartite_graph_pickel_file)
    else:
        twitter_bipartite_G = createBipartiteGraph(G, twitter_bipartite_graph_title, twitter_bipartite_graph_name)
        degree_to_remove = 0
        # Identify and remove nodes with the specified degree
        nodes_to_remove = [node for node in twitter_bipartite_G.nodes if twitter_bipartite_G.degree[node] == degree_to_remove]
        twitter_bipartite_G.remove_nodes_from(nodes_to_remove)

        save_as_pickle(twitter_bipartite_graph_pickel_file, twitter_bipartite_G)
        sign_bitcoin_tripartite_edges = twitter_bipartite_G.edges

        pd.DataFrame(sign_bitcoin_tripartite_edges).to_csv(twitter_edges_csv, encoding='utf-8', float_format='%f')

        source_nodes = []
        zero_target_nodes = []
        zero_source_nodes = []
        target_nodes = []
        for node in twitter_bipartite_G.nodes:
            if str(node).startswith("S"):
                source_nodes.append(node)
                if twitter_bipartite_G.degree(node) == 0:
                    zero_source_nodes.append(node)
            elif str(node).startswith("T"):
                target_nodes.append(node)
                if twitter_bipartite_G.degree(node) == 0:
                    zero_target_nodes.append(node)

        row = []
        row.append("Total No Of Nodes in Directed Graph:")
        row.append(str(len(G.nodes)))
        row.append("Total No Of Edges in Directed Graph:")
        row.append(str(len(G.edges)))
        row.append("Total No of Source_SDM Nodes: ")
        row.append(str(len(source_nodes)))
        row.append("Total No of Target Nodes: ")
        row.append(str(len(target_nodes)))
        row.append("Total No of Edges in Multipartite Graph: ")
        row.append(str(len(twitter_bipartite_G.edges)))
        row.append("Edge Density of Original Multipartite Graph: ")
        row.append(str(len(twitter_bipartite_G.edges) / (len(source_nodes) * len(target_nodes))))
        row.append("Total No Of Zero Degree Source_SDM Nodes:")
        row.append(str(len(zero_source_nodes)))
        row.append("Total No Of Zero Degree Target Nodes:")
        row.append(str(len(zero_target_nodes)))
        row.append("Total No Of Zero Degree Nodes in Multipartite:")
        row.append(str(len(zero_source_nodes) + len(zero_target_nodes)))
        pd.DataFrame(row).to_csv(twitter_graph_info_csv)

    pd.DataFrame(twitter_bipartite_G.edges()).to_csv("twitter_edges.csv", index=False)
    relation_csv = pd.read_csv("twitter_edges.csv")
    relations = [tuple(x) for x in relation_csv.iloc[:, :2].values.tolist()]


    if exists(twitter_2D_poincare_pkl) == False and exists(twitter_2D_poincare_node_colors_pkl) == False and exists(twitter_2D_poincare_label_pkl) == False:
        model = plot_poincare_2d_visualization(relations)
    else:
        model = read_pickel(twitter_2D_poincare_pkl)
        node_colors = read_pickel(twitter_2D_poincare_node_colors_pkl)
        node_labels = read_pickel(twitter_2D_poincare_label_pkl)
        tsne_visualization(model, twitter_bipartite_G)


