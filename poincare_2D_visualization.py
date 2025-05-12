import copy
import pickle
from os.path import exists
from gensim.viz.poincare import poincare_2d_visualization
from gensim.models.poincare import PoincareModel
import pandas as pd
from resource.constants import sign_bitcoin_poincare_2D_Visualization, sign_bitcoin_2D_poincare_pkl, \
    sign_bitcoin_poincare_soucre_2D_Visualization, sign_bitcoin_poincare_positive_target_2D_Visualization, \
    sign_bitcoin_poincare_negative_target_2D_Visualization, sign_bitcoin_2D_poincare_label_pkl, \
    sign_bitcoin_2D_poincare_node_colors_pkl, sign_bitcoin_2D_poincare_spectral_clustering_pickel, no_of_clusters, \
    sign_bitcoin_tripartite_graph_pickel_file, sign_bitcoin_2D_poincare_spectral_colors_pickel, \
    sign_bitcoin_poincare_2D_spectral_Visualization, sign_bitcoin_2D_poincare_spectral_clustering_precomputed_pickel, \
    sign_bitcoin_2D_poincare_spectral_colors_precomputed_pickel, \
    sign_bitcoin_poincare_2D_spectral_precomputed_Visualization, \
    sign_bitcoin_2D_poincare_spectral_colors_precomputed_graph_based_pickel, \
    sign_bitcoin_2D_poincare_spectral_clustering_precomputed_graph_based_pickel, \
    sign_bitcoin_poincare_2D_spectral_precomputed_graph_based_Visualization, sign_bitcoin_2D_poincare_DBSCAN_pickel, \
    sign_bitcoin_2D_poincare_DBSCAN_colors_pickel, sign_bitcoin_poincare_2D_DBSCAN_Visualization
from sklearn.cluster import SpectralClustering, DBSCAN
import numpy as np


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

def plot_poincare_2d_visualization(pairs):
    if exists(sign_bitcoin_2D_poincare_pkl) == False:
        model = PoincareModel(pairs, size=2, negative=4)
        model.train(epochs=100)
        save_as_pickle(sign_bitcoin_2D_poincare_pkl, model)
        print("Finished Training")
    else:
        model = read_pickel(sign_bitcoin_2D_poincare_pkl)
    fig = poincare_2d_visualization(model, pairs, "Poincare Hierarchy", show_node_labels=pairs)
    # Extract node labels and assign colors based on prefix
    labels = list(model.kv.index2word)
    save_as_pickle(sign_bitcoin_2D_poincare_label_pkl,labels)
    def get_color(label):
        if label.startswith("N"):
            return 'red'
        elif label.startswith("S"):
            return 'green'
        elif label.startswith("P"):
            return 'blue'
        else:
            return 'gray'  # fallback color

    fig_source = copy.deepcopy(fig)
    fig_positive_target = copy.deepcopy(fig)
    fig_negative_target = copy.deepcopy(fig)

    node_colors = [get_color(label) for label in labels]
    save_as_pickle(sign_bitcoin_2D_poincare_node_colors_pkl, node_colors)
    # Update the marker color in the trace that has both markers and text (the node labels)
    for trace in fig.data:
        trace.marker.color = node_colors
        trace.marker.colorbar = None  # Hide colorbar

    print("Show 2D Image")
    fig.show()
    fig.write_image(sign_bitcoin_poincare_2D_Visualization)

    source_colors = ["green" if get_color(label) == "green" else "gray" for label in labels]
    positive_target_colors = ["blue" if get_color(label) == "blue" else "gray" for label in labels]
    negative_target_colors = ["red" if get_color(label) == "red" else "gray" for label in labels]

    # Update the marker color in the trace that has both markers and text (the node labels)
    for trace in fig.data:
        trace.marker.color = source_colors
        trace.marker.colorbar = None  # Hide colorbar

    print("Show Source Nodes 2D Image")
    fig.show()
    fig.write_image(sign_bitcoin_poincare_soucre_2D_Visualization)

    # Update the marker color in the trace that has both markers and text (the node labels)
    for trace in fig.data:
        trace.marker.color = positive_target_colors
        trace.marker.colorbar = None  # Hide colorbar

    print("Show Positive Target Nodes 2D Image")
    fig.show()
    fig.write_image(sign_bitcoin_poincare_positive_target_2D_Visualization)

    # Update the marker color in the trace that has both markers and text (the node labels)
    for trace in fig.data:
        trace.marker.color = negative_target_colors
        trace.marker.colorbar = None  # Hide colorbar

    print("Show Negative Target Nodes 2D Image")
    fig.show()
    fig.write_image(sign_bitcoin_poincare_negative_target_2D_Visualization)

def spectral_bitcoin(middleDimentions, no_of_clusters, pairs, tripartite_G):
    if exists(sign_bitcoin_2D_poincare_spectral_clustering_pickel):
        clusters = read_pickel(sign_bitcoin_2D_poincare_spectral_clustering_pickel)
    else:
        spectral = SpectralClustering(n_clusters=no_of_clusters, affinity='nearest_neighbors', n_neighbors=50, random_state=0)
        clusters = spectral.fit_predict(middleDimentions)
        save_as_pickle(sign_bitcoin_2D_poincare_spectral_clustering_pickel, clusters)

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
        save_as_pickle(sign_bitcoin_2D_poincare_spectral_colors_pickel, selected_colors)
        
        model = read_pickel(sign_bitcoin_2D_poincare_pkl)
        fig = poincare_2d_visualization(model, pairs, "Poincare Hierarchy", show_node_labels=pairs)
        for trace in fig.data:
            trace.marker.color = selected_colors
            trace.marker.colorbar = None  # Hide colorbar

        fig.show()
        fig.write_image(sign_bitcoin_poincare_2D_spectral_Visualization)


def bitcoin_spectral_poincare(embeddings, pairs):
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
    save_as_pickle(sign_bitcoin_2D_poincare_spectral_clustering_precomputed_pickel, clusters)
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
    save_as_pickle(sign_bitcoin_2D_poincare_spectral_colors_precomputed_pickel, selected_colors)

    model = read_pickel(sign_bitcoin_2D_poincare_pkl)
    fig = poincare_2d_visualization(model, pairs, "Poincare Hierarchy", show_node_labels=pairs)
    for trace in fig.data:
        trace.marker.color = selected_colors
        trace.marker.colorbar = None  # Hide colorbar

    fig.show()
    fig.write_image(sign_bitcoin_poincare_2D_spectral_precomputed_Visualization)


def bitcoin_spectral_graph_based(adjacency_matrix, pairs):
    clustering = SpectralClustering(n_clusters=no_of_clusters, affinity='precomputed')
    clusters = clustering.fit_predict(adjacency_matrix)

    save_as_pickle(sign_bitcoin_2D_poincare_spectral_clustering_precomputed_graph_based_pickel, clusters)
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
    save_as_pickle(sign_bitcoin_2D_poincare_spectral_colors_precomputed_graph_based_pickel, selected_colors)

    model = read_pickel(sign_bitcoin_2D_poincare_pkl)
    fig = poincare_2d_visualization(model, pairs, "Poincare Hierarchy", show_node_labels=pairs)
    for trace in fig.data:
        trace.marker.color = selected_colors
        trace.marker.colorbar = None  # Hide colorbar

    fig.show()
    fig.write_image(sign_bitcoin_poincare_2D_spectral_precomputed_graph_based_Visualization)


def DBSCAN_Clustering(embedding_vectors, pairs):

    def poincare_distance(u, v):
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        euclidean = np.linalg.norm(u - v)
        return np.arccosh(1 + 2 * (euclidean ** 2) / ((1 - norm_u ** 2) * (1 - norm_v ** 2)))

    dbscan = DBSCAN(eps=0.3, min_samples=5, metric=poincare_distance)
    clusters = dbscan.fit_predict(embedding_vectors)

    save_as_pickle(sign_bitcoin_2D_poincare_DBSCAN_pickel, clusters)
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
    save_as_pickle(sign_bitcoin_2D_poincare_DBSCAN_colors_pickel, selected_colors)

    model = read_pickel(sign_bitcoin_2D_poincare_pkl)
    fig = poincare_2d_visualization(model, pairs, "Poincare Hierarchy", show_node_labels=pairs)
    for trace in fig.data:
        trace.marker.color = selected_colors
        trace.marker.colorbar = None  # Hide colorbar

    fig.show()
    fig.write_image(sign_bitcoin_poincare_2D_DBSCAN_Visualization)


bitocin_tripartite = read_pickel(sign_bitcoin_tripartite_graph_pickel_file)

relation_csv = pd.read_csv("edges.csv")
relations = [tuple(x) for x in relation_csv.iloc[:,:2].values.tolist()]

if exists(sign_bitcoin_2D_poincare_pkl) == False and exists(sign_bitcoin_2D_poincare_node_colors_pkl) == False and exists(sign_bitcoin_2D_poincare_label_pkl) == False:
    plot_poincare_2d_visualization(relations)
else:
    middleDimentions = read_pickel(sign_bitcoin_2D_poincare_pkl)
    node_colors = read_pickel(sign_bitcoin_2D_poincare_node_colors_pkl)
    node_labels = read_pickel(sign_bitcoin_2D_poincare_label_pkl)

if exists(sign_bitcoin_2D_poincare_spectral_clustering_pickel) == False:
    spectral_bitcoin(np.array(middleDimentions.kv.vectors), no_of_clusters, relations,bitocin_tripartite)


#bitcoin_spectral_poincare(np.array(middleDimentions.kv.vectors), relations)

#A = nx.to_numpy_array(bitocin_tripartite)

#bitcoin_spectral_graph_based(A, relations)

DBSCAN_Clustering(np.array(middleDimentions.kv.vectors), relations)



