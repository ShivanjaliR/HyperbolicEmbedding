'''
https://www.kaggle.com/datasets/shichenyang/citeseer?select=nodes.csv
'''
import copy
import pickle
import sys
import pandas as pd
import networkx as nx
import numpy as np
from os.path import exists
from gensim.models.poincare import PoincareModel
from gensim.viz.poincare import poincare_2d_visualization
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from resource.citeseer_citation_constants import citeseer_citation_original_graph_pkl, \
    citeseer_citation_csv_node_inputfile, citeseer_citation_csv_edges_inputfile, citeseer_citation_bipartite_graph_pkl, \
    citeseer_citation_bipartite_edges_csv, citeseer_citation_bipartite_adj_matrix_csv, \
    citeseer_citation_bipartite_graph_name, citeseer_citation_network_graph, citeseer_citation_network_title, \
    citeseer_citation_bipartite_title, original_citeseer_citation_graph_adj_matrix_csv, \
    node_to_label_map_pkl, citeseer_output_file, citeseer_bipartite_graph_edge_density_info, citeseer_2D_poincare_pkl, \
    citeseer_poincare_2D_Visualization, citeseer_poincare_soucre_2D_Visualization, \
    citeseer_poincare_target_2D_Visualization, citeseer_poincare_title, citeseer_tsne_poincare, \
    citeseer_target_poincare, citeseer_source_node_poincare, citeseer_poincare_2


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

def createDirectedGraph(nodeId, source_id, target_id):
    G = nx.DiGraph()
    G.add_nodes_from(nodeId)

    for i in range(len(source_id)):
        if G.nodes().__contains__(source_id[i]) == False:
            G.add_node(source_id[i])
        if G.nodes().__contains__(target_id[i]) == False:
            G.add_node(target_id[i])
        G.add_edge(source_id[i], target_id[i])
    A = nx.adjacency_matrix(G)
    adj_matrix = pd.DataFrame(A.toarray(), columns=np.array(G.nodes()), index=np.array(G.nodes()))
    adj_matrix.to_csv(original_citeseer_citation_graph_adj_matrix_csv, encoding='utf-8')
    # set figure size
    plt.figure(figsize=(10, 10))

    # define position of nodes in figure
    # pos = nx.spring_layout(G)
    pos = nx.circular_layout(G)

    # draw nodes and edges
    nx.draw_networkx(G, pos=pos, with_labels=False)

    # plot the title (if any)
    plt.title(citeseer_citation_network_title)
    plt.savefig(citeseer_citation_network_graph)
    plt.show()
    return G


def createBipartiteGraph(graph, bipartite_graph_name):
    edges = graph.edges()
    nodes = graph.nodes()
    G = nx.Graph()
    G.add_nodes_from([("S" + str(list(nodes)[i]), {"color": "blue"}) for i in range(len(nodes))], bipartite=0)
    G.add_nodes_from([("T" + str(list(nodes)[i]), {"color": "red"}) for i in range(len(nodes))], bipartite=1)

    color_map = [G.nodes()._nodes[list(G.nodes())[i]]['color'] for i in range(len(G.nodes()))]
    new_edges = []
    for edge in edges:
        source = "S" + str(edge[0])
        target = "T" + str(edge[1])
        new_edges.append((source, target))
    G.add_edges_from(new_edges)

    # set figure size
    plt.figure()

    # define position of nodes in figure
    # pos = nx.nx_agraph.graphviz_layout(G)
    # pos = nx.spring_layout(G)

    # draw nodes and edges
    nodes_list = np.array(pd.DataFrame(G.nodes).keys())
    pos = nx.drawing.layout.bipartite_layout(G, list(G.nodes())[0:len(graph.nodes())], scale=10, aspect_ratio=5)
    nx.draw_networkx(G, pos=pos, node_color=color_map, node_size=50)
    # plot the title (if any)
    plt.title(citeseer_citation_bipartite_title)
    plt.savefig(citeseer_citation_bipartite_graph_name, format="PNG")
    plt.show()
    return G

def plot_poincare_2d_visualization(pairs):
    if exists(citeseer_2D_poincare_pkl) == False:
        model = PoincareModel(pairs, size=2, negative=4)
        model.train(epochs=100)
        save_as_pickle(citeseer_2D_poincare_pkl, model)
        print("Finished Training")
    else:
        model = read_pickel(citeseer_2D_poincare_pkl)
    fig = poincare_2d_visualization(model, pairs, "Poincare Hierarchy", show_node_labels=pairs)
    # Extract node labels and assign colors based on prefix
    labels = list(model.kv.index2word)

    def get_color(label):
        if label.startswith("T"):
            return 'red'
        elif label.startswith("S"):
            return 'blue'
        else:
            return 'gray'  # fallback color

    fig_source = copy.deepcopy(fig)
    fig_target = copy.deepcopy(fig)

    node_colors = [get_color(label) for label in labels]

    # Update the marker color in the trace that has both markers and text (the node labels)
    for trace in fig.data:
        trace.marker.color = node_colors
        trace.marker.colorbar = None  # Hide colorbar

    print("Show 2D Image")
    fig.show()
    fig.write_image(citeseer_poincare_2D_Visualization)

    source_colors = ["blue" if get_color(label) == "blue" else "gray" for label in labels]
    target_colors = ["red" if get_color(label) == "red" else "gray" for label in labels]

    # Update the marker color in the trace that has both markers and text (the node labels)
    for trace in fig.data:
        trace.marker.color = source_colors
        trace.marker.colorbar = None  # Hide colorbar

    print("Show Source Nodes 2D Image")
    fig.show()
    fig.write_image(citeseer_poincare_soucre_2D_Visualization)

    # Update the marker color in the trace that has both markers and text (the node labels)
    for trace in fig.data:
        trace.marker.color = target_colors
        trace.marker.colorbar = None  # Hide colorbar

    print("Show Positive Target Nodes 2D Image")
    fig.show()
    fig.write_image(citeseer_poincare_target_2D_Visualization)

def plot_embedding(points, labels, ra):
    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1],
               c=labels,
               marker=".",
               cmap="tab10")
    ax.add_patch(plt.Circle((0, 0), radius=1, edgecolor="b", facecolor="None"))
    ax.axis("square")
    plt.savefig(citeseer_poincare_2)
    
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
        elif colors[i] == "blue":
            normal_x_embedding.append(embeddings_2d[i, 0])
            normal_y_embedding.append(embeddings_2d[i, 1])
            normal_labels.append(keys[i])
            normal_colors.append("blue")

    fig, ax2 = plt.subplots(figsize=(6, 6))
    plot_embedding(embeddings_2d, colors, ax2)

    figure = plt.figure(figsize=(15, 15))
    ax = figure.add_subplot(111)
    postive_scatter = ax.scatter(target_x_embedding, target_y_embedding, cmap="Set2", c=target_colors, label="Targets")
    normal_scatter = ax.scatter(normal_x_embedding, normal_y_embedding, cmap="Set2", c=normal_colors, label="Source Nodes")
    plt.title(citeseer_poincare_title)
    plt.legend()
    plt.savefig(citeseer_tsne_poincare)
    # plt.show()

    figure1 = plt.figure(figsize=(15, 15))
    ax1 = figure1.add_subplot(111)
    source_scatter1 = ax1.scatter(target_x_embedding, target_y_embedding, cmap="Set2", c=target_colors, label="Target Nodes")
    plt.title("Target Nodes Embedding")
    plt.legend()
    plt.savefig(citeseer_target_poincare)
    # plt.show()

    figure3 = plt.figure(figsize=(15, 15))
    ax2 = figure3.add_subplot(111)
    target_scatter1 = ax2.scatter(normal_x_embedding, normal_y_embedding, cmap="Set2", c=normal_colors, label="Source Nodes")
    plt.title("Source Nodes Node Embedding")
    plt.legend()
    plt.savefig(citeseer_source_node_poincare)
    # plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sys.stdout = open(citeseer_output_file, "w+")
    if exists(citeseer_citation_original_graph_pkl):
        original_citation_graph = read_pickel(citeseer_citation_original_graph_pkl)
        node_label_map = read_pickel(node_to_label_map_pkl)
    else:
        node_content = pd.read_csv(citeseer_citation_csv_node_inputfile)
        nodeId = node_content['nodeID']
        label = node_content['label']
        node_label_map = node_content.set_index(nodeId).to_dict()['label']
        connection_content = pd.read_csv(citeseer_citation_csv_edges_inputfile)
        source_id = connection_content['source']
        target_id = connection_content['target']
        original_citation_graph = createDirectedGraph(nodeId, source_id, target_id)
        # Specify the degree you want to remove (0 degrees)
        degree_to_remove = 0

        # Identify and remove nodes with the specified degree
        nodes_to_remove = [node for node in original_citation_graph.nodes if original_citation_graph.degree[node] == degree_to_remove]
        original_citation_graph.remove_nodes_from(nodes_to_remove)

        save_as_pickle(citeseer_citation_original_graph_pkl, original_citation_graph)
        save_as_pickle(node_to_label_map_pkl,node_label_map)


    if exists(citeseer_citation_bipartite_graph_pkl):
        citation_bipartite_G = read_pickel(citeseer_citation_bipartite_graph_pkl)
    else:
        citation_bipartite_G = createBipartiteGraph(original_citation_graph, citeseer_citation_bipartite_graph_name)

        # Specify the degree you want to remove (0 degrees)
        degree_to_remove = 0
        nodes_to_remove = [node for node in citation_bipartite_G.nodes if citation_bipartite_G.degree[node] == degree_to_remove]
        print("Number of zero degree nodes:" + str(len(nodes_to_remove)))

        # Identify and remove nodes with the specified degree

        citation_bipartite_G.remove_nodes_from(nodes_to_remove)

        save_as_pickle(citeseer_citation_bipartite_graph_pkl, citation_bipartite_G)
        citation_bipartite_edges = citation_bipartite_G.edges

        pd.DataFrame(citation_bipartite_G.edges()).to_csv("citeseer_edges.csv", index=False)

        pd.DataFrame(citation_bipartite_edges).to_csv(citeseer_citation_bipartite_edges_csv, encoding='utf-8', float_format='%f')
        bipartite_A = nx.adjacency_matrix(citation_bipartite_G)
        bipartite_adj_matrix = pd.DataFrame(bipartite_A.toarray(), columns=np.array(citation_bipartite_G.nodes),
                                            index=np.array(citation_bipartite_G.nodes))
        pd.DataFrame(bipartite_adj_matrix).to_csv(citeseer_citation_bipartite_adj_matrix_csv, encoding='utf-8', float_format='%f')

        source_nodes = []
        target_nodes = []
        zero_src_nodes = []
        zero_trg_nodes = []
        for node in citation_bipartite_G.nodes:
            if str(node).startswith("S"):
                source_nodes.append(node)
                if citation_bipartite_G.degree(node) == 0:
                    zero_src_nodes.append(node)
            elif str(node).startswith("T"):
                target_nodes.append(node)
                if citation_bipartite_G.degree(node) == 0:
                    zero_trg_nodes.append(node)

        print("Total No of Nodes in Directed Graph: " + str(len(original_citation_graph.nodes)))
        print("Total No of Edges in Directed Graph: " + str(len(original_citation_graph.edges)))
        print("Total No of Source_SDM Nodes in Bipartite Graph: " + str(len(source_nodes)))
        print("Total No of Target Nodes in Bipartite Graph: " + str(len(target_nodes)))
        print("Total No of Edges in Bipartite Graph: " + str(len(citation_bipartite_G.edges)))
        print("Edge Density of Original Bipartite Graph in Bipartite Graph: " + str(len(citation_bipartite_G.edges) / (len(source_nodes) * len(target_nodes))))
        print("Total No Of Zero Degree Source_SDM Nodes in Bipartite Graph:" + str(len(zero_src_nodes)))
        print("Total No Of Zero Degree Target Nodes in Bipartite Graph:" + str(len(zero_trg_nodes)))
        print("Total No Of Zero Degree Nodes in Bipartite Graph:" + str(len(zero_src_nodes) + len(zero_trg_nodes)))

        original_bitcoin_graph_rows = []
        row = [len(original_citation_graph.nodes), len(original_citation_graph.edges), len(source_nodes),
               len(target_nodes), len(zero_src_nodes), len(zero_trg_nodes),
               len(zero_src_nodes) + len(zero_trg_nodes), len(citation_bipartite_G.edges),
               str(len(citation_bipartite_G.edges) / (len(source_nodes) * len(target_nodes)))]
        original_bitcoin_graph_rows.append(row)
        original_graph_cols = ["No Of Nodes in Directed Graph", "No Of Edges in Directed Graph",
                               "No Of Source_SDM Nodes", "No Of Target Nodes",
                               "Total No Of Zero Degree Source_SDM Nodes",
                               "Total No Of Zero Degree Target Nodes",
                               "Total No Of Zero Degree Nodes", "No Of Edges in Bipartite Graph",
                               "Edge Density Of Bipartite Graph"]
        pd.DataFrame(original_bitcoin_graph_rows, columns=original_graph_cols).to_csv(citeseer_bipartite_graph_edge_density_info)


relation_csv = pd.read_csv("citeseer_edges.csv")
relations = [tuple(x) for x in relation_csv.iloc[:,:2].values.tolist()]

if exists(citeseer_2D_poincare_pkl) == False:
    plot_poincare_2d_visualization(relations)
else:
    model = read_pickel(citeseer_2D_poincare_pkl)
    tsne_visualization(model, citation_bipartite_G)


