import sys
from os.path import exists
import pandas as pd
import numpy as np
from resource.constants import sign_bitcoin_output_file, sign_bitcoin_original_graph_pickel_file, inputfile, \
    sign_bitcoin_tripartite_graph_pickel_file, sign_bitcoin_tripartite_graph_title, sign_bitcoin_tripartite_graph_name
from utils import read_pickel, save_as_pickle, create_original_graph, createTripartiteGraph, \
    poincareEmbedding

if __name__ == '__main__':
    sys.stdout = open(sign_bitcoin_output_file, "w+")
    # Open the first file for appending

    if exists(sign_bitcoin_original_graph_pickel_file):
        G = read_pickel(sign_bitcoin_original_graph_pickel_file)
    else:
        node_content = pd.read_csv(inputfile)
        source_nd = np.array(node_content)[:, 0]
        target_nd = np.array(node_content)[:, 1]
        rating = np.array(node_content)[:, 2]
        G, A = create_original_graph(source_nd, target_nd, rating)
        save_as_pickle(sign_bitcoin_original_graph_pickel_file, G)

    if exists(sign_bitcoin_tripartite_graph_pickel_file):
        bitcoin_tripartite_G = read_pickel(sign_bitcoin_tripartite_graph_pickel_file)
    else:
        bitcoin_tripartite_G, pos_edges, neg_edges = createTripartiteGraph(G, sign_bitcoin_tripartite_graph_title, sign_bitcoin_tripartite_graph_name)
        degree_to_remove = 0
        # Identify and remove nodes with the specified degree
        nodes_to_remove = [node for node in bitcoin_tripartite_G.nodes if bitcoin_tripartite_G.degree[node] == degree_to_remove]
        nodes_to_remove = [node for node in bitcoin_tripartite_G.nodes if bitcoin_tripartite_G.degree[node] == degree_to_remove]
        bitcoin_tripartite_G.remove_nodes_from(nodes_to_remove)
        save_as_pickle(sign_bitcoin_tripartite_graph_pickel_file, bitcoin_tripartite_G)

    '''if exists(sign_bitcoin_tripartite_middle_dimension_pkl):
        tripartite_middleDimentions = read_pickel(sign_bitcoin_tripartite_middle_dimension_pkl)
    else:
        tripartite_middleDimentions, key_to_indexes, model_wv = featureExtraction_tripartite(bitcoin_tripartite_G,
                                                                                             sign_bitcoin_TSNE_tripartite_node_embedding_title,
                     hgb                                                                        sign_bitcoin_TSNE_tripartite_node_embedding_scatter_plot)
        save_as_pickle(sign_bitcoin_tripartite_middle_dimension_pkl, tripartite_middleDimentions)'''

    '''if exists(sign_bitcoin_isomap_pkl):
        isomap_embedding = read_pickel(sign_bitcoin_isomap_pkl)
    else:
        A = nx.to_numpy_array(bitcoin_tripartite_G, weight="weight")
        sign_bitcoin_tripartite_Adj_matrix = pd.DataFrame(A, columns=np.array(bitcoin_tripartite_G.nodes), index=np.array(bitcoin_tripartite_G.nodes))
        X_transformed = IsomapEmbedding(sign_bitcoin_tripartite_Adj_matrix, bitcoin_tripartite_G)
        save_as_pickle(sign_bitcoin_isomap_pkl, X_transformed)'''

    '''if exists(sign_bitcoin_hope_pkl):
        hope_embedding = read_pickel(sign_bitcoin_hope_pkl)
    else:
        hope = HOPE(dimension=20)
        hope_embedding = hope.train(bitcoin_tripartite_G)
        save_as_pickle(sign_bitcoin_hope_pkl, hope_embedding)
        hope_display(hope_embedding, bitcoin_tripartite_G)'''

    #if exists(sign_bitcoin_poincare_pkl):
    #    poincare = read_pickel(sign_bitcoin_poincare_pkl)
    #else:
    poincare_embeddings = poincareEmbedding(bitcoin_tripartite_G)




