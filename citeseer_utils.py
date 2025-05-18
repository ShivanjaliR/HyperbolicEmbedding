from os.path import exists
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering, DBSCAN, KMeans
from sklearn.manifold import TSNE
import copy
from resource.citeseer_citation_constants import citeseer_hub_plot_name, citeseer_authority_plot_name, \
    citeseer_clusters_list_pkl, min_samples, citeseer_spectral_dbscan_plot_name, citeseer_dbscan_cluster_csv_file, \
    citeseer_hit_authority_score_csv, citeseer_cust_dbscan_cluster_csv_file, citeseer_5th_nearest_neighbor_distance_csv, \
    citeseer_speparate_cluster_path, citesser_TSNE_clusterwise_source_target_plot, \
    citeseer_TSNE_clusterwise_source_target_title, citesser_TSNE_clusterwise_source_with_cluster_labels_plot, \
    citeseer_customized_dbscan_clusters, citeseer_customized_dbscan_clusters_plot, citeseer_zero_degree_node_analysis, \
    citeseer_node_embedding_title, citeseer_TSNE_clusterwise_source_with_cluster_title, \
    citeseer_cust_dbscan_cluster_nodes, closer_2, citeseer_dbscan_cluster_nodes_labels, citeseer_dbscan_cluster_info, \
    citeseer_cluster_nodes, citeseer_clusterwise_embedding_src_nodes, citeseer_cust_clusterwise_embedding_nodes, \
    citeseer_cluster_dict, citeseer_embedding_map, citeseer_source_nodes_map, citeseer_avg_distance_map, closer_3, \
    citeseer_original_hub_csv, citeseer_original_authority_csv, citeseer_original_hub_plot, \
    citeseer_original_authority_plot, citeseer_original_hub_title, citeseer_original_authority_title, \
    citeseer_spectral_cluster, citeseer_source_nodes_only_title, citesser_clusterwise_target_plot, \
    citeseer_target_nodes_only_title, citesser_clusterwise_source_plot, citeseer_citation_main_TSNE_embedding, \
    citesser_bipartite_graph_degree_distribution, citesser_cluster_degree_distribution, \
    target_citeseer_node_degree_list, \
    citeseer_node_degree_list, source_target_citeseer_node_degree_list, citeseer_node_degree_summary, \
    target_citeseer_node_degree_summary, \
    source_citeseer_node_degree_summary, citeseer_dbscan_cluster_node_edge_csv_file, \
    citeseer_pair_removed_cust_dbscan_cluster_csv_file, citeseer_dbscan_cluster_node_edge_pkl, \
    citeseer_nodes_not_in_cluster_csv, citeseer_nodes_not_part_of_any_community
from utils import save_as_pickle, read_pickel, average_distance, key_func
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")

def get_customized_clusters(cust_dbscan_cluster_nodes, middleDimentions, bitcoin_directed_A, closer):
    for p in range(len(cust_dbscan_cluster_nodes)):
        dbscan_names = set(cust_dbscan_cluster_nodes[p]['nodes'])
        while len(dbscan_names) > 0:
            last_element = dbscan_names.pop()
            last_element_index = cust_dbscan_cluster_nodes[p]['nodes'].index(last_element)
            last_element_status = cust_dbscan_cluster_nodes[p]['status'][last_element_index]
            if last_element_status == "unvisited":
                cust_dbscan_cluster_nodes[p]['status'][last_element_index] = "visited"
                if last_element.startswith("S"):
                    new_src = int(last_element[1:])
                    src_trg = bitcoin_directed_A.loc[new_src]
                    indices = [i for i, value in enumerate(src_trg) if value == 1]
                    dataframe_indices = list(src_trg.index)
                    trg_list = [dataframe_indices[trg_index] for trg_index in indices]
                    for trg in trg_list:
                        # if list(cust_dbscan_cluster_nodes[p]['nodes']).__contains__("T"+str(trg)) == False and list(cluster_nodes[i]['name']).__contains__("T"+str(trg)) == True:
                        if list(cust_dbscan_cluster_nodes[p]['nodes']).__contains__("T" + str(trg)) == False:
                            src_trg_neighbors = bitcoin_directed_A[trg]
                            indices = [i for i, value in enumerate(src_trg_neighbors) if value == 1]
                            dataframe_indices = list(src_trg_neighbors.index)
                            src_list = [dataframe_indices[src_index] for src_index in indices]
                            src_trg_neighbors = set(src_list)
                            cls_src_list = [int(src[1:]) for src in dbscan_names if src.startswith("S")]
                            cls_src_list = set(cls_src_list)
                            common_src = list(cls_src_list.intersection(src_trg_neighbors))
                            if len(common_src) >= 2:
                                included_trg = "T" + str(trg)
                                cust_dbscan_cluster_nodes[p]['nodes'].append(included_trg)
                                cust_dbscan_cluster_nodes[p]['status'].append("unvisited")
                                cust_dbscan_cluster_nodes[p]['original_embedding'].append(list(middleDimentions.loc[included_trg]))
                                cust_dbscan_cluster_nodes[p]['label'].append(
                                    list(set(cust_dbscan_cluster_nodes[p]['label']))[0])
                                dbscan_names.update([included_trg])
                elif last_element.startswith("T"):
                    new_trg = int(last_element[1:])
                    trg_src = bitcoin_directed_A[new_trg]
                    indices = [i for i, value in enumerate(trg_src) if value == 1]
                    dataframe_indices = list(trg_src.index)
                    src_list = [dataframe_indices[src_index] for src_index in indices]
                    for src in src_list:
                        # if list(cust_dbscan_cluster_nodes[p]['nodes']).__contains__("S"+str(src)) == False and list(cluster_nodes[i]['name']).__contains__("S"+str(src)) == True:
                        if list(cust_dbscan_cluster_nodes[p]['nodes']).__contains__("S" + str(src)) == False:
                            trg_src_neighbors = bitcoin_directed_A.loc[src]
                            indices = [i for i, value in enumerate(trg_src_neighbors) if value == 1]
                            dataframe_indices = list(trg_src_neighbors.index)
                            trg_list = [dataframe_indices[trg_index] for trg_index in indices]
                            trg_src_neighbors = set(trg_list)
                            cls_trg_list = [int(src[1:]) for src in dbscan_names if src.startswith("T")]
                            cls_trg_list = set(cls_trg_list)
                            common_trg = list(cls_trg_list.intersection(trg_src_neighbors))
                            if len(common_trg) >= closer:
                                included_src = "S" + str(src)
                                cust_dbscan_cluster_nodes[p]['nodes'].append(included_src)
                                cust_dbscan_cluster_nodes[p]['status'].append("unvisited")
                                cust_dbscan_cluster_nodes[p]['original_embedding'].append(list(middleDimentions.loc[included_src]))
                                cust_dbscan_cluster_nodes[p]['label'].append(
                                    list(set(cust_dbscan_cluster_nodes[p]['label']))[0])
                                dbscan_names.update([included_src])
    return cust_dbscan_cluster_nodes

def get_cust_table(cust_dbscan_cluster_nodes, sorted_hubs, sorted_authorities, i, dbscan_cluster_info, bipartite_G, cust_clusterwise_embedding_nodes, dbscan_rows):
    cust_nodes = []
    cust_labels = []
    for h in range(len(cust_dbscan_cluster_nodes)):
        cust_nodes.extend(list(cust_dbscan_cluster_nodes[h]['nodes']))
        cust_labels.extend(list(cust_dbscan_cluster_nodes[h]['label']))
    cust_clusterwise_embedding_nodes[i]['name'] = cust_nodes
    cust_clusterwise_embedding_nodes[i]['label'] = cust_labels

    for k in range(len(cust_dbscan_cluster_nodes)):
        hub_bar_list = {}
        authority_bar_list = {}
        line_x_values = {}
        for bar, bar_value in zip(list(sorted_hubs.keys()), list(sorted_hubs.values())):
            hub_bar_list[bar] = 0
            authority_bar_list[bar] = 0
            line_x_values[bar] = ""
        row = []
        row.append(i + 1)
        row.append(k + 1)
        row.append(len(cust_dbscan_cluster_nodes[k]['nodes']))
        row.append(len(dbscan_cluster_info[k]['names']))
        sr_no = 0
        tr_no = 0
        pair_nodes = set()
        degree_cnt = 0
        edges = set()
        cls_nodes = set(cust_dbscan_cluster_nodes[k]['nodes'])
        hits_score_list = {}
        authority_score_list = {}
        percentile_score_list = {}
        for nd in cust_dbscan_cluster_nodes[k]['nodes']:
            if nd.startswith("S"):
                sr_no += 1
                node_no = nd[1:]
                tr_node = "T" + str(node_no)
                if list(cust_dbscan_cluster_nodes[k]['nodes']).__contains__(tr_node):
                    pair_nodes.add(node_no)
            elif nd.startswith("T"):
                tr_no += 1
                node_no = nd[1:]
                sr_node = "S" + str(node_no)
                if list(cust_dbscan_cluster_nodes[k]['nodes']).__contains__(sr_node):
                    pair_nodes.add(node_no)
            edge_list = list(bipartite_G.neighbors(nd))
            common_neighbors = set(edge_list).intersection(cls_nodes)
            hits_score_list[nd] = sorted_hubs[nd]
            authority_score_list[nd] = sorted_authorities[nd]
            hub_bar_list[nd] = max(sorted_hubs.values())
            authority_bar_list[nd] = max(sorted_authorities.values())
            percentile = np.percentile(list(sorted_hubs.values()),
                                       (list(sorted_hubs.keys()).index(nd) / len(list(sorted_hubs.values()))) * 100)
            percentile_score_list[nd] = percentile
            for cm_ed in common_neighbors:
                if nd.startswith("S"):
                    edge = str(nd) + "-" + str(cm_ed)
                elif nd.startswith("T"):
                    edge = str(cm_ed) + "-" + str(nd)
                edges.add(edge)

        row.append(sr_no)
        row.append(tr_no)
        row.append(len(edges))
        row.append(len(pair_nodes))
        row.append(cust_dbscan_cluster_nodes[k]['nodes'])
        row.append(edges)
        row.append(hits_score_list)
        row.append(authority_score_list)

        y_min = min(list(sorted_hubs.values()))
        y_max = max(list(sorted_hubs.values()))
        figure_cluster_hub = plt.figure(figsize=(15, 15))
        ax_cluster_hub = figure_cluster_hub.add_subplot(111)
        ax_cluster_hub.set_ylim(y_min, y_max)
        ax_cluster_hub.plot(list(sorted_hubs.keys()), list(sorted_hubs.values()), label="Bipartite Graph Hub Score")
        ax_cluster_hub.bar(list(sorted_hubs.keys()), list(hub_bar_list.values()), color='red',
                           label="Cluster Hub Score Location")
        plt.legend()
        plt.title("Node Hub Score Position(Source_SDM Clustering) For Spectral Cluster No:" + str(i + 1) + "DBSCAN Cluster No:" + str(k + 1))
        # plt.show()
        plt.savefig(citeseer_speparate_cluster_path + "node_hub_score_position_spectral_cls_source_clustering" + str(i + 1) + "dbscan_cls_" + str(k + 1) + ".png")

        y_min = min(list(sorted_authorities.values()))
        y_max = max(list(sorted_authorities.values()))
        figure_cluster_authority = plt.figure(figsize=(15, 15))
        ax_cluster_authority = figure_cluster_authority.add_subplot(111)
        ax_cluster_authority.set_ylim(y_min, y_max)
        ax_cluster_authority.plot(list(sorted_authorities.keys()), list(sorted_authorities.values()), label="Bipartite Graph Authority Score")
        ax_cluster_authority.bar(list(sorted_authorities.keys()), list(authority_bar_list.values()), color='red', label="Cluster Authority Score Location")
        plt.legend()
        plt.title("Node Authority Score Position(Source_SDM Clustering)  For Spectral Cluster No:" + str(i + 1) + "DBSCAN Cluster No:" + str(k + 1))
        # plt.show()
        plt.savefig(citeseer_speparate_cluster_path + "node_authority_score_position_spectral_cls_source_clustering" + str(i + 1) + "dbscan_cls_" + str(k + 1) + ".png")
        dbscan_rows.append(row)
    return dbscan_rows


def spectral_dbscan_clustering_citeseer_source(middleDimentions, no_of_clusters, citation_spectral_clustering_csv, title,
                                spectral_cluster_plot_file_name, bipartite_G, edge_file_name,
                                nodes_not_in_cls_pkl, original_citeseer_graph):
    sorted_hubs, sorted_authorities = hit_score_original_graph(original_citeseer_graph, citeseer_original_hub_csv,
                                                               citeseer_original_authority_csv,citeseer_hit_authority_score_csv,
                                                               citeseer_original_hub_plot,
                                                               citeseer_original_authority_plot,
                                                               citeseer_original_hub_title,
                                                               citeseer_original_authority_title)

    nodes_not_in_cls = []
    if exists(citeseer_clusters_list_pkl):
        clusters = read_pickel(citeseer_clusters_list_pkl)
    else:
        spectral = SpectralClustering(n_clusters=no_of_clusters, affinity='nearest_neighbors', n_neighbors=50, random_state=0)
        clusters = spectral.fit_predict(middleDimentions)
        save_as_pickle(citeseer_clusters_list_pkl, clusters)

    if exists(citeseer_cluster_nodes) and exists(citeseer_clusterwise_embedding_src_nodes) and exists(citeseer_cust_clusterwise_embedding_nodes) and exists(citeseer_cluster_dict):
        cluster_nodes = read_pickel(citeseer_cluster_nodes)
        clusterwise_embedding_src_nodes = read_pickel(citeseer_clusterwise_embedding_src_nodes)
        cust_clusterwise_embedding_nodes = read_pickel(citeseer_cust_clusterwise_embedding_nodes)
        cluster_dict = read_pickel(citeseer_cluster_dict)
    else:
        cluster_dict = {}
        cluster_nodes = {}
        clusterwise_embedding_src_nodes = {}
        cust_clusterwise_embedding_nodes = {}
        for q in range(no_of_clusters):
            cluster_dict[q] = []
            cluster_nodes[q] = {}
            cluster_nodes[q]['nodes'] = []
            cluster_nodes[q]['label'] = []
            cluster_nodes[q]['name'] = []
            cluster_nodes[q]['status'] = []
            cluster_nodes[q]['cluster_no'] = []
            cluster_nodes[q]['index'] = []
            clusterwise_embedding_src_nodes[q] = {}
            clusterwise_embedding_src_nodes[q]['name'] = []
            clusterwise_embedding_src_nodes[q]['label'] = []
            cust_clusterwise_embedding_nodes[q] = {}
            cust_clusterwise_embedding_nodes[q]['name'] = []
            cust_clusterwise_embedding_nodes[q]['label'] = []

        for w in range(len(clusters)):
            cluster_dict[clusters[w]].append(list(middleDimentions.index)[w])
            cluster_nodes[clusters[w]]['nodes'].append(list(middleDimentions.iloc[w]))
            cluster_nodes[clusters[w]]['label'].append('unlabeled')
            cluster_nodes[clusters[w]]['name'].append(list(middleDimentions.index)[w])
            cluster_nodes[clusters[w]]['status'].append('unvisited')
            cluster_nodes[clusters[w]]['cluster_no'].append(-1)
            cluster_nodes[clusters[w]]['index'].append(list(middleDimentions.index).index(list(middleDimentions.index)[w]))

        save_as_pickle(citeseer_cluster_nodes, cluster_nodes)
        save_as_pickle(citeseer_clusterwise_embedding_src_nodes, clusterwise_embedding_src_nodes)
        save_as_pickle(citeseer_cust_clusterwise_embedding_nodes, cust_clusterwise_embedding_nodes)
        save_as_pickle(citeseer_cluster_dict, cluster_dict)

    if exists(citeseer_dbscan_cluster_nodes_labels) and exists(citeseer_embedding_map) and exists(citeseer_source_nodes_map) and exists(citeseer_avg_distance_map):
        cluster_labels_map = read_pickel(citeseer_dbscan_cluster_nodes_labels)
        embedding_map = read_pickel(citeseer_embedding_map)
        source_nodes_map = read_pickel(citeseer_source_nodes_map)
        avg_distance_map = read_pickel(citeseer_avg_distance_map)
    else:
        cluster_labels_map = {}
        avg_distances_rows = []
        embedding_map = {}
        source_nodes_map = {}
        avg_distance_map = {}
        avg_elblow_distance = {}
        for u in range(len(cluster_nodes)):
            source_nodes = {}
            source_nodes['nodes'] = []
            source_nodes['name'] = []
            source_nodes['label'] = []
            source_nodes['status'] = []
            source_nodes['cluster_no'] = []
            source_nodes['index'] = []
            source_nodes['embedding'] = []
            for j in range(len(cluster_nodes[u]['nodes'])):
                if cluster_nodes[u]['name'][j].startswith("S"):
                    source_nodes['nodes'].append(cluster_nodes[u]['nodes'][j])
                    source_nodes['name'].append(cluster_nodes[u]['name'][j])
                    source_nodes['label'].append(cluster_nodes[u]['label'][j])
                    source_nodes['status'].append(cluster_nodes[u]['status'][j])
                    source_nodes['cluster_no'].append(cluster_nodes[u]['cluster_no'][j])
                    source_nodes['index'].append(cluster_nodes[u]['index'][j])
                    source_nodes['embedding'].append(list(middleDimentions.iloc[cluster_nodes[u]['index'][j]]))
            source_nodes_map[u] = source_nodes
            avg, avg_elbow_point = average_distance(source_nodes, u+1)
            avg_distance_map[u] = avg
            avg_distances_row = []
            avg_distances_row.append(u + 1)
            #avg_distances_row.append(sorted_names)
            #avg_distances_row.append(sorted_values)
            avg_distances_rows.append(avg_distances_row)

            dbscan = DBSCAN(eps=avg, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(source_nodes['nodes'])
            cluster_labels_map[u] = cluster_labels

            tsne = TSNE(n_components=2, perplexity=5)
            embeddings_2d = tsne.fit_transform(np.array(source_nodes['nodes']))
            embedding_map[u] = embeddings_2d

        save_as_pickle(citeseer_dbscan_cluster_nodes_labels, cluster_labels_map)
        save_as_pickle(citeseer_embedding_map, embedding_map)
        save_as_pickle(citeseer_source_nodes_map,source_nodes_map)
        save_as_pickle(citeseer_avg_distance_map,avg_distance_map)
        #avg_distance_cols = ["Cluster No", "Source_SDM Names", "5th Nearest Neighbor Distance"]
        #pd.DataFrame(avg_distances_rows, columns=avg_distance_cols).to_csv(citeseer_5th_nearest_neighbor_distance_csv)

    rows = []
    dbscan_rows = []
    avg_distances_rows = []
    node_edge_rows = []
    dbscan_pair_removed_rows = []
    node_degree_list = []
    trg_node_degree_list = []
    degree_summary_info = []
    degree_summary_info_trg = []
    degree_summary_info_s = []
    s_trg_node_degree_list = []
    for i in range(len(cluster_nodes)):

        cluster_labels = cluster_labels_map[i]
        citeseer_directed_A = nx.adjacency_matrix(original_citeseer_graph)
        bitcoin_directed_A = pd.DataFrame(citeseer_directed_A.toarray(), columns=original_citeseer_graph.nodes(), index=original_citeseer_graph.nodes())

        embeddings_2d = embedding_map[i]
        #print("embedding len" +str(len(embeddings_2d)))
        #print("cluster label len"+str(len(np.array(cluster_labels))))
        #print(cluster_labels)
        print("Spectral Cluster No:" + str(i+1))
        '''x_values = [embeddings_2d[j, 0] for j in range(len(embeddings_2d[:, 0])) if np.array(cluster_labels)[j] != -1]
        y_values = [embeddings_2d[j, 1] for j in range(len(embeddings_2d[:, 0])) if np.array(cluster_labels)[j] != -1]       
        source_lables = [int(lab) for lab in np.array(cluster_labels) if lab != -1]      
        source_names = [source_nodes['name'][j] for j in range(len(source_nodes['name'])) if cluster_labels[j] != -1]
        src_cls_embedding = [source_nodes['embedding'][j] for j in range(len(source_nodes['name'])) if cluster_labels[j] != -1]'''
        x_values = []
        y_values = []
        source_lables = []
        source_names = []
        src_cls_embedding = []
        for b in range(len(np.array(cluster_labels))):
            if cluster_labels[b] != -1:
                x_values.append(embeddings_2d[b, 0])
                y_values.append(embeddings_2d[b, 1])
                source_lables.append(int(cluster_labels[b]))
                source_nodes = source_nodes_map[i]
                source_names.append(source_nodes['name'][b])
                src_cls_embedding.append(source_nodes['embedding'][b])

        figure5 = plt.figure(figsize=(15, 15))
        ax5 = figure5.add_subplot(111)
        ax5.scatter(x_values, y_values, c=source_lables)
        plt.legend()
        plt.title("DBSCAN Inside Spectral Clustering (Cluster No: " + str(i+1) + ") ( No of Clusters " + str(len(set(source_lables))) + ")")
        plt.savefig(citeseer_spectral_dbscan_plot_name + "_" + str(i + 1) + ".png")
        #plt.show()

        clusterwise_embedding_src_nodes[i]['name'] = source_names
        clusterwise_embedding_src_nodes[i]['label'] = source_lables

        dbscan_cluster_info = {}
        for lab in list(set(source_lables)):
            dbscan_cluster_info[lab] = {}
            dbscan_cluster_info[lab]['x_value'] = []
            dbscan_cluster_info[lab]['y_value'] = []
            dbscan_cluster_info[lab]['names'] = []
            dbscan_cluster_info[lab]['embedding'] = []
            dbscan_cluster_info[lab]['status'] = []
            dbscan_cluster_info[lab]['label'] = []

        for lab in set(source_lables):
            dbscan_cluster_info[lab]['x_value'] = [x_values[k] for k in range(len(source_lables)) if source_lables[k] == lab]
            dbscan_cluster_info[lab]['y_value'] = [y_values[k] for k in range(len(source_lables)) if source_lables[k] == lab]
            dbscan_cluster_info[lab]['names'] = [source_names[k] for k in range(len(source_names)) if source_lables[k] == lab]
            dbscan_cluster_info[lab]['embedding'] = [src_cls_embedding[k] for k in range(len(source_lables)) if source_lables[k] == lab]
            dbscan_cluster_info[lab]['label'] = [lab for k in range(len(source_lables)) if source_lables[k] == lab]

        for k in range(len(dbscan_cluster_info)):
            row = []
            row.append(i + 1)
            row.append(k + 1)
            row.append(len(dbscan_cluster_info[k]['names']))
            row.append(min_samples)
            row.append(avg_distance_map[i])
            row.append(dbscan_cluster_info[k]['names'])
            row.append(dbscan_cluster_info[k]['embedding'])
            row.append(dbscan_cluster_info[k]['x_value'])
            row.append(dbscan_cluster_info[k]['y_value'])
            rows.append(row)

        cust_dbscan_cluster_nodes = {}
        cust_dbscan_cluster_nodes_pair_removed = {}
        for k in range(len(dbscan_cluster_info)):
            cust_dbscan_cluster_nodes[k] = {}
            cust_dbscan_cluster_nodes[k]['nodes'] = []
            cust_dbscan_cluster_nodes[k]['original_embedding'] = []
            cust_dbscan_cluster_nodes[k]['src_x_embedding'] = []
            cust_dbscan_cluster_nodes[k]['src_y_embedding'] = []
            cust_dbscan_cluster_nodes[k]['status'] = []
            cust_dbscan_cluster_nodes[k]['label'] = []
            cust_dbscan_cluster_nodes[k]['no_src'] = 0
            cust_dbscan_cluster_nodes[k]['no_trg'] = 0
            cust_dbscan_cluster_nodes[k]['no_pair'] = 0
            cust_dbscan_cluster_nodes[k]['avg_degree'] = 0
            cust_dbscan_cluster_nodes[k]['iterations'] = 0
            cust_dbscan_cluster_nodes_pair_removed[k] = {}
            cust_dbscan_cluster_nodes_pair_removed[k]['nodes'] = []
            cust_dbscan_cluster_nodes_pair_removed[k]['original_embedding'] = []
            cust_dbscan_cluster_nodes_pair_removed[k]['src_x_embedding'] = []
            cust_dbscan_cluster_nodes_pair_removed[k]['src_y_embedding'] = []
            cust_dbscan_cluster_nodes_pair_removed[k]['status'] = []
            cust_dbscan_cluster_nodes_pair_removed[k]['label'] = []
            cust_dbscan_cluster_nodes_pair_removed[k]['pair'] = []
            cust_dbscan_cluster_nodes_pair_removed[k]['no_src'] = 0
            cust_dbscan_cluster_nodes_pair_removed[k]['no_trg'] = 0
            cust_dbscan_cluster_nodes_pair_removed[k]['no_pair'] = 0
            cust_dbscan_cluster_nodes_pair_removed[k]['avg_degree'] = 0
            cust_dbscan_cluster_nodes_pair_removed[k]['iterations'] = 0

        for k in range(len(dbscan_cluster_info)):
            cust_dbscan_cluster_nodes[k]['nodes'] = copy.deepcopy(dbscan_cluster_info[k]['names'])
            cust_dbscan_cluster_nodes[k]['original_embedding'] = copy.deepcopy(dbscan_cluster_info[k]['embedding'])
            cust_dbscan_cluster_nodes[k]['src_x_embedding'] = copy.deepcopy(dbscan_cluster_info[k]['x_value'])
            cust_dbscan_cluster_nodes[k]['src_y_embedding'] = copy.deepcopy(dbscan_cluster_info[k]['y_value'])
            cust_dbscan_cluster_nodes[k]['status'] = ["unvisited" for k in range(len(dbscan_cluster_info[k]['names']))]
            cust_dbscan_cluster_nodes[k]['label'] = copy.deepcopy(dbscan_cluster_info[k]['label'])


        for p in range(len(cust_dbscan_cluster_nodes)):
            print("DBSCAN Cluster No:"+str(p+1))
            #dbscan_names = set(cust_dbscan_cluster_nodes[p]['nodes'])
            dbscan_names = cust_dbscan_cluster_nodes[p]['nodes']
            iteration_flag = False
            previous_flag = ""
            iteration = 0
            while len(dbscan_names) > 0:
                #last_element = dbscan_names.pop()
                last_element = dbscan_names[0]
                dbscan_names = dbscan_names[1:]
                last_element_index = cust_dbscan_cluster_nodes[p]['nodes'].index(last_element)
                last_element_status = cust_dbscan_cluster_nodes[p]['status'][last_element_index]

                if previous_flag =="":
                    previous_flag = last_element[0]
                    iteration += 1
                #elif previous_flag != last_element[0]:
                #    iteration += 1

                if last_element_status == "unvisited":
                    cust_dbscan_cluster_nodes[p]['status'][last_element_index] = "visited"
                    if last_element.startswith("S"):
                        new_src = int(last_element[1:])
                        src_trg = bitcoin_directed_A.loc[new_src]
                        indices = [i for i, value in enumerate(src_trg) if value == 1]
                        dataframe_indices = list(src_trg.index)
                        trg_list = [dataframe_indices[trg_index] for trg_index in indices]
                        for trg in trg_list:
                            #if list(cust_dbscan_cluster_nodes[p]['nodes']).__contains__("T"+str(trg)) == False and list(cluster_nodes[i]['name']).__contains__("T"+str(trg)) == True:
                            if list(cust_dbscan_cluster_nodes[p]['nodes']).__contains__("T" + str(trg)) == False:
                                src_trg_neighbors = bitcoin_directed_A[trg]
                                indices = [i for i, value in enumerate(src_trg_neighbors) if value == 1]
                                dataframe_indices = list(src_trg_neighbors.index)
                                src_list = [dataframe_indices[src_index] for src_index in indices]
                                src_trg_neighbors = set(src_list)
                                cls_src_list = [int(src[1:]) for src in cust_dbscan_cluster_nodes[p]['nodes'] if src.startswith("S")]
                                cls_src_list.append(last_element[1:])
                                cls_src_list = set(cls_src_list)
                                common_src = list(cls_src_list.intersection(src_trg_neighbors))
                                if len(common_src) >=closer_2:
                                    included_trg="T"+str(trg)
                                    cust_dbscan_cluster_nodes[p]['nodes'].append(included_trg)
                                    cust_dbscan_cluster_nodes[p]['status'].append("unvisited")
                                    cust_dbscan_cluster_nodes[p]['original_embedding'].append(list(middleDimentions.loc[included_trg]))
                                    cust_dbscan_cluster_nodes[p]['label'].append(list(set(cust_dbscan_cluster_nodes[p]['label']))[0])
                                    #dbscan_names.update([included_trg])
                                    dbscan_names.append(included_trg)
                                    if previous_flag != last_element[0]:
                                        previous_flag = last_element[0]
                                        iteration += 1
                    elif last_element.startswith("T"):
                        new_trg = int(last_element[1:])
                        trg_src = bitcoin_directed_A[new_trg]
                        indices = [i for i, value in enumerate(trg_src) if value == 1]
                        dataframe_indices = list(trg_src.index)
                        src_list = [dataframe_indices[src_index] for src_index in indices]
                        for src in src_list:
                            #if list(cust_dbscan_cluster_nodes[p]['nodes']).__contains__("S"+str(src)) == False and list(cluster_nodes[i]['name']).__contains__("S"+str(src)) == True:
                            if list(cust_dbscan_cluster_nodes[p]['nodes']).__contains__("S" + str(src)) == False:
                                trg_src_neighbors = bitcoin_directed_A.loc[src]
                                indices = [i for i, value in enumerate(trg_src_neighbors) if value == 1]
                                dataframe_indices = list(trg_src_neighbors.index)
                                trg_list = [dataframe_indices[trg_index] for trg_index in indices]
                                trg_src_neighbors = set(trg_list)
                                cls_trg_list = [int(src[1:]) for src in cust_dbscan_cluster_nodes[p]['nodes'] if src.startswith("T")]
                                cls_trg_list.append(last_element[1:])
                                cls_trg_list = set(cls_trg_list)
                                common_trg = list(cls_trg_list.intersection(trg_src_neighbors))
                                if len(common_trg) >= closer_2:
                                    included_src = "S" + str(src)
                                    cust_dbscan_cluster_nodes[p]['nodes'].append(included_src)
                                    cust_dbscan_cluster_nodes[p]['status'].append("unvisited")
                                    cust_dbscan_cluster_nodes[p]['original_embedding'].append(list(middleDimentions.loc[included_src]))
                                    cust_dbscan_cluster_nodes[p]['label'].append(list(set(cust_dbscan_cluster_nodes[p]['label']))[0])
                                    #dbscan_names.update([included_src])
                                    dbscan_names.append(included_src)
                                    if previous_flag != last_element[0]:
                                        previous_flag = last_element[0]
                                        iteration += 1

            print("Spectral Cluster " + str(i+1) +"DBSCAN Cluster " + str(p+1) +" Number Of Iterations:" + str(iteration))
            cust_dbscan_cluster_nodes[p]['iterations'] = iteration

        cust_nodes = []
        cust_labels = []
        for h in range(len(cust_dbscan_cluster_nodes)):
            cust_nodes.extend(list(cust_dbscan_cluster_nodes[h]['nodes']))
            cust_labels.extend(list(cust_dbscan_cluster_nodes[h]['label']))
        cust_clusterwise_embedding_nodes[i]['name'] = cust_nodes
        cust_clusterwise_embedding_nodes[i]['label'] = cust_labels

        for k in range(len(cust_dbscan_cluster_nodes)):
            hub_bar_list = {}
            authority_bar_list = {}
            line_x_values = {}
            hub_color = {}
            authority_color = {}
            for bar, bar_value in zip(list(sorted_hubs.keys()), list(sorted_hubs.values())):
                hub_bar_list[bar] = 0
                authority_bar_list[bar] = 0
                line_x_values[bar] = ""
                hub_color[bar] ="red"
                authority_color[bar] = "red"

            sr_no = 0
            tr_no = 0
            pair_nodes = set()
            degree_cnt = 0
            edges = set()
            cls_nodes = set(cust_dbscan_cluster_nodes[k]['nodes'])
            hits_score_list = {}
            authority_score_list = {}
            percentile_score_list = {}
            s_degree_dict = {}
            trg_degree_dict = {}
            degree_dict = {}
            pair_node_list = set()
            pair_removal_edges = set()
            pop_index = []
            pop_element = []
            for nd in cust_dbscan_cluster_nodes[k]['nodes']:
                if nd.startswith("S"):
                    edge_list = list(bipartite_G.neighbors(nd))
                    common_neighbors = set(edge_list).intersection(cust_dbscan_cluster_nodes[k]['nodes'])
                    if len(common_neighbors) == 0:
                        get_index_src = cust_dbscan_cluster_nodes[k]['nodes'].index(nd)
                        pop_index.append(get_index_src)
                        pop_element.append(nd)
                    else:
                        sr_no += 1
                        if degree_dict.__contains__(len(common_neighbors)) == False:
                            degree_dict[len(common_neighbors)] = []
                            degree_dict[len(common_neighbors)].append(nd)
                        else:
                            degree_dict[len(common_neighbors)].append(nd)
                        if s_degree_dict.__contains__(len(common_neighbors)) == False:
                            s_degree_dict[len(common_neighbors)] = []
                            s_degree_dict[len(common_neighbors)].append(nd)
                        else:
                            s_degree_dict[len(common_neighbors)].append(nd)
                        node_no = nd[1:]
                        tr_node = "T" + str(node_no)
                        if list(cust_dbscan_cluster_nodes[k]['nodes']).__contains__(
                                tr_node) and pair_nodes.__contains__(node_no) == False:
                            pair_nodes.add(node_no)
                elif nd.startswith("T"):
                    tr_no += 1
                    node_no = nd[1:]
                    sr_node = "S" + str(node_no)
                    if list(cust_dbscan_cluster_nodes[k]['nodes']).__contains__(sr_node) and pair_nodes.__contains__(
                            node_no) == False:
                        pair_nodes.add(node_no)
                    edge_list = list(bipartite_G.neighbors(nd))
                    common_neighbors = set(edge_list).intersection(cust_dbscan_cluster_nodes[k]['nodes'])
                    if degree_dict.__contains__(len(common_neighbors)) == False:
                        degree_dict[len(common_neighbors)] = []
                        degree_dict[len(common_neighbors)].append(nd)
                    else:
                        degree_dict[len(common_neighbors)].append(nd)

                    if trg_degree_dict.__contains__(len(common_neighbors)) == False:
                        trg_degree_dict[len(common_neighbors)] = []
                        trg_degree_dict[len(common_neighbors)].append(nd)
                    else:
                        trg_degree_dict[len(common_neighbors)].append(nd)
                edge_list = list(bipartite_G.neighbors(nd))
                common_neighbors = set(edge_list).intersection(cls_nodes)
                hits_score_list[nd[1:]] = sorted_hubs[nd[1:]] * 1e-4
                hub_bar_list[nd[1:]] = max(sorted_hubs.values())
                authority_bar_list[nd[1:]] = max(sorted_authorities.values())
                line_x_values[nd[1:]] = nd[1:]
                authority_score_list[nd[1:]] = sorted_authorities[nd[1:]] * 1e-4
                if nd.startswith("S"):
                    hub_color[nd[1:]] = "blue"
                    authority_color[nd[1:]] = "blue"

                for cm_ed in common_neighbors:
                    if (nd.startswith("S") == True) and (str(cm_ed).startswith("S") == False):
                        edge = str(nd) + "-" + str(cm_ed)
                        if pair_node_list.__contains__(cm_ed) == False and pair_removal_edges.__contains__(
                                str(nd) + "-" + str(cm_ed)) == False:
                            removed_edges = str(nd) + "-" + str(cm_ed)
                    elif nd.startswith("T") and (str(cm_ed).startswith("S") == True):
                        edge = str(cm_ed) + "-" + str(nd)
                        if pair_node_list.__contains__(nd) == False and pair_removal_edges.__contains__(
                                str(cm_ed) + "-" + str(nd)) == False:
                            removed_edges = str(cm_ed) + "-" + str(nd)
                    edges.add(edge)
                    pair_removal_edges.add(removed_edges)

            for nd in pop_element:
                get_index_src = cust_dbscan_cluster_nodes[k]['nodes'].index(nd)
                cust_dbscan_cluster_nodes[k]['nodes'].pop(get_index_src)
                cust_dbscan_cluster_nodes[k]['original_embedding'].pop(get_index_src)
                cust_dbscan_cluster_nodes[k]['status'].pop(get_index_src)
                cust_dbscan_cluster_nodes[k]['label'].pop(get_index_src)

            sorted_degree_dict = dict(sorted(degree_dict.items()))
            for key, value in sorted_degree_dict.items():
                degree_node_row = []
                degree_node_row.append(i + 1)
                degree_node_row.append(k + 1)
                degree_node_row.append(key)
                degree_node_row.append(len(value))
                degree_node_row.append(value)
                node_degree_list.append(degree_node_row)

            sorted_degree_dict = dict(sorted(trg_degree_dict.items()))
            for key, value in sorted_degree_dict.items():
                degree_node_row = []
                degree_node_row.append(i + 1)
                degree_node_row.append(k + 1)
                degree_node_row.append(key)
                degree_node_row.append(len(value))
                degree_node_row.append(value)
                trg_node_degree_list.append(degree_node_row)

            sorted_degree_dict = dict(sorted(s_degree_dict.items()))
            for key, value in sorted_degree_dict.items():
                degree_node_row = []
                degree_node_row.append(i + 1)
                degree_node_row.append(k + 1)
                degree_node_row.append(key)
                degree_node_row.append(len(value))
                degree_node_row.append(value)
                s_trg_node_degree_list.append(degree_node_row)

            degree_summary_info_row = []
            degree_summary_info_row.append(i + 1)
            degree_summary_info_row.append(k + 1)
            degree_list = list(degree_dict.keys())
            if len(degree_list) > 0:
                min_degree = min(degree_list)
                max_degree = max(degree_list)
                avg_degree = sum(degree_list) / len(degree_list)
                degree_summary_info_row.append(min_degree)
                degree_summary_info_row.append(max_degree)
                degree_summary_info_row.append(avg_degree)
                degree_summary_info_row.append(str() + "/" + str() + "/" + str())
                degree_summary_info.append(degree_summary_info_row)
            else:
                degree_summary_info_row.append(0)
                degree_summary_info_row.append(0)
                degree_summary_info_row.append(0)
                degree_summary_info_row.append(0)
                degree_summary_info.append(degree_summary_info_row)

            degree_summary_info_row = []
            degree_summary_info_row.append(i + 1)
            degree_summary_info_row.append(k + 1)
            degree_list = list(trg_degree_dict.keys())
            if len(degree_list) > 0:
                min_degree = min(degree_list)
                max_degree = max(degree_list)
                avg_degree = sum(degree_list) / len(degree_list)
                degree_summary_info_row.append(min_degree)
                degree_summary_info_row.append(max_degree)
                degree_summary_info_row.append(avg_degree)
                degree_summary_info_row.append(str(min_degree) + "/" + str(max_degree) + "/" + str(avg_degree))
                degree_summary_info_trg.append(degree_summary_info_row)
            else:
                degree_summary_info_row.append(0)
                degree_summary_info_row.append(0)
                degree_summary_info_row.append(0)
                degree_summary_info_row.append(0)
                degree_summary_info_trg.append(degree_summary_info_row)

            degree_summary_info_row = []
            degree_summary_info_row.append(i + 1)
            degree_summary_info_row.append(k + 1)
            degree_list = list(s_degree_dict.keys())
            if len(degree_list) > 0:
                min_degree = min(degree_list)
                max_degree = max(degree_list)
                avg_degree = sum(degree_list) / len(degree_list)
                degree_summary_info_row.append(min_degree)
                degree_summary_info_row.append(max_degree)
                degree_summary_info_row.append(avg_degree)
                degree_summary_info_row.append(str(min_degree) + "/" + str(max_degree) + "/" + str(avg_degree))
                degree_summary_info_s.append(degree_summary_info_row)
            else:
                degree_summary_info_row.append(0)
                degree_summary_info_row.append(0)
                degree_summary_info_row.append(0)
                degree_summary_info_row.append(0)
                degree_summary_info_s.append(degree_summary_info_row)

            cust_dbscan_cluster_nodes[k]['pair'] = pair_node_list
            cust_dbscan_cluster_nodes_pair_removed[k]['nodes'] = set(cust_dbscan_cluster_nodes[k]['nodes']) - pair_node_list
            cust_dbscan_cluster_nodes_pair_removed[k]['iterations'] = cust_dbscan_cluster_nodes[k]['iterations']
            cust_dbscan_cluster_nodes_pair_removed[k]['pair'] = pair_node_list

            row = []
            row.append(i + 1)
            row.append(k + 1)
            row.append(len(cust_dbscan_cluster_nodes[k]['nodes']))
            row.append(len(dbscan_cluster_info[k]['names']))
            row.append(sr_no)
            row.append(tr_no)
            row.append(len(edges))
            row.append(len(pair_nodes))
            row.append(cust_dbscan_cluster_nodes[k]['iterations'])
            row.append(cust_dbscan_cluster_nodes[k]['nodes'])
            # row.append(edges)
            # row.append(hits_score_list)
            # row.append(authority_score_list)
            dbscan_rows.append(row)

            row = []
            row.append(i + 1)
            row.append(k + 1)
            row.append(len(cust_dbscan_cluster_nodes[k]['nodes']))
            row.append(len(cust_dbscan_cluster_nodes[k]['nodes']) - len(pair_node_list))
            row.append(len(dbscan_cluster_info[k]['names']))
            row.append(sr_no)
            row.append(tr_no)
            row.append(sr_no - len(pair_nodes))
            row.append(tr_no - len(pair_nodes))
            row.append(len(edges))
            row.append(len(pair_removal_edges))
            row.append(len(pair_nodes))
            row.append(cust_dbscan_cluster_nodes[k]['iterations'])
            dbscan_pair_removed_rows.append(row)

            list_row = []
            list_row.append(i + 1)
            list_row.append(k + 1)
            list_row.append(pair_nodes)
            list_row.append(cust_dbscan_cluster_nodes[k]['nodes'])
            list_row.append(edges)
            node_edge_rows.append(list_row)

            y_min = min(list(sorted_hubs.values()))
            y_max = max(list(sorted_hubs.values()))
            figure_cluster_hub = plt.figure(figsize=(15, 15))
            ax_cluster_hub = figure_cluster_hub.add_subplot(111)
            ax_cluster_hub.set_ylim(y_min, y_max)
            ax_cluster_hub.plot(list(sorted_hubs.keys()), list(sorted_hubs.values()), label="Bipartite Graph Hub Score")
            ax_cluster_hub.bar(list(sorted_hubs.keys()), list(hub_bar_list.values()), color=list(hub_color.values()), label="Cluster Hub Score Location")
            plt.legend()
            plt.title("Node Hub Score Position(Source_SDM Clustering) For Spectral Cluster No:" + str(i+1) + "DBSCAN Cluster No:" + str(k+1))
            #plt.show()
            plt.savefig(citeseer_spectral_cluster + "node_hub_score_position_spectral_cls_source_clustering"+str(i+1) + "dbscan_cls_" + str(k+1) +".png")

            y_min = min(list(sorted_authorities.values()))
            y_max = max(list(sorted_authorities.values()))
            figure_cluster_authority = plt.figure(figsize=(15, 15))
            ax_cluster_authority = figure_cluster_authority.add_subplot(111)
            ax_cluster_authority.set_ylim(y_min, y_max)
            ax_cluster_authority.plot(list(sorted_authorities.keys()), list(sorted_authorities.values()), label="Bipartite Graph Authority Score")
            ax_cluster_authority.bar(list(sorted_authorities.keys()), list(authority_bar_list.values()), color=list(authority_color.values()), label="Cluster Authority Score Location")
            plt.legend()
            plt.title("Node Authority Score Position(Source_SDM Clustering)  For Spectral Cluster No:" + str(i + 1) + "DBSCAN Cluster No:" + str(k + 1))
            #plt.show()
            plt.savefig(citeseer_spectral_cluster + "node_authority_score_position_spectral_cls_source_clustering" + str(i + 1) + "dbscan_cls_" + str(k + 1) + ".png")



    degree_list_col = ['Spectral Cluster No', 'DBSCAN Cluster No', 'Degree Of Trg Node', 'No Of Degree Trg Node', 'Trg Node List']
    pd.DataFrame(trg_node_degree_list, columns=degree_list_col).to_csv(target_citeseer_node_degree_list)

    degree_list_col = ['Spectral Cluster No', 'DBSCAN Cluster No', 'Degree Of Node', 'No Of Degree Node', 'Node List']
    pd.DataFrame(node_degree_list, columns=degree_list_col).to_csv(citeseer_node_degree_list)

    degree_list_col = ['Spectral Cluster No', 'DBSCAN Cluster No', 'Degree Of Src- Node', 'No Of Degree Src- Node',
                       'Src- Node List']
    pd.DataFrame(s_trg_node_degree_list, columns=degree_list_col).to_csv(source_target_citeseer_node_degree_list)

    pd.DataFrame(degree_summary_info,
                 columns=['Spectral Cluster No', 'DBSCAN Cluster No', 'Min Degree', 'Max Degree', 'Avg Degree',
                          'Combined Result']).to_csv(citeseer_node_degree_summary)
    pd.DataFrame(degree_summary_info_trg,
                 columns=['Spectral Cluster No', 'DBSCAN Cluster No', 'Min Degree in Target', 'Max Degree in Target',
                          'Avg Degree in Target', 'Combined Result']).to_csv(target_citeseer_node_degree_summary)
    pd.DataFrame(degree_summary_info_s,
                 columns=['Spectral Cluster No', 'DBSCAN Cluster No', 'Min Degree in Source_SDM', 'Max Degree in Source_SDM',
                          'Avg Degree in Source_SDM', 'Combined Result']).to_csv(source_citeseer_node_degree_summary)

    # target clustering
    dbscan_cols = ['Spectral Cluster No', 'DBSCAN Cluster No', 'No Of Target Nodes in Cluster', 'Min Sample', 'Eps',
                   'List Of Nodes', 'Node2vec 20-D Embedding', 'X Value(TSNE)', 'Y Value(TSNE)']
    pd.DataFrame(rows, columns=dbscan_cols).to_csv(citeseer_dbscan_cluster_csv_file)

    nodd_edge_cols = ['Spectral Cluster No', 'DBSCAN Cluster No', "Pair Removed", "List Of Nodes", "List Of Edges"]
    pd.DataFrame(node_edge_rows, columns=nodd_edge_cols).to_csv(citeseer_dbscan_cluster_node_edge_csv_file)
    save_as_pickle(citeseer_dbscan_cluster_node_edge_pkl, node_edge_rows)

    # target columns
    '''cust_dbscan_cols = ["Spectral Cluster No", "DBSCAN Cluster No", "No Of Nodes in Cluster", "Original No Of Sources","No Of Source_SDM Nodes","No Of Target Nodes",
                        "No Of Edges","No Of Pairs", "Iteration for Convergence", "List Of Nodes", "List Of Edges", "Original Bipartite Graph Hub Score", "Original Bipartite Graph Authority Score"]'''
    cust_dbscan_cols = ["Spectral Cluster No", "DBSCAN Cluster No", "No Of Nodes in Cluster", "Original No Of Sources",
                        "No Of Source_SDM Nodes", "No Of Target Nodes",
                        "No Of Edges", "No Of Pairs", "Iteration for Convergence", "List Of Nodes"]

    pd.DataFrame(dbscan_rows, columns=cust_dbscan_cols).to_csv(citeseer_cust_dbscan_cluster_csv_file)

    cust_dbscan_pair_removed_cols = ["Spectral Cluster No", "DBSCAN Community No", "No Of Nodes in Community",
                                     "no Of Nodes in Community After Pair Removed",
                                     "No Of Source_SDM Nodes After DBSCAN", "No Of Source_SDM Nodes",
                                     "No Of Target Nodes", "No Of Source_SDM Nodes After Pair Removed",
                                     "No Of Target Nodes After Pair Removed",
                                     "No Of Edges",
                                     "No Of Edges After Pair Removed", "No Of Pairs", "No Of Iterations"]
    pd.DataFrame(dbscan_pair_removed_rows, columns=cust_dbscan_pair_removed_cols).to_csv(citeseer_pair_removed_cust_dbscan_cluster_csv_file)

    avg_distance_cols = ["Cluster No", "Source_SDM Names", "5th Nearest Neighbor Distance"]
    pd.DataFrame(avg_distances_rows, columns=avg_distance_cols).to_csv(citeseer_5th_nearest_neighbor_distance_csv)

    not_included_nodes = []
    src_nodes_not_included = []
    trg_nodes_not_included = []
    flag = 0
    for gn in bipartite_G.nodes():
        for k in range(len(node_edge_rows)):
            cls_nodes = list(node_edge_rows[k][3])
            if cls_nodes.__contains__(gn) == True:
                flag = 1
                break
        if flag == 0:
            not_included_nodes.append(gn)
            if gn.startswith("S"):
                src_nodes_not_included.append(gn)
            elif gn.startswith("T"):
                trg_nodes_not_included.append(gn)
        else:
            flag = 0

    row = []
    row.append("Total No of Nodes which are not part of any community")
    row.append(len(not_included_nodes))
    row.append("No of Source_SDM Nodes which are not part of any community")
    row.append(len(src_nodes_not_included))
    row.append("No of Target Nodes which are not part of any community")
    row.append(len(trg_nodes_not_included))
    row.append("List of nodes which are not part of any community")
    row.append(not_included_nodes)
    save_as_pickle(citeseer_nodes_not_part_of_any_community, row)
    pd.DataFrame(row).to_csv(citeseer_nodes_not_in_cluster_csv)

    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 0),  # Maroon
        (0, 128, 0),  # Green
        (0, 0, 128),  # Navy
        (128, 128, 0),  # Olive
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
        (255, 165, 0),  # Orange
        (128, 128, 128),  # Gray
        (255, 192, 203),  # Pink
        (0, 128, 0),  # Green
        (128, 0, 0),  # Maroon
        (0, 0, 128),  # Navy
        (0, 128, 128),  # Teal
        (128, 0, 128),  # Purple
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
    ]

    normalized_colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]

    selected_colors = [normalized_colors[i] for i in clusters]

    tsne = TSNE(n_components=2, perplexity=25)
    embeddings_2d = tsne.fit_transform(np.array(middleDimentions))

    x_min = min(embeddings_2d[:, 0])
    x_max = max(embeddings_2d[:, 0])
    y_min = min(embeddings_2d[:, 1])
    y_max = max(embeddings_2d[:, 1])
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    figure = plt.figure(figsize=(15, 15))
    ax = figure.add_subplot(111)
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=selected_colors, cmap='jet')
    plt.title(title + "( No of Clusters " + str(len(set(clusters))) + ")")
    plt.savefig(spectral_cluster_plot_file_name)
    plt.legend()
    # plt.show()

    embeddings_2d = pd.DataFrame(embeddings_2d, index=middleDimentions.index)

    '''Original Embedding'''
    k = 0
    for cls_nodes in cluster_dict.values():
        source_x_embedding = []
        source_y_embedding = []
        target_x_embedding = []
        target_y_embedding = []
        source_labels = []
        target_labels = []
        source_colors = []
        target_colors = []
        figure1 = plt.figure(figsize=(15, 15))
        ax1 = figure1.add_subplot(111)
        color = [k for j in range(len(cls_nodes))]

        for cls_node in cls_nodes:
            scatter = ax1.scatter(embeddings_2d.loc[cls_node][0], embeddings_2d.loc[cls_node][1], c=normalized_colors[k], cmap="jet")
            if cls_node.startswith("S"):
                source_x_embedding.append(embeddings_2d.loc[cls_node][0])
                source_y_embedding.append(embeddings_2d.loc[cls_node][1])
                source_labels.append(cls_node)
                source_colors.append("blue")
            elif cls_node.startswith("T"):
                target_x_embedding.append(embeddings_2d.loc[cls_node][0])
                target_y_embedding.append(embeddings_2d.loc[cls_node][1])
                target_labels.append(cls_node)
                target_colors.append("red")

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.title(citeseer_node_embedding_title + " "+ str(k+1))
        cluster_file_name = citeseer_spectral_cluster + str(k + 1)
        plt.savefig(cluster_file_name)
        #plt.show()

        figure2 = plt.figure(figsize=(15, 15))
        ax2 = figure2.add_subplot(111)
        source_scatter = ax2.scatter(source_x_embedding, source_y_embedding, cmap="Set2", c=source_colors, label="Source_SDM")
        target_scatter = ax2.scatter(target_x_embedding, target_y_embedding, cmap="Set2", c=target_colors, label="Target")
        plt.title(citeseer_TSNE_clusterwise_source_target_title + str(k+1))
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.savefig(citesser_TSNE_clusterwise_source_target_plot + str(k + 1))
        #plt.show()

        figure22 = plt.figure(figsize=(15, 15))
        ax2 = figure22.add_subplot(111)
        source_scatter = ax2.scatter(source_x_embedding, source_y_embedding, cmap="Set2", c=source_colors, label="Source_SDM")
        plt.title(citeseer_source_nodes_only_title + str(k + 1))
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.savefig(citesser_clusterwise_source_plot + str(k + 1))
        # plt.show()

        cluster_wise_embedding_x = []
        cluster_wise_embedding_y = []
        cluster_wise_embedding_label = []
        m = 0
        for cls_name in clusterwise_embedding_src_nodes[k]['name']:
            cluster_wise_embedding_x.append(embeddings_2d.loc[cls_name][0])
            cluster_wise_embedding_y.append(embeddings_2d.loc[cls_name][1])
            cluster_wise_embedding_label.append(clusterwise_embedding_src_nodes[k]['label'][m])
            m += 1
        figure3 = plt.figure(figsize=(15, 15))
        ax3 = figure3.add_subplot(111)
        clusterwise_source_scatter = ax3.scatter(cluster_wise_embedding_x, cluster_wise_embedding_y, cmap="Set2", c=cluster_wise_embedding_label, label="Source_SDM Nodes")
        plt.title(citeseer_TSNE_clusterwise_source_with_cluster_title + str(k+1) + " No Of Clusters:" + str(len(set(cluster_wise_embedding_label))))
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.savefig(citesser_TSNE_clusterwise_source_with_cluster_labels_plot + str(k + 1))
        #plt.show()

        cust_cluster_wise_embedding_x = []
        cust_cluster_wise_embedding_y = []
        cust_cluster_wise_embedding_label = []
        m = 0
        for cls_name in cust_clusterwise_embedding_nodes[k]['name']:
            cust_cluster_wise_embedding_x.append(embeddings_2d.loc[cls_name][0])
            cust_cluster_wise_embedding_y.append(embeddings_2d.loc[cls_name][1])
            cust_cluster_wise_embedding_label.append(cust_clusterwise_embedding_nodes[k]['label'][m])
            m += 1
        figure10 = plt.figure(figsize=(15, 15))
        ax10 = figure10.add_subplot(111)
        cust_clusterwise_source_scatter = ax10.scatter(cust_cluster_wise_embedding_x, cust_cluster_wise_embedding_y, cmap="Set2", c=cust_cluster_wise_embedding_label)
        plt.title(citeseer_customized_dbscan_clusters + str(k + 1) + "( No of Clusters " + str(len(list(set(cust_clusterwise_embedding_nodes[k]['label'])))) + ")")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.savefig(citeseer_customized_dbscan_clusters_plot + str(k + 1))
        plt.legend()
        #plt.show()

        k += 1


    cols = ["Cluster No",
            "No of Nodes in Cluster",
            "No of Sources",
            "No Of Zero Degree Source_SDM",
            "No of Sources - No Of Zero Degree Source_SDM",
            "No of Targets",
            "No Of Zero Degree Target",
            "No of Targets - No Of Zero Degree Target",
            "No of Zero-degree Nodes",
            "No of Outgoing Edges in Cluster (to nodes only within cluster)",
            "No of Incoming Edges in Cluster (from nodes only within cluster)",
            "No of possible edges",
            "No of Only Know Each Other (As Source_SDM)",
            "No of Only Know Each Other (As Target)",
            "Edge Percentage",
            "Average Degree Of Cluster",
            "Hub/Authority Score",
            "Highest Graph Source_SDM Degree",
            "Highest Graph Target Degree",
            "Highest Cluster Source_SDM Degree",
            "Highest Cluster Target Degree",
            "No of Source_SDM and Target Pair available",
            "No of Target and Source_SDM Pair available",
            "Source_SDM Target Pairs",
            "Target Source_SDM Pairs",
            "List of Nodes in Cluster",
            "List of Zero-degree Nodes in Cluster",
            "List of Only Know Each Other (As Source_SDM)",
            "List of Only Know Each Other (As Target)",
            "Singleton Nodes",
            "List of Connected Target Nodes Out-off Cluster",
            "No of Connected Target Nodes Out-off Cluster",
            "List of Connected Source_SDM Nodes Out-off Cluster",
            "No of Connected Source_SDM Nodes Out-off Cluster"]
    rows = []
    new_rows = []
    edges_rows = []
    cluster_wise_label_map = {str(class_name): {} for class_name in range(len(cluster_dict))}
    node_degree_dict = dict(bipartite_G.degree())
    total_degree_row = []
    total_cluster_degree_row = []
    total_cluster_degree = []
    zero_degree_rows = []
    for i in range(len(cluster_dict)):
        row = []
        edge_row = []
        row.append(i + 1)
        source_no = 0
        target_no = 0
        source_list = []
        target_list = []
        original_graph_edge_dict = {}
        source_target_pair = []
        target_source_pair = []
        src_high_degree = -1
        trg_high_degree = -1
        degree_list = []
        zero_src_nodes = []
        zero_trg_nodes = []
        for node in cluster_dict[i]:
            if node.startswith("S"):
                if node_degree_dict[node] > src_high_degree:
                    src_high_degree = node_degree_dict[node]
                if bipartite_G.degree(node) == 0:
                    zero_src_nodes.append(node)
                source_list.append(node)
                source_no += 1
            elif node.startswith("T"):
                if node_degree_dict[node] > trg_high_degree:
                    trg_high_degree = node_degree_dict[node]
                if bipartite_G.degree(node) == 0:
                    zero_trg_nodes.append(node)
                target_list.append(node)
                target_no += 1
            degree_list.append(node_degree_dict[node])
            original_graph_edge_dict[node] = list(bipartite_G.neighbors(node))

        unique_degrees = set(degree_list)
        unique_degree_dict = {}
        for z in range(len(degree_list)):
            if unique_degree_dict.keys().__contains__("Degree " + str(list(degree_list)[z])):
                unique_degree_dict["Degree " + str(list(degree_list)[z])] = unique_degree_dict["Degree " + str(
                    list(degree_list)[z])] + 1
            else:
                unique_degree_dict["Degree " + str(list(degree_list)[z])] = 1

        '''bin_width = 5
        num_bins = max(int((max(degree_list) - min(degree_list)) / bin_width), 1)
        n, bins, patches = plt.hist(degree_list, bins=num_bins)'''
        fig = plt.figure(figsize=(10, 8))
        plt.bar(list(unique_degree_dict.keys()), list(unique_degree_dict.values()), align='center')
        plt.xticks(list(unique_degree_dict.keys()), list(unique_degree_dict.keys()), rotation=90, ha='right')

        for k in range(len(list(unique_degree_dict.keys()))):
            plt.annotate(list(unique_degree_dict.values())[k],
                         (list(unique_degree_dict.keys())[k], list(unique_degree_dict.values())[k]), ha='center',
                         va='bottom')

        # plt.xlim(min(list(unique_degree_dict.keys())) - 0.5, max(list(unique_degree_dict.keys())) + 0.5)
        # Set labels and title
        plt.xlabel('Degree Level')
        plt.ylabel('No of Nodes in Degree')
        plt.title('Bipartite Graph Degree Distribution' + str(i + 1))
        plt.savefig(citesser_bipartite_graph_degree_distribution + str(i + 1) + ".png")
        # Display the plot
        # plt.show()

        degree_cols = ["Cluster No", "Node List", "Degree List", "Bipartite Graph Degree Distribution"]
        degree_row = []
        degree_row.append(i + 1)
        degree_row.append(cluster_dict[i])
        degree_row.append(degree_list)
        degree_row.append(unique_degree_dict)
        total_degree_row.append(degree_row)
        '''for j in range(len(n)):
            if n[j] > 0:
                plt.text(bins[j] + bin_width / 2, n[j], str(int(n[j])), ha='center', va='bottom')

        bin_ranges = ['{:.1f}-{:.1f}'.format(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]
        tick_locations = bins[:-1] + bin_width/2  # Adjust tick positions
        plt.xticks(tick_locations, bin_ranges, rotation=45, ha='right')'''

        total_edges_in_cluster_as_source = 0
        total_edges_in_graph_with_source = 0
        total_connected_target_nodes_not_in_cls = []
        source_edges = []
        cluster_degree = {}
        node_edges_count = {}
        highest_src_cls_degree = -1
        source_list_without_zero_degree = []
        zero_degree_nodes = []
        only_know_each_other_as_source = []
        singleton_nodes = []
        for source in source_list:
            source_neighbor_list = original_graph_edge_dict[source]
            common_targets_in_cluster = set(source_neighbor_list) & set(target_list)

            if len(common_targets_in_cluster) == 0:
                zero_degree_row = []
                zero_degree_row.append(i + 1)
                zero_degree_row.append(source)
                zero_degree_row.append("Degree " + str(len(common_targets_in_cluster)))
                zero_degree_row.append(len(source_neighbor_list))
                zero_degree_rows.append(zero_degree_row)
                zero_degree_nodes.append(source)
            else:
                source_list_without_zero_degree.append(source)

                if len(common_targets_in_cluster) == 1:
                    only_source = original_graph_edge_dict[list(common_targets_in_cluster)[0]]
                    only_source_list = set(only_source) & set(source_list)
                    if len(only_source_list) == 1:
                        if list(only_source_list)[0] == source:
                            only_know_each_other_as_source.append(source + "->" + list(common_targets_in_cluster)[0])
                            singleton_nodes.append(source)
                            singleton_nodes.append(list(common_targets_in_cluster)[0])
                            source_list_without_zero_degree.remove(source)

                if len(common_targets_in_cluster) > highest_src_cls_degree:
                    highest_src_cls_degree = len(common_targets_in_cluster)
                node_edges_count[source] = common_targets_in_cluster
                if cluster_degree.keys().__contains__("Degree " + str(len(common_targets_in_cluster))):
                    cluster_degree["Degree " + str(len(common_targets_in_cluster))] = cluster_degree["Degree " + str(
                        len(common_targets_in_cluster))] + 1
                else:
                    cluster_degree["Degree " + str(len(common_targets_in_cluster))] = 1

                connected_target_nodes_not_in_cls = set(source_neighbor_list) - set(common_targets_in_cluster)
                for not_cls_trg in connected_target_nodes_not_in_cls:
                    if not_cls_trg not in total_connected_target_nodes_not_in_cls:
                        total_connected_target_nodes_not_in_cls.append(not_cls_trg)
                total_edges_in_cluster_as_source += len(common_targets_in_cluster)
                total_edges_in_graph_with_source += len(source_neighbor_list)
                '''source_suffix = source[1:]
                target_string = "T" + source_suffix
                if target_list.__contains__(target_string):
                    source_target_pair.append(source + "-" + target_string)'''
                for trg in common_targets_in_cluster:
                    source_edges.append(source + '-' + trg)

        total_edges_in_cluster_as_target = 0
        total_edges_in_graph_with_target = 0
        total_connected_source_nodes_not_in_cls = []
        target_edges = []
        highest_trg_cls_degree = -1

        target_list_without_zero_degree = []
        only_know_each_other_as_target = []
        for target in target_list:
            target_neighbor_list = original_graph_edge_dict[target]
            common_source_in_cluster = set(target_neighbor_list) & set(source_list)
            if len(common_source_in_cluster) > highest_trg_cls_degree:
                highest_trg_cls_degree = len(common_source_in_cluster)

            zero_degree_row = []
            if len(common_source_in_cluster) == 0:
                zero_degree_row.append(i + 1)
                zero_degree_row.append(target)
                zero_degree_row.append("Degree " + str(len(common_source_in_cluster)))
                zero_degree_row.append(len(target_neighbor_list))
                zero_degree_rows.append(zero_degree_row)
                zero_degree_nodes.append(target)
            else:
                target_list_without_zero_degree.append(target)

                if len(common_source_in_cluster) == 1:
                    only_target = original_graph_edge_dict[list(common_source_in_cluster)[0]]
                    only_target_list = set(only_target) & set(target_list)
                    if len(only_target_list) == 1:
                        if list(only_target_list)[0] == target:
                            only_know_each_other_as_target.append(list(common_source_in_cluster)[0] + "->" + target)
                            singleton_nodes.append(target)
                            singleton_nodes.append(list(common_source_in_cluster)[0])
                            target_list_without_zero_degree.remove(target)

                node_edges_count[target] = common_source_in_cluster
                if cluster_degree.keys().__contains__("Degree " + str(len(common_source_in_cluster))):
                    cluster_degree["Degree " + str(len(common_source_in_cluster))] = cluster_degree["Degree " + str(len(common_source_in_cluster))] + 1
                else:
                    cluster_degree["Degree " + str(len(common_source_in_cluster))] = 1
                connected_source_nodes_not_in_cls = set(target_neighbor_list) - set(common_source_in_cluster)

                for not_cls_src in connected_source_nodes_not_in_cls:
                    if not_cls_src not in total_connected_source_nodes_not_in_cls:
                        total_connected_source_nodes_not_in_cls.append(not_cls_src)

                # total_connected_source_nodes_not_in_cls.extend(list(connected_source_nodes_not_in_cls))
                total_edges_in_cluster_as_target += len(common_source_in_cluster)
                total_edges_in_graph_with_target += len(target_neighbor_list)
                '''target_suffix = target[1:]
                source_string = "S" + target_suffix
                if source_list.__contains__(source_string):
                    target_source_pair.append(source_string + "-" + target)'''
                for src in common_source_in_cluster:
                    target_edges.append(src + '-' + target)

        fig = plt.figure(figsize=(10, 8))
        plt.bar(list(cluster_degree.keys()), list(cluster_degree.values()), align='center')
        plt.xticks(list(cluster_degree.keys()), list(cluster_degree.keys()), rotation=90, ha='right')

        for k in range(len(list(cluster_degree.keys()))):
            plt.annotate(list(cluster_degree.values())[k],
                         (list(cluster_degree.keys())[k], list(cluster_degree.values())[k]), ha='center', va='bottom')

        # plt.xlim(min(list(unique_degree_dict.keys())) - 0.5, max(list(unique_degree_dict.keys())) + 0.5)
        # Set labels and title
        plt.xlabel('Degree Level')
        plt.ylabel('No of Nodes in Degree')
        plt.title('Cluster Degree Distribution of Current Nodes in Cluster' + str(i + 1))
        plt.savefig(citesser_cluster_degree_distribution + str(i + 1) + ".png")
        # Display the plot
        # plt.show()

        for source in source_list_without_zero_degree:
            source_suffix = source[1:]
            target_string = "T" + source_suffix
            if target_list_without_zero_degree.__contains__(target_string):
                source_target_pair.append(source + "-" + target_string)
            '''for trg in common_targets_in_cluster:
                source_edges.append(source + '-' + trg)'''

        for target in target_list_without_zero_degree:
            target_suffix = target[1:]
            source_string = "S" + target_suffix
            if source_list_without_zero_degree.__contains__(source_string):
                target_source_pair.append(source_string + "-" + target)
            '''for src in common_source_in_cluster:
                target_edges.append(src + '-' + target)'''

        cluster_degree_row = []
        cluster_degree_row.append(i + 1)
        cluster_degree_row.append(cluster_dict[i])
        cluster_degree_row.append(node_edges_count)
        cluster_degree_row.append(cluster_degree)
        total_cluster_degree_row.append(cluster_degree_row)
        cluster_degree_cols = ["Cluster No", "Node List", "Node Edge Count", "Cluster Degree Distribution"]

        removed_zero_degree_nodes = set(cluster_dict[i]) - set(zero_degree_nodes)
        removed_singleton_nodes = set(removed_zero_degree_nodes) - set(singleton_nodes)
        row.append(len(removed_singleton_nodes))
        # row.append(source_no)
        # row.append(target_no)
        row.append(len(source_list_without_zero_degree))
        row.append(len(zero_src_nodes))
        row.append(len(source_list_without_zero_degree) - len(zero_src_nodes))
        row.append(len(target_list_without_zero_degree))
        row.append(len(zero_trg_nodes))
        row.append(len(target_list_without_zero_degree) - len(zero_trg_nodes))
        row.append(len(zero_degree_nodes))
        row.append(total_edges_in_cluster_as_source)
        row.append(total_edges_in_cluster_as_target)
        row.append(len(source_list_without_zero_degree) * len(target_list_without_zero_degree))
        row.append(len(only_know_each_other_as_source))
        row.append(len(only_know_each_other_as_target))
        if len(source_list_without_zero_degree) != 0 and len(target_list_without_zero_degree) != 0:
            row.append(total_edges_in_cluster_as_source / (len(source_list_without_zero_degree) * len(target_list_without_zero_degree)))
        else:
            row.append(0)
        row.append(total_edges_in_cluster_as_source / (source_no + target_no))
        if source_no > target_no:
            row.append(total_edges_in_cluster_as_source / target_no)
        elif source_no < target_no:
            row.append(total_edges_in_cluster_as_source / source_no)
        row.append(src_high_degree)
        row.append(trg_high_degree)
        row.append(highest_src_cls_degree)
        row.append(highest_trg_cls_degree)
        row.append(len(source_target_pair))
        row.append(len(target_source_pair))
        row.append(source_target_pair)
        row.append(target_source_pair)
        # row.append(total_edges_in_graph_with_source)
        # row.append(total_edges_in_graph_with_target)
        # TODO: Consider List Of Nodes after removing zero degree nodes and singleton nodes or Original Node List
        row.append(sorted(cluster_dict[i], key=key_func))
        row.append(sorted(zero_degree_nodes, key=key_func))
        row.append(only_know_each_other_as_source)
        row.append(only_know_each_other_as_target)
        row.append(set(singleton_nodes))
        # row.append(sorted(cluster_dict[i], key=key_func))
        row.append(total_connected_target_nodes_not_in_cls)
        row.append(len(total_connected_target_nodes_not_in_cls))
        row.append(total_connected_source_nodes_not_in_cls)
        row.append(len(total_connected_source_nodes_not_in_cls))
        rows.append(row)

        nodes_not_in_cls_row = []
        nodes_not_in_cls_row.append(i)
        nodes_not_in_cls_row.append(total_connected_source_nodes_not_in_cls)
        nodes_not_in_cls_row.append(total_connected_target_nodes_not_in_cls)
        nodes_not_in_cls.append(nodes_not_in_cls_row)

        edge_row.append(i + 1)
        edge_row.append(source_edges)
        edge_row.append(target_edges)
        edges_rows.append(edge_row)

    nodes_not_in_cls_col = ['Cluster No', 'Source_SDM Nodes Not In Cluster', 'Target Nodes Not In Cluster']
    nodes_not_in_cls_dict = pd.DataFrame(nodes_not_in_cls, columns=nodes_not_in_cls_col)
    save_as_pickle(nodes_not_in_cls_pkl, nodes_not_in_cls_dict)
    zero_degree_cols = ['Cluster No', 'Node', 'Zero Degree in Cluster', 'Degree Ouside Cluster']
    pd.DataFrame(zero_degree_rows, columns=zero_degree_cols).to_csv(citeseer_zero_degree_node_analysis)
    result = pd.DataFrame(rows, columns=cols)
    #pd.DataFrame(result).to_csv(citation_spectral_clustering_csv, encoding='utf-8', float_format='%f')
    pd.DataFrame(total_degree_row, columns=degree_cols).to_csv("degreeList_out_off_cluster_nodes.csv", encoding='utf-8', float_format='%f')
    pd.DataFrame(total_cluster_degree_row, columns=cluster_degree_cols).to_csv("cluster_degree_out_off_cluster_nodes.csv", encoding='utf-8', float_format='%f')
    edge_cols = ['Cluster No', 'S-T edges', 'T-S edges']
    edge_rows_df = pd.DataFrame(edges_rows, columns=edge_cols)
    pd.DataFrame(edge_rows_df).to_csv(edge_file_name, encoding='utf-8', float_format='%f')

    return clusters, cluster_dict


def spectral_dbscan_clustering_citeseer_target(middleDimentions, no_of_clusters, citation_spectral_clustering_csv, title,
                                spectral_cluster_plot_file_name, bipartite_G, edge_file_name,
                                nodes_not_in_cls_pkl, original_citeseer_graph):
    sorted_hubs, sorted_authorities = hit_score_original_graph(original_citeseer_graph, citeseer_original_hub_csv,
                                                               citeseer_original_authority_csv,citeseer_hit_authority_score_csv,
                                                               citeseer_original_hub_plot,
                                                               citeseer_original_authority_plot,
                                                               citeseer_original_hub_title,
                                                               citeseer_original_authority_title)

    nodes_not_in_cls = []
    if exists(citeseer_clusters_list_pkl):
        clusters = read_pickel(citeseer_clusters_list_pkl)
    else:
        spectral = SpectralClustering(n_clusters=no_of_clusters, affinity='nearest_neighbors', n_neighbors=50, random_state=0)
        clusters = spectral.fit_predict(middleDimentions)
        save_as_pickle(citeseer_clusters_list_pkl, clusters)

    if exists(citeseer_cluster_nodes) and exists(citeseer_clusterwise_embedding_src_nodes) and exists(citeseer_cust_clusterwise_embedding_nodes) and exists(citeseer_cluster_dict):
        cluster_nodes = read_pickel(citeseer_cluster_nodes)
        clusterwise_embedding_src_nodes = read_pickel(citeseer_clusterwise_embedding_src_nodes)
        cust_clusterwise_embedding_nodes = read_pickel(citeseer_cust_clusterwise_embedding_nodes)
        cluster_dict = read_pickel(citeseer_cluster_dict)
    else:
        cluster_dict = {}
        cluster_nodes = {}
        clusterwise_embedding_src_nodes = {}
        cust_clusterwise_embedding_nodes = {}
        for q in range(no_of_clusters):
            cluster_dict[q] = []
            cluster_nodes[q] = {}
            cluster_nodes[q]['nodes'] = []
            cluster_nodes[q]['label'] = []
            cluster_nodes[q]['name'] = []
            cluster_nodes[q]['status'] = []
            cluster_nodes[q]['cluster_no'] = []
            cluster_nodes[q]['index'] = []
            clusterwise_embedding_src_nodes[q] = {}
            clusterwise_embedding_src_nodes[q]['name'] = []
            clusterwise_embedding_src_nodes[q]['label'] = []
            cust_clusterwise_embedding_nodes[q] = {}
            cust_clusterwise_embedding_nodes[q]['name'] = []
            cust_clusterwise_embedding_nodes[q]['label'] = []

        for w in range(len(clusters)):
            cluster_dict[clusters[w]].append(list(middleDimentions.index)[w])
            cluster_nodes[clusters[w]]['nodes'].append(list(middleDimentions.iloc[w]))
            cluster_nodes[clusters[w]]['label'].append('unlabeled')
            cluster_nodes[clusters[w]]['name'].append(list(middleDimentions.index)[w])
            cluster_nodes[clusters[w]]['status'].append('unvisited')
            cluster_nodes[clusters[w]]['cluster_no'].append(-1)
            cluster_nodes[clusters[w]]['index'].append(list(middleDimentions.index).index(list(middleDimentions.index)[w]))

        save_as_pickle(citeseer_cluster_nodes, cluster_nodes)
        save_as_pickle(citeseer_clusterwise_embedding_src_nodes, clusterwise_embedding_src_nodes)
        save_as_pickle(citeseer_cust_clusterwise_embedding_nodes, cust_clusterwise_embedding_nodes)
        save_as_pickle(citeseer_cluster_dict, cluster_dict)

    if exists(citeseer_dbscan_cluster_nodes_labels) and exists(citeseer_embedding_map) and exists(citeseer_source_nodes_map) and exists(citeseer_avg_distance_map):
        cluster_labels_map = read_pickel(citeseer_dbscan_cluster_nodes_labels)
        embedding_map = read_pickel(citeseer_embedding_map)
        source_nodes_map = read_pickel(citeseer_source_nodes_map)
        avg_distance_map = read_pickel(citeseer_avg_distance_map)
    else:
        cluster_labels_map = {}
        avg_distances_rows = []
        embedding_map = {}
        source_nodes_map = {}
        avg_distance_map = {}
        avg_elblow_distance = {}
        for u in range(len(cluster_nodes)):
            source_nodes = {}
            source_nodes['nodes'] = []
            source_nodes['name'] = []
            source_nodes['label'] = []
            source_nodes['status'] = []
            source_nodes['cluster_no'] = []
            source_nodes['index'] = []
            source_nodes['embedding'] = []
            for j in range(len(cluster_nodes[u]['nodes'])):
                if cluster_nodes[u]['name'][j].startswith("T"):
                    source_nodes['nodes'].append(cluster_nodes[u]['nodes'][j])
                    source_nodes['name'].append(cluster_nodes[u]['name'][j])
                    source_nodes['label'].append(cluster_nodes[u]['label'][j])
                    source_nodes['status'].append(cluster_nodes[u]['status'][j])
                    source_nodes['cluster_no'].append(cluster_nodes[u]['cluster_no'][j])
                    source_nodes['index'].append(cluster_nodes[u]['index'][j])
                    source_nodes['embedding'].append(list(middleDimentions.iloc[cluster_nodes[u]['index'][j]]))
            source_nodes_map[u] = source_nodes
            avg, avg_elbow_point = average_distance(source_nodes, u+1)
            avg_distance_map[u] = avg
            avg_distances_row = []
            avg_distances_row.append(u + 1)
            #avg_distances_row.append(sorted_names)
            #avg_distances_row.append(sorted_values)
            avg_distances_rows.append(avg_distances_row)

            dbscan = DBSCAN(eps=avg, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(source_nodes['nodes'])
            cluster_labels_map[u] = cluster_labels

            tsne = TSNE(n_components=2, perplexity=5)
            embeddings_2d = tsne.fit_transform(np.array(source_nodes['nodes']))
            embedding_map[u] = embeddings_2d

        save_as_pickle(citeseer_dbscan_cluster_nodes_labels, cluster_labels_map)
        save_as_pickle(citeseer_embedding_map, embedding_map)
        save_as_pickle(citeseer_source_nodes_map,source_nodes_map)
        save_as_pickle(citeseer_avg_distance_map,avg_distance_map)
        #avg_distance_cols = ["Cluster No", "Source_SDM Names", "5th Nearest Neighbor Distance"]
        #pd.DataFrame(avg_distances_rows, columns=avg_distance_cols).to_csv(citeseer_5th_nearest_neighbor_distance_csv)

    rows = []
    dbscan_rows = []
    avg_distances_rows = []
    node_edge_rows = []
    dbscan_pair_removed_rows = []
    node_degree_list = []
    trg_node_degree_list = []
    degree_summary_info = []
    degree_summary_info_trg = []
    degree_summary_info_s = []
    s_trg_node_degree_list = []
    for i in range(len(cluster_nodes)):

        cluster_labels = cluster_labels_map[i]
        citeseer_directed_A = nx.adjacency_matrix(original_citeseer_graph)
        bitcoin_directed_A = pd.DataFrame(citeseer_directed_A.toarray(), columns=original_citeseer_graph.nodes(), index=original_citeseer_graph.nodes())

        embeddings_2d = embedding_map[i]
        #print("embedding len" +str(len(embeddings_2d)))
        #print("cluster label len"+str(len(np.array(cluster_labels))))
        #print(cluster_labels)
        print("Spectral Cluster No:" + str(i+1))
        '''x_values = [embeddings_2d[j, 0] for j in range(len(embeddings_2d[:, 0])) if np.array(cluster_labels)[j] != -1]
        y_values = [embeddings_2d[j, 1] for j in range(len(embeddings_2d[:, 0])) if np.array(cluster_labels)[j] != -1]       
        source_lables = [int(lab) for lab in np.array(cluster_labels) if lab != -1]      
        source_names = [source_nodes['name'][j] for j in range(len(source_nodes['name'])) if cluster_labels[j] != -1]
        src_cls_embedding = [source_nodes['embedding'][j] for j in range(len(source_nodes['name'])) if cluster_labels[j] != -1]'''
        x_values = []
        y_values = []
        source_lables = []
        source_names = []
        src_cls_embedding = []
        for b in range(len(np.array(cluster_labels))):
            if cluster_labels[b] != -1:
                x_values.append(embeddings_2d[b, 0])
                y_values.append(embeddings_2d[b, 1])
                source_lables.append(int(cluster_labels[b]))
                source_nodes = source_nodes_map[i]
                source_names.append(source_nodes['name'][b])
                src_cls_embedding.append(source_nodes['embedding'][b])

        figure5 = plt.figure(figsize=(15, 15))
        ax5 = figure5.add_subplot(111)
        ax5.scatter(x_values, y_values, c=source_lables)
        plt.legend()
        plt.title("DBSCAN Inside Spectral Clustering (Cluster No: " + str(i+1) + ") ( No of Clusters " + str(len(set(source_lables))) + ")")
        plt.savefig(citeseer_spectral_dbscan_plot_name + "_" + str(i + 1) + ".png")
        #plt.show()

        clusterwise_embedding_src_nodes[i]['name'] = source_names
        clusterwise_embedding_src_nodes[i]['label'] = source_lables

        dbscan_cluster_info = {}
        for lab in list(set(source_lables)):
            dbscan_cluster_info[lab] = {}
            dbscan_cluster_info[lab]['x_value'] = []
            dbscan_cluster_info[lab]['y_value'] = []
            dbscan_cluster_info[lab]['names'] = []
            dbscan_cluster_info[lab]['embedding'] = []
            dbscan_cluster_info[lab]['status'] = []
            dbscan_cluster_info[lab]['label'] = []

        for lab in set(source_lables):
            dbscan_cluster_info[lab]['x_value'] = [x_values[k] for k in range(len(source_lables)) if source_lables[k] == lab]
            dbscan_cluster_info[lab]['y_value'] = [y_values[k] for k in range(len(source_lables)) if source_lables[k] == lab]
            dbscan_cluster_info[lab]['names'] = [source_names[k] for k in range(len(source_names)) if source_lables[k] == lab]
            dbscan_cluster_info[lab]['embedding'] = [src_cls_embedding[k] for k in range(len(source_lables)) if source_lables[k] == lab]
            dbscan_cluster_info[lab]['label'] = [lab for k in range(len(source_lables)) if source_lables[k] == lab]

        for k in range(len(dbscan_cluster_info)):
            row = []
            row.append(i + 1)
            row.append(k + 1)
            row.append(len(dbscan_cluster_info[k]['names']))
            row.append(min_samples)
            row.append(avg_distance_map[i])
            row.append(dbscan_cluster_info[k]['names'])
            row.append(dbscan_cluster_info[k]['embedding'])
            row.append(dbscan_cluster_info[k]['x_value'])
            row.append(dbscan_cluster_info[k]['y_value'])
            rows.append(row)

        cust_dbscan_cluster_nodes = {}
        cust_dbscan_cluster_nodes_pair_removed = {}
        for k in range(len(dbscan_cluster_info)):
            cust_dbscan_cluster_nodes[k] = {}
            cust_dbscan_cluster_nodes[k]['nodes'] = []
            cust_dbscan_cluster_nodes[k]['original_embedding'] = []
            cust_dbscan_cluster_nodes[k]['src_x_embedding'] = []
            cust_dbscan_cluster_nodes[k]['src_y_embedding'] = []
            cust_dbscan_cluster_nodes[k]['status'] = []
            cust_dbscan_cluster_nodes[k]['label'] = []
            cust_dbscan_cluster_nodes[k]['no_src'] = 0
            cust_dbscan_cluster_nodes[k]['no_trg'] = 0
            cust_dbscan_cluster_nodes[k]['no_pair'] = 0
            cust_dbscan_cluster_nodes[k]['avg_degree'] = 0
            cust_dbscan_cluster_nodes[k]['iterations'] = 0
            cust_dbscan_cluster_nodes_pair_removed[k] = {}
            cust_dbscan_cluster_nodes_pair_removed[k]['nodes'] = []
            cust_dbscan_cluster_nodes_pair_removed[k]['original_embedding'] = []
            cust_dbscan_cluster_nodes_pair_removed[k]['src_x_embedding'] = []
            cust_dbscan_cluster_nodes_pair_removed[k]['src_y_embedding'] = []
            cust_dbscan_cluster_nodes_pair_removed[k]['status'] = []
            cust_dbscan_cluster_nodes_pair_removed[k]['label'] = []
            cust_dbscan_cluster_nodes_pair_removed[k]['pair'] = []
            cust_dbscan_cluster_nodes_pair_removed[k]['no_src'] = 0
            cust_dbscan_cluster_nodes_pair_removed[k]['no_trg'] = 0
            cust_dbscan_cluster_nodes_pair_removed[k]['no_pair'] = 0
            cust_dbscan_cluster_nodes_pair_removed[k]['avg_degree'] = 0
            cust_dbscan_cluster_nodes_pair_removed[k]['iterations'] = 0

        for k in range(len(dbscan_cluster_info)):
            cust_dbscan_cluster_nodes[k]['nodes'] = copy.deepcopy(dbscan_cluster_info[k]['names'])
            cust_dbscan_cluster_nodes[k]['original_embedding'] = copy.deepcopy(dbscan_cluster_info[k]['embedding'])
            cust_dbscan_cluster_nodes[k]['src_x_embedding'] = copy.deepcopy(dbscan_cluster_info[k]['x_value'])
            cust_dbscan_cluster_nodes[k]['src_y_embedding'] = copy.deepcopy(dbscan_cluster_info[k]['y_value'])
            cust_dbscan_cluster_nodes[k]['status'] = ["unvisited" for k in range(len(dbscan_cluster_info[k]['names']))]
            cust_dbscan_cluster_nodes[k]['label'] = copy.deepcopy(dbscan_cluster_info[k]['label'])


        for p in range(len(cust_dbscan_cluster_nodes)):
            print("DBSCAN Cluster No:"+str(p+1))
            #dbscan_names = set(cust_dbscan_cluster_nodes[p]['nodes'])
            dbscan_names = cust_dbscan_cluster_nodes[p]['nodes']
            iteration_flag = False
            previous_flag = ""
            iteration = 0
            while len(dbscan_names) > 0:
                #last_element = dbscan_names.pop()
                last_element = dbscan_names[0]
                dbscan_names = dbscan_names[1:]
                last_element_index = cust_dbscan_cluster_nodes[p]['nodes'].index(last_element)
                last_element_status = cust_dbscan_cluster_nodes[p]['status'][last_element_index]

                if previous_flag =="":
                    previous_flag = last_element[0]
                    iteration += 1
                #elif previous_flag != last_element[0]:
                #    iteration += 1

                if last_element_status == "unvisited":
                    cust_dbscan_cluster_nodes[p]['status'][last_element_index] = "visited"
                    if last_element.startswith("S"):
                        new_src = int(last_element[1:])
                        src_trg = bitcoin_directed_A.loc[new_src]
                        indices = [i for i, value in enumerate(src_trg) if value == 1]
                        dataframe_indices = list(src_trg.index)
                        trg_list = [dataframe_indices[trg_index] for trg_index in indices]
                        for trg in trg_list:
                            #if list(cust_dbscan_cluster_nodes[p]['nodes']).__contains__("T"+str(trg)) == False and list(cluster_nodes[i]['name']).__contains__("T"+str(trg)) == True:
                            if list(cust_dbscan_cluster_nodes[p]['nodes']).__contains__("T" + str(trg)) == False:
                                src_trg_neighbors = bitcoin_directed_A[trg]
                                indices = [i for i, value in enumerate(src_trg_neighbors) if value == 1]
                                dataframe_indices = list(src_trg_neighbors.index)
                                src_list = [dataframe_indices[src_index] for src_index in indices]
                                src_trg_neighbors = set(src_list)
                                cls_src_list = [int(src[1:]) for src in cust_dbscan_cluster_nodes[p]['nodes'] if src.startswith("S")]
                                cls_src_list.append(last_element[1:])
                                cls_src_list = set(cls_src_list)
                                common_src = list(cls_src_list.intersection(src_trg_neighbors))
                                if len(common_src) >=closer_2:
                                    included_trg="T"+str(trg)
                                    cust_dbscan_cluster_nodes[p]['nodes'].append(included_trg)
                                    cust_dbscan_cluster_nodes[p]['status'].append("unvisited")
                                    cust_dbscan_cluster_nodes[p]['original_embedding'].append(list(middleDimentions.loc[included_trg]))
                                    cust_dbscan_cluster_nodes[p]['label'].append(list(set(cust_dbscan_cluster_nodes[p]['label']))[0])
                                    #dbscan_names.update([included_trg])
                                    dbscan_names.append(included_trg)
                                    if previous_flag != last_element[0]:
                                        previous_flag = last_element[0]
                                        iteration += 1
                    elif last_element.startswith("T"):
                        new_trg = int(last_element[1:])
                        trg_src = bitcoin_directed_A[new_trg]
                        indices = [i for i, value in enumerate(trg_src) if value == 1]
                        dataframe_indices = list(trg_src.index)
                        src_list = [dataframe_indices[src_index] for src_index in indices]
                        for src in src_list:
                            #if list(cust_dbscan_cluster_nodes[p]['nodes']).__contains__("S"+str(src)) == False and list(cluster_nodes[i]['name']).__contains__("S"+str(src)) == True:
                            if list(cust_dbscan_cluster_nodes[p]['nodes']).__contains__("S" + str(src)) == False:
                                trg_src_neighbors = bitcoin_directed_A.loc[src]
                                indices = [i for i, value in enumerate(trg_src_neighbors) if value == 1]
                                dataframe_indices = list(trg_src_neighbors.index)
                                trg_list = [dataframe_indices[trg_index] for trg_index in indices]
                                trg_src_neighbors = set(trg_list)
                                cls_trg_list = [int(src[1:]) for src in cust_dbscan_cluster_nodes[p]['nodes'] if src.startswith("T")]
                                cls_trg_list.append(last_element[1:])
                                cls_trg_list = set(cls_trg_list)
                                common_trg = list(cls_trg_list.intersection(trg_src_neighbors))
                                if len(common_trg) >= closer_2:
                                    included_src = "S" + str(src)
                                    cust_dbscan_cluster_nodes[p]['nodes'].append(included_src)
                                    cust_dbscan_cluster_nodes[p]['status'].append("unvisited")
                                    cust_dbscan_cluster_nodes[p]['original_embedding'].append(list(middleDimentions.loc[included_src]))
                                    cust_dbscan_cluster_nodes[p]['label'].append(list(set(cust_dbscan_cluster_nodes[p]['label']))[0])
                                    #dbscan_names.update([included_src])
                                    dbscan_names.append(included_src)
                                    if previous_flag != last_element[0]:
                                        previous_flag = last_element[0]
                                        iteration += 1

            print("Spectral Cluster " + str(i+1) +"DBSCAN Cluster " + str(p+1) +" Number Of Iterations:" + str(iteration))
            cust_dbscan_cluster_nodes[p]['iterations'] = iteration

        cust_nodes = []
        cust_labels = []
        for h in range(len(cust_dbscan_cluster_nodes)):
            cust_nodes.extend(list(cust_dbscan_cluster_nodes[h]['nodes']))
            cust_labels.extend(list(cust_dbscan_cluster_nodes[h]['label']))
        cust_clusterwise_embedding_nodes[i]['name'] = cust_nodes
        cust_clusterwise_embedding_nodes[i]['label'] = cust_labels

        for k in range(len(cust_dbscan_cluster_nodes)):
            hub_bar_list = {}
            authority_bar_list = {}
            line_x_values = {}
            hub_color = {}
            authority_color = {}
            for bar, bar_value in zip(list(sorted_hubs.keys()), list(sorted_hubs.values())):
                hub_bar_list[bar] = 0
                authority_bar_list[bar] = 0
                line_x_values[bar] = ""
                hub_color[bar] ="red"
                authority_color[bar] = "red"

            sr_no = 0
            tr_no = 0
            pair_nodes = set()
            degree_cnt = 0
            edges = set()
            cls_nodes = set(cust_dbscan_cluster_nodes[k]['nodes'])
            hits_score_list = {}
            authority_score_list = {}
            percentile_score_list = {}
            s_degree_dict = {}
            trg_degree_dict = {}
            degree_dict = {}
            pair_node_list = set()
            pair_removal_edges = set()
            pop_index = []
            pop_element = []
            for nd in cust_dbscan_cluster_nodes[k]['nodes']:
                if nd.startswith("S"):
                    edge_list = list(bipartite_G.neighbors(nd))
                    common_neighbors = set(edge_list).intersection(cust_dbscan_cluster_nodes[k]['nodes'])
                    if len(common_neighbors) == 0:
                        get_index_src = cust_dbscan_cluster_nodes[k]['nodes'].index(nd)
                        pop_index.append(get_index_src)
                        pop_element.append(nd)
                    else:
                        sr_no += 1
                        if degree_dict.__contains__(len(common_neighbors)) == False:
                            degree_dict[len(common_neighbors)] = []
                            degree_dict[len(common_neighbors)].append(nd)
                        else:
                            degree_dict[len(common_neighbors)].append(nd)
                        if s_degree_dict.__contains__(len(common_neighbors)) == False:
                            s_degree_dict[len(common_neighbors)] = []
                            s_degree_dict[len(common_neighbors)].append(nd)
                        else:
                            s_degree_dict[len(common_neighbors)].append(nd)
                        node_no = nd[1:]
                        tr_node = "T" + str(node_no)
                        if list(cust_dbscan_cluster_nodes[k]['nodes']).__contains__(
                                tr_node) and pair_nodes.__contains__(node_no) == False:
                            pair_nodes.add(node_no)
                elif nd.startswith("T"):
                    tr_no += 1
                    node_no = nd[1:]
                    sr_node = "S" + str(node_no)
                    if list(cust_dbscan_cluster_nodes[k]['nodes']).__contains__(sr_node) and pair_nodes.__contains__(
                            node_no) == False:
                        pair_nodes.add(node_no)
                    edge_list = list(bipartite_G.neighbors(nd))
                    common_neighbors = set(edge_list).intersection(cust_dbscan_cluster_nodes[k]['nodes'])
                    if degree_dict.__contains__(len(common_neighbors)) == False:
                        degree_dict[len(common_neighbors)] = []
                        degree_dict[len(common_neighbors)].append(nd)
                    else:
                        degree_dict[len(common_neighbors)].append(nd)

                    if trg_degree_dict.__contains__(len(common_neighbors)) == False:
                        trg_degree_dict[len(common_neighbors)] = []
                        trg_degree_dict[len(common_neighbors)].append(nd)
                    else:
                        trg_degree_dict[len(common_neighbors)].append(nd)
                edge_list = list(bipartite_G.neighbors(nd))
                common_neighbors = set(edge_list).intersection(cls_nodes)
                hits_score_list[nd[1:]] = sorted_hubs[nd[1:]] * 1e-4
                hub_bar_list[nd[1:]] = max(sorted_hubs.values())
                authority_bar_list[nd[1:]] = max(sorted_authorities.values())
                line_x_values[nd[1:]] = nd[1:]
                authority_score_list[nd[1:]] = sorted_authorities[nd[1:]] * 1e-4
                if nd.startswith("S"):
                    hub_color[nd[1:]] = "blue"
                    authority_color[nd[1:]] = "blue"

                for cm_ed in common_neighbors:
                    if (nd.startswith("S") == True) and (str(cm_ed).startswith("S") == False):
                        edge = str(nd) + "-" + str(cm_ed)
                        if pair_node_list.__contains__(cm_ed) == False and pair_removal_edges.__contains__(
                                str(nd) + "-" + str(cm_ed)) == False:
                            removed_edges = str(nd) + "-" + str(cm_ed)
                    elif nd.startswith("T") and (str(cm_ed).startswith("S") == True):
                        edge = str(cm_ed) + "-" + str(nd)
                        if pair_node_list.__contains__(nd) == False and pair_removal_edges.__contains__(
                                str(cm_ed) + "-" + str(nd)) == False:
                            removed_edges = str(cm_ed) + "-" + str(nd)
                    edges.add(edge)
                    pair_removal_edges.add(removed_edges)

            for nd in pop_element:
                get_index_src = cust_dbscan_cluster_nodes[k]['nodes'].index(nd)
                cust_dbscan_cluster_nodes[k]['nodes'].pop(get_index_src)
                cust_dbscan_cluster_nodes[k]['original_embedding'].pop(get_index_src)
                cust_dbscan_cluster_nodes[k]['status'].pop(get_index_src)
                cust_dbscan_cluster_nodes[k]['label'].pop(get_index_src)

            sorted_degree_dict = dict(sorted(degree_dict.items()))
            for key, value in sorted_degree_dict.items():
                degree_node_row = []
                degree_node_row.append(i + 1)
                degree_node_row.append(k + 1)
                degree_node_row.append(key)
                degree_node_row.append(len(value))
                degree_node_row.append(value)
                node_degree_list.append(degree_node_row)

            sorted_degree_dict = dict(sorted(trg_degree_dict.items()))
            for key, value in sorted_degree_dict.items():
                degree_node_row = []
                degree_node_row.append(i + 1)
                degree_node_row.append(k + 1)
                degree_node_row.append(key)
                degree_node_row.append(len(value))
                degree_node_row.append(value)
                trg_node_degree_list.append(degree_node_row)

            sorted_degree_dict = dict(sorted(s_degree_dict.items()))
            for key, value in sorted_degree_dict.items():
                degree_node_row = []
                degree_node_row.append(i + 1)
                degree_node_row.append(k + 1)
                degree_node_row.append(key)
                degree_node_row.append(len(value))
                degree_node_row.append(value)
                s_trg_node_degree_list.append(degree_node_row)

            degree_summary_info_row = []
            degree_summary_info_row.append(i + 1)
            degree_summary_info_row.append(k + 1)
            degree_list = list(degree_dict.keys())
            if len(degree_list) > 0:
                min_degree = min(degree_list)
                max_degree = max(degree_list)
                avg_degree = sum(degree_list) / len(degree_list)
                degree_summary_info_row.append(min_degree)
                degree_summary_info_row.append(max_degree)
                degree_summary_info_row.append(avg_degree)
                degree_summary_info_row.append(str() + "/" + str() + "/" + str())
                degree_summary_info.append(degree_summary_info_row)
            else:
                degree_summary_info_row.append(0)
                degree_summary_info_row.append(0)
                degree_summary_info_row.append(0)
                degree_summary_info_row.append(0)
                degree_summary_info.append(degree_summary_info_row)

            degree_summary_info_row = []
            degree_summary_info_row.append(i + 1)
            degree_summary_info_row.append(k + 1)
            degree_list = list(trg_degree_dict.keys())
            if len(degree_list) > 0:
                min_degree = min(degree_list)
                max_degree = max(degree_list)
                avg_degree = sum(degree_list) / len(degree_list)
                degree_summary_info_row.append(min_degree)
                degree_summary_info_row.append(max_degree)
                degree_summary_info_row.append(avg_degree)
                degree_summary_info_row.append(str(min_degree) + "/" + str(max_degree) + "/" + str(avg_degree))
                degree_summary_info_trg.append(degree_summary_info_row)
            else:
                degree_summary_info_row.append(0)
                degree_summary_info_row.append(0)
                degree_summary_info_row.append(0)
                degree_summary_info_row.append(0)
                degree_summary_info_trg.append(degree_summary_info_row)

            degree_summary_info_row = []
            degree_summary_info_row.append(i + 1)
            degree_summary_info_row.append(k + 1)
            degree_list = list(s_degree_dict.keys())
            if len(degree_list) > 0:
                min_degree = min(degree_list)
                max_degree = max(degree_list)
                avg_degree = sum(degree_list) / len(degree_list)
                degree_summary_info_row.append(min_degree)
                degree_summary_info_row.append(max_degree)
                degree_summary_info_row.append(avg_degree)
                degree_summary_info_row.append(str(min_degree) + "/" + str(max_degree) + "/" + str(avg_degree))
                degree_summary_info_s.append(degree_summary_info_row)
            else:
                degree_summary_info_row.append(0)
                degree_summary_info_row.append(0)
                degree_summary_info_row.append(0)
                degree_summary_info_row.append(0)
                degree_summary_info_s.append(degree_summary_info_row)

            cust_dbscan_cluster_nodes[k]['pair'] = pair_node_list
            cust_dbscan_cluster_nodes_pair_removed[k]['nodes'] = set(cust_dbscan_cluster_nodes[k]['nodes']) - pair_node_list
            cust_dbscan_cluster_nodes_pair_removed[k]['iterations'] = cust_dbscan_cluster_nodes[k]['iterations']
            cust_dbscan_cluster_nodes_pair_removed[k]['pair'] = pair_node_list

            row = []
            row.append(i + 1)
            row.append(k + 1)
            row.append(len(cust_dbscan_cluster_nodes[k]['nodes']))
            row.append(len(dbscan_cluster_info[k]['names']))
            row.append(sr_no)
            row.append(tr_no)
            row.append(len(edges))
            row.append(len(pair_nodes))
            row.append(cust_dbscan_cluster_nodes[k]['iterations'])
            row.append(cust_dbscan_cluster_nodes[k]['nodes'])
            # row.append(edges)
            # row.append(hits_score_list)
            # row.append(authority_score_list)
            dbscan_rows.append(row)

            row = []
            row.append(i + 1)
            row.append(k + 1)
            row.append(len(cust_dbscan_cluster_nodes[k]['nodes']))
            row.append(len(cust_dbscan_cluster_nodes[k]['nodes']) - len(pair_node_list))
            row.append(len(dbscan_cluster_info[k]['names']))
            row.append(sr_no)
            row.append(tr_no)
            row.append(sr_no - len(pair_nodes))
            row.append(tr_no - len(pair_nodes))
            row.append(len(edges))
            row.append(len(pair_removal_edges))
            row.append(len(pair_nodes))
            row.append(cust_dbscan_cluster_nodes[k]['iterations'])
            dbscan_pair_removed_rows.append(row)

            list_row = []
            list_row.append(i + 1)
            list_row.append(k + 1)
            list_row.append(pair_nodes)
            list_row.append(cust_dbscan_cluster_nodes[k]['nodes'])
            list_row.append(edges)
            node_edge_rows.append(list_row)

            y_min = min(list(sorted_hubs.values()))
            y_max = max(list(sorted_hubs.values()))
            figure_cluster_hub = plt.figure(figsize=(15, 15))
            ax_cluster_hub = figure_cluster_hub.add_subplot(111)
            ax_cluster_hub.set_ylim(y_min, y_max)
            ax_cluster_hub.plot(list(sorted_hubs.keys()), list(sorted_hubs.values()), label="Bipartite Graph Hub Score")
            ax_cluster_hub.bar(list(sorted_hubs.keys()), list(hub_bar_list.values()), color=list(hub_color.values()), label="Cluster Hub Score Location")
            plt.legend()
            plt.title("Node Hub Score Position(Source_SDM Clustering) For Spectral Cluster No:" + str(i+1) + "DBSCAN Cluster No:" + str(k+1))
            #plt.show()
            plt.savefig(citeseer_spectral_cluster + "node_hub_score_position_spectral_cls_source_clustering"+str(i+1) + "dbscan_cls_" + str(k+1) +".png")

            y_min = min(list(sorted_authorities.values()))
            y_max = max(list(sorted_authorities.values()))
            figure_cluster_authority = plt.figure(figsize=(15, 15))
            ax_cluster_authority = figure_cluster_authority.add_subplot(111)
            ax_cluster_authority.set_ylim(y_min, y_max)
            ax_cluster_authority.plot(list(sorted_authorities.keys()), list(sorted_authorities.values()), label="Bipartite Graph Authority Score")
            ax_cluster_authority.bar(list(sorted_authorities.keys()), list(authority_bar_list.values()), color=list(authority_color.values()), label="Cluster Authority Score Location")
            plt.legend()
            plt.title("Node Authority Score Position(Source_SDM Clustering)  For Spectral Cluster No:" + str(i + 1) + "DBSCAN Cluster No:" + str(k + 1))
            #plt.show()
            plt.savefig(citeseer_spectral_cluster + "node_authority_score_position_spectral_cls_source_clustering" + str(i + 1) + "dbscan_cls_" + str(k + 1) + ".png")



    degree_list_col = ['Spectral Cluster No', 'DBSCAN Cluster No', 'Degree Of Trg Node', 'No Of Degree Trg Node', 'Trg Node List']
    pd.DataFrame(trg_node_degree_list, columns=degree_list_col).to_csv(target_citeseer_node_degree_list)

    degree_list_col = ['Spectral Cluster No', 'DBSCAN Cluster No', 'Degree Of Node', 'No Of Degree Node', 'Node List']
    pd.DataFrame(node_degree_list, columns=degree_list_col).to_csv(citeseer_node_degree_list)

    degree_list_col = ['Spectral Cluster No', 'DBSCAN Cluster No', 'Degree Of Src- Node', 'No Of Degree Src- Node',
                       'Src- Node List']
    pd.DataFrame(s_trg_node_degree_list, columns=degree_list_col).to_csv(source_target_citeseer_node_degree_list)

    pd.DataFrame(degree_summary_info,
                 columns=['Spectral Cluster No', 'DBSCAN Cluster No', 'Min Degree', 'Max Degree', 'Avg Degree',
                          'Combined Result']).to_csv(citeseer_node_degree_summary)
    pd.DataFrame(degree_summary_info_trg,
                 columns=['Spectral Cluster No', 'DBSCAN Cluster No', 'Min Degree in Target', 'Max Degree in Target',
                          'Avg Degree in Target', 'Combined Result']).to_csv(target_citeseer_node_degree_summary)
    pd.DataFrame(degree_summary_info_s,
                 columns=['Spectral Cluster No', 'DBSCAN Cluster No', 'Min Degree in Source_SDM', 'Max Degree in Source_SDM',
                          'Avg Degree in Source_SDM', 'Combined Result']).to_csv(source_citeseer_node_degree_summary)

    # target clustering
    dbscan_cols = ['Spectral Cluster No', 'DBSCAN Cluster No', 'No Of Target Nodes in Cluster', 'Min Sample', 'Eps',
                   'List Of Nodes', 'Node2vec 20-D Embedding', 'X Value(TSNE)', 'Y Value(TSNE)']
    pd.DataFrame(rows, columns=dbscan_cols).to_csv(citeseer_dbscan_cluster_csv_file)

    nodd_edge_cols = ['Spectral Cluster No', 'DBSCAN Cluster No', "Pair Removed", "List Of Nodes", "List Of Edges"]
    pd.DataFrame(node_edge_rows, columns=nodd_edge_cols).to_csv(citeseer_dbscan_cluster_node_edge_csv_file)
    save_as_pickle(citeseer_dbscan_cluster_node_edge_pkl, node_edge_rows)

    # target columns
    '''cust_dbscan_cols = ["Spectral Cluster No", "DBSCAN Cluster No", "No Of Nodes in Cluster", "Original No Of Sources","No Of Source_SDM Nodes","No Of Target Nodes",
                        "No Of Edges","No Of Pairs", "Iteration for Convergence", "List Of Nodes", "List Of Edges", "Original Bipartite Graph Hub Score", "Original Bipartite Graph Authority Score"]'''
    cust_dbscan_cols = ["Spectral Cluster No", "DBSCAN Cluster No", "No Of Nodes in Cluster", "Original No Of Sources",
                        "No Of Source_SDM Nodes", "No Of Target Nodes",
                        "No Of Edges", "No Of Pairs", "Iteration for Convergence", "List Of Nodes"]

    pd.DataFrame(dbscan_rows, columns=cust_dbscan_cols).to_csv(citeseer_cust_dbscan_cluster_csv_file)

    cust_dbscan_pair_removed_cols = ["Spectral Cluster No", "DBSCAN Community No", "No Of Nodes in Community",
                                     "no Of Nodes in Community After Pair Removed",
                                     "No Of Source_SDM Nodes After DBSCAN", "No Of Source_SDM Nodes",
                                     "No Of Target Nodes", "No Of Source_SDM Nodes After Pair Removed",
                                     "No Of Target Nodes After Pair Removed",
                                     "No Of Edges",
                                     "No Of Edges After Pair Removed", "No Of Pairs", "No Of Iterations"]
    pd.DataFrame(dbscan_pair_removed_rows, columns=cust_dbscan_pair_removed_cols).to_csv(citeseer_pair_removed_cust_dbscan_cluster_csv_file)

    avg_distance_cols = ["Cluster No", "Source_SDM Names", "5th Nearest Neighbor Distance"]
    pd.DataFrame(avg_distances_rows, columns=avg_distance_cols).to_csv(citeseer_5th_nearest_neighbor_distance_csv)

    not_included_nodes = []
    src_nodes_not_included = []
    trg_nodes_not_included = []
    flag = 0
    for gn in bipartite_G.nodes():
        for k in range(len(node_edge_rows)):
            cls_nodes = list(node_edge_rows[k][3])
            if cls_nodes.__contains__(gn) == True:
                flag = 1
                break
        if flag == 0:
            not_included_nodes.append(gn)
            if gn.startswith("S"):
                src_nodes_not_included.append(gn)
            elif gn.startswith("T"):
                trg_nodes_not_included.append(gn)
        else:
            flag = 0

    row = []
    row.append("Total No of Nodes which are not part of any community")
    row.append(len(not_included_nodes))
    row.append("No of Source_SDM Nodes which are not part of any community")
    row.append(len(src_nodes_not_included))
    row.append("No of Target Nodes which are not part of any community")
    row.append(len(trg_nodes_not_included))
    row.append("List of nodes which are not part of any community")
    row.append(not_included_nodes)
    save_as_pickle(citeseer_nodes_not_part_of_any_community, row)
    pd.DataFrame(row).to_csv(citeseer_nodes_not_in_cluster_csv)

    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 0),  # Maroon
        (0, 128, 0),  # Green
        (0, 0, 128),  # Navy
        (128, 128, 0),  # Olive
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
        (255, 165, 0),  # Orange
        (128, 128, 128),  # Gray
        (255, 192, 203),  # Pink
        (0, 128, 0),  # Green
        (128, 0, 0),  # Maroon
        (0, 0, 128),  # Navy
        (0, 128, 128),  # Teal
        (128, 0, 128),  # Purple
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
    ]

    normalized_colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]

    selected_colors = [normalized_colors[i] for i in clusters]

    tsne = TSNE(n_components=2, perplexity=25)
    embeddings_2d = tsne.fit_transform(np.array(middleDimentions))

    x_min = min(embeddings_2d[:, 0])
    x_max = max(embeddings_2d[:, 0])
    y_min = min(embeddings_2d[:, 1])
    y_max = max(embeddings_2d[:, 1])
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    figure = plt.figure(figsize=(15, 15))
    ax = figure.add_subplot(111)
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=selected_colors, cmap='jet')
    plt.title(title + "( No of Clusters " + str(len(set(clusters))) + ")")
    plt.savefig(spectral_cluster_plot_file_name)
    plt.legend()
    # plt.show()

    embeddings_2d = pd.DataFrame(embeddings_2d, index=middleDimentions.index)

    '''Original Embedding'''
    k = 0
    for cls_nodes in cluster_dict.values():
        source_x_embedding = []
        source_y_embedding = []
        target_x_embedding = []
        target_y_embedding = []
        source_labels = []
        target_labels = []
        source_colors = []
        target_colors = []
        figure1 = plt.figure(figsize=(15, 15))
        ax1 = figure1.add_subplot(111)
        color = [k for j in range(len(cls_nodes))]

        for cls_node in cls_nodes:
            scatter = ax1.scatter(embeddings_2d.loc[cls_node][0], embeddings_2d.loc[cls_node][1], c=normalized_colors[k], cmap="jet")
            if cls_node.startswith("S"):
                source_x_embedding.append(embeddings_2d.loc[cls_node][0])
                source_y_embedding.append(embeddings_2d.loc[cls_node][1])
                source_labels.append(cls_node)
                source_colors.append("blue")
            elif cls_node.startswith("T"):
                target_x_embedding.append(embeddings_2d.loc[cls_node][0])
                target_y_embedding.append(embeddings_2d.loc[cls_node][1])
                target_labels.append(cls_node)
                target_colors.append("red")

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.title(citeseer_node_embedding_title + " "+ str(k+1))
        cluster_file_name = citeseer_spectral_cluster + str(k + 1)
        plt.savefig(cluster_file_name)
        #plt.show()

        figure2 = plt.figure(figsize=(15, 15))
        ax2 = figure2.add_subplot(111)
        source_scatter = ax2.scatter(source_x_embedding, source_y_embedding, cmap="Set2", c=source_colors, label="Source_SDM")
        target_scatter = ax2.scatter(target_x_embedding, target_y_embedding, cmap="Set2", c=target_colors, label="Target")
        plt.title(citeseer_TSNE_clusterwise_source_target_title + str(k+1))
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.savefig(citesser_TSNE_clusterwise_source_target_plot + str(k + 1))
        #plt.show()

        figure22 = plt.figure(figsize=(15, 15))
        ax2 = figure22.add_subplot(111)
        source_scatter = ax2.scatter(source_x_embedding, source_y_embedding, cmap="Set2", c=source_colors, label="Source_SDM")
        plt.title(citeseer_source_nodes_only_title + str(k + 1))
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.savefig(citesser_clusterwise_source_plot + str(k + 1))
        # plt.show()

        cluster_wise_embedding_x = []
        cluster_wise_embedding_y = []
        cluster_wise_embedding_label = []
        m = 0
        for cls_name in clusterwise_embedding_src_nodes[k]['name']:
            cluster_wise_embedding_x.append(embeddings_2d.loc[cls_name][0])
            cluster_wise_embedding_y.append(embeddings_2d.loc[cls_name][1])
            cluster_wise_embedding_label.append(clusterwise_embedding_src_nodes[k]['label'][m])
            m += 1
        figure3 = plt.figure(figsize=(15, 15))
        ax3 = figure3.add_subplot(111)
        clusterwise_source_scatter = ax3.scatter(cluster_wise_embedding_x, cluster_wise_embedding_y, cmap="Set2", c=cluster_wise_embedding_label, label="Source_SDM Nodes")
        plt.title(citeseer_TSNE_clusterwise_source_with_cluster_title + str(k+1) + " No Of Clusters:" + str(len(set(cluster_wise_embedding_label))))
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.savefig(citesser_TSNE_clusterwise_source_with_cluster_labels_plot + str(k + 1))
        #plt.show()

        cust_cluster_wise_embedding_x = []
        cust_cluster_wise_embedding_y = []
        cust_cluster_wise_embedding_label = []
        m = 0
        for cls_name in cust_clusterwise_embedding_nodes[k]['name']:
            cust_cluster_wise_embedding_x.append(embeddings_2d.loc[cls_name][0])
            cust_cluster_wise_embedding_y.append(embeddings_2d.loc[cls_name][1])
            cust_cluster_wise_embedding_label.append(cust_clusterwise_embedding_nodes[k]['label'][m])
            m += 1
        figure10 = plt.figure(figsize=(15, 15))
        ax10 = figure10.add_subplot(111)
        cust_clusterwise_source_scatter = ax10.scatter(cust_cluster_wise_embedding_x, cust_cluster_wise_embedding_y, cmap="Set2", c=cust_cluster_wise_embedding_label)
        plt.title(citeseer_customized_dbscan_clusters + str(k + 1) + "( No of Clusters " + str(len(list(set(cust_clusterwise_embedding_nodes[k]['label'])))) + ")")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.savefig(citeseer_customized_dbscan_clusters_plot + str(k + 1))
        plt.legend()
        #plt.show()

        k += 1


    cols = ["Cluster No",
            "No of Nodes in Cluster",
            "No of Sources",
            "No Of Zero Degree Source_SDM",
            "No of Sources - No Of Zero Degree Source_SDM",
            "No of Targets",
            "No Of Zero Degree Target",
            "No of Targets - No Of Zero Degree Target",
            "No of Zero-degree Nodes",
            "No of Outgoing Edges in Cluster (to nodes only within cluster)",
            "No of Incoming Edges in Cluster (from nodes only within cluster)",
            "No of possible edges",
            "No of Only Know Each Other (As Source_SDM)",
            "No of Only Know Each Other (As Target)",
            "Edge Percentage",
            "Average Degree Of Cluster",
            "Hub/Authority Score",
            "Highest Graph Source_SDM Degree",
            "Highest Graph Target Degree",
            "Highest Cluster Source_SDM Degree",
            "Highest Cluster Target Degree",
            "No of Source_SDM and Target Pair available",
            "No of Target and Source_SDM Pair available",
            "Source_SDM Target Pairs",
            "Target Source_SDM Pairs",
            "List of Nodes in Cluster",
            "List of Zero-degree Nodes in Cluster",
            "List of Only Know Each Other (As Source_SDM)",
            "List of Only Know Each Other (As Target)",
            "Singleton Nodes",
            "List of Connected Target Nodes Out-off Cluster",
            "No of Connected Target Nodes Out-off Cluster",
            "List of Connected Source_SDM Nodes Out-off Cluster",
            "No of Connected Source_SDM Nodes Out-off Cluster"]
    rows = []
    new_rows = []
    edges_rows = []
    cluster_wise_label_map = {str(class_name): {} for class_name in range(len(cluster_dict))}
    node_degree_dict = dict(bipartite_G.degree())
    total_degree_row = []
    total_cluster_degree_row = []
    total_cluster_degree = []
    zero_degree_rows = []
    for i in range(len(cluster_dict)):
        row = []
        edge_row = []
        row.append(i + 1)
        source_no = 0
        target_no = 0
        source_list = []
        target_list = []
        original_graph_edge_dict = {}
        source_target_pair = []
        target_source_pair = []
        src_high_degree = -1
        trg_high_degree = -1
        degree_list = []
        zero_src_nodes = []
        zero_trg_nodes = []
        for node in cluster_dict[i]:
            if node.startswith("S"):
                if node_degree_dict[node] > src_high_degree:
                    src_high_degree = node_degree_dict[node]
                if bipartite_G.degree(node) == 0:
                    zero_src_nodes.append(node)
                source_list.append(node)
                source_no += 1
            elif node.startswith("T"):
                if node_degree_dict[node] > trg_high_degree:
                    trg_high_degree = node_degree_dict[node]
                if bipartite_G.degree(node) == 0:
                    zero_trg_nodes.append(node)
                target_list.append(node)
                target_no += 1
            degree_list.append(node_degree_dict[node])
            original_graph_edge_dict[node] = list(bipartite_G.neighbors(node))

        unique_degrees = set(degree_list)
        unique_degree_dict = {}
        for z in range(len(degree_list)):
            if unique_degree_dict.keys().__contains__("Degree " + str(list(degree_list)[z])):
                unique_degree_dict["Degree " + str(list(degree_list)[z])] = unique_degree_dict["Degree " + str(
                    list(degree_list)[z])] + 1
            else:
                unique_degree_dict["Degree " + str(list(degree_list)[z])] = 1

        '''bin_width = 5
        num_bins = max(int((max(degree_list) - min(degree_list)) / bin_width), 1)
        n, bins, patches = plt.hist(degree_list, bins=num_bins)'''
        fig = plt.figure(figsize=(10, 8))
        plt.bar(list(unique_degree_dict.keys()), list(unique_degree_dict.values()), align='center')
        plt.xticks(list(unique_degree_dict.keys()), list(unique_degree_dict.keys()), rotation=90, ha='right')

        for k in range(len(list(unique_degree_dict.keys()))):
            plt.annotate(list(unique_degree_dict.values())[k],
                         (list(unique_degree_dict.keys())[k], list(unique_degree_dict.values())[k]), ha='center',
                         va='bottom')

        # plt.xlim(min(list(unique_degree_dict.keys())) - 0.5, max(list(unique_degree_dict.keys())) + 0.5)
        # Set labels and title
        plt.xlabel('Degree Level')
        plt.ylabel('No of Nodes in Degree')
        plt.title('Bipartite Graph Degree Distribution' + str(i + 1))
        plt.savefig(citesser_bipartite_graph_degree_distribution + str(i + 1) + ".png")
        # Display the plot
        # plt.show()

        degree_cols = ["Cluster No", "Node List", "Degree List", "Bipartite Graph Degree Distribution"]
        degree_row = []
        degree_row.append(i + 1)
        degree_row.append(cluster_dict[i])
        degree_row.append(degree_list)
        degree_row.append(unique_degree_dict)
        total_degree_row.append(degree_row)
        '''for j in range(len(n)):
            if n[j] > 0:
                plt.text(bins[j] + bin_width / 2, n[j], str(int(n[j])), ha='center', va='bottom')

        bin_ranges = ['{:.1f}-{:.1f}'.format(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]
        tick_locations = bins[:-1] + bin_width/2  # Adjust tick positions
        plt.xticks(tick_locations, bin_ranges, rotation=45, ha='right')'''

        total_edges_in_cluster_as_source = 0
        total_edges_in_graph_with_source = 0
        total_connected_target_nodes_not_in_cls = []
        source_edges = []
        cluster_degree = {}
        node_edges_count = {}
        highest_src_cls_degree = -1
        source_list_without_zero_degree = []
        zero_degree_nodes = []
        only_know_each_other_as_source = []
        singleton_nodes = []
        for source in source_list:
            source_neighbor_list = original_graph_edge_dict[source]
            common_targets_in_cluster = set(source_neighbor_list) & set(target_list)

            if len(common_targets_in_cluster) == 0:
                zero_degree_row = []
                zero_degree_row.append(i + 1)
                zero_degree_row.append(source)
                zero_degree_row.append("Degree " + str(len(common_targets_in_cluster)))
                zero_degree_row.append(len(source_neighbor_list))
                zero_degree_rows.append(zero_degree_row)
                zero_degree_nodes.append(source)
            else:
                source_list_without_zero_degree.append(source)

                if len(common_targets_in_cluster) == 1:
                    only_source = original_graph_edge_dict[list(common_targets_in_cluster)[0]]
                    only_source_list = set(only_source) & set(source_list)
                    if len(only_source_list) == 1:
                        if list(only_source_list)[0] == source:
                            only_know_each_other_as_source.append(source + "->" + list(common_targets_in_cluster)[0])
                            singleton_nodes.append(source)
                            singleton_nodes.append(list(common_targets_in_cluster)[0])
                            source_list_without_zero_degree.remove(source)

                if len(common_targets_in_cluster) > highest_src_cls_degree:
                    highest_src_cls_degree = len(common_targets_in_cluster)
                node_edges_count[source] = common_targets_in_cluster
                if cluster_degree.keys().__contains__("Degree " + str(len(common_targets_in_cluster))):
                    cluster_degree["Degree " + str(len(common_targets_in_cluster))] = cluster_degree["Degree " + str(
                        len(common_targets_in_cluster))] + 1
                else:
                    cluster_degree["Degree " + str(len(common_targets_in_cluster))] = 1

                connected_target_nodes_not_in_cls = set(source_neighbor_list) - set(common_targets_in_cluster)
                for not_cls_trg in connected_target_nodes_not_in_cls:
                    if not_cls_trg not in total_connected_target_nodes_not_in_cls:
                        total_connected_target_nodes_not_in_cls.append(not_cls_trg)
                total_edges_in_cluster_as_source += len(common_targets_in_cluster)
                total_edges_in_graph_with_source += len(source_neighbor_list)
                '''source_suffix = source[1:]
                target_string = "T" + source_suffix
                if target_list.__contains__(target_string):
                    source_target_pair.append(source + "-" + target_string)'''
                for trg in common_targets_in_cluster:
                    source_edges.append(source + '-' + trg)

        total_edges_in_cluster_as_target = 0
        total_edges_in_graph_with_target = 0
        total_connected_source_nodes_not_in_cls = []
        target_edges = []
        highest_trg_cls_degree = -1

        target_list_without_zero_degree = []
        only_know_each_other_as_target = []
        for target in target_list:
            target_neighbor_list = original_graph_edge_dict[target]
            common_source_in_cluster = set(target_neighbor_list) & set(source_list)
            if len(common_source_in_cluster) > highest_trg_cls_degree:
                highest_trg_cls_degree = len(common_source_in_cluster)

            zero_degree_row = []
            if len(common_source_in_cluster) == 0:
                zero_degree_row.append(i + 1)
                zero_degree_row.append(target)
                zero_degree_row.append("Degree " + str(len(common_source_in_cluster)))
                zero_degree_row.append(len(target_neighbor_list))
                zero_degree_rows.append(zero_degree_row)
                zero_degree_nodes.append(target)
            else:
                target_list_without_zero_degree.append(target)

                if len(common_source_in_cluster) == 1:
                    only_target = original_graph_edge_dict[list(common_source_in_cluster)[0]]
                    only_target_list = set(only_target) & set(target_list)
                    if len(only_target_list) == 1:
                        if list(only_target_list)[0] == target:
                            only_know_each_other_as_target.append(list(common_source_in_cluster)[0] + "->" + target)
                            singleton_nodes.append(target)
                            singleton_nodes.append(list(common_source_in_cluster)[0])
                            target_list_without_zero_degree.remove(target)

                node_edges_count[target] = common_source_in_cluster
                if cluster_degree.keys().__contains__("Degree " + str(len(common_source_in_cluster))):
                    cluster_degree["Degree " + str(len(common_source_in_cluster))] = cluster_degree["Degree " + str(len(common_source_in_cluster))] + 1
                else:
                    cluster_degree["Degree " + str(len(common_source_in_cluster))] = 1
                connected_source_nodes_not_in_cls = set(target_neighbor_list) - set(common_source_in_cluster)

                for not_cls_src in connected_source_nodes_not_in_cls:
                    if not_cls_src not in total_connected_source_nodes_not_in_cls:
                        total_connected_source_nodes_not_in_cls.append(not_cls_src)

                # total_connected_source_nodes_not_in_cls.extend(list(connected_source_nodes_not_in_cls))
                total_edges_in_cluster_as_target += len(common_source_in_cluster)
                total_edges_in_graph_with_target += len(target_neighbor_list)
                '''target_suffix = target[1:]
                source_string = "S" + target_suffix
                if source_list.__contains__(source_string):
                    target_source_pair.append(source_string + "-" + target)'''
                for src in common_source_in_cluster:
                    target_edges.append(src + '-' + target)

        fig = plt.figure(figsize=(10, 8))
        plt.bar(list(cluster_degree.keys()), list(cluster_degree.values()), align='center')
        plt.xticks(list(cluster_degree.keys()), list(cluster_degree.keys()), rotation=90, ha='right')

        for k in range(len(list(cluster_degree.keys()))):
            plt.annotate(list(cluster_degree.values())[k],
                         (list(cluster_degree.keys())[k], list(cluster_degree.values())[k]), ha='center', va='bottom')

        # plt.xlim(min(list(unique_degree_dict.keys())) - 0.5, max(list(unique_degree_dict.keys())) + 0.5)
        # Set labels and title
        plt.xlabel('Degree Level')
        plt.ylabel('No of Nodes in Degree')
        plt.title('Cluster Degree Distribution of Current Nodes in Cluster' + str(i + 1))
        plt.savefig(citesser_cluster_degree_distribution + str(i + 1) + ".png")
        # Display the plot
        # plt.show()

        for source in source_list_without_zero_degree:
            source_suffix = source[1:]
            target_string = "T" + source_suffix
            if target_list_without_zero_degree.__contains__(target_string):
                source_target_pair.append(source + "-" + target_string)
            '''for trg in common_targets_in_cluster:
                source_edges.append(source + '-' + trg)'''

        for target in target_list_without_zero_degree:
            target_suffix = target[1:]
            source_string = "S" + target_suffix
            if source_list_without_zero_degree.__contains__(source_string):
                target_source_pair.append(source_string + "-" + target)
            '''for src in common_source_in_cluster:
                target_edges.append(src + '-' + target)'''

        cluster_degree_row = []
        cluster_degree_row.append(i + 1)
        cluster_degree_row.append(cluster_dict[i])
        cluster_degree_row.append(node_edges_count)
        cluster_degree_row.append(cluster_degree)
        total_cluster_degree_row.append(cluster_degree_row)
        cluster_degree_cols = ["Cluster No", "Node List", "Node Edge Count", "Cluster Degree Distribution"]

        removed_zero_degree_nodes = set(cluster_dict[i]) - set(zero_degree_nodes)
        removed_singleton_nodes = set(removed_zero_degree_nodes) - set(singleton_nodes)
        row.append(len(removed_singleton_nodes))
        # row.append(source_no)
        # row.append(target_no)
        row.append(len(source_list_without_zero_degree))
        row.append(len(zero_src_nodes))
        row.append(len(source_list_without_zero_degree) - len(zero_src_nodes))
        row.append(len(target_list_without_zero_degree))
        row.append(len(zero_trg_nodes))
        row.append(len(target_list_without_zero_degree) - len(zero_trg_nodes))
        row.append(len(zero_degree_nodes))
        row.append(total_edges_in_cluster_as_source)
        row.append(total_edges_in_cluster_as_target)
        row.append(len(source_list_without_zero_degree) * len(target_list_without_zero_degree))
        row.append(len(only_know_each_other_as_source))
        row.append(len(only_know_each_other_as_target))
        if len(source_list_without_zero_degree) != 0 and len(target_list_without_zero_degree) != 0:
            row.append(total_edges_in_cluster_as_source / (len(source_list_without_zero_degree) * len(target_list_without_zero_degree)))
        else:
            row.append(0)
        row.append(total_edges_in_cluster_as_source / (source_no + target_no))
        if source_no > target_no:
            row.append(total_edges_in_cluster_as_source / target_no)
        elif source_no < target_no:
            row.append(total_edges_in_cluster_as_source / source_no)
        row.append(src_high_degree)
        row.append(trg_high_degree)
        row.append(highest_src_cls_degree)
        row.append(highest_trg_cls_degree)
        row.append(len(source_target_pair))
        row.append(len(target_source_pair))
        row.append(source_target_pair)
        row.append(target_source_pair)
        # row.append(total_edges_in_graph_with_source)
        # row.append(total_edges_in_graph_with_target)
        # TODO: Consider List Of Nodes after removing zero degree nodes and singleton nodes or Original Node List
        row.append(sorted(cluster_dict[i], key=key_func))
        row.append(sorted(zero_degree_nodes, key=key_func))
        row.append(only_know_each_other_as_source)
        row.append(only_know_each_other_as_target)
        row.append(set(singleton_nodes))
        # row.append(sorted(cluster_dict[i], key=key_func))
        row.append(total_connected_target_nodes_not_in_cls)
        row.append(len(total_connected_target_nodes_not_in_cls))
        row.append(total_connected_source_nodes_not_in_cls)
        row.append(len(total_connected_source_nodes_not_in_cls))
        rows.append(row)

        nodes_not_in_cls_row = []
        nodes_not_in_cls_row.append(i)
        nodes_not_in_cls_row.append(total_connected_source_nodes_not_in_cls)
        nodes_not_in_cls_row.append(total_connected_target_nodes_not_in_cls)
        nodes_not_in_cls.append(nodes_not_in_cls_row)

        edge_row.append(i + 1)
        edge_row.append(source_edges)
        edge_row.append(target_edges)
        edges_rows.append(edge_row)

    nodes_not_in_cls_col = ['Cluster No', 'Source_SDM Nodes Not In Cluster', 'Target Nodes Not In Cluster']
    nodes_not_in_cls_dict = pd.DataFrame(nodes_not_in_cls, columns=nodes_not_in_cls_col)
    save_as_pickle(nodes_not_in_cls_pkl, nodes_not_in_cls_dict)
    zero_degree_cols = ['Cluster No', 'Node', 'Zero Degree in Cluster', 'Degree Ouside Cluster']
    pd.DataFrame(zero_degree_rows, columns=zero_degree_cols).to_csv(citeseer_zero_degree_node_analysis)
    result = pd.DataFrame(rows, columns=cols)
    #pd.DataFrame(result).to_csv(citation_spectral_clustering_csv, encoding='utf-8', float_format='%f')
    pd.DataFrame(total_degree_row, columns=degree_cols).to_csv("degreeList_out_off_cluster_nodes.csv", encoding='utf-8', float_format='%f')
    pd.DataFrame(total_cluster_degree_row, columns=cluster_degree_cols).to_csv("cluster_degree_out_off_cluster_nodes.csv", encoding='utf-8', float_format='%f')
    edge_cols = ['Cluster No', 'S-T edges', 'T-S edges']
    edge_rows_df = pd.DataFrame(edges_rows, columns=edge_cols)
    pd.DataFrame(edge_rows_df).to_csv(edge_file_name, encoding='utf-8', float_format='%f')

    return clusters, cluster_dict

def spectral_dbscan_clustering_bitcoin_target_correct_final_one(middleDimentions, no_of_clusters, citation_spectral_clustering_csv, title,
                                              spectral_cluster_plot_file_name, bipartite_G, edge_file_name,
                                              nodes_not_in_cls_pkl, original_citeseer_graph):
    # print("Original Graph Hits score:"+ str(nx.hits(original_bitcoin_graph)))
    hubs, authorities = nx.hits(bipartite_G)
    # Sort the Hub and Authority scores by value in descending order
    sorted_hubs = dict(sorted(hubs.items(), key=lambda item: item[1], reverse=True))
    sorted_authorities = dict(sorted(authorities.items(), key=lambda item: item[1], reverse=True))

    figure12 = plt.figure(figsize=(15, 15))
    ax12 = figure12.add_subplot(111)
    ax12.plot(list(sorted_hubs.keys()), list(sorted_hubs.values()))
    plt.legend()
    plt.title("Hub score plot")
    plt.savefig(citeseer_hub_plot_name + ".png")
    # plt.show()

    figure13 = plt.figure(figsize=(15, 15))
    ax13 = figure13.add_subplot(111)
    ax13.plot(list(sorted_authorities.keys()), list(sorted_authorities.values()))
    plt.legend()
    plt.title("Authority score plot")
    plt.savefig(citeseer_authority_plot_name + ".png")
    # plt.show()

    first_key_hub = list(sorted_hubs.keys())[0]
    first_value_hub = sorted_hubs[first_key_hub]

    last_key_hub = list(sorted_hubs.keys())[len(list(sorted_hubs.keys())) - 1]
    last_value_hub = sorted_hubs[last_key_hub]

    first_key_authority = list(sorted_authorities.keys())[0]
    first_value_authority = sorted_authorities[first_key_authority]

    last_key_authority = list(sorted_authorities.keys())[len(list(sorted_authorities.keys())) - 1]
    last_value_authority = sorted_authorities[last_key_authority]

    print("Bipartite Graph hubs Score:" + str(sorted_hubs))
    print("Highest Hub Score of Bipartite Graph " + str(first_key_hub) + ":" + str(first_value_hub))
    print("Lowest Hub Score of Bipartite Graph " + str(last_key_hub) + ":" + str(last_value_hub))
    print("Bipartite Graph authorities Score:" + str(sorted_authorities))
    print("Highest Authorities Score of Bipartite Graph " + str(first_key_authority) + ":" + str(first_value_authority))
    print("Lowest Authorities Score of Bipartite Graph " + str(last_key_authority) + ":" + str(last_value_authority))

    nodes_not_in_cls = []
    # Bitcoin cluster pickel file
    '''if exists(bitcoin_spectral_clustering_pickel):
        clusters = read_pickel(bitcoin_spectral_clustering_pickel)
    else:
        spectral = SpectralClustering(n_clusters=no_of_clusters, affinity='nearest_neighbors', n_neighbors=50, random_state=0)
        clusters = spectral.fit_predict(middleDimentions)
        save_as_pickle(bitcoin_spectral_clustering_pickel, clusters) '''

    if exists(citeseer_clusters_list_pkl):
        clusters = read_pickel(citeseer_clusters_list_pkl)
    else:
        spectral = SpectralClustering(n_clusters=no_of_clusters, affinity='nearest_neighbors', n_neighbors=50, random_state=0)
        clusters = spectral.fit_predict(middleDimentions)
        save_as_pickle(citeseer_clusters_list_pkl, clusters)

    if exists(citeseer_cluster_nodes) and exists(citeseer_clusterwise_embedding_src_nodes) and exists(
            citeseer_cust_clusterwise_embedding_nodes) and exists(citeseer_cluster_dict):
        cluster_nodes = read_pickel(citeseer_cluster_nodes)
        clusterwise_embedding_src_nodes = read_pickel(citeseer_clusterwise_embedding_src_nodes)
        cust_clusterwise_embedding_nodes = read_pickel(citeseer_cust_clusterwise_embedding_nodes)
        cluster_dict = read_pickel(citeseer_cluster_dict)
    else:
        cluster_dict = {}
        cluster_nodes = {}
        clusterwise_embedding_src_nodes = {}
        cust_clusterwise_embedding_nodes = {}
        for i in range(no_of_clusters):
            cluster_dict[i] = []
            cluster_nodes[i] = {}
            cluster_nodes[i]['nodes'] = []
            cluster_nodes[i]['label'] = []
            cluster_nodes[i]['name'] = []
            cluster_nodes[i]['status'] = []
            cluster_nodes[i]['cluster_no'] = []
            cluster_nodes[i]['index'] = []
            clusterwise_embedding_src_nodes[i] = {}
            clusterwise_embedding_src_nodes[i]['name'] = []
            clusterwise_embedding_src_nodes[i]['label'] = []
            cust_clusterwise_embedding_nodes[i] = {}
            cust_clusterwise_embedding_nodes[i]['name'] = []
            cust_clusterwise_embedding_nodes[i]['label'] = []

        for i in range(len(clusters)):
            cluster_dict[clusters[i]].append(list(middleDimentions.index)[i])
            cluster_nodes[clusters[i]]['nodes'].append(list(middleDimentions.iloc[i]))
            cluster_nodes[clusters[i]]['label'].append('unlabeled')
            cluster_nodes[clusters[i]]['name'].append(list(middleDimentions.index)[i])
            cluster_nodes[clusters[i]]['status'].append('unvisited')
            cluster_nodes[clusters[i]]['cluster_no'].append(-1)
            cluster_nodes[clusters[i]]['index'].append(list(middleDimentions.index).index(list(middleDimentions.index)[i]))

        save_as_pickle(citeseer_cluster_nodes, cluster_nodes)
        save_as_pickle(citeseer_clusterwise_embedding_src_nodes, clusterwise_embedding_src_nodes)
        save_as_pickle(citeseer_cust_clusterwise_embedding_nodes, cust_clusterwise_embedding_nodes)
        save_as_pickle(citeseer_cluster_dict, cluster_dict)

    if exists(citeseer_dbscan_cluster_nodes_labels) and exists(citeseer_embedding_map) and exists(citeseer_source_nodes_map) and exists(citeseer_avg_distance_map):
        cluster_labels_map = read_pickel(citeseer_dbscan_cluster_nodes_labels)
        embedding_map = read_pickel(citeseer_embedding_map)
        source_nodes_map = read_pickel(citeseer_source_nodes_map)
        avg_distance_map = read_pickel(citeseer_avg_distance_map)
    else:
        cluster_labels_map = {}
        avg_distances_rows = []
        embedding_map = {}
        source_nodes_map = {}
        avg_distance_map = {}
        for u in range(len(cluster_nodes)):
            source_nodes = {}
            source_nodes['nodes'] = []
            source_nodes['name'] = []
            source_nodes['label'] = []
            source_nodes['status'] = []
            source_nodes['cluster_no'] = []
            source_nodes['index'] = []
            source_nodes['embedding'] = []
            for j in range(len(cluster_nodes[u]['nodes'])):
                if cluster_nodes[u]['name'][j].startswith("T"):
                    source_nodes['nodes'].append(cluster_nodes[u]['nodes'][j])
                    source_nodes['name'].append(cluster_nodes[u]['name'][j])
                    source_nodes['label'].append(cluster_nodes[u]['label'][j])
                    source_nodes['status'].append(cluster_nodes[u]['status'][j])
                    source_nodes['cluster_no'].append(cluster_nodes[u]['cluster_no'][j])
                    source_nodes['index'].append(cluster_nodes[u]['index'][j])
                    source_nodes['embedding'].append(list(middleDimentions.iloc[cluster_nodes[u]['index'][j]]))
            source_nodes_map[u] = source_nodes
            avg, sorted_names, sorted_values = average_distance(source_nodes)
            avg_distance_map[u] = avg
            avg_distances_row = []
            avg_distances_row.append(u + 1)
            avg_distances_row.append(sorted_names)
            avg_distances_row.append(sorted_values)
            avg_distances_rows.append(avg_distances_row)

            dbscan = DBSCAN(eps=avg, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(source_nodes['nodes'])
            cluster_labels_map[u] = cluster_labels

            tsne = TSNE(n_components=2, perplexity=5)
            embeddings_2d = tsne.fit_transform(np.array(source_nodes['nodes']))
            embedding_map[u] = embeddings_2d

        save_as_pickle(citeseer_dbscan_cluster_nodes_labels, cluster_labels_map)
        save_as_pickle(citeseer_embedding_map, embedding_map)
        save_as_pickle(citeseer_source_nodes_map, source_nodes_map)
        save_as_pickle(citeseer_avg_distance_map, avg_distance_map)
        avg_distance_cols = ["Cluster No", "Source_SDM Names", "5th Nearest Neighbor Distance"]
        pd.DataFrame(avg_distances_rows, columns=avg_distance_cols).to_csv(citeseer_5th_nearest_neighbor_distance_csv)

    rows = []
    dbscan_rows = []
    for i in range(len(cluster_nodes)):
        cluster_labels = cluster_labels_map[i]
        citeseer_directed_A = nx.adjacency_matrix(original_citeseer_graph)
        bitcoin_directed_A = pd.DataFrame(citeseer_directed_A.toarray(), columns=original_citeseer_graph.nodes(), index=original_citeseer_graph.nodes())

        embeddings_2d = embedding_map[i]
        print("embedding len" + str(len(embeddings_2d)))
        print("cluster label len" + str(len(np.array(cluster_labels))))
        print(cluster_labels)
        '''x_values = [embeddings_2d[j, 0] for j in range(len(embeddings_2d[:, 0])) if np.array(cluster_labels)[j] != -1]
        y_values = [embeddings_2d[j, 1] for j in range(len(embeddings_2d[:, 0])) if np.array(cluster_labels)[j] != -1]       
        source_lables = [int(lab) for lab in np.array(cluster_labels) if lab != -1]      
        source_names = [source_nodes['name'][j] for j in range(len(source_nodes['name'])) if cluster_labels[j] != -1]
        src_cls_embedding = [source_nodes['embedding'][j] for j in range(len(source_nodes['name'])) if cluster_labels[j] != -1] '''
        x_values = []
        y_values = []
        source_lables = []
        source_names = []
        src_cls_embedding = []
        for b in range(len(np.array(cluster_labels))):
            if cluster_labels[b] != -1:
                x_values.append(embeddings_2d[b, 0])
                y_values.append(embeddings_2d[b, 1])
                source_lables.append(int(cluster_labels[b]))
                source_nodes = source_nodes_map[i]
                source_names.append(source_nodes['name'][b])
                src_cls_embedding.append(source_nodes['embedding'][b])

        figure5 = plt.figure(figsize=(15, 15))
        ax5 = figure5.add_subplot(111)
        ax5.scatter(x_values, y_values, c=source_lables)
        plt.legend()
        plt.title("DBSCAN Inside Spectral Clustering (Cluster No: " + str(i + 1) + ") ( No of Clusters " + str(
            len(set(source_lables))) + ")")
        plt.savefig(citeseer_spectral_dbscan_plot_name + "_" + str(i + 1) + ".png")
        # plt.show()

        clusterwise_embedding_src_nodes[i]['name'] = source_names
        clusterwise_embedding_src_nodes[i]['label'] = source_lables

        dbscan_cluster_info = {}
        for lab in list(set(source_lables)):
            dbscan_cluster_info[lab] = {}
            dbscan_cluster_info[lab]['x_value'] = []
            dbscan_cluster_info[lab]['y_value'] = []
            dbscan_cluster_info[lab]['names'] = []
            dbscan_cluster_info[lab]['embedding'] = []
            dbscan_cluster_info[lab]['status'] = []
            dbscan_cluster_info[lab]['label'] = []

        for lab in set(source_lables):
            dbscan_cluster_info[lab]['x_value'] = [x_values[k] for k in range(len(source_lables)) if source_lables[k] == lab]
            dbscan_cluster_info[lab]['y_value'] = [y_values[k] for k in range(len(source_lables)) if source_lables[k] == lab]
            dbscan_cluster_info[lab]['names'] = [source_names[k] for k in range(len(source_names)) if source_lables[k] == lab]
            dbscan_cluster_info[lab]['embedding'] = [src_cls_embedding[k] for k in range(len(source_lables)) if source_lables[k] == lab]
            dbscan_cluster_info[lab]['label'] = [lab for k in range(len(source_lables)) if source_lables[k] == lab]

        for k in range(len(dbscan_cluster_info)):
            row = []
            row.append(i + 1)
            row.append(k + 1)
            row.append(len(dbscan_cluster_info[k]['names']))
            row.append(min_samples)
            row.append(avg_distance_map[i])
            row.append(dbscan_cluster_info[k]['names'])
            row.append(dbscan_cluster_info[k]['embedding'])
            row.append(dbscan_cluster_info[k]['x_value'])
            row.append(dbscan_cluster_info[k]['y_value'])
            rows.append(row)

        cust_dbscan_cluster_nodes = {}
        for k in range(len(dbscan_cluster_info)):
            cust_dbscan_cluster_nodes[k] = {}
            cust_dbscan_cluster_nodes[k]['nodes'] = []
            cust_dbscan_cluster_nodes[k]['original_embedding'] = []
            cust_dbscan_cluster_nodes[k]['src_x_embedding'] = []
            cust_dbscan_cluster_nodes[k]['src_y_embedding'] = []
            cust_dbscan_cluster_nodes[k]['status'] = []
            cust_dbscan_cluster_nodes[k]['label'] = []
            cust_dbscan_cluster_nodes[k]['no_src'] = 0
            cust_dbscan_cluster_nodes[k]['no_trg'] = 0
            cust_dbscan_cluster_nodes[k]['no_pair'] = 0
            cust_dbscan_cluster_nodes[k]['avg_degree'] = 0

        for k in range(len(dbscan_cluster_info)):
            cust_dbscan_cluster_nodes[k]['nodes'] = copy.deepcopy(dbscan_cluster_info[k]['names'])
            cust_dbscan_cluster_nodes[k]['original_embedding'] = copy.deepcopy(dbscan_cluster_info[k]['embedding'])
            cust_dbscan_cluster_nodes[k]['src_x_embedding'] = copy.deepcopy(dbscan_cluster_info[k]['x_value'])
            cust_dbscan_cluster_nodes[k]['src_y_embedding'] = copy.deepcopy(dbscan_cluster_info[k]['y_value'])
            cust_dbscan_cluster_nodes[k]['status'] = ["unvisited" for k in range(len(dbscan_cluster_info[k]['names']))]
            cust_dbscan_cluster_nodes[k]['label'] = copy.deepcopy(dbscan_cluster_info[k]['label'])

        converged_cnt = 0
        for p in range(len(cust_dbscan_cluster_nodes)):
            dbscan_names = set(cust_dbscan_cluster_nodes[p]['nodes'])
            while len(dbscan_names) > 0:
                last_element = dbscan_names.pop()
                last_element_index = cust_dbscan_cluster_nodes[p]['nodes'].index(last_element)
                last_element_status = cust_dbscan_cluster_nodes[p]['status'][last_element_index]
                if last_element_status == "unvisited":
                    cust_dbscan_cluster_nodes[p]['status'][last_element_index] = "visited"
                    if last_element.startswith("S"):
                        new_src = int(last_element[1:])
                        src_trg = bitcoin_directed_A.loc[new_src]
                        indices = [i for i, value in enumerate(src_trg) if value == 1]
                        dataframe_indices = list(src_trg.index)
                        trg_list = [dataframe_indices[trg_index] for trg_index in indices]
                        for trg in trg_list:
                            # if list(cust_dbscan_cluster_nodes[p]['nodes']).__contains__("T"+str(trg)) == False and list(cluster_nodes[i]['name']).__contains__("T"+str(trg)) == True:
                            if list(cust_dbscan_cluster_nodes[p]['nodes']).__contains__("T" + str(trg)) == False:
                                src_trg_neighbors = bitcoin_directed_A[trg]
                                indices = [i for i, value in enumerate(src_trg_neighbors) if value == 1]
                                dataframe_indices = list(src_trg_neighbors.index)
                                src_list = [dataframe_indices[src_index] for src_index in indices]
                                src_trg_neighbors = set(src_list)
                                cls_src_list = [int(src[1:]) for src in dbscan_names if src.startswith("S")]
                                cls_src_list = set(cls_src_list)
                                common_src = list(cls_src_list.intersection(src_trg_neighbors))
                                if len(common_src) >= closer_3:
                                    included_trg = "T" + str(trg)
                                    cust_dbscan_cluster_nodes[p]['nodes'].append(included_trg)
                                    cust_dbscan_cluster_nodes[p]['status'].append("unvisited")
                                    cust_dbscan_cluster_nodes[p]['original_embedding'].append(
                                        list(middleDimentions.loc[included_trg]))
                                    cust_dbscan_cluster_nodes[p]['label'].append(
                                        list(set(cust_dbscan_cluster_nodes[p]['label']))[0])
                                    dbscan_names.update([included_trg])
                    elif last_element.startswith("T"):
                        new_trg = int(last_element[1:])
                        trg_src = bitcoin_directed_A[new_trg]
                        indices = [i for i, value in enumerate(trg_src) if value == 1]
                        dataframe_indices = list(trg_src.index)
                        src_list = [dataframe_indices[src_index] for src_index in indices]
                        for src in src_list:
                            # if list(cust_dbscan_cluster_nodes[p]['nodes']).__contains__("S"+str(src)) == False and list(cluster_nodes[i]['name']).__contains__("S"+str(src)) == True:
                            if list(cust_dbscan_cluster_nodes[p]['nodes']).__contains__("S" + str(src)) == False:
                                trg_src_neighbors = bitcoin_directed_A.loc[src]
                                indices = [i for i, value in enumerate(trg_src_neighbors) if value == 1]
                                dataframe_indices = list(trg_src_neighbors.index)
                                trg_list = [dataframe_indices[trg_index] for trg_index in indices]
                                trg_src_neighbors = set(trg_list)
                                cls_trg_list = [int(src[1:]) for src in dbscan_names if src.startswith("T")]
                                cls_trg_list = set(cls_trg_list)
                                common_trg = list(cls_trg_list.intersection(trg_src_neighbors))
                                if len(common_trg) >= closer_3:
                                    included_src = "S" + str(src)
                                    cust_dbscan_cluster_nodes[p]['nodes'].append(included_src)
                                    cust_dbscan_cluster_nodes[p]['status'].append("unvisited")
                                    cust_dbscan_cluster_nodes[p]['original_embedding'].append(
                                        list(middleDimentions.loc[included_src]))
                                    cust_dbscan_cluster_nodes[p]['label'].append(
                                        list(set(cust_dbscan_cluster_nodes[p]['label']))[0])
                                    dbscan_names.update([included_src])

        cust_nodes = []
        cust_labels = []
        for h in range(len(cust_dbscan_cluster_nodes)):
            cust_nodes.extend(list(cust_dbscan_cluster_nodes[h]['nodes']))
            cust_labels.extend(list(cust_dbscan_cluster_nodes[h]['label']))
        cust_clusterwise_embedding_nodes[i]['name'] = cust_nodes
        cust_clusterwise_embedding_nodes[i]['label'] = cust_labels
        '''tsne = TSNE(n_components=2, perplexity=25)
        embeddings_2d = tsne.fit_transform(np.array(middleDimentions))
        x_min = min(embeddings_2d[:, 0])
        x_max = max(embeddings_2d[:, 0])
        y_min = min(embeddings_2d[:, 1])
        y_max = max(embeddings_2d[:, 1])
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        figure10 = plt.figure(figsize=(15, 15))
        ax10 = figure10.add_subplot(111)
        embeddings_2d = pd.DataFrame(embeddings_2d, index=middleDimentions.index)
        cust_x = []
        cust_y = []
        cust_label = []
        for k in range(len(cust_dbscan_cluster_nodes)):
            dbscan_cls_names = list(cust_dbscan_cluster_nodes[k]['nodes'])
            dbscan_labels = list(cust_dbscan_cluster_nodes[k]['label'])
            for p in range(len(dbscan_cls_names)):
                cust_x.append(embeddings_2d.loc[dbscan_cls_names[p]][0])
                cust_y.append(embeddings_2d.loc[dbscan_cls_names[p]][1])
                cust_label.append(dbscan_labels[p])
        ax10.scatter(cust_x,cust_y, c=cust_label, cmap="jet")
        plt.title(bitcoin_customized_dbscan_clusters + str(i+1) +"( No of Clusters " + str(len(cust_dbscan_cluster_nodes)) + ")")
        plt.savefig(bitcoin_customized_dbscan_clusters_plot + str(i+1))
        plt.legend()
        plt.show()'''

        for k in range(len(cust_dbscan_cluster_nodes)):
            hub_bar_list = {}
            authority_bar_list = {}
            line_x_values = {}
            for bar, bar_value in zip(list(sorted_hubs.keys()), list(sorted_hubs.values())):
                hub_bar_list[bar]= 0
                authority_bar_list[bar] = 0
                line_x_values[bar] = ""
            row = []
            row.append(i + 1)
            row.append(k + 1)
            row.append(len(cust_dbscan_cluster_nodes[k]['nodes']))
            row.append(len(dbscan_cluster_info[k]['names']))
            sr_no = 0
            tr_no = 0
            pair_nodes = set()
            degree_cnt = 0
            edges = set()
            cls_nodes = set(cust_dbscan_cluster_nodes[k]['nodes'])
            hits_score_list = {}
            authority_score_list = {}
            percentile_score_list = {}
            for nd in cust_dbscan_cluster_nodes[k]['nodes']:
                if nd.startswith("S"):
                    sr_no += 1
                    node_no = nd[1:]
                    tr_node = "T" + str(node_no)
                    if list(cust_dbscan_cluster_nodes[k]['nodes']).__contains__(tr_node):
                        pair_nodes.add(node_no)
                elif nd.startswith("T"):
                    tr_no += 1
                    node_no = nd[1:]
                    sr_node = "S" + str(node_no)
                    if list(cust_dbscan_cluster_nodes[k]['nodes']).__contains__(sr_node):
                        pair_nodes.add(node_no)
                edge_list = list(bipartite_G.neighbors(nd))
                common_neighbors = set(edge_list).intersection(cls_nodes)
                hits_score_list[nd] = sorted_hubs[nd] * 1e-4
                hub_bar_list[nd] = max(sorted_hubs.values())
                authority_bar_list[nd] = max(sorted_authorities.values())
                line_x_values[nd] = nd
                authority_score_list[nd] = sorted_authorities[nd] * 1e-4
                percentile = np.percentile(list(sorted_hubs.values()),(list(sorted_hubs.keys()).index(nd) / len(list(sorted_hubs.values()))) * 100)
                percentile_score_list[nd] = percentile
                for cm_ed in common_neighbors:
                    if nd.startswith("S"):
                        edge = str(nd) + "-" + str(cm_ed)
                    elif nd.startswith("T"):
                        edge = str(cm_ed) + "-" + str(nd)
                    edges.add(edge)

            row.append(sr_no)
            row.append(tr_no)
            row.append(len(edges))
            row.append(len(pair_nodes))
            row.append(cust_dbscan_cluster_nodes[k]['nodes'])
            row.append(edges)
            bipartite_G_copy = bipartite_G.copy()
            #subgraph = bipartite_G_copy.subgraph(cust_dbscan_cluster_nodes[k]['nodes'])
            '''bipartite_G_copy.freeze()
            for edge in edges:
                subgraph.add_edge(*edge)'''
            #sub_graph_hubs, sub_graph_authorities = nx.hits(subgraph)
            # Sort the Hub and Authority scores by value in descending order
            #sub_graph_sorted_hubs = dict(sorted(sub_graph_hubs.items(), key=lambda item: item[1], reverse=True))
            #sub_graph_sorted_authorities = dict(sorted(sub_graph_authorities.items(), key=lambda item: item[1], reverse=True))
            #row.append(cust_dbscan_cluster_nodes[k]['status'])
            #row.append(cust_dbscan_cluster_nodes[k]['original_embedding'])
            #row.append(cust_dbscan_cluster_nodes[k]['label'])
            row.append(hits_score_list)
            row.append(authority_score_list)
            #row.append(percentile_score_list)
            #row.append(sub_graph_sorted_hubs)
            #row.append(sub_graph_sorted_authorities)

            y_min = min(list(sorted_hubs.values()))
            y_max = max(list(sorted_hubs.values()))
            figure_cluster_hub = plt.figure(figsize=(15, 15))
            ax_cluster_hub = figure_cluster_hub.add_subplot(111)
            ax_cluster_hub.set_ylim(y_min, y_max)
            ax_cluster_hub.plot(list(sorted_hubs.keys()), list(sorted_hubs.values()), label="Bipartite Graph Hub Score")
            ax_cluster_hub.bar(list(sorted_hubs.keys()), list(hub_bar_list.values()), color='red', label="Cluster Hub Score Location")
            plt.legend()
            plt.title("Node Hub Score Position For Spectral Cluster No:" + str(i+1) + "DBSCAN Cluster No:" + str(k+1) + " Highest Hub Score in Cluster:"+str(max(list(sorted_hubs.values()))) + " Lowest Hub Score in Cluster:" +str(min(list(sorted_hubs.values())))
                      +" Highest Authority Score in Cluster:"+str(max(list(authority_score_list.values()))) + " Lowest Authority Score in Cluster:"+str(min(list(authority_score_list.values()))))
            #plt.show()
            plt.savefig(citeseer_speparate_cluster_path + "node_hub_score_position_spectral_cls"+str(i+1) + "dbscan_cls_" + str(k+1) +".png")

            y_min = min(list(sorted_authorities.values()))
            y_max = max(list(sorted_authorities.values()))
            figure_cluster_authority = plt.figure(figsize=(15, 15))
            ax_cluster_authority = figure_cluster_authority.add_subplot(111)
            ax_cluster_authority.set_ylim(y_min, y_max)
            ax_cluster_authority.plot(list(sorted_authorities.keys()), list(sorted_authorities.values()), label="Bipartite Graph Authority Score")
            ax_cluster_authority.bar(list(sorted_authorities.keys()), list(authority_bar_list.values()), color='red', label="Cluster Authority Score Location")
            plt.legend()
            plt.title("Node Authority Score Position For Spectral Cluster No:" + str(i + 1) + "  DBSCAN Cluster No:" + str(k + 1))
            # plt.show()
            plt.savefig(citeseer_speparate_cluster_path + "node_authority_score_position_spectral_cls" + str(i + 1) + "dbscan_cls_" + str(k + 1) + ".png")
            dbscan_rows.append(row)

    # target clustering
    dbscan_cols = ['Spectral Cluster No', 'DBSCAN Cluster No', 'No Of Target Nodes in Cluster', 'Min Sample', 'Eps','List Of Nodes',
                   'Node2vec 20-D Embedding','X Value(TSNE)', 'Y Value(TSNE)']
    pd.DataFrame(rows, columns=dbscan_cols).to_csv(citeseer_dbscan_cluster_csv_file)

    hit_authority_score_info_row = []
    hit_authority_score_info_rows = []
    hit_authority_score_info_row.append("Highest Hub Score")
    hit_authority_score_info_rows.append(hit_authority_score_info_row)
    hit_authority_score_info_row = []
    hit_authority_score_info_row.append(str(first_key_hub) + ":" + str(first_value_hub))
    hit_authority_score_info_rows.append(hit_authority_score_info_row)
    hit_authority_score_info_row = []
    hit_authority_score_info_row.append("Lowest Hub Score")
    hit_authority_score_info_rows.append(hit_authority_score_info_row)
    hit_authority_score_info_row = []
    hit_authority_score_info_row.append(str(last_key_hub) + ":" + str(last_value_hub))
    hit_authority_score_info_rows.append(hit_authority_score_info_row)
    hit_authority_score_info_row = []
    hit_authority_score_info_row.append("Highest Authority Score")
    hit_authority_score_info_rows.append(hit_authority_score_info_row)
    hit_authority_score_info_row = []
    hit_authority_score_info_row.append(str(first_key_authority) + ":" + str(first_value_authority))
    hit_authority_score_info_rows.append(hit_authority_score_info_row)
    hit_authority_score_info_row = []
    hit_authority_score_info_row.append("Lowest Authority Score")
    hit_authority_score_info_rows.append(hit_authority_score_info_row)
    hit_authority_score_info_row = []
    hit_authority_score_info_row.append(str(last_key_authority) + ":" + str(last_value_authority))
    hit_authority_score_info_rows.append(hit_authority_score_info_row)

    pd.DataFrame(hit_authority_score_info_rows).to_csv(citeseer_hit_authority_score_csv)

    # target columns
    '''cust_dbscan_cols = ["Spectral Cluster No", "DBSCAN Cluster No", "No Of Nodes in Cluster", "Original No Of Targets","No Of Source_SDM Nodes","No Of Target Nodes",  "No Of Edges","No Of Pairs", "List Of Nodes",
                        "List Of Edges", "Status Of Nodes", "Original Embedding", "Labels",
                        "Original Bipartite Graph Hub Score", "Original Bipartite Graph Authority Score","Subgraph Hub Score","Subgraph Authority Score",  "Percentile"]'''
    cust_dbscan_cols = ["Spectral Cluster No", "DBSCAN Cluster No", "No Of Nodes in Cluster", "Original No Of Targets",
                        "No Of Source_SDM Nodes", "No Of Target Nodes", "No Of Edges", "No Of Pairs", "List Of Nodes",
                        "List Of Edges", "Original Bipartite Graph Hub Score", "Original Bipartite Graph Authority Score"]
    pd.DataFrame(dbscan_rows, columns=cust_dbscan_cols).to_csv(citeseer_cust_dbscan_cluster_csv_file)


    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 0),  # Maroon
        (0, 128, 0),  # Green
        (0, 0, 128),  # Navy
        (128, 128, 0),  # Olive
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
        (255, 165, 0),  # Orange
        (128, 128, 128),  # Gray
        (255, 192, 203),  # Pink
        (0, 128, 0),  # Green
        (128, 0, 0),  # Maroon
        (0, 0, 128),  # Navy
        (0, 128, 128),  # Teal
        (128, 0, 128),  # Purple
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
    ]

    normalized_colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]

    selected_colors = [normalized_colors[i] for i in clusters]
    tsne = TSNE(n_components=2, perplexity=25)
    embeddings_2d = tsne.fit_transform(np.array(middleDimentions))
    x_min = min(embeddings_2d[:, 0])
    x_max = max(embeddings_2d[:, 0])
    y_min = min(embeddings_2d[:, 1])
    y_max = max(embeddings_2d[:, 1])
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    figure = plt.figure(figsize=(15, 15))
    ax = figure.add_subplot(111)
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=selected_colors, cmap='jet')
    plt.title(title + "( No of Clusters " + str(len(set(clusters))) + ")")
    plt.savefig(spectral_cluster_plot_file_name)
    plt.legend()
    # plt.show()

    embeddings_2d = pd.DataFrame(embeddings_2d, index=middleDimentions.index)

    '''Original Embedding'''
    k = 0
    for cls_nodes in cluster_dict.values():
        source_x_embedding = []
        source_y_embedding = []
        target_x_embedding = []
        target_y_embedding = []
        source_labels = []
        target_labels = []
        source_colors = []
        target_colors = []
        figure1 = plt.figure(figsize=(15, 15))
        ax1 = figure1.add_subplot(111)
        color = [k for j in range(len(cls_nodes))]

        for cls_node in cls_nodes:
            scatter = ax1.scatter(embeddings_2d.loc[cls_node][0], embeddings_2d.loc[cls_node][1], c=normalized_colors[k], cmap="jet")
            if cls_node.startswith("T"):
                source_x_embedding.append(embeddings_2d.loc[cls_node][0])
                source_y_embedding.append(embeddings_2d.loc[cls_node][1])
                source_labels.append(cls_node)
                source_colors.append("blue")
            elif cls_node.startswith("T"):
                target_x_embedding.append(embeddings_2d.loc[cls_node][0])
                target_y_embedding.append(embeddings_2d.loc[cls_node][1])
                target_labels.append(cls_node)
                target_colors.append("red")

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.title(citeseer_node_embedding_title + " " + str(k + 1))
        cluster_file_name = citeseer_speparate_cluster_path + str(k + 1)
        plt.savefig(cluster_file_name)
        # plt.show()

        figure2 = plt.figure(figsize=(15, 15))
        ax2 = figure2.add_subplot(111)
        source_scatter = ax2.scatter(source_x_embedding, source_y_embedding, cmap="Set2", c=source_colors, label="Source_SDM")
        target_scatter = ax2.scatter(target_x_embedding, target_y_embedding, cmap="Set2", c=target_colors, label="Target")
        plt.title(citeseer_TSNE_clusterwise_source_target_title + str(k + 1))
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.savefig(citesser_TSNE_clusterwise_source_target_plot + str(k + 1))
        # plt.show()

        cluster_wise_embedding_x = []
        cluster_wise_embedding_y = []
        cluster_wise_embedding_label = []
        m = 0
        for cls_name in clusterwise_embedding_src_nodes[k]['name']:
            cluster_wise_embedding_x.append(embeddings_2d.loc[cls_name][0])
            cluster_wise_embedding_y.append(embeddings_2d.loc[cls_name][1])
            cluster_wise_embedding_label.append(clusterwise_embedding_src_nodes[k]['label'][m])
            m += 1
        figure3 = plt.figure(figsize=(15, 15))
        ax3 = figure3.add_subplot(111)
        clusterwise_source_scatter = ax3.scatter(cluster_wise_embedding_x, cluster_wise_embedding_y, cmap="Set2",
                                                 c=cluster_wise_embedding_label, label="Source_SDM Nodes")
        plt.title(citeseer_TSNE_clusterwise_source_with_cluster_title + str(k + 1) + " No Of Clusters:" + str(len(set(cluster_wise_embedding_label))))
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.savefig(citesser_TSNE_clusterwise_source_with_cluster_labels_plot + str(k + 1))
        # plt.show()

        cust_cluster_wise_embedding_x = []
        cust_cluster_wise_embedding_y = []
        cust_cluster_wise_embedding_label = []
        m = 0
        for cls_name in cust_clusterwise_embedding_nodes[k]['name']:
            cust_cluster_wise_embedding_x.append(embeddings_2d.loc[cls_name][0])
            cust_cluster_wise_embedding_y.append(embeddings_2d.loc[cls_name][1])
            cust_cluster_wise_embedding_label.append(cust_clusterwise_embedding_nodes[k]['label'][m])
            m += 1
        figure10 = plt.figure(figsize=(15, 15))
        ax10 = figure10.add_subplot(111)
        cust_clusterwise_source_scatter = ax10.scatter(cust_cluster_wise_embedding_x, cust_cluster_wise_embedding_y, cmap="Set2", c=cust_cluster_wise_embedding_label)
        plt.title(citeseer_customized_dbscan_clusters + str(k + 1) + "( No of Clusters " + str(len(list(set(cust_clusterwise_embedding_nodes[k]['label'])))) + ")")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.savefig(citeseer_customized_dbscan_clusters_plot + str(k + 1))
        plt.legend()
        # plt.show()

        k += 1

    cols = ["Cluster No",
            "No of Nodes in Cluster",
            "No of Sources",
            "No Of Zero Degree Source_SDM",
            "No of Sources - No Of Zero Degree Source_SDM",
            "No of Targets",
            "No Of Zero Degree Target",
            "No of Targets - No Of Zero Degree Target",
            "No of Zero-degree Nodes",
            "No of Outgoing Edges in Cluster (to nodes only within cluster)",
            "No of Incoming Edges in Cluster (from nodes only within cluster)",
            "No of possible edges",
            "No of Only Know Each Other (As Source_SDM)",
            "No of Only Know Each Other (As Target)",
            "Edge Percentage",
            "Average Degree Of Cluster",
            "Hub/Authority Score",
            "Highest Graph Source_SDM Degree",
            "Highest Graph Target Degree",
            "Highest Cluster Source_SDM Degree",
            "Highest Cluster Target Degree",
            "No of Source_SDM and Target Pair available",
            "No of Target and Source_SDM Pair available",
            "Source_SDM Target Pairs",
            "Target Source_SDM Pairs",
            "List of Nodes in Cluster",
            "List of Zero-degree Nodes in Cluster",
            "List of Only Know Each Other (As Source_SDM)",
            "List of Only Know Each Other (As Target)",
            "Singleton Nodes",
            "List of Connected Target Nodes Out-off Cluster",
            "No of Connected Target Nodes Out-off Cluster",
            "List of Connected Source_SDM Nodes Out-off Cluster",
            "No of Connected Source_SDM Nodes Out-off Cluster"]
    rows = []
    new_rows = []
    edges_rows = []
    cluster_wise_label_map = {str(class_name): {} for class_name in range(len(cluster_dict))}
    node_degree_dict = dict(bipartite_G.degree())
    total_degree_row = []
    total_cluster_degree_row = []
    total_cluster_degree = []
    zero_degree_rows = []
    for i in range(len(cluster_dict)):
        row = []
        edge_row = []
        row.append(i + 1)
        source_no = 0
        target_no = 0
        source_list = []
        target_list = []
        original_graph_edge_dict = {}
        source_target_pair = []
        target_source_pair = []
        src_high_degree = -1
        trg_high_degree = -1
        degree_list = []
        zero_src_nodes = []
        zero_trg_nodes = []
        for node in cluster_dict[i]:
            if node.startswith("S"):
                if node_degree_dict[node] > src_high_degree:
                    src_high_degree = node_degree_dict[node]
                if bipartite_G.degree(node) == 0:
                    zero_src_nodes.append(node)
                source_list.append(node)
                source_no += 1
            elif node.startswith("T"):
                if node_degree_dict[node] > trg_high_degree:
                    trg_high_degree = node_degree_dict[node]
                if bipartite_G.degree(node) == 0:
                    zero_trg_nodes.append(node)
                target_list.append(node)
                target_no += 1
            degree_list.append(node_degree_dict[node])
            original_graph_edge_dict[node] = list(bipartite_G.neighbors(node))

        unique_degrees = set(degree_list)
        unique_degree_dict = {}
        for z in range(len(degree_list)):
            if unique_degree_dict.keys().__contains__("Degree " + str(list(degree_list)[z])):
                unique_degree_dict["Degree " + str(list(degree_list)[z])] = unique_degree_dict["Degree " + str(
                    list(degree_list)[z])] + 1
            else:
                unique_degree_dict["Degree " + str(list(degree_list)[z])] = 1

        '''bin_width = 5
        num_bins = max(int((max(degree_list) - min(degree_list)) / bin_width), 1)
        n, bins, patches = plt.hist(degree_list, bins=num_bins)'''
        fig = plt.figure(figsize=(10, 8))
        plt.bar(list(unique_degree_dict.keys()), list(unique_degree_dict.values()), align='center')
        plt.xticks(list(unique_degree_dict.keys()), list(unique_degree_dict.keys()), rotation=90, ha='right')

        for k in range(len(list(unique_degree_dict.keys()))):
            plt.annotate(list(unique_degree_dict.values())[k], (list(unique_degree_dict.keys())[k], list(unique_degree_dict.values())[k]), ha='center', va='bottom')

        # plt.xlim(min(list(unique_degree_dict.keys())) - 0.5, max(list(unique_degree_dict.keys())) + 0.5)
        # Set labels and title
        plt.xlabel('Degree Level')
        plt.ylabel('No of Nodes in Degree')
        plt.title('Bipartite Graph Degree Distribution' + str(i + 1))
        plt.savefig("Bipartite Graph Degree_Distribution" + str(i + 1) + ".png")
        # Display the plot
        # plt.show()

        degree_cols = ["Cluster No", "Node List", "Degree List", "Bipartite Graph Degree Distribution"]
        degree_row = []
        degree_row.append(i + 1)
        degree_row.append(cluster_dict[i])
        degree_row.append(degree_list)
        degree_row.append(unique_degree_dict)
        total_degree_row.append(degree_row)
        '''for j in range(len(n)):
            if n[j] > 0:
                plt.text(bins[j] + bin_width / 2, n[j], str(int(n[j])), ha='center', va='bottom')

        bin_ranges = ['{:.1f}-{:.1f}'.format(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]
        tick_locations = bins[:-1] + bin_width/2  # Adjust tick positions
        plt.xticks(tick_locations, bin_ranges, rotation=45, ha='right')'''

        total_edges_in_cluster_as_source = 0
        total_edges_in_graph_with_source = 0
        total_connected_target_nodes_not_in_cls = []
        source_edges = []
        cluster_degree = {}
        node_edges_count = {}
        highest_src_cls_degree = -1
        source_list_without_zero_degree = []
        zero_degree_nodes = []
        only_know_each_other_as_source = []
        singleton_nodes = []
        for source in source_list:
            source_neighbor_list = original_graph_edge_dict[source]
            common_targets_in_cluster = set(source_neighbor_list) & set(target_list)

            if len(common_targets_in_cluster) == 0:
                zero_degree_row = []
                zero_degree_row.append(i + 1)
                zero_degree_row.append(source)
                zero_degree_row.append("Degree " + str(len(common_targets_in_cluster)))
                zero_degree_row.append(len(source_neighbor_list))
                zero_degree_rows.append(zero_degree_row)
                zero_degree_nodes.append(source)
            else:
                source_list_without_zero_degree.append(source)

                if len(common_targets_in_cluster) == 1:
                    only_source = original_graph_edge_dict[list(common_targets_in_cluster)[0]]
                    only_source_list = set(only_source) & set(source_list)
                    if len(only_source_list) == 1:
                        if list(only_source_list)[0] == source:
                            only_know_each_other_as_source.append(source + "->" + list(common_targets_in_cluster)[0])
                            singleton_nodes.append(source)
                            singleton_nodes.append(list(common_targets_in_cluster)[0])
                            source_list_without_zero_degree.remove(source)

                if len(common_targets_in_cluster) > highest_src_cls_degree:
                    highest_src_cls_degree = len(common_targets_in_cluster)
                node_edges_count[source] = common_targets_in_cluster
                if cluster_degree.keys().__contains__("Degree " + str(len(common_targets_in_cluster))):
                    cluster_degree["Degree " + str(len(common_targets_in_cluster))] = cluster_degree["Degree " + str(
                        len(common_targets_in_cluster))] + 1
                else:
                    cluster_degree["Degree " + str(len(common_targets_in_cluster))] = 1

                connected_target_nodes_not_in_cls = set(source_neighbor_list) - set(common_targets_in_cluster)
                for not_cls_trg in connected_target_nodes_not_in_cls:
                    if not_cls_trg not in total_connected_target_nodes_not_in_cls:
                        total_connected_target_nodes_not_in_cls.append(not_cls_trg)
                total_edges_in_cluster_as_source += len(common_targets_in_cluster)
                total_edges_in_graph_with_source += len(source_neighbor_list)
                '''source_suffix = source[1:]
                target_string = "T" + source_suffix
                if target_list.__contains__(target_string):
                    source_target_pair.append(source + "-" + target_string)'''
                for trg in common_targets_in_cluster:
                    source_edges.append(source + '-' + trg)

        total_edges_in_cluster_as_target = 0
        total_edges_in_graph_with_target = 0
        total_connected_source_nodes_not_in_cls = []
        target_edges = []
        highest_trg_cls_degree = -1

        target_list_without_zero_degree = []
        only_know_each_other_as_target = []
        for target in target_list:
            target_neighbor_list = original_graph_edge_dict[target]
            common_source_in_cluster = set(target_neighbor_list) & set(source_list)
            if len(common_source_in_cluster) > highest_trg_cls_degree:
                highest_trg_cls_degree = len(common_source_in_cluster)

            zero_degree_row = []
            if len(common_source_in_cluster) == 0:
                zero_degree_row.append(i + 1)
                zero_degree_row.append(target)
                zero_degree_row.append("Degree " + str(len(common_source_in_cluster)))
                zero_degree_row.append(len(target_neighbor_list))
                zero_degree_rows.append(zero_degree_row)
                zero_degree_nodes.append(target)
            else:
                target_list_without_zero_degree.append(target)

                if len(common_source_in_cluster) == 1:
                    only_target = original_graph_edge_dict[list(common_source_in_cluster)[0]]
                    only_target_list = set(only_target) & set(target_list)
                    if len(only_target_list) == 1:
                        if list(only_target_list)[0] == target:
                            only_know_each_other_as_target.append(list(common_source_in_cluster)[0] + "->" + target)
                            singleton_nodes.append(target)
                            singleton_nodes.append(list(common_source_in_cluster)[0])
                            target_list_without_zero_degree.remove(target)

                node_edges_count[target] = common_source_in_cluster
                if cluster_degree.keys().__contains__("Degree " + str(len(common_source_in_cluster))):
                    cluster_degree["Degree " + str(len(common_source_in_cluster))] = cluster_degree["Degree " + str(
                        len(common_source_in_cluster))] + 1
                else:
                    cluster_degree["Degree " + str(len(common_source_in_cluster))] = 1
                connected_source_nodes_not_in_cls = set(target_neighbor_list) - set(common_source_in_cluster)

                for not_cls_src in connected_source_nodes_not_in_cls:
                    if not_cls_src not in total_connected_source_nodes_not_in_cls:
                        total_connected_source_nodes_not_in_cls.append(not_cls_src)

                # total_connected_source_nodes_not_in_cls.extend(list(connected_source_nodes_not_in_cls))
                total_edges_in_cluster_as_target += len(common_source_in_cluster)
                total_edges_in_graph_with_target += len(target_neighbor_list)
                '''target_suffix = target[1:]
                source_string = "S" + target_suffix
                if source_list.__contains__(source_string):
                    target_source_pair.append(source_string + "-" + target)'''
                for src in common_source_in_cluster:
                    target_edges.append(src + '-' + target)

        fig = plt.figure(figsize=(10, 8))
        plt.bar(list(cluster_degree.keys()), list(cluster_degree.values()), align='center')
        plt.xticks(list(cluster_degree.keys()), list(cluster_degree.keys()), rotation=90, ha='right')

        for k in range(len(list(cluster_degree.keys()))):
            plt.annotate(list(cluster_degree.values())[k], (list(cluster_degree.keys())[k], list(cluster_degree.values())[k]), ha='center', va='bottom')

        # plt.xlim(min(list(unique_degree_dict.keys())) - 0.5, max(list(unique_degree_dict.keys())) + 0.5)
        # Set labels and title
        plt.xlabel('Degree Level')
        plt.ylabel('No of Nodes in Degree')
        plt.title('Cluster Degree Distribution of Current Nodes in Cluster' + str(i + 1))
        plt.savefig("Cluster Degree_Distribution of Current Nodes in Cluster" + str(i + 1) + ".png")
        # Display the plot
        # plt.show()

        for source in source_list_without_zero_degree:
            source_suffix = source[1:]
            target_string = "T" + source_suffix
            if target_list_without_zero_degree.__contains__(target_string):
                source_target_pair.append(source + "-" + target_string)
            '''for trg in common_targets_in_cluster:
                source_edges.append(source + '-' + trg)'''

        for target in target_list_without_zero_degree:
            target_suffix = target[1:]
            source_string = "S" + target_suffix
            if source_list_without_zero_degree.__contains__(source_string):
                target_source_pair.append(source_string + "-" + target)
            '''for src in common_source_in_cluster:
                target_edges.append(src + '-' + target)'''

        cluster_degree_row = []
        cluster_degree_row.append(i + 1)
        cluster_degree_row.append(cluster_dict[i])
        cluster_degree_row.append(node_edges_count)
        cluster_degree_row.append(cluster_degree)
        total_cluster_degree_row.append(cluster_degree_row)
        cluster_degree_cols = ["Cluster No", "Node List", "Node Edge Count", "Cluster Degree Distribution"]

        removed_zero_degree_nodes = set(cluster_dict[i]) - set(zero_degree_nodes)
        removed_singleton_nodes = set(removed_zero_degree_nodes) - set(singleton_nodes)
        row.append(len(removed_singleton_nodes))
        # row.append(source_no)
        # row.append(target_no)
        row.append(len(source_list_without_zero_degree))
        row.append(len(zero_src_nodes))
        row.append(len(source_list_without_zero_degree) - len(zero_src_nodes))
        row.append(len(target_list_without_zero_degree))
        row.append(len(zero_trg_nodes))
        row.append(len(target_list_without_zero_degree) - len(zero_trg_nodes))
        row.append(len(zero_degree_nodes))
        row.append(total_edges_in_cluster_as_source)
        row.append(total_edges_in_cluster_as_target)
        row.append(len(source_list_without_zero_degree) * len(target_list_without_zero_degree))
        row.append(len(only_know_each_other_as_source))
        row.append(len(only_know_each_other_as_target))
        if len(source_list_without_zero_degree) != 0 and len(target_list_without_zero_degree) != 0:
            row.append(total_edges_in_cluster_as_source / (
                    len(source_list_without_zero_degree) * len(target_list_without_zero_degree)))
        else:
            row.append(0)
        row.append(total_edges_in_cluster_as_source / (source_no + target_no))
        if source_no > target_no:
            row.append(total_edges_in_cluster_as_source / target_no)
        elif source_no < target_no:
            row.append(total_edges_in_cluster_as_source / source_no)
        row.append(src_high_degree)
        row.append(trg_high_degree)
        row.append(highest_src_cls_degree)
        row.append(highest_trg_cls_degree)
        row.append(len(source_target_pair))
        row.append(len(target_source_pair))
        row.append(source_target_pair)
        row.append(target_source_pair)
        # row.append(total_edges_in_graph_with_source)
        # row.append(total_edges_in_graph_with_target)
        # TODO: Consider List Of Nodes after removing zero degree nodes and singleton nodes or Original Node List
        row.append(sorted(cluster_dict[i], key=key_func))
        row.append(sorted(zero_degree_nodes, key=key_func))
        row.append(only_know_each_other_as_source)
        row.append(only_know_each_other_as_target)
        row.append(set(singleton_nodes))
        # row.append(sorted(cluster_dict[i], key=key_func))
        row.append(total_connected_target_nodes_not_in_cls)
        row.append(len(total_connected_target_nodes_not_in_cls))
        row.append(total_connected_source_nodes_not_in_cls)
        row.append(len(total_connected_source_nodes_not_in_cls))
        rows.append(row)

        nodes_not_in_cls_row = []
        nodes_not_in_cls_row.append(i)
        nodes_not_in_cls_row.append(total_connected_source_nodes_not_in_cls)
        nodes_not_in_cls_row.append(total_connected_target_nodes_not_in_cls)
        nodes_not_in_cls.append(nodes_not_in_cls_row)

        edge_row.append(i + 1)
        edge_row.append(source_edges)
        edge_row.append(target_edges)
        edges_rows.append(edge_row)

    nodes_not_in_cls_col = ['Cluster No', 'Source_SDM Nodes Not In Cluster', 'Target Nodes Not In Cluster']
    nodes_not_in_cls_dict = pd.DataFrame(nodes_not_in_cls, columns=nodes_not_in_cls_col)
    save_as_pickle(nodes_not_in_cls_pkl, nodes_not_in_cls_dict)
    zero_degree_cols = ['Cluster No', 'Node', 'Zero Degree in Cluster', 'Degree Ouside Cluster']
    # pd.DataFrame(zero_degree_rows, columns=zero_degree_cols).to_csv(bitcoin_zero_degree_node_analysis)
    pd.DataFrame(zero_degree_rows, columns=zero_degree_cols).to_csv(citeseer_zero_degree_node_analysis)
    result = pd.DataFrame(rows, columns=cols)
    # pd.DataFrame(result).to_csv(citation_spectral_clustering_csv, encoding='utf-8', float_format='%f')
    pd.DataFrame(total_degree_row, columns=degree_cols).to_csv("degreeList_out_off_cluster_nodes.csv", encoding='utf-8',
                                                               float_format='%f')
    pd.DataFrame(total_cluster_degree_row, columns=cluster_degree_cols).to_csv(
        "cluster_degree_out_off_cluster_nodes.csv", encoding='utf-8', float_format='%f')
    edge_cols = ['Cluster No', 'S-T edges', 'T-S edges']
    edge_rows_df = pd.DataFrame(edges_rows, columns=edge_cols)
    pd.DataFrame(edge_rows_df).to_csv(edge_file_name, encoding='utf-8', float_format='%f')

    return clusters, cluster_dict

def hit_score_original_graph(G, hub_csv, authority_csv, original_hit_authority_score_csv, hub_plot, authority_plot, hub_title, authority_title):
    # print("Original Graph Hits score:"+ str(nx.hits(original_bitcoin_graph)))
    hubs, authorities = nx.hits(G)
    # Sort the Hub and Authority scores by value in descending order
    sorted_hubs_items = sorted(hubs.items(), key=lambda item: item[1], reverse=True)
    sorted_hubs = {str(key): value for key, value in sorted_hubs_items}
    sorted_authorities_items = sorted(authorities.items(), key=lambda item: item[1], reverse=True)
    sorted_authorities = {str(key): value for key, value in sorted_authorities_items}

    figure12 = plt.figure(figsize=(15, 15))
    ax12 = figure12.add_subplot(111)
    ax12.plot(list(sorted_hubs.keys()), list(sorted_hubs.values()))
    plt.legend()
    plt.title(hub_title)
    plt.savefig(hub_plot)
    #plt.show()

    figure13 = plt.figure(figsize=(15, 15))
    ax13 = figure13.add_subplot(111)
    ax13.plot(list(sorted_authorities.keys()), list(sorted_authorities.values()))
    plt.legend()
    plt.title(authority_title)
    plt.savefig(authority_plot)
    #plt.show()

    first_key_hub = list(sorted_hubs.keys())[0]
    first_value_hub = sorted_hubs[first_key_hub]

    last_key_hub = list(sorted_hubs.keys())[len(list(sorted_hubs.keys())) - 1]
    last_value_hub = sorted_hubs[last_key_hub]

    first_key_authority = list(sorted_authorities.keys())[0]
    first_value_authority = sorted_authorities[first_key_authority]

    last_key_authority = list(sorted_authorities.keys())[len(list(sorted_authorities.keys())) - 1]
    last_value_authority = sorted_authorities[last_key_authority]

    print("Bipartite Graph hubs Score:" + str(sorted_hubs))
    print("Highest Hub Score of Bipartite Graph " + str(first_key_hub) + ":" + str(first_value_hub))
    print("Lowest Hub Score of Bipartite Graph " + str(last_key_hub) + ":" + str(last_value_hub))
    print("Bipartite Graph authorities Score:" + str(sorted_authorities))
    print("Highest Authorities Score of Bipartite Graph " + str(first_key_authority) + ":" + str(first_value_authority))
    print("Lowest Authorities Score of Bipartite Graph " + str(last_key_authority) + ":" + str(last_value_authority))

    pd.DataFrame({'Nodes': list(sorted_hubs.keys()), 'Hub Scores' : list(sorted_hubs.values())}).to_csv(hub_csv)
    pd.DataFrame({'Nodes': list(sorted_authorities.keys()), 'Authority Scores': list(sorted_authorities.values())}).to_csv(authority_csv)

    hit_authority_score_info_row = []
    hit_authority_score_info_rows = []
    hit_authority_score_info_row.append("Highest Hub Score")
    hit_authority_score_info_rows.append(hit_authority_score_info_row)
    hit_authority_score_info_row = []
    hit_authority_score_info_row.append(str(first_key_hub) + ":" + str(first_value_hub))
    hit_authority_score_info_rows.append(hit_authority_score_info_row)
    hit_authority_score_info_row = []
    hit_authority_score_info_row.append("Lowest Hub Score")
    hit_authority_score_info_rows.append(hit_authority_score_info_row)
    hit_authority_score_info_row = []
    hit_authority_score_info_row.append(str(last_key_hub) + ":" + str(last_value_hub))
    hit_authority_score_info_rows.append(hit_authority_score_info_row)
    hit_authority_score_info_row = []
    hit_authority_score_info_row.append("Highest Authority Score")
    hit_authority_score_info_rows.append(hit_authority_score_info_row)
    hit_authority_score_info_row = []
    hit_authority_score_info_row.append(str(first_key_authority) + ":" + str(first_value_authority))
    hit_authority_score_info_rows.append(hit_authority_score_info_row)
    hit_authority_score_info_row = []
    hit_authority_score_info_row.append("Lowest Authority Score")
    hit_authority_score_info_rows.append(hit_authority_score_info_row)
    hit_authority_score_info_row = []
    hit_authority_score_info_row.append(str(last_key_authority) + ":" + str(last_value_authority))
    hit_authority_score_info_rows.append(hit_authority_score_info_row)

    pd.DataFrame(hit_authority_score_info_rows).to_csv(original_hit_authority_score_csv)

    return sorted_hubs, sorted_authorities

def spectral_dbscan_clustering_bitcoin_target(middleDimentions, no_of_clusters, citation_spectral_clustering_csv, title,
                                              spectral_cluster_plot_file_name, bipartite_G, edge_file_name,
                                              nodes_not_in_cls_pkl, original_citeseer_graph):
    sorted_hubs, sorted_authorities = hit_score_original_graph(original_citeseer_graph, citeseer_original_hub_csv,
                                                               citeseer_original_authority_csv,
                                                               citeseer_original_hub_plot,
                                                               citeseer_original_authority_plot,
                                                               citeseer_original_hub_title,
                                                               citeseer_original_authority_title)

    nodes_not_in_cls = []

    if exists(citeseer_clusters_list_pkl):
        clusters = read_pickel(citeseer_clusters_list_pkl)
    else:
        spectral = SpectralClustering(n_clusters=no_of_clusters, affinity='nearest_neighbors', n_neighbors=50, random_state=0)
        clusters = spectral.fit_predict(middleDimentions)
        save_as_pickle(citeseer_clusters_list_pkl, clusters)

    if exists(citeseer_cluster_nodes) and exists(citeseer_clusterwise_embedding_src_nodes) and exists(citeseer_cust_clusterwise_embedding_nodes) and exists(citeseer_cluster_dict):
        cluster_nodes = read_pickel(citeseer_cluster_nodes)
        clusterwise_embedding_src_nodes = read_pickel(citeseer_clusterwise_embedding_src_nodes)
        cust_clusterwise_embedding_nodes = read_pickel(citeseer_cust_clusterwise_embedding_nodes)
        cluster_dict = read_pickel(citeseer_cluster_dict)
    else:
        cluster_dict = {}
        cluster_nodes = {}
        clusterwise_embedding_src_nodes = {}
        cust_clusterwise_embedding_nodes = {}
        for i in range(no_of_clusters):
            cluster_dict[i] = []
            cluster_nodes[i] = {}
            cluster_nodes[i]['nodes'] = []
            cluster_nodes[i]['label'] = []
            cluster_nodes[i]['name'] = []
            cluster_nodes[i]['status'] = []
            cluster_nodes[i]['cluster_no'] = []
            cluster_nodes[i]['index'] = []
            clusterwise_embedding_src_nodes[i] = {}
            clusterwise_embedding_src_nodes[i]['name'] = []
            clusterwise_embedding_src_nodes[i]['label'] = []
            cust_clusterwise_embedding_nodes[i] = {}
            cust_clusterwise_embedding_nodes[i]['name'] = []
            cust_clusterwise_embedding_nodes[i]['label'] = []

        for i in range(len(clusters)):
            cluster_dict[clusters[i]].append(list(middleDimentions.index)[i])
            cluster_nodes[clusters[i]]['nodes'].append(list(middleDimentions.iloc[i]))
            cluster_nodes[clusters[i]]['label'].append('unlabeled')
            cluster_nodes[clusters[i]]['name'].append(list(middleDimentions.index)[i])
            cluster_nodes[clusters[i]]['status'].append('unvisited')
            cluster_nodes[clusters[i]]['cluster_no'].append(-1)
            cluster_nodes[clusters[i]]['index'].append(list(middleDimentions.index).index(list(middleDimentions.index)[i]))

        save_as_pickle(citeseer_cluster_nodes, cluster_nodes)
        save_as_pickle(citeseer_clusterwise_embedding_src_nodes, clusterwise_embedding_src_nodes)
        save_as_pickle(citeseer_cust_clusterwise_embedding_nodes, cust_clusterwise_embedding_nodes)
        save_as_pickle(citeseer_cluster_dict, cluster_dict)

    if exists(citeseer_dbscan_cluster_nodes_labels) and exists(citeseer_embedding_map) and exists(citeseer_source_nodes_map) and exists(citeseer_avg_distance_map):
        cluster_labels_map = read_pickel(citeseer_dbscan_cluster_nodes_labels)
        embedding_map = read_pickel(citeseer_embedding_map)
        source_nodes_map = read_pickel(citeseer_source_nodes_map)
        avg_distance_map = read_pickel(citeseer_avg_distance_map)
    else:
        cluster_labels_map = {}
        avg_distances_rows = []
        embedding_map = {}
        source_nodes_map = {}
        avg_distance_map = {}
        for u in range(len(cluster_nodes)):
            source_nodes = {}
            source_nodes['nodes'] = []
            source_nodes['name'] = []
            source_nodes['label'] = []
            source_nodes['status'] = []
            source_nodes['cluster_no'] = []
            source_nodes['index'] = []
            source_nodes['embedding'] = []
            for j in range(len(cluster_nodes[u]['nodes'])):
                if cluster_nodes[u]['name'][j].startswith("T"):
                    source_nodes['nodes'].append(cluster_nodes[u]['nodes'][j])
                    source_nodes['name'].append(cluster_nodes[u]['name'][j])
                    source_nodes['label'].append(cluster_nodes[u]['label'][j])
                    source_nodes['status'].append(cluster_nodes[u]['status'][j])
                    source_nodes['cluster_no'].append(cluster_nodes[u]['cluster_no'][j])
                    source_nodes['index'].append(cluster_nodes[u]['index'][j])
                    source_nodes['embedding'].append(list(middleDimentions.iloc[cluster_nodes[u]['index'][j]]))
            source_nodes_map[u] = source_nodes
            avg, sorted_names, sorted_values = average_distance(source_nodes)
            avg_distance_map[u] = avg
            avg_distances_row = []
            avg_distances_row.append(u + 1)
            avg_distances_row.append(sorted_names)
            avg_distances_row.append(sorted_values)
            avg_distances_rows.append(avg_distances_row)

            dbscan = DBSCAN(eps=avg, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(source_nodes['nodes'])
            cluster_labels_map[u] = cluster_labels

            tsne = TSNE(n_components=2, perplexity=5)
            embeddings_2d = tsne.fit_transform(np.array(source_nodes['nodes']))
            embedding_map[u] = embeddings_2d

        save_as_pickle(citeseer_dbscan_cluster_nodes_labels, cluster_labels_map)
        save_as_pickle(citeseer_embedding_map, embedding_map)
        save_as_pickle(citeseer_source_nodes_map, source_nodes_map)
        save_as_pickle(citeseer_avg_distance_map, avg_distance_map)
        avg_distance_cols = ["Cluster No", "Source_SDM Names", "5th Nearest Neighbor Distance"]
        pd.DataFrame(avg_distances_rows, columns=avg_distance_cols).to_csv(citeseer_5th_nearest_neighbor_distance_csv)

    rows = []
    dbscan_rows = []
    for i in range(len(cluster_nodes)):
        cluster_labels = cluster_labels_map[i]
        citeseer_directed_A = nx.adjacency_matrix(original_citeseer_graph)
        bitcoin_directed_A = pd.DataFrame(citeseer_directed_A.toarray(), columns=original_citeseer_graph.nodes(), index=original_citeseer_graph.nodes())

        embeddings_2d = embedding_map[i]
        print("embedding len" + str(len(embeddings_2d)))
        print("cluster label len" + str(len(np.array(cluster_labels))))
        print(cluster_labels)
        '''x_values = [embeddings_2d[j, 0] for j in range(len(embeddings_2d[:, 0])) if np.array(cluster_labels)[j] != -1]
        y_values = [embeddings_2d[j, 1] for j in range(len(embeddings_2d[:, 0])) if np.array(cluster_labels)[j] != -1]       
        source_lables = [int(lab) for lab in np.array(cluster_labels) if lab != -1]      
        source_names = [source_nodes['name'][j] for j in range(len(source_nodes['name'])) if cluster_labels[j] != -1]
        src_cls_embedding = [source_nodes['embedding'][j] for j in range(len(source_nodes['name'])) if cluster_labels[j] != -1] '''
        x_values = []
        y_values = []
        source_lables = []
        source_names = []
        src_cls_embedding = []
        for b in range(len(np.array(cluster_labels))):
            if cluster_labels[b] != -1:
                x_values.append(embeddings_2d[b, 0])
                y_values.append(embeddings_2d[b, 1])
                source_lables.append(int(cluster_labels[b]))
                source_nodes = source_nodes_map[i]
                source_names.append(source_nodes['name'][b])
                src_cls_embedding.append(source_nodes['embedding'][b])

        figure5 = plt.figure(figsize=(15, 15))
        ax5 = figure5.add_subplot(111)
        ax5.scatter(x_values, y_values, c=source_lables)
        plt.legend()
        plt.title("DBSCAN Inside Spectral Clustering (Cluster No: " + str(i + 1) + ") ( No of Clusters " + str(
            len(set(source_lables))) + ")")
        plt.savefig(citeseer_spectral_dbscan_plot_name + "_" + str(i + 1) + ".png")
        # plt.show()

        clusterwise_embedding_src_nodes[i]['name'] = source_names
        clusterwise_embedding_src_nodes[i]['label'] = source_lables

        dbscan_cluster_info = {}
        for lab in list(set(source_lables)):
            dbscan_cluster_info[lab] = {}
            dbscan_cluster_info[lab]['x_value'] = []
            dbscan_cluster_info[lab]['y_value'] = []
            dbscan_cluster_info[lab]['names'] = []
            dbscan_cluster_info[lab]['embedding'] = []
            dbscan_cluster_info[lab]['status'] = []
            dbscan_cluster_info[lab]['label'] = []

        for lab in set(source_lables):
            dbscan_cluster_info[lab]['x_value'] = [x_values[k] for k in range(len(source_lables)) if source_lables[k] == lab]
            dbscan_cluster_info[lab]['y_value'] = [y_values[k] for k in range(len(source_lables)) if source_lables[k] == lab]
            dbscan_cluster_info[lab]['names'] = [source_names[k] for k in range(len(source_names)) if source_lables[k] == lab]
            dbscan_cluster_info[lab]['embedding'] = [src_cls_embedding[k] for k in range(len(source_lables)) if source_lables[k] == lab]
            dbscan_cluster_info[lab]['label'] = [lab for k in range(len(source_lables)) if source_lables[k] == lab]

        for k in range(len(dbscan_cluster_info)):
            row = []
            row.append(i + 1)
            row.append(k + 1)
            row.append(len(dbscan_cluster_info[k]['names']))
            row.append(min_samples)
            row.append(avg_distance_map[i])
            row.append(dbscan_cluster_info[k]['names'])
            row.append(dbscan_cluster_info[k]['embedding'])
            row.append(dbscan_cluster_info[k]['x_value'])
            row.append(dbscan_cluster_info[k]['y_value'])
            rows.append(row)

        cust_dbscan_cluster_nodes = {}
        for k in range(len(dbscan_cluster_info)):
            cust_dbscan_cluster_nodes[k] = {}
            cust_dbscan_cluster_nodes[k]['nodes'] = []
            cust_dbscan_cluster_nodes[k]['original_embedding'] = []
            cust_dbscan_cluster_nodes[k]['src_x_embedding'] = []
            cust_dbscan_cluster_nodes[k]['src_y_embedding'] = []
            cust_dbscan_cluster_nodes[k]['status'] = []
            cust_dbscan_cluster_nodes[k]['label'] = []
            cust_dbscan_cluster_nodes[k]['no_src'] = 0
            cust_dbscan_cluster_nodes[k]['no_trg'] = 0
            cust_dbscan_cluster_nodes[k]['no_pair'] = 0
            cust_dbscan_cluster_nodes[k]['avg_degree'] = 0

        for k in range(len(dbscan_cluster_info)):
            cust_dbscan_cluster_nodes[k]['nodes'] = copy.deepcopy(dbscan_cluster_info[k]['names'])
            cust_dbscan_cluster_nodes[k]['original_embedding'] = copy.deepcopy(dbscan_cluster_info[k]['embedding'])
            cust_dbscan_cluster_nodes[k]['src_x_embedding'] = copy.deepcopy(dbscan_cluster_info[k]['x_value'])
            cust_dbscan_cluster_nodes[k]['src_y_embedding'] = copy.deepcopy(dbscan_cluster_info[k]['y_value'])
            cust_dbscan_cluster_nodes[k]['status'] = ["unvisited" for k in range(len(dbscan_cluster_info[k]['names']))]
            cust_dbscan_cluster_nodes[k]['label'] = copy.deepcopy(dbscan_cluster_info[k]['label'])

        converged_cnt = 0
        for p in range(len(cust_dbscan_cluster_nodes)):
            dbscan_names = set(cust_dbscan_cluster_nodes[p]['nodes'])
            while len(dbscan_names) > 0:
                last_element = dbscan_names.pop()
                last_element_index = cust_dbscan_cluster_nodes[p]['nodes'].index(last_element)
                last_element_status = cust_dbscan_cluster_nodes[p]['status'][last_element_index]
                if last_element_status == "unvisited":
                    cust_dbscan_cluster_nodes[p]['status'][last_element_index] = "visited"
                    if last_element.startswith("S"):
                        new_src = int(last_element[1:])
                        src_trg = bitcoin_directed_A.loc[new_src]
                        indices = [i for i, value in enumerate(src_trg) if value == 1]
                        dataframe_indices = list(src_trg.index)
                        trg_list = [dataframe_indices[trg_index] for trg_index in indices]
                        for trg in trg_list:
                            # if list(cust_dbscan_cluster_nodes[p]['nodes']).__contains__("T"+str(trg)) == False and list(cluster_nodes[i]['name']).__contains__("T"+str(trg)) == True:
                            if list(cust_dbscan_cluster_nodes[p]['nodes']).__contains__("T" + str(trg)) == False:
                                src_trg_neighbors = bitcoin_directed_A[trg]
                                indices = [i for i, value in enumerate(src_trg_neighbors) if value == 1]
                                dataframe_indices = list(src_trg_neighbors.index)
                                src_list = [dataframe_indices[src_index] for src_index in indices]
                                src_trg_neighbors = set(src_list)
                                cls_src_list = [int(src[1:]) for src in dbscan_names if src.startswith("S")]
                                cls_src_list = set(cls_src_list)
                                common_src = list(cls_src_list.intersection(src_trg_neighbors))
                                if len(common_src) >= closer_3:
                                    included_trg = "T" + str(trg)
                                    cust_dbscan_cluster_nodes[p]['nodes'].append(included_trg)
                                    cust_dbscan_cluster_nodes[p]['status'].append("unvisited")
                                    cust_dbscan_cluster_nodes[p]['original_embedding'].append(
                                        list(middleDimentions.loc[included_trg]))
                                    cust_dbscan_cluster_nodes[p]['label'].append(
                                        list(set(cust_dbscan_cluster_nodes[p]['label']))[0])
                                    dbscan_names.update([included_trg])
                    elif last_element.startswith("T"):
                        new_trg = int(last_element[1:])
                        trg_src = bitcoin_directed_A[new_trg]
                        indices = [i for i, value in enumerate(trg_src) if value == 1]
                        dataframe_indices = list(trg_src.index)
                        src_list = [dataframe_indices[src_index] for src_index in indices]
                        for src in src_list:
                            # if list(cust_dbscan_cluster_nodes[p]['nodes']).__contains__("S"+str(src)) == False and list(cluster_nodes[i]['name']).__contains__("S"+str(src)) == True:
                            if list(cust_dbscan_cluster_nodes[p]['nodes']).__contains__("S" + str(src)) == False:
                                trg_src_neighbors = bitcoin_directed_A.loc[src]
                                indices = [i for i, value in enumerate(trg_src_neighbors) if value == 1]
                                dataframe_indices = list(trg_src_neighbors.index)
                                trg_list = [dataframe_indices[trg_index] for trg_index in indices]
                                trg_src_neighbors = set(trg_list)
                                cls_trg_list = [int(src[1:]) for src in dbscan_names if src.startswith("T")]
                                cls_trg_list = set(cls_trg_list)
                                common_trg = list(cls_trg_list.intersection(trg_src_neighbors))
                                if len(common_trg) >= closer_3:
                                    included_src = "S" + str(src)
                                    cust_dbscan_cluster_nodes[p]['nodes'].append(included_src)
                                    cust_dbscan_cluster_nodes[p]['status'].append("unvisited")
                                    cust_dbscan_cluster_nodes[p]['original_embedding'].append(
                                        list(middleDimentions.loc[included_src]))
                                    cust_dbscan_cluster_nodes[p]['label'].append(
                                        list(set(cust_dbscan_cluster_nodes[p]['label']))[0])
                                    dbscan_names.update([included_src])

        cust_nodes = []
        cust_labels = []
        for h in range(len(cust_dbscan_cluster_nodes)):
            cust_nodes.extend(list(cust_dbscan_cluster_nodes[h]['nodes']))
            cust_labels.extend(list(cust_dbscan_cluster_nodes[h]['label']))
        cust_clusterwise_embedding_nodes[i]['name'] = cust_nodes
        cust_clusterwise_embedding_nodes[i]['label'] = cust_labels
        '''tsne = TSNE(n_components=2, perplexity=25)
        embeddings_2d = tsne.fit_transform(np.array(middleDimentions))
        x_min = min(embeddings_2d[:, 0])
        x_max = max(embeddings_2d[:, 0])
        y_min = min(embeddings_2d[:, 1])
        y_max = max(embeddings_2d[:, 1])
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        figure10 = plt.figure(figsize=(15, 15))
        ax10 = figure10.add_subplot(111)
        embeddings_2d = pd.DataFrame(embeddings_2d, index=middleDimentions.index)
        cust_x = []
        cust_y = []
        cust_label = []
        for k in range(len(cust_dbscan_cluster_nodes)):
            dbscan_cls_names = list(cust_dbscan_cluster_nodes[k]['nodes'])
            dbscan_labels = list(cust_dbscan_cluster_nodes[k]['label'])
            for p in range(len(dbscan_cls_names)):
                cust_x.append(embeddings_2d.loc[dbscan_cls_names[p]][0])
                cust_y.append(embeddings_2d.loc[dbscan_cls_names[p]][1])
                cust_label.append(dbscan_labels[p])
        ax10.scatter(cust_x,cust_y, c=cust_label, cmap="jet")
        plt.title(bitcoin_customized_dbscan_clusters + str(i+1) +"( No of Clusters " + str(len(cust_dbscan_cluster_nodes)) + ")")
        plt.savefig(bitcoin_customized_dbscan_clusters_plot + str(i+1))
        plt.legend()
        plt.show()'''

        for k in range(len(cust_dbscan_cluster_nodes)):
            hub_bar_list = {}
            authority_bar_list = {}
            line_x_values = {}
            for bar, bar_value in zip(list(sorted_hubs.keys()), list(sorted_hubs.values())):
                hub_bar_list[bar]= 0
                authority_bar_list[bar] = 0
                line_x_values[bar] = ""
            row = []
            row.append(i + 1)
            row.append(k + 1)
            row.append(len(cust_dbscan_cluster_nodes[k]['nodes']))
            row.append(len(dbscan_cluster_info[k]['names']))
            sr_no = 0
            tr_no = 0
            pair_nodes = set()
            degree_cnt = 0
            edges = set()
            cls_nodes = set(cust_dbscan_cluster_nodes[k]['nodes'])
            hits_score_list = {}
            authority_score_list = {}
            percentile_score_list = {}
            for nd in cust_dbscan_cluster_nodes[k]['nodes']:
                if nd.startswith("S"):
                    sr_no += 1
                    node_no = nd[1:]
                    tr_node = "T" + str(node_no)
                    if list(cust_dbscan_cluster_nodes[k]['nodes']).__contains__(tr_node):
                        pair_nodes.add(node_no)
                elif nd.startswith("T"):
                    tr_no += 1
                    node_no = nd[1:]
                    sr_node = "S" + str(node_no)
                    if list(cust_dbscan_cluster_nodes[k]['nodes']).__contains__(sr_node):
                        pair_nodes.add(node_no)
                edge_list = list(bipartite_G.neighbors(nd))
                common_neighbors = set(edge_list).intersection(cls_nodes)
                hits_score_list[nd[1:]] = sorted_hubs[nd[1:]] * 1e-4
                hub_bar_list[nd[1:]] = max(sorted_hubs.values())
                authority_bar_list[nd[1:]] = max(sorted_authorities.values())
                line_x_values[nd[1:]] = nd[1:]
                authority_score_list[nd[1:]] = sorted_authorities[nd[1:]] * 1e-4
                #percentile = np.percentile(list(sorted_hubs.values()),(list(sorted_hubs.keys()).index(nd) / len(list(sorted_hubs.values()))) * 100)
                #percentile_score_list[nd] = percentile
                for cm_ed in common_neighbors:
                    if nd.startswith("S"):
                        edge = str(nd) + "-" + str(cm_ed)
                    elif nd.startswith("T"):
                        edge = str(cm_ed) + "-" + str(nd)
                    edges.add(edge)

            row.append(sr_no)
            row.append(tr_no)
            row.append(len(edges))
            row.append(len(pair_nodes))
            row.append(cust_dbscan_cluster_nodes[k]['nodes'])
            row.append(edges)
            bipartite_G_copy = bipartite_G.copy()
            #subgraph = bipartite_G_copy.subgraph(cust_dbscan_cluster_nodes[k]['nodes'])
            '''bipartite_G_copy.freeze()
            for edge in edges:
                subgraph.add_edge(*edge)'''
            #sub_graph_hubs, sub_graph_authorities = nx.hits(subgraph)
            # Sort the Hub and Authority scores by value in descending order
            #sub_graph_sorted_hubs = dict(sorted(sub_graph_hubs.items(), key=lambda item: item[1], reverse=True))
            #sub_graph_sorted_authorities = dict(sorted(sub_graph_authorities.items(), key=lambda item: item[1], reverse=True))
            #row.append(cust_dbscan_cluster_nodes[k]['status'])
            #row.append(cust_dbscan_cluster_nodes[k]['original_embedding'])
            #row.append(cust_dbscan_cluster_nodes[k]['label'])
            row.append(hits_score_list)
            row.append(authority_score_list)
            #row.append(percentile_score_list)
            #row.append(sub_graph_sorted_hubs)
            #row.append(sub_graph_sorted_authorities)

            y_min = min(list(sorted_hubs.values()))
            y_max = max(list(sorted_hubs.values()))
            figure_cluster_hub = plt.figure(figsize=(15, 15))
            ax_cluster_hub = figure_cluster_hub.add_subplot(111)
            ax_cluster_hub.set_ylim(y_min, y_max)
            ax_cluster_hub.plot(list(sorted_hubs.keys()), list(sorted_hubs.values()), label="Bipartite Graph Hub Score")
            ax_cluster_hub.bar(list(sorted_hubs.keys()), list(hub_bar_list.values()), color='red', label="Cluster Hub Score Location")
            plt.legend()
            plt.title("Node Hub Score Position For Spectral Cluster No:" + str(i+1) + "DBSCAN Cluster No:" + str(k+1) + " Highest Hub Score in Cluster:"+str(max(list(sorted_hubs.values()))) + " Lowest Hub Score in Cluster:" +str(min(list(sorted_hubs.values())))
                      +" Highest Authority Score in Cluster:"+str(max(list(authority_score_list.values()))) + " Lowest Authority Score in Cluster:"+str(min(list(authority_score_list.values()))))
            #plt.show()
            plt.savefig(citeseer_speparate_cluster_path + "node_hub_score_position_spectral_cls"+str(i+1) + "dbscan_cls_" + str(k+1) +".png")

            y_min = min(list(sorted_authorities.values()))
            y_max = max(list(sorted_authorities.values()))
            figure_cluster_authority = plt.figure(figsize=(15, 15))
            ax_cluster_authority = figure_cluster_authority.add_subplot(111)
            ax_cluster_authority.set_ylim(y_min, y_max)
            ax_cluster_authority.plot(list(sorted_authorities.keys()), list(sorted_authorities.values()), label="Bipartite Graph Authority Score")
            ax_cluster_authority.bar(list(sorted_authorities.keys()), list(authority_bar_list.values()), color='red', label="Cluster Authority Score Location")
            plt.legend()
            plt.title("Node Authority Score Position For Spectral Cluster No:" + str(i + 1) + "  DBSCAN Cluster No:" + str(k + 1))
            # plt.show()
            plt.savefig(citeseer_speparate_cluster_path + "node_authority_score_position_spectral_cls" + str(i + 1) + "dbscan_cls_" + str(k + 1) + ".png")
            dbscan_rows.append(row)

    # target clustering
    dbscan_cols = ['Spectral Cluster No', 'DBSCAN Cluster No', 'No Of Target Nodes in Cluster', 'Min Sample', 'Eps','List Of Nodes',
                   'Node2vec 20-D Embedding','X Value(TSNE)', 'Y Value(TSNE)']
    pd.DataFrame(rows, columns=dbscan_cols).to_csv(citeseer_dbscan_cluster_csv_file)

    # target columns
    '''cust_dbscan_cols = ["Spectral Cluster No", "DBSCAN Cluster No", "No Of Nodes in Cluster", "Original No Of Targets","No Of Source_SDM Nodes","No Of Target Nodes",  "No Of Edges","No Of Pairs", "List Of Nodes",
                        "List Of Edges", "Status Of Nodes", "Original Embedding", "Labels",
                        "Original Bipartite Graph Hub Score", "Original Bipartite Graph Authority Score","Subgraph Hub Score","Subgraph Authority Score",  "Percentile"]'''
    cust_dbscan_cols = ["Spectral Cluster No", "DBSCAN Cluster No", "No Of Nodes in Cluster", "Original No Of Targets",
                        "No Of Source_SDM Nodes", "No Of Target Nodes", "No Of Edges", "No Of Pairs", "List Of Nodes",
                        "List Of Edges", "Original Bipartite Graph Hub Score", "Original Bipartite Graph Authority Score"]
    pd.DataFrame(dbscan_rows, columns=cust_dbscan_cols).to_csv(citeseer_cust_dbscan_cluster_csv_file)


    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 0),  # Maroon
        (0, 128, 0),  # Green
        (0, 0, 128),  # Navy
        (128, 128, 0),  # Olive
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
        (255, 165, 0),  # Orange
        (128, 128, 128),  # Gray
        (255, 192, 203),  # Pink
        (0, 128, 0),  # Green
        (128, 0, 0),  # Maroon
        (0, 0, 128),  # Navy
        (0, 128, 128),  # Teal
        (128, 0, 128),  # Purple
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
    ]

    normalized_colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]

    selected_colors = [normalized_colors[i] for i in clusters]
    tsne = TSNE(n_components=2, perplexity=25)
    embeddings_2d = tsne.fit_transform(np.array(middleDimentions))
    x_min = min(embeddings_2d[:, 0])
    x_max = max(embeddings_2d[:, 0])
    y_min = min(embeddings_2d[:, 1])
    y_max = max(embeddings_2d[:, 1])
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    figure = plt.figure(figsize=(15, 15))
    ax = figure.add_subplot(111)
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=selected_colors, cmap='jet')
    plt.title(title + "( No of Clusters " + str(len(set(clusters))) + ")")
    plt.savefig(spectral_cluster_plot_file_name)
    plt.legend()
    # plt.show()

    embeddings_2d = pd.DataFrame(embeddings_2d, index=middleDimentions.index)

    '''Original Embedding'''
    k = 0
    for cls_nodes in cluster_dict.values():
        source_x_embedding = []
        source_y_embedding = []
        target_x_embedding = []
        target_y_embedding = []
        source_labels = []
        target_labels = []
        source_colors = []
        target_colors = []
        figure1 = plt.figure(figsize=(15, 15))
        ax1 = figure1.add_subplot(111)
        color = [k for j in range(len(cls_nodes))]

        for cls_node in cls_nodes:
            scatter = ax1.scatter(embeddings_2d.loc[cls_node][0], embeddings_2d.loc[cls_node][1], c=normalized_colors[k], cmap="jet")
            if cls_node.startswith("T"):
                source_x_embedding.append(embeddings_2d.loc[cls_node][0])
                source_y_embedding.append(embeddings_2d.loc[cls_node][1])
                source_labels.append(cls_node)
                source_colors.append("blue")
            elif cls_node.startswith("T"):
                target_x_embedding.append(embeddings_2d.loc[cls_node][0])
                target_y_embedding.append(embeddings_2d.loc[cls_node][1])
                target_labels.append(cls_node)
                target_colors.append("red")

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.title(citeseer_node_embedding_title + " " + str(k + 1))
        cluster_file_name = citeseer_speparate_cluster_path + str(k + 1)
        plt.savefig(cluster_file_name)
        # plt.show()

        figure2 = plt.figure(figsize=(15, 15))
        ax2 = figure2.add_subplot(111)
        source_scatter = ax2.scatter(source_x_embedding, source_y_embedding, cmap="Set2", c=source_colors, label="Source_SDM")
        target_scatter = ax2.scatter(target_x_embedding, target_y_embedding, cmap="Set2", c=target_colors, label="Target")
        plt.title(citeseer_TSNE_clusterwise_source_target_title + str(k + 1))
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.savefig(citesser_TSNE_clusterwise_source_target_plot + str(k + 1))
        # plt.show()

        cluster_wise_embedding_x = []
        cluster_wise_embedding_y = []
        cluster_wise_embedding_label = []
        m = 0
        for cls_name in clusterwise_embedding_src_nodes[k]['name']:
            cluster_wise_embedding_x.append(embeddings_2d.loc[cls_name][0])
            cluster_wise_embedding_y.append(embeddings_2d.loc[cls_name][1])
            cluster_wise_embedding_label.append(clusterwise_embedding_src_nodes[k]['label'][m])
            m += 1
        figure3 = plt.figure(figsize=(15, 15))
        ax3 = figure3.add_subplot(111)
        clusterwise_source_scatter = ax3.scatter(cluster_wise_embedding_x, cluster_wise_embedding_y, cmap="Set2", c=cluster_wise_embedding_label, label="Source_SDM Nodes")
        plt.title(citeseer_TSNE_clusterwise_source_with_cluster_title + str(k + 1) + " No Of Clusters:" + str(len(set(cluster_wise_embedding_label))))
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.savefig(citesser_TSNE_clusterwise_source_with_cluster_labels_plot + str(k + 1))
        # plt.show()

        cust_cluster_wise_embedding_x = []
        cust_cluster_wise_embedding_y = []
        cust_cluster_wise_embedding_label = []
        m = 0
        for cls_name in cust_clusterwise_embedding_nodes[k]['name']:
            cust_cluster_wise_embedding_x.append(embeddings_2d.loc[cls_name][0])
            cust_cluster_wise_embedding_y.append(embeddings_2d.loc[cls_name][1])
            cust_cluster_wise_embedding_label.append(cust_clusterwise_embedding_nodes[k]['label'][m])
            m += 1
        figure10 = plt.figure(figsize=(15, 15))
        ax10 = figure10.add_subplot(111)
        cust_clusterwise_source_scatter = ax10.scatter(cust_cluster_wise_embedding_x, cust_cluster_wise_embedding_y, cmap="Set2", c=cust_cluster_wise_embedding_label)
        plt.title(citeseer_customized_dbscan_clusters + str(k + 1) + "( No of Clusters " + str(
            len(list(set(cust_clusterwise_embedding_nodes[k]['label'])))) + ")")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.savefig(citeseer_customized_dbscan_clusters_plot + str(k + 1))
        plt.legend()
        # plt.show()

        k += 1

    cols = ["Cluster No",
            "No of Nodes in Cluster",
            "No of Sources",
            "No Of Zero Degree Source_SDM",
            "No of Sources - No Of Zero Degree Source_SDM",
            "No of Targets",
            "No Of Zero Degree Target",
            "No of Targets - No Of Zero Degree Target",
            "No of Zero-degree Nodes",
            "No of Outgoing Edges in Cluster (to nodes only within cluster)",
            "No of Incoming Edges in Cluster (from nodes only within cluster)",
            "No of possible edges",
            "No of Only Know Each Other (As Source_SDM)",
            "No of Only Know Each Other (As Target)",
            "Edge Percentage",
            "Average Degree Of Cluster",
            "Hub/Authority Score",
            "Highest Graph Source_SDM Degree",
            "Highest Graph Target Degree",
            "Highest Cluster Source_SDM Degree",
            "Highest Cluster Target Degree",
            "No of Source_SDM and Target Pair available",
            "No of Target and Source_SDM Pair available",
            "Source_SDM Target Pairs",
            "Target Source_SDM Pairs",
            "List of Nodes in Cluster",
            "List of Zero-degree Nodes in Cluster",
            "List of Only Know Each Other (As Source_SDM)",
            "List of Only Know Each Other (As Target)",
            "Singleton Nodes",
            "List of Connected Target Nodes Out-off Cluster",
            "No of Connected Target Nodes Out-off Cluster",
            "List of Connected Source_SDM Nodes Out-off Cluster",
            "No of Connected Source_SDM Nodes Out-off Cluster"]
    rows = []
    new_rows = []
    edges_rows = []
    cluster_wise_label_map = {str(class_name): {} for class_name in range(len(cluster_dict))}
    node_degree_dict = dict(bipartite_G.degree())
    total_degree_row = []
    total_cluster_degree_row = []
    total_cluster_degree = []
    zero_degree_rows = []
    for i in range(len(cluster_dict)):
        row = []
        edge_row = []
        row.append(i + 1)
        source_no = 0
        target_no = 0
        source_list = []
        target_list = []
        original_graph_edge_dict = {}
        source_target_pair = []
        target_source_pair = []
        src_high_degree = -1
        trg_high_degree = -1
        degree_list = []
        zero_src_nodes = []
        zero_trg_nodes = []
        for node in cluster_dict[i]:
            if node.startswith("S"):
                if node_degree_dict[node] > src_high_degree:
                    src_high_degree = node_degree_dict[node]
                if bipartite_G.degree(node) == 0:
                    zero_src_nodes.append(node)
                source_list.append(node)
                source_no += 1
            elif node.startswith("T"):
                if node_degree_dict[node] > trg_high_degree:
                    trg_high_degree = node_degree_dict[node]
                if bipartite_G.degree(node) == 0:
                    zero_trg_nodes.append(node)
                target_list.append(node)
                target_no += 1
            degree_list.append(node_degree_dict[node])
            original_graph_edge_dict[node] = list(bipartite_G.neighbors(node))

        unique_degrees = set(degree_list)
        unique_degree_dict = {}
        for z in range(len(degree_list)):
            if unique_degree_dict.keys().__contains__("Degree " + str(list(degree_list)[z])):
                unique_degree_dict["Degree " + str(list(degree_list)[z])] = unique_degree_dict["Degree " + str(
                    list(degree_list)[z])] + 1
            else:
                unique_degree_dict["Degree " + str(list(degree_list)[z])] = 1

        '''bin_width = 5
        num_bins = max(int((max(degree_list) - min(degree_list)) / bin_width), 1)
        n, bins, patches = plt.hist(degree_list, bins=num_bins)'''
        fig = plt.figure(figsize=(10, 8))
        plt.bar(list(unique_degree_dict.keys()), list(unique_degree_dict.values()), align='center')
        plt.xticks(list(unique_degree_dict.keys()), list(unique_degree_dict.keys()), rotation=90, ha='right')

        for k in range(len(list(unique_degree_dict.keys()))):
            plt.annotate(list(unique_degree_dict.values())[k], (list(unique_degree_dict.keys())[k], list(unique_degree_dict.values())[k]), ha='center', va='bottom')

        # plt.xlim(min(list(unique_degree_dict.keys())) - 0.5, max(list(unique_degree_dict.keys())) + 0.5)
        # Set labels and title
        plt.xlabel('Degree Level')
        plt.ylabel('No of Nodes in Degree')
        plt.title('Bipartite Graph Degree Distribution' + str(i + 1))
        plt.savefig("Bipartite Graph Degree_Distribution" + str(i + 1) + ".png")
        # Display the plot
        # plt.show()

        degree_cols = ["Cluster No", "Node List", "Degree List", "Bipartite Graph Degree Distribution"]
        degree_row = []
        degree_row.append(i + 1)
        degree_row.append(cluster_dict[i])
        degree_row.append(degree_list)
        degree_row.append(unique_degree_dict)
        total_degree_row.append(degree_row)
        '''for j in range(len(n)):
            if n[j] > 0:
                plt.text(bins[j] + bin_width / 2, n[j], str(int(n[j])), ha='center', va='bottom')

        bin_ranges = ['{:.1f}-{:.1f}'.format(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]
        tick_locations = bins[:-1] + bin_width/2  # Adjust tick positions
        plt.xticks(tick_locations, bin_ranges, rotation=45, ha='right')'''

        total_edges_in_cluster_as_source = 0
        total_edges_in_graph_with_source = 0
        total_connected_target_nodes_not_in_cls = []
        source_edges = []
        cluster_degree = {}
        node_edges_count = {}
        highest_src_cls_degree = -1
        source_list_without_zero_degree = []
        zero_degree_nodes = []
        only_know_each_other_as_source = []
        singleton_nodes = []
        for source in source_list:
            source_neighbor_list = original_graph_edge_dict[source]
            common_targets_in_cluster = set(source_neighbor_list) & set(target_list)

            if len(common_targets_in_cluster) == 0:
                zero_degree_row = []
                zero_degree_row.append(i + 1)
                zero_degree_row.append(source)
                zero_degree_row.append("Degree " + str(len(common_targets_in_cluster)))
                zero_degree_row.append(len(source_neighbor_list))
                zero_degree_rows.append(zero_degree_row)
                zero_degree_nodes.append(source)
            else:
                source_list_without_zero_degree.append(source)

                if len(common_targets_in_cluster) == 1:
                    only_source = original_graph_edge_dict[list(common_targets_in_cluster)[0]]
                    only_source_list = set(only_source) & set(source_list)
                    if len(only_source_list) == 1:
                        if list(only_source_list)[0] == source:
                            only_know_each_other_as_source.append(source + "->" + list(common_targets_in_cluster)[0])
                            singleton_nodes.append(source)
                            singleton_nodes.append(list(common_targets_in_cluster)[0])
                            source_list_without_zero_degree.remove(source)

                if len(common_targets_in_cluster) > highest_src_cls_degree:
                    highest_src_cls_degree = len(common_targets_in_cluster)
                node_edges_count[source] = common_targets_in_cluster
                if cluster_degree.keys().__contains__("Degree " + str(len(common_targets_in_cluster))):
                    cluster_degree["Degree " + str(len(common_targets_in_cluster))] = cluster_degree["Degree " + str(
                        len(common_targets_in_cluster))] + 1
                else:
                    cluster_degree["Degree " + str(len(common_targets_in_cluster))] = 1

                connected_target_nodes_not_in_cls = set(source_neighbor_list) - set(common_targets_in_cluster)
                for not_cls_trg in connected_target_nodes_not_in_cls:
                    if not_cls_trg not in total_connected_target_nodes_not_in_cls:
                        total_connected_target_nodes_not_in_cls.append(not_cls_trg)
                total_edges_in_cluster_as_source += len(common_targets_in_cluster)
                total_edges_in_graph_with_source += len(source_neighbor_list)
                '''source_suffix = source[1:]
                target_string = "T" + source_suffix
                if target_list.__contains__(target_string):
                    source_target_pair.append(source + "-" + target_string)'''
                for trg in common_targets_in_cluster:
                    source_edges.append(source + '-' + trg)

        total_edges_in_cluster_as_target = 0
        total_edges_in_graph_with_target = 0
        total_connected_source_nodes_not_in_cls = []
        target_edges = []
        highest_trg_cls_degree = -1

        target_list_without_zero_degree = []
        only_know_each_other_as_target = []
        for target in target_list:
            target_neighbor_list = original_graph_edge_dict[target]
            common_source_in_cluster = set(target_neighbor_list) & set(source_list)
            if len(common_source_in_cluster) > highest_trg_cls_degree:
                highest_trg_cls_degree = len(common_source_in_cluster)

            zero_degree_row = []
            if len(common_source_in_cluster) == 0:
                zero_degree_row.append(i + 1)
                zero_degree_row.append(target)
                zero_degree_row.append("Degree " + str(len(common_source_in_cluster)))
                zero_degree_row.append(len(target_neighbor_list))
                zero_degree_rows.append(zero_degree_row)
                zero_degree_nodes.append(target)
            else:
                target_list_without_zero_degree.append(target)

                if len(common_source_in_cluster) == 1:
                    only_target = original_graph_edge_dict[list(common_source_in_cluster)[0]]
                    only_target_list = set(only_target) & set(target_list)
                    if len(only_target_list) == 1:
                        if list(only_target_list)[0] == target:
                            only_know_each_other_as_target.append(list(common_source_in_cluster)[0] + "->" + target)
                            singleton_nodes.append(target)
                            singleton_nodes.append(list(common_source_in_cluster)[0])
                            target_list_without_zero_degree.remove(target)

                node_edges_count[target] = common_source_in_cluster
                if cluster_degree.keys().__contains__("Degree " + str(len(common_source_in_cluster))):
                    cluster_degree["Degree " + str(len(common_source_in_cluster))] = cluster_degree["Degree " + str(
                        len(common_source_in_cluster))] + 1
                else:
                    cluster_degree["Degree " + str(len(common_source_in_cluster))] = 1
                connected_source_nodes_not_in_cls = set(target_neighbor_list) - set(common_source_in_cluster)

                for not_cls_src in connected_source_nodes_not_in_cls:
                    if not_cls_src not in total_connected_source_nodes_not_in_cls:
                        total_connected_source_nodes_not_in_cls.append(not_cls_src)

                # total_connected_source_nodes_not_in_cls.extend(list(connected_source_nodes_not_in_cls))
                total_edges_in_cluster_as_target += len(common_source_in_cluster)
                total_edges_in_graph_with_target += len(target_neighbor_list)
                '''target_suffix = target[1:]
                source_string = "S" + target_suffix
                if source_list.__contains__(source_string):
                    target_source_pair.append(source_string + "-" + target)'''
                for src in common_source_in_cluster:
                    target_edges.append(src + '-' + target)

        fig = plt.figure(figsize=(10, 8))
        plt.bar(list(cluster_degree.keys()), list(cluster_degree.values()), align='center')
        plt.xticks(list(cluster_degree.keys()), list(cluster_degree.keys()), rotation=90, ha='right')

        for k in range(len(list(cluster_degree.keys()))):
            plt.annotate(list(cluster_degree.values())[k], (list(cluster_degree.keys())[k], list(cluster_degree.values())[k]), ha='center', va='bottom')

        # plt.xlim(min(list(unique_degree_dict.keys())) - 0.5, max(list(unique_degree_dict.keys())) + 0.5)
        # Set labels and title
        plt.xlabel('Degree Level')
        plt.ylabel('No of Nodes in Degree')
        plt.title('Cluster Degree Distribution of Current Nodes in Cluster' + str(i + 1))
        plt.savefig("Cluster Degree_Distribution of Current Nodes in Cluster" + str(i + 1) + ".png")
        # Display the plot
        # plt.show()

        for source in source_list_without_zero_degree:
            source_suffix = source[1:]
            target_string = "T" + source_suffix
            if target_list_without_zero_degree.__contains__(target_string):
                source_target_pair.append(source + "-" + target_string)
            '''for trg in common_targets_in_cluster:
                source_edges.append(source + '-' + trg)'''

        for target in target_list_without_zero_degree:
            target_suffix = target[1:]
            source_string = "S" + target_suffix
            if source_list_without_zero_degree.__contains__(source_string):
                target_source_pair.append(source_string + "-" + target)
            '''for src in common_source_in_cluster:
                target_edges.append(src + '-' + target)'''

        cluster_degree_row = []
        cluster_degree_row.append(i + 1)
        cluster_degree_row.append(cluster_dict[i])
        cluster_degree_row.append(node_edges_count)
        cluster_degree_row.append(cluster_degree)
        total_cluster_degree_row.append(cluster_degree_row)
        cluster_degree_cols = ["Cluster No", "Node List", "Node Edge Count", "Cluster Degree Distribution"]

        removed_zero_degree_nodes = set(cluster_dict[i]) - set(zero_degree_nodes)
        removed_singleton_nodes = set(removed_zero_degree_nodes) - set(singleton_nodes)
        row.append(len(removed_singleton_nodes))
        # row.append(source_no)
        # row.append(target_no)
        row.append(len(source_list_without_zero_degree))
        row.append(len(zero_src_nodes))
        row.append(len(source_list_without_zero_degree) - len(zero_src_nodes))
        row.append(len(target_list_without_zero_degree))
        row.append(len(zero_trg_nodes))
        row.append(len(target_list_without_zero_degree) - len(zero_trg_nodes))
        row.append(len(zero_degree_nodes))
        row.append(total_edges_in_cluster_as_source)
        row.append(total_edges_in_cluster_as_target)
        row.append(len(source_list_without_zero_degree) * len(target_list_without_zero_degree))
        row.append(len(only_know_each_other_as_source))
        row.append(len(only_know_each_other_as_target))
        if len(source_list_without_zero_degree) != 0 and len(target_list_without_zero_degree) != 0:
            row.append(total_edges_in_cluster_as_source / (
                    len(source_list_without_zero_degree) * len(target_list_without_zero_degree)))
        else:
            row.append(0)
        row.append(total_edges_in_cluster_as_source / (source_no + target_no))
        if source_no > target_no:
            row.append(total_edges_in_cluster_as_source / target_no)
        elif source_no < target_no:
            row.append(total_edges_in_cluster_as_source / source_no)
        row.append(src_high_degree)
        row.append(trg_high_degree)
        row.append(highest_src_cls_degree)
        row.append(highest_trg_cls_degree)
        row.append(len(source_target_pair))
        row.append(len(target_source_pair))
        row.append(source_target_pair)
        row.append(target_source_pair)
        # row.append(total_edges_in_graph_with_source)
        # row.append(total_edges_in_graph_with_target)
        # TODO: Consider List Of Nodes after removing zero degree nodes and singleton nodes or Original Node List
        row.append(sorted(cluster_dict[i], key=key_func))
        row.append(sorted(zero_degree_nodes, key=key_func))
        row.append(only_know_each_other_as_source)
        row.append(only_know_each_other_as_target)
        row.append(set(singleton_nodes))
        # row.append(sorted(cluster_dict[i], key=key_func))
        row.append(total_connected_target_nodes_not_in_cls)
        row.append(len(total_connected_target_nodes_not_in_cls))
        row.append(total_connected_source_nodes_not_in_cls)
        row.append(len(total_connected_source_nodes_not_in_cls))
        rows.append(row)

        nodes_not_in_cls_row = []
        nodes_not_in_cls_row.append(i)
        nodes_not_in_cls_row.append(total_connected_source_nodes_not_in_cls)
        nodes_not_in_cls_row.append(total_connected_target_nodes_not_in_cls)
        nodes_not_in_cls.append(nodes_not_in_cls_row)

        edge_row.append(i + 1)
        edge_row.append(source_edges)
        edge_row.append(target_edges)
        edges_rows.append(edge_row)

    nodes_not_in_cls_col = ['Cluster No', 'Source_SDM Nodes Not In Cluster', 'Target Nodes Not In Cluster']
    nodes_not_in_cls_dict = pd.DataFrame(nodes_not_in_cls, columns=nodes_not_in_cls_col)
    save_as_pickle(nodes_not_in_cls_pkl, nodes_not_in_cls_dict)
    zero_degree_cols = ['Cluster No', 'Node', 'Zero Degree in Cluster', 'Degree Ouside Cluster']
    # pd.DataFrame(zero_degree_rows, columns=zero_degree_cols).to_csv(bitcoin_zero_degree_node_analysis)
    pd.DataFrame(zero_degree_rows, columns=zero_degree_cols).to_csv(citeseer_zero_degree_node_analysis)
    result = pd.DataFrame(rows, columns=cols)
    # pd.DataFrame(result).to_csv(citation_spectral_clustering_csv, encoding='utf-8', float_format='%f')
    pd.DataFrame(total_degree_row, columns=degree_cols).to_csv("degreeList_out_off_cluster_nodes.csv", encoding='utf-8',
                                                               float_format='%f')
    pd.DataFrame(total_cluster_degree_row, columns=cluster_degree_cols).to_csv(
        "cluster_degree_out_off_cluster_nodes.csv", encoding='utf-8', float_format='%f')
    edge_cols = ['Cluster No', 'S-T edges', 'T-S edges']
    edge_rows_df = pd.DataFrame(edges_rows, columns=edge_cols)
    pd.DataFrame(edge_rows_df).to_csv(edge_file_name, encoding='utf-8', float_format='%f')

    return clusters, cluster_dict

def spectral_dbscan_clustering_bitcoin_source_correct_hit(middleDimentions, no_of_clusters, citation_spectral_clustering_csv, title,
                                              spectral_cluster_plot_file_name, bipartite_G, edge_file_name,
                                              nodes_not_in_cls_pkl, original_citeseer_graph):
    sorted_hubs, sorted_authorities = hit_score_original_graph(original_citeseer_graph, citeseer_original_hub_csv, citeseer_original_authority_csv,
                                      citeseer_hit_authority_score_csv, citeseer_original_hub_plot, citeseer_original_authority_plot, citeseer_original_hub_title,
                                     citeseer_original_authority_title)


    nodes_not_in_cls = []
    # Bitcoin cluster pickel file
    '''if exists(bitcoin_spectral_clustering_pickel):
        clusters = read_pickel(bitcoin_spectral_clustering_pickel)
    else:
        spectral = SpectralClustering(n_clusters=no_of_clusters, affinity='nearest_neighbors', n_neighbors=50, random_state=0)
        clusters = spectral.fit_predict(middleDimentions)
        save_as_pickle(bitcoin_spectral_clustering_pickel, clusters) '''

    if exists(citeseer_clusters_list_pkl):
        clusters = read_pickel(citeseer_clusters_list_pkl)
    else:
        spectral = SpectralClustering(n_clusters=no_of_clusters, affinity='nearest_neighbors', n_neighbors=50, random_state=0)
        clusters = spectral.fit_predict(middleDimentions)
        save_as_pickle(citeseer_clusters_list_pkl, clusters)

    if exists(citeseer_cluster_nodes) and exists(citeseer_clusterwise_embedding_src_nodes) and exists(citeseer_cust_clusterwise_embedding_nodes) and exists(citeseer_cluster_dict):
        cluster_nodes = read_pickel(citeseer_cluster_nodes)
        clusterwise_embedding_src_nodes = read_pickel(citeseer_clusterwise_embedding_src_nodes)
        cust_clusterwise_embedding_nodes = read_pickel(citeseer_cust_clusterwise_embedding_nodes)
        cluster_dict = read_pickel(citeseer_cluster_dict)
    else:
        cluster_dict = {}
        cluster_nodes = {}
        clusterwise_embedding_src_nodes = {}
        cust_clusterwise_embedding_nodes = {}
        for i in range(no_of_clusters):
            cluster_dict[i] = []
            cluster_nodes[i] = {}
            cluster_nodes[i]['nodes'] = []
            cluster_nodes[i]['label'] = []
            cluster_nodes[i]['name'] = []
            cluster_nodes[i]['status'] = []
            cluster_nodes[i]['cluster_no'] = []
            cluster_nodes[i]['index'] = []
            clusterwise_embedding_src_nodes[i] = {}
            clusterwise_embedding_src_nodes[i]['name'] = []
            clusterwise_embedding_src_nodes[i]['label'] = []
            cust_clusterwise_embedding_nodes[i] = {}
            cust_clusterwise_embedding_nodes[i]['name'] = []
            cust_clusterwise_embedding_nodes[i]['label'] = []

        for i in range(len(clusters)):
            cluster_dict[clusters[i]].append(list(middleDimentions.index)[i])
            cluster_nodes[clusters[i]]['nodes'].append(list(middleDimentions.iloc[i]))
            cluster_nodes[clusters[i]]['label'].append('unlabeled')
            cluster_nodes[clusters[i]]['name'].append(list(middleDimentions.index)[i])
            cluster_nodes[clusters[i]]['status'].append('unvisited')
            cluster_nodes[clusters[i]]['cluster_no'].append(-1)
            cluster_nodes[clusters[i]]['index'].append(list(middleDimentions.index).index(list(middleDimentions.index)[i]))

        save_as_pickle(citeseer_cluster_nodes, cluster_nodes)
        save_as_pickle(citeseer_clusterwise_embedding_src_nodes, clusterwise_embedding_src_nodes)
        save_as_pickle(citeseer_cust_clusterwise_embedding_nodes, cust_clusterwise_embedding_nodes)
        save_as_pickle(citeseer_cluster_dict, cluster_dict)

    if exists(citeseer_dbscan_cluster_nodes_labels) and exists(citeseer_embedding_map) and exists(citeseer_source_nodes_map) and exists(citeseer_avg_distance_map):
        cluster_labels_map = read_pickel(citeseer_dbscan_cluster_nodes_labels)
        embedding_map = read_pickel(citeseer_embedding_map)
        source_nodes_map = read_pickel(citeseer_source_nodes_map)
        avg_distance_map = read_pickel(citeseer_avg_distance_map)
    else:
        cluster_labels_map = {}
        avg_distances_rows = []
        embedding_map = {}
        source_nodes_map = {}
        avg_distance_map = {}
        avg_distance_elblow = {}
        for u in range(len(cluster_nodes)):
            source_nodes = {}
            source_nodes['nodes'] = []
            source_nodes['name'] = []
            source_nodes['label'] = []
            source_nodes['status'] = []
            source_nodes['cluster_no'] = []
            source_nodes['index'] = []
            source_nodes['embedding'] = []
            for j in range(len(cluster_nodes[u]['nodes'])):
                if cluster_nodes[u]['name'][j].startswith("S"):
                    source_nodes['nodes'].append(cluster_nodes[u]['nodes'][j])
                    source_nodes['name'].append(cluster_nodes[u]['name'][j])
                    source_nodes['label'].append(cluster_nodes[u]['label'][j])
                    source_nodes['status'].append(cluster_nodes[u]['status'][j])
                    source_nodes['cluster_no'].append(cluster_nodes[u]['cluster_no'][j])
                    source_nodes['index'].append(cluster_nodes[u]['index'][j])
                    source_nodes['embedding'].append(list(middleDimentions.iloc[cluster_nodes[u]['index'][j]]))
            source_nodes_map[u] = source_nodes
            #avg, sorted_names, sorted_values = average_distance(source_nodes)
            avg, avg_elblow_point = average_distance(source_nodes, u+1)
            avg_distance_map[u] = avg
            avg_distance_elblow[u] = avg_elblow_point
            avg_distances_row = []
            avg_distances_row.append(u + 1)
            #avg_distances_row.append(sorted_names)
            #avg_distances_row.append(sorted_values)
            avg_distances_rows.append(avg_distances_row)

            #dbscan = DBSCAN(eps=avg, min_samples=min_samples)
            dbscan = DBSCAN(eps=avg, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(source_nodes['nodes'])
            cluster_labels_map[u] = cluster_labels

            tsne = TSNE(n_components=2, perplexity=5)
            embeddings_2d = tsne.fit_transform(np.array(source_nodes['nodes']))
            embedding_map[u] = embeddings_2d

        save_as_pickle(citeseer_dbscan_cluster_nodes_labels, cluster_labels_map)
        save_as_pickle(citeseer_embedding_map, embedding_map)
        save_as_pickle(citeseer_source_nodes_map, source_nodes_map)
        save_as_pickle(citeseer_avg_distance_map, avg_distance_map)
        avg_distance_cols = ["Cluster No", "Source_SDM Names", "5th Nearest Neighbor Distance"]
        #pd.DataFrame(avg_distances_rows, columns=avg_distance_cols).to_csv(citeseer_5th_nearest_neighbor_distance_csv)

    rows = []
    dbscan_rows = []
    for i in range(len(cluster_nodes)):
        cluster_labels = cluster_labels_map[i]
        citeseer_directed_A = nx.adjacency_matrix(original_citeseer_graph)
        bitcoin_directed_A = pd.DataFrame(citeseer_directed_A.toarray(), columns=original_citeseer_graph.nodes(), index=original_citeseer_graph.nodes())

        embeddings_2d = embedding_map[i]
        print("embedding len" + str(len(embeddings_2d)))
        print("cluster label len" + str(len(np.array(cluster_labels))))
        print(cluster_labels)
        '''x_values = [embeddings_2d[j, 0] for j in range(len(embeddings_2d[:, 0])) if np.array(cluster_labels)[j] != -1]
        y_values = [embeddings_2d[j, 1] for j in range(len(embeddings_2d[:, 0])) if np.array(cluster_labels)[j] != -1]       
        source_lables = [int(lab) for lab in np.array(cluster_labels) if lab != -1]      
        source_names = [source_nodes['name'][j] for j in range(len(source_nodes['name'])) if cluster_labels[j] != -1]
        src_cls_embedding = [source_nodes['embedding'][j] for j in range(len(source_nodes['name'])) if cluster_labels[j] != -1] '''
        x_values = []
        y_values = []
        source_lables = []
        source_names = []
        src_cls_embedding = []
        for b in range(len(np.array(cluster_labels))):
            if cluster_labels[b] != -1:
                x_values.append(embeddings_2d[b, 0])
                y_values.append(embeddings_2d[b, 1])
                source_lables.append(int(cluster_labels[b]))
                source_nodes = source_nodes_map[i]
                source_names.append(source_nodes['name'][b])
                src_cls_embedding.append(source_nodes['embedding'][b])

        figure5 = plt.figure(figsize=(15, 15))
        ax5 = figure5.add_subplot(111)
        ax5.scatter(x_values, y_values, c=source_lables)
        plt.legend()
        plt.title("DBSCAN Inside Spectral Clustering (Cluster No: " + str(i + 1) + ") ( No of Clusters " + str(
            len(set(source_lables))) + ")")
        plt.savefig(citeseer_spectral_dbscan_plot_name + "_" + str(i + 1) + ".png")
        # plt.show()

        clusterwise_embedding_src_nodes[i]['name'] = source_names
        clusterwise_embedding_src_nodes[i]['label'] = source_lables

        dbscan_cluster_info = {}
        for lab in list(set(source_lables)):
            dbscan_cluster_info[lab] = {}
            dbscan_cluster_info[lab]['x_value'] = []
            dbscan_cluster_info[lab]['y_value'] = []
            dbscan_cluster_info[lab]['names'] = []
            dbscan_cluster_info[lab]['embedding'] = []
            dbscan_cluster_info[lab]['status'] = []
            dbscan_cluster_info[lab]['label'] = []

        for lab in set(source_lables):
            dbscan_cluster_info[lab]['x_value'] = [x_values[k] for k in range(len(source_lables)) if source_lables[k] == lab]
            dbscan_cluster_info[lab]['y_value'] = [y_values[k] for k in range(len(source_lables)) if source_lables[k] == lab]
            dbscan_cluster_info[lab]['names'] = [source_names[k] for k in range(len(source_names)) if source_lables[k] == lab]
            dbscan_cluster_info[lab]['embedding'] = [src_cls_embedding[k] for k in range(len(source_lables)) if source_lables[k] == lab]
            dbscan_cluster_info[lab]['label'] = [lab for k in range(len(source_lables)) if source_lables[k] == lab]

        for k in range(len(dbscan_cluster_info)):
            row = []
            row.append(i + 1)
            row.append(k + 1)
            row.append(len(dbscan_cluster_info[k]['names']))
            row.append(min_samples)
            row.append(avg_distance_map[i])
            row.append(dbscan_cluster_info[k]['names'])
            row.append(dbscan_cluster_info[k]['embedding'])
            row.append(dbscan_cluster_info[k]['x_value'])
            row.append(dbscan_cluster_info[k]['y_value'])
            rows.append(row)

        cust_dbscan_cluster_nodes = {}
        for k in range(len(dbscan_cluster_info)):
            cust_dbscan_cluster_nodes[k] = {}
            cust_dbscan_cluster_nodes[k]['nodes'] = []
            cust_dbscan_cluster_nodes[k]['original_embedding'] = []
            cust_dbscan_cluster_nodes[k]['src_x_embedding'] = []
            cust_dbscan_cluster_nodes[k]['src_y_embedding'] = []
            cust_dbscan_cluster_nodes[k]['status'] = []
            cust_dbscan_cluster_nodes[k]['label'] = []
            cust_dbscan_cluster_nodes[k]['no_src'] = 0
            cust_dbscan_cluster_nodes[k]['no_trg'] = 0
            cust_dbscan_cluster_nodes[k]['no_pair'] = 0
            cust_dbscan_cluster_nodes[k]['avg_degree'] = 0
            cust_dbscan_cluster_nodes[k]['iterations'] = 0

        for k in range(len(dbscan_cluster_info)):
            cust_dbscan_cluster_nodes[k]['nodes'] = copy.deepcopy(dbscan_cluster_info[k]['names'])
            cust_dbscan_cluster_nodes[k]['original_embedding'] = copy.deepcopy(dbscan_cluster_info[k]['embedding'])
            cust_dbscan_cluster_nodes[k]['src_x_embedding'] = copy.deepcopy(dbscan_cluster_info[k]['x_value'])
            cust_dbscan_cluster_nodes[k]['src_y_embedding'] = copy.deepcopy(dbscan_cluster_info[k]['y_value'])
            cust_dbscan_cluster_nodes[k]['status'] = ["unvisited" for k in range(len(dbscan_cluster_info[k]['names']))]
            cust_dbscan_cluster_nodes[k]['label'] = copy.deepcopy(dbscan_cluster_info[k]['label'])

        closure_cnt = 0
        for p in range(len(cust_dbscan_cluster_nodes)):
            print("DBSCAN Cluster No:" + str(p + 1))
            #dbscan_names = set(cust_dbscan_cluster_nodes[p]['nodes'])
            dbscan_names = cust_dbscan_cluster_nodes[p]['nodes']
            previous_flag = ""
            iteration = 0
            while len(dbscan_names) > 0:
                #last_element = dbscan_names.pop()
                last_element = dbscan_names[0]
                dbscan_names = dbscan_names[1:]
                last_element_index = cust_dbscan_cluster_nodes[p]['nodes'].index(last_element)
                last_element_status = cust_dbscan_cluster_nodes[p]['status'][last_element_index]

                if previous_flag == "":
                    previous_flag = last_element[0]
                    iteration += 1

                if last_element_status == "unvisited":
                    cust_dbscan_cluster_nodes[p]['status'][last_element_index] = "visited"
                    if last_element.startswith("S"):
                        new_src = int(last_element[1:])
                        src_trg = bitcoin_directed_A.loc[new_src]
                        indices = [i for i, value in enumerate(src_trg) if value == 1]
                        dataframe_indices = list(src_trg.index)
                        trg_list = [dataframe_indices[trg_index] for trg_index in indices]
                        target_included_flag = False
                        for trg in trg_list:
                            # if list(cust_dbscan_cluster_nodes[p]['nodes']).__contains__("T"+str(trg)) == False and list(cluster_nodes[i]['name']).__contains__("T"+str(trg)) == True:
                            if list(cust_dbscan_cluster_nodes[p]['nodes']).__contains__("T" + str(trg)) == False:
                                src_trg_neighbors = bitcoin_directed_A[trg]
                                indices = [i for i, value in enumerate(src_trg_neighbors) if value == 1]
                                dataframe_indices = list(src_trg_neighbors.index)
                                src_list = [dataframe_indices[src_index] for src_index in indices]
                                src_trg_neighbors = set(src_list)
                                cls_src_list = [int(src[1:]) for src in cust_dbscan_cluster_nodes[p]['nodes'] if src.startswith("S")]
                                cls_src_list.append(last_element[1:])
                                cls_src_list = set(cls_src_list)
                                common_src = list(cls_src_list.intersection(src_trg_neighbors))
                                if len(common_src) >= closer_3:
                                    included_trg = "T" + str(trg)
                                    cust_dbscan_cluster_nodes[p]['nodes'].append(included_trg)
                                    cust_dbscan_cluster_nodes[p]['status'].append("unvisited")
                                    cust_dbscan_cluster_nodes[p]['original_embedding'].append(list(middleDimentions.loc[included_trg]))
                                    cust_dbscan_cluster_nodes[p]['label'].append(list(set(cust_dbscan_cluster_nodes[p]['label']))[0])
                                    #dbscan_names.update([included_trg])
                                    dbscan_names.append(included_trg)
                                    if previous_flag != last_element[0]:
                                        previous_flag = last_element[0]
                                        iteration += 1
                    elif last_element.startswith("T"):
                        new_trg = int(last_element[1:])
                        trg_src = bitcoin_directed_A[new_trg]
                        indices = [i for i, value in enumerate(trg_src) if value == 1]
                        dataframe_indices = list(trg_src.index)
                        src_list = [dataframe_indices[src_index] for src_index in indices]
                        source_included = False
                        for src in src_list:
                            # if list(cust_dbscan_cluster_nodes[p]['nodes']).__contains__("S"+str(src)) == False and list(cluster_nodes[i]['name']).__contains__("S"+str(src)) == True:
                            if list(cust_dbscan_cluster_nodes[p]['nodes']).__contains__("S" + str(src)) == False:
                                trg_src_neighbors = bitcoin_directed_A.loc[src]
                                indices = [i for i, value in enumerate(trg_src_neighbors) if value == 1]
                                dataframe_indices = list(trg_src_neighbors.index)
                                trg_list = [dataframe_indices[trg_index] for trg_index in indices]
                                trg_src_neighbors = set(trg_list)
                                cls_trg_list = [int(src[1:]) for src in cust_dbscan_cluster_nodes[p]['nodes'] if src.startswith("T")]
                                cls_trg_list.append(last_element[1:])
                                cls_trg_list = set(cls_trg_list)
                                common_trg = list(cls_trg_list.intersection(trg_src_neighbors))
                                if len(common_trg) >= closer_3:
                                    included_src = "S" + str(src)
                                    cust_dbscan_cluster_nodes[p]['nodes'].append(included_src)
                                    cust_dbscan_cluster_nodes[p]['status'].append("unvisited")
                                    cust_dbscan_cluster_nodes[p]['original_embedding'].append(list(middleDimentions.loc[included_src]))
                                    cust_dbscan_cluster_nodes[p]['label'].append(list(set(cust_dbscan_cluster_nodes[p]['label']))[0])
                                    #dbscan_names.update([included_src])
                                    dbscan_names.append(included_src)
                                    if previous_flag != last_element[0]:
                                        previous_flag = last_element[0]
                                        iteration += 1

            print("Spectral Cluster " + str(i + 1) + "DBSCAN Cluster " + str(p + 1) + " Number Of Iterations:" + str(iteration))
            cust_dbscan_cluster_nodes[p]['iterations'] = iteration

        cust_nodes = []
        cust_labels = []
        for h in range(len(cust_dbscan_cluster_nodes)):
            cust_nodes.extend(list(cust_dbscan_cluster_nodes[h]['nodes']))
            cust_labels.extend(list(cust_dbscan_cluster_nodes[h]['label']))
        cust_clusterwise_embedding_nodes[i]['name'] = cust_nodes
        cust_clusterwise_embedding_nodes[i]['label'] = cust_labels
        '''tsne = TSNE(n_components=2, perplexity=25)
        embeddings_2d = tsne.fit_transform(np.array(middleDimentions))
        x_min = min(embeddings_2d[:, 0])
        x_max = max(embeddings_2d[:, 0])
        y_min = min(embeddings_2d[:, 1])
        y_max = max(embeddings_2d[:, 1])
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        figure10 = plt.figure(figsize=(15, 15))
        ax10 = figure10.add_subplot(111)
        embeddings_2d = pd.DataFrame(embeddings_2d, index=middleDimentions.index)
        cust_x = []
        cust_y = []
        cust_label = []
        for k in range(len(cust_dbscan_cluster_nodes)):
            dbscan_cls_names = list(cust_dbscan_cluster_nodes[k]['nodes'])
            dbscan_labels = list(cust_dbscan_cluster_nodes[k]['label'])
            for p in range(len(dbscan_cls_names)):
                cust_x.append(embeddings_2d.loc[dbscan_cls_names[p]][0])
                cust_y.append(embeddings_2d.loc[dbscan_cls_names[p]][1])
                cust_label.append(dbscan_labels[p])
        ax10.scatter(cust_x,cust_y, c=cust_label, cmap="jet")
        plt.title(bitcoin_customized_dbscan_clusters + str(i+1) +"( No of Clusters " + str(len(cust_dbscan_cluster_nodes)) + ")")
        plt.savefig(bitcoin_customized_dbscan_clusters_plot + str(i+1))
        plt.legend()
        plt.show()'''

        colors = []
        for k in range(len(cust_dbscan_cluster_nodes)):
            hub_bar_list = {}
            authority_bar_list = {}
            line_x_values = {}
            hub_color = {}
            authority_color = {}
            for bar, bar_value in zip(list(sorted_hubs.keys()), list(sorted_hubs.values())):
                hub_bar_list[bar]= 0
                authority_bar_list[bar] = 0
                colors.append('#FFFFFF')
                line_x_values[bar] = ""
                hub_color[bar] = "red"
                authority_color[bar] = "red"
            row = []
            row.append(i + 1)
            row.append(k + 1)
            row.append(len(cust_dbscan_cluster_nodes[k]['nodes']))
            row.append(len(dbscan_cluster_info[k]['names']))
            sr_no = 0
            tr_no = 0
            pair_nodes = set()
            degree_cnt = 0
            edges = set()
            cls_nodes = set(cust_dbscan_cluster_nodes[k]['nodes'])
            hits_score_list = {}
            authority_score_list = {}
            percentile_score_list = {}
            for nd in cust_dbscan_cluster_nodes[k]['nodes']:
                if nd.startswith("S"):
                    sr_no += 1
                    node_no = nd[1:]
                    tr_node = "T" + str(node_no)
                    if list(cust_dbscan_cluster_nodes[k]['nodes']).__contains__(tr_node):
                        pair_nodes.add(node_no)
                elif nd.startswith("T"):
                    tr_no += 1
                    node_no = nd[1:]
                    sr_node = "S" + str(node_no)
                    if list(cust_dbscan_cluster_nodes[k]['nodes']).__contains__(sr_node):
                        pair_nodes.add(node_no)
                edge_list = list(bipartite_G.neighbors(nd))
                common_neighbors = set(edge_list).intersection(cls_nodes)
                hits_score_list[nd[1:]] = sorted_hubs[nd[1:]] * 1e-4
                hub_bar_list[nd[1:]] = max(sorted_hubs.values())
                if nd.startswith("T"):
                    colors.append('#FF0000')
                elif nd.startswith("S"):
                    colors.append('#0000FF')

                authority_bar_list[nd[1:]] = max(sorted_authorities.values())
                line_x_values[nd[1:]] = nd[1:]
                authority_score_list[nd[1:]] = sorted_authorities[nd[1:]] * 1e-4
                if nd.startswith("S"):
                    hub_color[nd[1:]] = "blue"
                    authority_color[nd[1:]] = "blue"
                for cm_ed in common_neighbors:
                    if nd.startswith("S"):
                        edge = str(nd) + "-" + str(cm_ed)
                    elif nd.startswith("T"):
                        edge = str(cm_ed) + "-" + str(nd)
                    edges.add(edge)

            row.append(sr_no)
            row.append(tr_no)
            row.append(len(edges))
            row.append(len(pair_nodes))
            row.append(cust_dbscan_cluster_nodes[k]['iterations'])
            row.append(cust_dbscan_cluster_nodes[k]['nodes'])
            row.append(edges)
            bipartite_G_copy = bipartite_G.copy()
            #subgraph = bipartite_G_copy.subgraph(cust_dbscan_cluster_nodes[k]['nodes'])
            '''bipartite_G_copy.freeze()
            for edge in edges:
                subgraph.add_edge(*edge)'''
            #sub_graph_hubs, sub_graph_authorities = nx.hits(subgraph)
            # Sort the Hub and Authority scores by value in descending order
            #sub_graph_sorted_hubs = dict(sorted(sub_graph_hubs.items(), key=lambda item: item[1], reverse=True))
            #sub_graph_sorted_authorities = dict(sorted(sub_graph_authorities.items(), key=lambda item: item[1], reverse=True))
            #row.append(cust_dbscan_cluster_nodes[k]['status'])
            #row.append(cust_dbscan_cluster_nodes[k]['original_embedding'])
            #row.append(cust_dbscan_cluster_nodes[k]['label'])
            row.append(hits_score_list)
            row.append(authority_score_list)
            #row.append(percentile_score_list)
            #row.append(sub_graph_sorted_hubs)
            #row.append(sub_graph_sorted_authorities)

            y_min = min(list(sorted_hubs.values()))
            y_max = max(list(sorted_hubs.values()))
            figure_cluster_hub = plt.figure(figsize=(15, 15))
            ax_cluster_hub = figure_cluster_hub.add_subplot(111)
            ax_cluster_hub.set_ylim(y_min, y_max)
            ax_cluster_hub.plot(list(sorted_hubs.keys()), list(sorted_hubs.values()), label="Bipartite Graph Hub Score")
            ax_cluster_hub.bar(list(sorted_hubs.keys()), list(hub_bar_list.values()), color=list(hub_color.values()), label="Cluster Hub Score Location")
            plt.legend()
            plt.title("Node Hub Score Position For Spectral Cluster No:" + str(i+1) + "DBSCAN Cluster No:" + str(k+1))
            #plt.show()
            plt.savefig(citeseer_spectral_cluster + "node_hub_score_position_spectral_cls"+str(i+1) + "dbscan_cls_" + str(k+1) +".png")

            y_min = min(list(sorted_authorities.values()))
            y_max = max(list(sorted_authorities.values()))
            figure_cluster_authority = plt.figure(figsize=(15, 15))
            ax_cluster_authority = figure_cluster_authority.add_subplot(111)
            ax_cluster_authority.set_ylim(y_min, y_max)
            ax_cluster_authority.plot(list(sorted_authorities.keys()), list(sorted_authorities.values()), label="Bipartite Graph Authority Score")
            ax_cluster_authority.bar(list(sorted_authorities.keys()), list(authority_bar_list.values()), color=list(authority_color.values()), label="Cluster Authority Score Location")
            plt.legend()
            plt.title("Node Authority Score Position For Spectral Cluster No:" + str(i + 1) + "  DBSCAN Cluster No:" + str(k + 1))
            # plt.show()
            plt.savefig(citeseer_spectral_cluster + "node_authority_score_position_spectral_cls" + str(i + 1) + "dbscan_cls_" + str(k + 1) + ".png")
            dbscan_rows.append(row)

    # target clustering
    dbscan_cols = ['Spectral Cluster No', 'DBSCAN Cluster No', 'No Of Source_SDM Nodes in Cluster', 'Min Sample', '(Eps)Avg of avg of all distances', 'List Of Nodes',
                   'Node2vec 20-D Embedding','X Value(TSNE)', 'Y Value(TSNE)']
    pd.DataFrame(rows, columns=dbscan_cols).to_csv(citeseer_dbscan_cluster_csv_file)


    # target columns
    cust_dbscan_cols = ["Spectral Cluster No", "DBSCAN Cluster No", "No Of Nodes in Cluster", "Original No Of Source_SDM",
                        "No Of Source_SDM Nodes", "No Of Target Nodes", "No Of Edges", "No Of Pairs", "Iteration for Convergence","List Of Nodes",
                        "List Of Edges", "Original Bipartite Graph Hub Score", "Original Bipartite Graph Authority Score"]
    pd.DataFrame(dbscan_rows, columns=cust_dbscan_cols).to_csv(citeseer_cust_dbscan_cluster_csv_file)


    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 0),  # Maroon
        (0, 128, 0),  # Green
        (0, 0, 128),  # Navy
        (128, 128, 0),  # Olive
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
        (255, 165, 0),  # Orange
        (128, 128, 128),  # Gray
        (255, 192, 203),  # Pink
        (0, 128, 0),  # Green
        (128, 0, 0),  # Maroon
        (0, 0, 128),  # Navy
        (0, 128, 128),  # Teal
        (128, 0, 128),  # Purple
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
    ]

    normalized_colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]

    selected_colors = [normalized_colors[i] for i in clusters]

    if exists(citeseer_citation_main_TSNE_embedding):
        embeddings_2d = read_pickel(citeseer_citation_main_TSNE_embedding)
    else:
        tsne = TSNE(n_components=2, perplexity=25)
        embeddings_2d = tsne.fit_transform(np.array(middleDimentions))
        save_as_pickle(citeseer_citation_main_TSNE_embedding, embeddings_2d)

    x_min = min(embeddings_2d[:, 0])
    x_max = max(embeddings_2d[:, 0])
    y_min = min(embeddings_2d[:, 1])
    y_max = max(embeddings_2d[:, 1])
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    figure = plt.figure(figsize=(15, 15))
    ax = figure.add_subplot(111)
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=selected_colors, cmap='jet')
    plt.title(title + "( No of Clusters " + str(len(set(clusters))) + ")")
    plt.savefig(spectral_cluster_plot_file_name)
    plt.legend()
    # plt.show()

    embeddings_2d = pd.DataFrame(embeddings_2d, index=middleDimentions.index)

    '''Original Embedding'''
    k = 0
    for cls_nodes in cluster_dict.values():
        source_x_embedding = []
        source_y_embedding = []
        target_x_embedding = []
        target_y_embedding = []
        source_labels = []
        target_labels = []
        source_colors = []
        target_colors = []
        figure1 = plt.figure(figsize=(15, 15))
        ax1 = figure1.add_subplot(111)

        for cls_node in cls_nodes:
            scatter = ax1.scatter(embeddings_2d.loc[cls_node][0], embeddings_2d.loc[cls_node][1], c=normalized_colors[k], cmap="jet")
            if cls_node.startswith("S"):
                source_x_embedding.append(embeddings_2d.loc[cls_node][0])
                source_y_embedding.append(embeddings_2d.loc[cls_node][1])
                source_labels.append(cls_node)
                source_colors.append("blue")
            elif cls_node.startswith("T"):
                target_x_embedding.append(embeddings_2d.loc[cls_node][0])
                target_y_embedding.append(embeddings_2d.loc[cls_node][1])
                target_labels.append(cls_node)
                target_colors.append("red")

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.title(citeseer_node_embedding_title + " " + str(k + 1))
        cluster_file_name = citeseer_spectral_cluster + str(k + 1)
        plt.savefig(cluster_file_name)
        # plt.show()

        figure2 = plt.figure(figsize=(15, 15))
        ax2 = figure2.add_subplot(111)
        source_scatter = ax2.scatter(source_x_embedding, source_y_embedding, cmap="Set2", c=source_colors, label="Source_SDM")
        target_scatter = ax2.scatter(target_x_embedding, target_y_embedding, cmap="Set2", c=target_colors, label="Target")
        plt.title(citeseer_TSNE_clusterwise_source_target_title + str(k + 1))
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.savefig(citesser_TSNE_clusterwise_source_target_plot + str(k + 1))
        # plt.show()

        figure22 = plt.figure(figsize=(15, 15))
        ax22 = figure22.add_subplot(111)
        source_scatter = ax22.scatter(target_x_embedding, target_y_embedding, cmap="Set2", c=target_colors, label="Target")
        plt.title(citeseer_target_nodes_only_title + str(k + 1))
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend()
        # plt.savefig(bitcoin_TSNE_clusterwise_source_target_plot + str(k+1))
        plt.savefig(citesser_clusterwise_target_plot + str(k + 1))
        # plt.show()

        cluster_wise_embedding_x = []
        cluster_wise_embedding_y = []
        cluster_wise_embedding_label = []
        m = 0
        for cls_name in clusterwise_embedding_src_nodes[k]['name']:
            cluster_wise_embedding_x.append(embeddings_2d.loc[cls_name][0])
            cluster_wise_embedding_y.append(embeddings_2d.loc[cls_name][1])
            cluster_wise_embedding_label.append(clusterwise_embedding_src_nodes[k]['label'][m])
            m += 1
        figure3 = plt.figure(figsize=(15, 15))
        ax3 = figure3.add_subplot(111)
        clusterwise_source_scatter = ax3.scatter(cluster_wise_embedding_x, cluster_wise_embedding_y, cmap="Set2", c=cluster_wise_embedding_label, label="Source_SDM Nodes")
        plt.title(citeseer_TSNE_clusterwise_source_with_cluster_title + str(k + 1) + " No Of Clusters:" + str(len(set(cluster_wise_embedding_label))))
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.savefig(citesser_TSNE_clusterwise_source_with_cluster_labels_plot + str(k + 1))
        # plt.show()

        cust_cluster_wise_embedding_x = []
        cust_cluster_wise_embedding_y = []
        cust_cluster_wise_embedding_label = []
        m = 0
        for cls_name in cust_clusterwise_embedding_nodes[k]['name']:
            cust_cluster_wise_embedding_x.append(embeddings_2d.loc[cls_name][0])
            cust_cluster_wise_embedding_y.append(embeddings_2d.loc[cls_name][1])
            cust_cluster_wise_embedding_label.append(cust_clusterwise_embedding_nodes[k]['label'][m])
            m += 1
        figure10 = plt.figure(figsize=(15, 15))
        ax10 = figure10.add_subplot(111)
        cust_clusterwise_source_scatter = ax10.scatter(cust_cluster_wise_embedding_x, cust_cluster_wise_embedding_y, cmap="Set2", c=cust_cluster_wise_embedding_label)
        plt.title(citeseer_customized_dbscan_clusters + str(k + 1) + "( No of Clusters " + str(len(list(set(cust_clusterwise_embedding_nodes[k]['label'])))) + ")")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.savefig(citeseer_customized_dbscan_clusters_plot + str(k + 1))
        plt.legend()
        # plt.show()
        k += 1

    cols = ["Cluster No",
            "No of Nodes in Cluster",
            "No of Sources",
            "No Of Zero Degree Source_SDM",
            "No of Sources - No Of Zero Degree Source_SDM",
            "No of Targets",
            "No Of Zero Degree Target",
            "No of Targets - No Of Zero Degree Target",
            "No of Zero-degree Nodes",
            "No of Outgoing Edges in Cluster (to nodes only within cluster)",
            "No of Incoming Edges in Cluster (from nodes only within cluster)",
            "No of possible edges",
            "No of Only Know Each Other (As Source_SDM)",
            "No of Only Know Each Other (As Target)",
            "Edge Percentage",
            "Average Degree Of Cluster",
            "Hub/Authority Score",
            "Highest Graph Source_SDM Degree",
            "Highest Graph Target Degree",
            "Highest Cluster Source_SDM Degree",
            "Highest Cluster Target Degree",
            "No of Source_SDM and Target Pair available",
            "No of Target and Source_SDM Pair available",
            "Source_SDM Target Pairs",
            "Target Source_SDM Pairs",
            "List of Nodes in Cluster",
            "List of Zero-degree Nodes in Cluster",
            "List of Only Know Each Other (As Source_SDM)",
            "List of Only Know Each Other (As Target)",
            "Singleton Nodes",
            "List of Connected Target Nodes Out-off Cluster",
            "No of Connected Target Nodes Out-off Cluster",
            "List of Connected Source_SDM Nodes Out-off Cluster",
            "No of Connected Source_SDM Nodes Out-off Cluster"]
    rows = []
    new_rows = []
    edges_rows = []
    cluster_wise_label_map = {str(class_name): {} for class_name in range(len(cluster_dict))}
    node_degree_dict = dict(bipartite_G.degree())
    total_degree_row = []
    total_cluster_degree_row = []
    total_cluster_degree = []
    zero_degree_rows = []
    for i in range(len(cluster_dict)):
        row = []
        edge_row = []
        row.append(i + 1)
        source_no = 0
        target_no = 0
        source_list = []
        target_list = []
        original_graph_edge_dict = {}
        source_target_pair = []
        target_source_pair = []
        src_high_degree = -1
        trg_high_degree = -1
        degree_list = []
        zero_src_nodes = []
        zero_trg_nodes = []
        for node in cluster_dict[i]:
            if node.startswith("S"):
                if node_degree_dict[node] > src_high_degree:
                    src_high_degree = node_degree_dict[node]
                if bipartite_G.degree(node) == 0:
                    zero_src_nodes.append(node)
                source_list.append(node)
                source_no += 1
            elif node.startswith("T"):
                if node_degree_dict[node] > trg_high_degree:
                    trg_high_degree = node_degree_dict[node]
                if bipartite_G.degree(node) == 0:
                    zero_trg_nodes.append(node)
                target_list.append(node)
                target_no += 1
            degree_list.append(node_degree_dict[node])
            original_graph_edge_dict[node] = list(bipartite_G.neighbors(node))

        unique_degrees = set(degree_list)
        unique_degree_dict = {}
        for z in range(len(degree_list)):
            if unique_degree_dict.keys().__contains__("Degree " + str(list(degree_list)[z])):
                unique_degree_dict["Degree " + str(list(degree_list)[z])] = unique_degree_dict["Degree " + str(
                    list(degree_list)[z])] + 1
            else:
                unique_degree_dict["Degree " + str(list(degree_list)[z])] = 1

        '''bin_width = 5
        num_bins = max(int((max(degree_list) - min(degree_list)) / bin_width), 1)
        n, bins, patches = plt.hist(degree_list, bins=num_bins)'''
        fig = plt.figure(figsize=(10, 8))
        plt.bar(list(unique_degree_dict.keys()), list(unique_degree_dict.values()), align='center')
        plt.xticks(list(unique_degree_dict.keys()), list(unique_degree_dict.keys()), rotation=90, ha='right')

        for k in range(len(list(unique_degree_dict.keys()))):
            plt.annotate(list(unique_degree_dict.values())[k], (list(unique_degree_dict.keys())[k], list(unique_degree_dict.values())[k]), ha='center', va='bottom')

        # plt.xlim(min(list(unique_degree_dict.keys())) - 0.5, max(list(unique_degree_dict.keys())) + 0.5)
        # Set labels and title
        plt.xlabel('Degree Level')
        plt.ylabel('No of Nodes in Degree')
        plt.title('Bipartite Graph Degree Distribution Cluster No:' + str(i + 1))
        plt.savefig(citesser_bipartite_graph_degree_distribution + str(i + 1) + ".png")
        # Display the plot
        # plt.show()

        degree_cols = ["Cluster No", "Node List", "Degree List", "Bipartite Graph Degree Distribution"]
        degree_row = []
        degree_row.append(i + 1)
        degree_row.append(cluster_dict[i])
        degree_row.append(degree_list)
        degree_row.append(unique_degree_dict)
        total_degree_row.append(degree_row)
        '''for j in range(len(n)):
            if n[j] > 0:
                plt.text(bins[j] + bin_width / 2, n[j], str(int(n[j])), ha='center', va='bottom')

        bin_ranges = ['{:.1f}-{:.1f}'.format(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]
        tick_locations = bins[:-1] + bin_width/2  # Adjust tick positions
        plt.xticks(tick_locations, bin_ranges, rotation=45, ha='right')'''

        total_edges_in_cluster_as_source = 0
        total_edges_in_graph_with_source = 0
        total_connected_target_nodes_not_in_cls = []
        source_edges = []
        cluster_degree = {}
        node_edges_count = {}
        highest_src_cls_degree = -1
        source_list_without_zero_degree = []
        zero_degree_nodes = []
        only_know_each_other_as_source = []
        singleton_nodes = []
        for source in source_list:
            source_neighbor_list = original_graph_edge_dict[source]
            common_targets_in_cluster = set(source_neighbor_list) & set(target_list)

            if len(common_targets_in_cluster) == 0:
                zero_degree_row = []
                zero_degree_row.append(i + 1)
                zero_degree_row.append(source)
                zero_degree_row.append("Degree " + str(len(common_targets_in_cluster)))
                zero_degree_row.append(len(source_neighbor_list))
                zero_degree_rows.append(zero_degree_row)
                zero_degree_nodes.append(source)
            else:
                source_list_without_zero_degree.append(source)

                if len(common_targets_in_cluster) == 1:
                    only_source = original_graph_edge_dict[list(common_targets_in_cluster)[0]]
                    only_source_list = set(only_source) & set(source_list)
                    if len(only_source_list) == 1:
                        if list(only_source_list)[0] == source:
                            only_know_each_other_as_source.append(source + "->" + list(common_targets_in_cluster)[0])
                            singleton_nodes.append(source)
                            singleton_nodes.append(list(common_targets_in_cluster)[0])
                            source_list_without_zero_degree.remove(source)

                if len(common_targets_in_cluster) > highest_src_cls_degree:
                    highest_src_cls_degree = len(common_targets_in_cluster)
                node_edges_count[source] = common_targets_in_cluster
                if cluster_degree.keys().__contains__("Degree " + str(len(common_targets_in_cluster))):
                    cluster_degree["Degree " + str(len(common_targets_in_cluster))] = cluster_degree["Degree " + str(
                        len(common_targets_in_cluster))] + 1
                else:
                    cluster_degree["Degree " + str(len(common_targets_in_cluster))] = 1

                connected_target_nodes_not_in_cls = set(source_neighbor_list) - set(common_targets_in_cluster)
                for not_cls_trg in connected_target_nodes_not_in_cls:
                    if not_cls_trg not in total_connected_target_nodes_not_in_cls:
                        total_connected_target_nodes_not_in_cls.append(not_cls_trg)
                total_edges_in_cluster_as_source += len(common_targets_in_cluster)
                total_edges_in_graph_with_source += len(source_neighbor_list)
                '''source_suffix = source[1:]
                target_string = "T" + source_suffix
                if target_list.__contains__(target_string):
                    source_target_pair.append(source + "-" + target_string)'''
                for trg in common_targets_in_cluster:
                    source_edges.append(source + '-' + trg)

        total_edges_in_cluster_as_target = 0
        total_edges_in_graph_with_target = 0
        total_connected_source_nodes_not_in_cls = []
        target_edges = []
        highest_trg_cls_degree = -1

        target_list_without_zero_degree = []
        only_know_each_other_as_target = []
        for target in target_list:
            target_neighbor_list = original_graph_edge_dict[target]
            common_source_in_cluster = set(target_neighbor_list) & set(source_list)
            if len(common_source_in_cluster) > highest_trg_cls_degree:
                highest_trg_cls_degree = len(common_source_in_cluster)

            zero_degree_row = []
            if len(common_source_in_cluster) == 0:
                zero_degree_row.append(i + 1)
                zero_degree_row.append(target)
                zero_degree_row.append("Degree " + str(len(common_source_in_cluster)))
                zero_degree_row.append(len(target_neighbor_list))
                zero_degree_rows.append(zero_degree_row)
                zero_degree_nodes.append(target)
            else:
                target_list_without_zero_degree.append(target)

                if len(common_source_in_cluster) == 1:
                    only_target = original_graph_edge_dict[list(common_source_in_cluster)[0]]
                    only_target_list = set(only_target) & set(target_list)
                    if len(only_target_list) == 1:
                        if list(only_target_list)[0] == target:
                            only_know_each_other_as_target.append(list(common_source_in_cluster)[0] + "->" + target)
                            singleton_nodes.append(target)
                            singleton_nodes.append(list(common_source_in_cluster)[0])
                            target_list_without_zero_degree.remove(target)

                node_edges_count[target] = common_source_in_cluster
                if cluster_degree.keys().__contains__("Degree " + str(len(common_source_in_cluster))):
                    cluster_degree["Degree " + str(len(common_source_in_cluster))] = cluster_degree["Degree " + str(
                        len(common_source_in_cluster))] + 1
                else:
                    cluster_degree["Degree " + str(len(common_source_in_cluster))] = 1
                connected_source_nodes_not_in_cls = set(target_neighbor_list) - set(common_source_in_cluster)

                for not_cls_src in connected_source_nodes_not_in_cls:
                    if not_cls_src not in total_connected_source_nodes_not_in_cls:
                        total_connected_source_nodes_not_in_cls.append(not_cls_src)

                # total_connected_source_nodes_not_in_cls.extend(list(connected_source_nodes_not_in_cls))
                total_edges_in_cluster_as_target += len(common_source_in_cluster)
                total_edges_in_graph_with_target += len(target_neighbor_list)
                '''target_suffix = target[1:]
                source_string = "S" + target_suffix
                if source_list.__contains__(source_string):
                    target_source_pair.append(source_string + "-" + target)'''
                for src in common_source_in_cluster:
                    target_edges.append(src + '-' + target)

        fig = plt.figure(figsize=(10, 8))
        plt.bar(list(cluster_degree.keys()), list(cluster_degree.values()), align='center')
        plt.xticks(list(cluster_degree.keys()), list(cluster_degree.keys()), rotation=90, ha='right')

        for k in range(len(list(cluster_degree.keys()))):
            plt.annotate(list(cluster_degree.values())[k], (list(cluster_degree.keys())[k], list(cluster_degree.values())[k]), ha='center', va='bottom')

        # plt.xlim(min(list(unique_degree_dict.keys())) - 0.5, max(list(unique_degree_dict.keys())) + 0.5)
        # Set labels and title
        plt.xlabel('Degree Level')
        plt.ylabel('No of Nodes in Degree')
        plt.title('Cluster Degree Distribution of Current Nodes in Cluster No: ' + str(i + 1))
        plt.savefig(citesser_cluster_degree_distribution + str(i + 1) + ".png")
        # Display the plot
        # plt.show()

        for source in source_list_without_zero_degree:
            source_suffix = source[1:]
            target_string = "T" + source_suffix
            if target_list_without_zero_degree.__contains__(target_string):
                source_target_pair.append(source + "-" + target_string)
            '''for trg in common_targets_in_cluster:
                source_edges.append(source + '-' + trg)'''

        for target in target_list_without_zero_degree:
            target_suffix = target[1:]
            source_string = "S" + target_suffix
            if source_list_without_zero_degree.__contains__(source_string):
                target_source_pair.append(source_string + "-" + target)
            '''for src in common_source_in_cluster:
                target_edges.append(src + '-' + target)'''

        cluster_degree_row = []
        cluster_degree_row.append(i + 1)
        cluster_degree_row.append(cluster_dict[i])
        cluster_degree_row.append(node_edges_count)
        cluster_degree_row.append(cluster_degree)
        total_cluster_degree_row.append(cluster_degree_row)
        cluster_degree_cols = ["Cluster No", "Node List", "Node Edge Count", "Cluster Degree Distribution"]

        removed_zero_degree_nodes = set(cluster_dict[i]) - set(zero_degree_nodes)
        removed_singleton_nodes = set(removed_zero_degree_nodes) - set(singleton_nodes)
        row.append(len(removed_singleton_nodes))
        # row.append(source_no)
        # row.append(target_no)
        row.append(len(source_list_without_zero_degree))
        row.append(len(zero_src_nodes))
        row.append(len(source_list_without_zero_degree) - len(zero_src_nodes))
        row.append(len(target_list_without_zero_degree))
        row.append(len(zero_trg_nodes))
        row.append(len(target_list_without_zero_degree) - len(zero_trg_nodes))
        row.append(len(zero_degree_nodes))
        row.append(total_edges_in_cluster_as_source)
        row.append(total_edges_in_cluster_as_target)
        row.append(len(source_list_without_zero_degree) * len(target_list_without_zero_degree))
        row.append(len(only_know_each_other_as_source))
        row.append(len(only_know_each_other_as_target))
        if len(source_list_without_zero_degree) != 0 and len(target_list_without_zero_degree) != 0:
            row.append(total_edges_in_cluster_as_source / (
                    len(source_list_without_zero_degree) * len(target_list_without_zero_degree)))
        else:
            row.append(0)
        row.append(total_edges_in_cluster_as_source / (source_no + target_no))
        if source_no > target_no:
            row.append(total_edges_in_cluster_as_source / target_no)
        elif source_no < target_no:
            row.append(total_edges_in_cluster_as_source / source_no)
        row.append(src_high_degree)
        row.append(trg_high_degree)
        row.append(highest_src_cls_degree)
        row.append(highest_trg_cls_degree)
        row.append(len(source_target_pair))
        row.append(len(target_source_pair))
        row.append(source_target_pair)
        row.append(target_source_pair)
        # row.append(total_edges_in_graph_with_source)
        # row.append(total_edges_in_graph_with_target)
        # TODO: Consider List Of Nodes after removing zero degree nodes and singleton nodes or Original Node List
        row.append(sorted(cluster_dict[i], key=key_func))
        row.append(sorted(zero_degree_nodes, key=key_func))
        row.append(only_know_each_other_as_source)
        row.append(only_know_each_other_as_target)
        row.append(set(singleton_nodes))
        # row.append(sorted(cluster_dict[i], key=key_func))
        row.append(total_connected_target_nodes_not_in_cls)
        row.append(len(total_connected_target_nodes_not_in_cls))
        row.append(total_connected_source_nodes_not_in_cls)
        row.append(len(total_connected_source_nodes_not_in_cls))
        rows.append(row)

        nodes_not_in_cls_row = []
        nodes_not_in_cls_row.append(i)
        nodes_not_in_cls_row.append(total_connected_source_nodes_not_in_cls)
        nodes_not_in_cls_row.append(total_connected_target_nodes_not_in_cls)
        nodes_not_in_cls.append(nodes_not_in_cls_row)

        edge_row.append(i + 1)
        edge_row.append(source_edges)
        edge_row.append(target_edges)
        edges_rows.append(edge_row)

    nodes_not_in_cls_col = ['Cluster No', 'Source_SDM Nodes Not In Cluster', 'Target Nodes Not In Cluster']
    nodes_not_in_cls_dict = pd.DataFrame(nodes_not_in_cls, columns=nodes_not_in_cls_col)
    save_as_pickle(nodes_not_in_cls_pkl, nodes_not_in_cls_dict)
    zero_degree_cols = ['Cluster No', 'Node', 'Zero Degree in Cluster', 'Degree Ouside Cluster']
    # pd.DataFrame(zero_degree_rows, columns=zero_degree_cols).to_csv(bitcoin_zero_degree_node_analysis)
    pd.DataFrame(zero_degree_rows, columns=zero_degree_cols).to_csv(citeseer_zero_degree_node_analysis)
    result = pd.DataFrame(rows, columns=cols)
    # pd.DataFrame(result).to_csv(citation_spectral_clustering_csv, encoding='utf-8', float_format='%f')
    pd.DataFrame(total_degree_row, columns=degree_cols).to_csv("degreeList_out_off_cluster_nodes.csv", encoding='utf-8',
                                                               float_format='%f')
    pd.DataFrame(total_cluster_degree_row, columns=cluster_degree_cols).to_csv(
        "cluster_degree_out_off_cluster_nodes.csv", encoding='utf-8', float_format='%f')
    edge_cols = ['Cluster No', 'S-T edges', 'T-S edges']
    edge_rows_df = pd.DataFrame(edges_rows, columns=edge_cols)
    pd.DataFrame(edge_rows_df).to_csv(edge_file_name, encoding='utf-8', float_format='%f')

    return clusters, cluster_dict


def spectral_dbscan_clustering_bitcoin_target_correct_hit(middleDimentions, no_of_clusters, citation_spectral_clustering_csv, title,
                                              spectral_cluster_plot_file_name, bipartite_G, edge_file_name,
                                              nodes_not_in_cls_pkl, original_citeseer_graph):
    sorted_hubs, sorted_authorities = hit_score_original_graph(original_citeseer_graph, citeseer_original_hub_csv, citeseer_original_authority_csv,
                                      citeseer_hit_authority_score_csv, citeseer_original_hub_plot, citeseer_original_authority_plot, citeseer_original_hub_title,
                                     citeseer_original_authority_title)


    nodes_not_in_cls = []
    # Bitcoin cluster pickel file
    '''if exists(bitcoin_spectral_clustering_pickel):
        clusters = read_pickel(bitcoin_spectral_clustering_pickel)
    else:
        spectral = SpectralClustering(n_clusters=no_of_clusters, affinity='nearest_neighbors', n_neighbors=50, random_state=0)
        clusters = spectral.fit_predict(middleDimentions)
        save_as_pickle(bitcoin_spectral_clustering_pickel, clusters) '''

    if exists(citeseer_clusters_list_pkl):
        clusters = read_pickel(citeseer_clusters_list_pkl)
    else:
        spectral = SpectralClustering(n_clusters=no_of_clusters, affinity='nearest_neighbors', n_neighbors=50, random_state=0)
        clusters = spectral.fit_predict(middleDimentions)
        save_as_pickle(citeseer_clusters_list_pkl, clusters)

    if exists(citeseer_cluster_nodes) and exists(citeseer_clusterwise_embedding_src_nodes) and exists(citeseer_cust_clusterwise_embedding_nodes) and exists(citeseer_cluster_dict):
        cluster_nodes = read_pickel(citeseer_cluster_nodes)
        clusterwise_embedding_src_nodes = read_pickel(citeseer_clusterwise_embedding_src_nodes)
        cust_clusterwise_embedding_nodes = read_pickel(citeseer_cust_clusterwise_embedding_nodes)
        cluster_dict = read_pickel(citeseer_cluster_dict)
    else:
        cluster_dict = {}
        cluster_nodes = {}
        clusterwise_embedding_src_nodes = {}
        cust_clusterwise_embedding_nodes = {}
        for i in range(no_of_clusters):
            cluster_dict[i] = []
            cluster_nodes[i] = {}
            cluster_nodes[i]['nodes'] = []
            cluster_nodes[i]['label'] = []
            cluster_nodes[i]['name'] = []
            cluster_nodes[i]['status'] = []
            cluster_nodes[i]['cluster_no'] = []
            cluster_nodes[i]['index'] = []
            clusterwise_embedding_src_nodes[i] = {}
            clusterwise_embedding_src_nodes[i]['name'] = []
            clusterwise_embedding_src_nodes[i]['label'] = []
            cust_clusterwise_embedding_nodes[i] = {}
            cust_clusterwise_embedding_nodes[i]['name'] = []
            cust_clusterwise_embedding_nodes[i]['label'] = []

        for i in range(len(clusters)):
            cluster_dict[clusters[i]].append(list(middleDimentions.index)[i])
            cluster_nodes[clusters[i]]['nodes'].append(list(middleDimentions.iloc[i]))
            cluster_nodes[clusters[i]]['label'].append('unlabeled')
            cluster_nodes[clusters[i]]['name'].append(list(middleDimentions.index)[i])
            cluster_nodes[clusters[i]]['status'].append('unvisited')
            cluster_nodes[clusters[i]]['cluster_no'].append(-1)
            cluster_nodes[clusters[i]]['index'].append(list(middleDimentions.index).index(list(middleDimentions.index)[i]))

        save_as_pickle(citeseer_cluster_nodes, cluster_nodes)
        save_as_pickle(citeseer_clusterwise_embedding_src_nodes, clusterwise_embedding_src_nodes)
        save_as_pickle(citeseer_cust_clusterwise_embedding_nodes, cust_clusterwise_embedding_nodes)
        save_as_pickle(citeseer_cluster_dict, cluster_dict)

    if exists(citeseer_dbscan_cluster_nodes_labels) and exists(citeseer_embedding_map) and exists(citeseer_source_nodes_map) and exists(citeseer_avg_distance_map):
        cluster_labels_map = read_pickel(citeseer_dbscan_cluster_nodes_labels)
        embedding_map = read_pickel(citeseer_embedding_map)
        source_nodes_map = read_pickel(citeseer_source_nodes_map)
        avg_distance_map = read_pickel(citeseer_avg_distance_map)
    else:
        cluster_labels_map = {}
        avg_distances_rows = []
        embedding_map = {}
        source_nodes_map = {}
        avg_distance_map = {}
        avg_distance_elblow = {}
        for u in range(len(cluster_nodes)):
            source_nodes = {}
            source_nodes['nodes'] = []
            source_nodes['name'] = []
            source_nodes['label'] = []
            source_nodes['status'] = []
            source_nodes['cluster_no'] = []
            source_nodes['index'] = []
            source_nodes['embedding'] = []
            for j in range(len(cluster_nodes[u]['nodes'])):
                if cluster_nodes[u]['name'][j].startswith("T"):
                    source_nodes['nodes'].append(cluster_nodes[u]['nodes'][j])
                    source_nodes['name'].append(cluster_nodes[u]['name'][j])
                    source_nodes['label'].append(cluster_nodes[u]['label'][j])
                    source_nodes['status'].append(cluster_nodes[u]['status'][j])
                    source_nodes['cluster_no'].append(cluster_nodes[u]['cluster_no'][j])
                    source_nodes['index'].append(cluster_nodes[u]['index'][j])
                    source_nodes['embedding'].append(list(middleDimentions.iloc[cluster_nodes[u]['index'][j]]))
            source_nodes_map[u] = source_nodes
            #avg, sorted_names, sorted_values = average_distance(source_nodes)
            avg, avg_elblow_point = average_distance(source_nodes, u+1)
            avg_distance_map[u] = avg
            avg_distance_elblow[u] = avg_elblow_point
            avg_distances_row = []
            avg_distances_row.append(u + 1)
            #avg_distances_row.append(sorted_names)
            #avg_distances_row.append(sorted_values)
            avg_distances_rows.append(avg_distances_row)

            #dbscan = DBSCAN(eps=avg, min_samples=min_samples)
            dbscan = DBSCAN(eps=avg, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(source_nodes['nodes'])
            cluster_labels_map[u] = cluster_labels

            tsne = TSNE(n_components=2, perplexity=5)
            embeddings_2d = tsne.fit_transform(np.array(source_nodes['nodes']))
            embedding_map[u] = embeddings_2d

        save_as_pickle(citeseer_dbscan_cluster_nodes_labels, cluster_labels_map)
        save_as_pickle(citeseer_embedding_map, embedding_map)
        save_as_pickle(citeseer_source_nodes_map, source_nodes_map)
        save_as_pickle(citeseer_avg_distance_map, avg_distance_map)
        avg_distance_cols = ["Cluster No", "Source_SDM Names", "5th Nearest Neighbor Distance"]
        #pd.DataFrame(avg_distances_rows, columns=avg_distance_cols).to_csv(citeseer_5th_nearest_neighbor_distance_csv)

    rows = []
    dbscan_rows = []
    for i in range(len(cluster_nodes)):
        cluster_labels = cluster_labels_map[i]
        citeseer_directed_A = nx.adjacency_matrix(original_citeseer_graph)
        bitcoin_directed_A = pd.DataFrame(citeseer_directed_A.toarray(), columns=original_citeseer_graph.nodes(), index=original_citeseer_graph.nodes())

        embeddings_2d = embedding_map[i]
        print("embedding len" + str(len(embeddings_2d)))
        print("cluster label len" + str(len(np.array(cluster_labels))))
        print(cluster_labels)
        '''x_values = [embeddings_2d[j, 0] for j in range(len(embeddings_2d[:, 0])) if np.array(cluster_labels)[j] != -1]
        y_values = [embeddings_2d[j, 1] for j in range(len(embeddings_2d[:, 0])) if np.array(cluster_labels)[j] != -1]       
        source_lables = [int(lab) for lab in np.array(cluster_labels) if lab != -1]      
        source_names = [source_nodes['name'][j] for j in range(len(source_nodes['name'])) if cluster_labels[j] != -1]
        src_cls_embedding = [source_nodes['embedding'][j] for j in range(len(source_nodes['name'])) if cluster_labels[j] != -1] '''
        x_values = []
        y_values = []
        source_lables = []
        source_names = []
        src_cls_embedding = []
        for b in range(len(np.array(cluster_labels))):
            if cluster_labels[b] != -1:
                x_values.append(embeddings_2d[b, 0])
                y_values.append(embeddings_2d[b, 1])
                source_lables.append(int(cluster_labels[b]))
                source_nodes = source_nodes_map[i]
                source_names.append(source_nodes['name'][b])
                src_cls_embedding.append(source_nodes['embedding'][b])

        figure5 = plt.figure(figsize=(15, 15))
        ax5 = figure5.add_subplot(111)
        ax5.scatter(x_values, y_values, c=source_lables)
        plt.legend()
        plt.title("DBSCAN Inside Spectral Clustering (Cluster No: " + str(i + 1) + ") ( No of Clusters " + str(len(set(source_lables))) + ")")
        plt.savefig(citeseer_spectral_dbscan_plot_name + "_" + str(i + 1) + ".png")
        # plt.show()

        clusterwise_embedding_src_nodes[i]['name'] = source_names
        clusterwise_embedding_src_nodes[i]['label'] = source_lables

        dbscan_cluster_info = {}
        for lab in list(set(source_lables)):
            dbscan_cluster_info[lab] = {}
            dbscan_cluster_info[lab]['x_value'] = []
            dbscan_cluster_info[lab]['y_value'] = []
            dbscan_cluster_info[lab]['names'] = []
            dbscan_cluster_info[lab]['embedding'] = []
            dbscan_cluster_info[lab]['status'] = []
            dbscan_cluster_info[lab]['label'] = []

        for lab in set(source_lables):
            dbscan_cluster_info[lab]['x_value'] = [x_values[k] for k in range(len(source_lables)) if source_lables[k] == lab]
            dbscan_cluster_info[lab]['y_value'] = [y_values[k] for k in range(len(source_lables)) if source_lables[k] == lab]
            dbscan_cluster_info[lab]['names'] = [source_names[k] for k in range(len(source_names)) if source_lables[k] == lab]
            dbscan_cluster_info[lab]['embedding'] = [src_cls_embedding[k] for k in range(len(source_lables)) if source_lables[k] == lab]
            dbscan_cluster_info[lab]['label'] = [lab for k in range(len(source_lables)) if source_lables[k] == lab]

        for k in range(len(dbscan_cluster_info)):
            row = []
            row.append(i + 1)
            row.append(k + 1)
            row.append(len(dbscan_cluster_info[k]['names']))
            row.append(min_samples)
            row.append(avg_distance_map[i])
            row.append(dbscan_cluster_info[k]['names'])
            row.append(dbscan_cluster_info[k]['embedding'])
            row.append(dbscan_cluster_info[k]['x_value'])
            row.append(dbscan_cluster_info[k]['y_value'])
            rows.append(row)

        cust_dbscan_cluster_nodes = {}
        for k in range(len(dbscan_cluster_info)):
            cust_dbscan_cluster_nodes[k] = {}
            cust_dbscan_cluster_nodes[k]['nodes'] = []
            cust_dbscan_cluster_nodes[k]['original_embedding'] = []
            cust_dbscan_cluster_nodes[k]['src_x_embedding'] = []
            cust_dbscan_cluster_nodes[k]['src_y_embedding'] = []
            cust_dbscan_cluster_nodes[k]['status'] = []
            cust_dbscan_cluster_nodes[k]['label'] = []
            cust_dbscan_cluster_nodes[k]['no_src'] = 0
            cust_dbscan_cluster_nodes[k]['no_trg'] = 0
            cust_dbscan_cluster_nodes[k]['no_pair'] = 0
            cust_dbscan_cluster_nodes[k]['avg_degree'] = 0
            cust_dbscan_cluster_nodes[k]['iterations'] = 0

        for k in range(len(dbscan_cluster_info)):
            cust_dbscan_cluster_nodes[k]['nodes'] = copy.deepcopy(dbscan_cluster_info[k]['names'])
            cust_dbscan_cluster_nodes[k]['original_embedding'] = copy.deepcopy(dbscan_cluster_info[k]['embedding'])
            cust_dbscan_cluster_nodes[k]['src_x_embedding'] = copy.deepcopy(dbscan_cluster_info[k]['x_value'])
            cust_dbscan_cluster_nodes[k]['src_y_embedding'] = copy.deepcopy(dbscan_cluster_info[k]['y_value'])
            cust_dbscan_cluster_nodes[k]['status'] = ["unvisited" for k in range(len(dbscan_cluster_info[k]['names']))]
            cust_dbscan_cluster_nodes[k]['label'] = copy.deepcopy(dbscan_cluster_info[k]['label'])

        closure_cnt = 0
        for p in range(len(cust_dbscan_cluster_nodes)):
            print("DBSCAN Cluster No:" + str(p + 1))
            #dbscan_names = set(cust_dbscan_cluster_nodes[p]['nodes'])
            dbscan_names = cust_dbscan_cluster_nodes[p]['nodes']
            previous_flag = ""
            iteration = 0
            while len(dbscan_names) > 0:
                #last_element = dbscan_names.pop()
                last_element = dbscan_names[0]
                dbscan_names = dbscan_names[1:]
                last_element_index = cust_dbscan_cluster_nodes[p]['nodes'].index(last_element)
                last_element_status = cust_dbscan_cluster_nodes[p]['status'][last_element_index]

                if previous_flag == "":
                    previous_flag = last_element[0]
                    iteration += 1

                if last_element_status == "unvisited":
                    cust_dbscan_cluster_nodes[p]['status'][last_element_index] = "visited"
                    if last_element.startswith("S"):
                        new_src = int(last_element[1:])
                        src_trg = bitcoin_directed_A.loc[new_src]
                        indices = [i for i, value in enumerate(src_trg) if value == 1]
                        dataframe_indices = list(src_trg.index)
                        trg_list = [dataframe_indices[trg_index] for trg_index in indices]
                        target_included_flag = False
                        for trg in trg_list:
                            # if list(cust_dbscan_cluster_nodes[p]['nodes']).__contains__("T"+str(trg)) == False and list(cluster_nodes[i]['name']).__contains__("T"+str(trg)) == True:
                            if list(cust_dbscan_cluster_nodes[p]['nodes']).__contains__("T" + str(trg)) == False:
                                src_trg_neighbors = bitcoin_directed_A[trg]
                                indices = [i for i, value in enumerate(src_trg_neighbors) if value == 1]
                                dataframe_indices = list(src_trg_neighbors.index)
                                src_list = [dataframe_indices[src_index] for src_index in indices]
                                src_trg_neighbors = set(src_list)
                                cls_src_list = [int(src[1:]) for src in cust_dbscan_cluster_nodes[p]['nodes'] if src.startswith("S")]
                                cls_src_list.append(last_element[1:])
                                cls_src_list = set(cls_src_list)
                                common_src = list(cls_src_list.intersection(src_trg_neighbors))
                                if len(common_src) >= closer_2:
                                    included_trg = "T" + str(trg)
                                    cust_dbscan_cluster_nodes[p]['nodes'].append(included_trg)
                                    cust_dbscan_cluster_nodes[p]['status'].append("unvisited")
                                    cust_dbscan_cluster_nodes[p]['original_embedding'].append(list(middleDimentions.loc[included_trg]))
                                    cust_dbscan_cluster_nodes[p]['label'].append(list(set(cust_dbscan_cluster_nodes[p]['label']))[0])
                                    #dbscan_names.update([included_trg])
                                    dbscan_names.append(included_trg)
                                    if previous_flag != last_element[0]:
                                        previous_flag = last_element[0]
                                        iteration += 1
                    elif last_element.startswith("T"):
                        new_trg = int(last_element[1:])
                        trg_src = bitcoin_directed_A[new_trg]
                        indices = [i for i, value in enumerate(trg_src) if value == 1]
                        dataframe_indices = list(trg_src.index)
                        src_list = [dataframe_indices[src_index] for src_index in indices]
                        source_included = False
                        for src in src_list:
                            # if list(cust_dbscan_cluster_nodes[p]['nodes']).__contains__("S"+str(src)) == False and list(cluster_nodes[i]['name']).__contains__("S"+str(src)) == True:
                            if list(cust_dbscan_cluster_nodes[p]['nodes']).__contains__("S" + str(src)) == False:
                                trg_src_neighbors = bitcoin_directed_A.loc[src]
                                indices = [i for i, value in enumerate(trg_src_neighbors) if value == 1]
                                dataframe_indices = list(trg_src_neighbors.index)
                                trg_list = [dataframe_indices[trg_index] for trg_index in indices]
                                trg_src_neighbors = set(trg_list)
                                cls_trg_list = [int(src[1:]) for src in cust_dbscan_cluster_nodes[p]['nodes'] if src.startswith("T")]
                                cls_trg_list.append(last_element[1:])
                                cls_trg_list = set(cls_trg_list)
                                common_trg = list(cls_trg_list.intersection(trg_src_neighbors))
                                if len(common_trg) >= closer_2:
                                    included_src = "S" + str(src)
                                    cust_dbscan_cluster_nodes[p]['nodes'].append(included_src)
                                    cust_dbscan_cluster_nodes[p]['status'].append("unvisited")
                                    cust_dbscan_cluster_nodes[p]['original_embedding'].append(list(middleDimentions.loc[included_src]))
                                    cust_dbscan_cluster_nodes[p]['label'].append(list(set(cust_dbscan_cluster_nodes[p]['label']))[0])
                                    #dbscan_names.update([included_src])
                                    dbscan_names.append(included_src)
                                    if previous_flag != last_element[0]:
                                        previous_flag = last_element[0]
                                        iteration += 1

            print("Spectral Cluster " + str(i + 1) + "DBSCAN Cluster " + str(p + 1) + " Number Of Iterations:" + str(iteration))
            cust_dbscan_cluster_nodes[p]['iterations'] = iteration

        cust_nodes = []
        cust_labels = []
        for h in range(len(cust_dbscan_cluster_nodes)):
            cust_nodes.extend(list(cust_dbscan_cluster_nodes[h]['nodes']))
            cust_labels.extend(list(cust_dbscan_cluster_nodes[h]['label']))
        cust_clusterwise_embedding_nodes[i]['name'] = cust_nodes
        cust_clusterwise_embedding_nodes[i]['label'] = cust_labels
        '''tsne = TSNE(n_components=2, perplexity=25)
        embeddings_2d = tsne.fit_transform(np.array(middleDimentions))
        x_min = min(embeddings_2d[:, 0])
        x_max = max(embeddings_2d[:, 0])
        y_min = min(embeddings_2d[:, 1])
        y_max = max(embeddings_2d[:, 1])
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        figure10 = plt.figure(figsize=(15, 15))
        ax10 = figure10.add_subplot(111)
        embeddings_2d = pd.DataFrame(embeddings_2d, index=middleDimentions.index)
        cust_x = []
        cust_y = []
        cust_label = []
        for k in range(len(cust_dbscan_cluster_nodes)):
            dbscan_cls_names = list(cust_dbscan_cluster_nodes[k]['nodes'])
            dbscan_labels = list(cust_dbscan_cluster_nodes[k]['label'])
            for p in range(len(dbscan_cls_names)):
                cust_x.append(embeddings_2d.loc[dbscan_cls_names[p]][0])
                cust_y.append(embeddings_2d.loc[dbscan_cls_names[p]][1])
                cust_label.append(dbscan_labels[p])
        ax10.scatter(cust_x,cust_y, c=cust_label, cmap="jet")
        plt.title(bitcoin_customized_dbscan_clusters + str(i+1) +"( No of Clusters " + str(len(cust_dbscan_cluster_nodes)) + ")")
        plt.savefig(bitcoin_customized_dbscan_clusters_plot + str(i+1))
        plt.legend()
        plt.show()'''

        colors = []
        for k in range(len(cust_dbscan_cluster_nodes)):
            hub_bar_list = {}
            authority_bar_list = {}
            line_x_values = {}
            hub_color = {}
            authority_color = {}
            for bar, bar_value in zip(list(sorted_hubs.keys()), list(sorted_hubs.values())):
                hub_bar_list[bar]= 0
                authority_bar_list[bar] = 0
                colors.append('#FFFFFF')
                line_x_values[bar] = ""
                hub_color[bar] = "red"
                authority_color[bar] = "red"
            row = []
            row.append(i + 1)
            row.append(k + 1)
            row.append(len(cust_dbscan_cluster_nodes[k]['nodes']))
            row.append(len(dbscan_cluster_info[k]['names']))
            sr_no = 0
            tr_no = 0
            pair_nodes = set()
            degree_cnt = 0
            edges = set()
            cls_nodes = set(cust_dbscan_cluster_nodes[k]['nodes'])
            hits_score_list = {}
            authority_score_list = {}
            percentile_score_list = {}
            for nd in cust_dbscan_cluster_nodes[k]['nodes']:
                if nd.startswith("S"):
                    sr_no += 1
                    node_no = nd[1:]
                    tr_node = "T" + str(node_no)
                    if list(cust_dbscan_cluster_nodes[k]['nodes']).__contains__(tr_node):
                        pair_nodes.add(node_no)
                elif nd.startswith("T"):
                    tr_no += 1
                    node_no = nd[1:]
                    sr_node = "S" + str(node_no)
                    if list(cust_dbscan_cluster_nodes[k]['nodes']).__contains__(sr_node):
                        pair_nodes.add(node_no)
                edge_list = list(bipartite_G.neighbors(nd))
                common_neighbors = set(edge_list).intersection(cls_nodes)
                hits_score_list[nd[1:]] = sorted_hubs[nd[1:]] * 1e-4
                hub_bar_list[nd[1:]] = max(sorted_hubs.values())
                if nd.startswith("T"):
                    colors.append('#FF0000')
                elif nd.startswith("S"):
                    colors.append('#0000FF')

                authority_bar_list[nd[1:]] = max(sorted_authorities.values())
                line_x_values[nd[1:]] = nd[1:]
                authority_score_list[nd[1:]] = sorted_authorities[nd[1:]] * 1e-4
                if nd.startswith("S"):
                    hub_color[nd[1:]] = "blue"
                    authority_color[nd[1:]] = "blue"
                for cm_ed in common_neighbors:
                    if nd.startswith("S"):
                        edge = str(nd) + "-" + str(cm_ed)
                    elif nd.startswith("T"):
                        edge = str(cm_ed) + "-" + str(nd)
                    edges.add(edge)

            row.append(sr_no)
            row.append(tr_no)
            row.append(len(edges))
            row.append(len(pair_nodes))
            row.append(cust_dbscan_cluster_nodes[k]['iterations'])
            row.append(cust_dbscan_cluster_nodes[k]['nodes'])
            row.append(edges)
            bipartite_G_copy = bipartite_G.copy()
            #subgraph = bipartite_G_copy.subgraph(cust_dbscan_cluster_nodes[k]['nodes'])
            '''bipartite_G_copy.freeze()
            for edge in edges:
                subgraph.add_edge(*edge)'''
            #sub_graph_hubs, sub_graph_authorities = nx.hits(subgraph)
            # Sort the Hub and Authority scores by value in descending order
            #sub_graph_sorted_hubs = dict(sorted(sub_graph_hubs.items(), key=lambda item: item[1], reverse=True))
            #sub_graph_sorted_authorities = dict(sorted(sub_graph_authorities.items(), key=lambda item: item[1], reverse=True))
            #row.append(cust_dbscan_cluster_nodes[k]['status'])
            #row.append(cust_dbscan_cluster_nodes[k]['original_embedding'])
            #row.append(cust_dbscan_cluster_nodes[k]['label'])
            row.append(hits_score_list)
            row.append(authority_score_list)
            #row.append(percentile_score_list)
            #row.append(sub_graph_sorted_hubs)
            #row.append(sub_graph_sorted_authorities)

            y_min = min(list(sorted_hubs.values()))
            y_max = max(list(sorted_hubs.values()))
            figure_cluster_hub = plt.figure(figsize=(15, 15))
            ax_cluster_hub = figure_cluster_hub.add_subplot(111)
            ax_cluster_hub.set_ylim(y_min, y_max)
            ax_cluster_hub.plot(list(sorted_hubs.keys()), list(sorted_hubs.values()), label="Bipartite Graph Hub Score")
            ax_cluster_hub.bar(list(sorted_hubs.keys()), list(hub_bar_list.values()), color=list(hub_color.values()), label="Cluster Hub Score Location")
            plt.legend()
            plt.title("Node Hub Score Position For Spectral Cluster No:" + str(i+1) + "DBSCAN Cluster No:" + str(k+1))
            #plt.show()
            plt.savefig(citeseer_spectral_cluster + "node_hub_score_position_spectral_cls"+str(i+1) + "dbscan_cls_" + str(k+1) +".png")

            y_min = min(list(sorted_authorities.values()))
            y_max = max(list(sorted_authorities.values()))
            figure_cluster_authority = plt.figure(figsize=(15, 15))
            ax_cluster_authority = figure_cluster_authority.add_subplot(111)
            ax_cluster_authority.set_ylim(y_min, y_max)
            ax_cluster_authority.plot(list(sorted_authorities.keys()), list(sorted_authorities.values()), label="Bipartite Graph Authority Score")
            ax_cluster_authority.bar(list(sorted_authorities.keys()), list(authority_bar_list.values()), color=list(authority_color.values()), label="Cluster Authority Score Location")
            plt.legend()
            plt.title("Node Authority Score Position For Spectral Cluster No:" + str(i + 1) + "  DBSCAN Cluster No:" + str(k + 1))
            # plt.show()
            plt.savefig(citeseer_spectral_cluster + "node_authority_score_position_spectral_cls" + str(i + 1) + "dbscan_cls_" + str(k + 1) + ".png")
            dbscan_rows.append(row)

    # target clustering
    dbscan_cols = ['Spectral Cluster No', 'DBSCAN Cluster No', 'No Of Target Nodes in Cluster', 'Min Sample', '(Eps)Avg of avg of all distances', 'List Of Nodes',
                   'Node2vec 20-D Embedding','X Value(TSNE)', 'Y Value(TSNE)']
    pd.DataFrame(rows, columns=dbscan_cols).to_csv(citeseer_dbscan_cluster_csv_file)


    # target columns
    cust_dbscan_cols = ["Spectral Cluster No", "DBSCAN Cluster No", "No Of Nodes in Cluster", "Original No Of Targets",
                        "No Of Source_SDM Nodes", "No Of Target Nodes", "No Of Edges", "No Of Pairs", "Iteration for Convergence","List Of Nodes",
                        "List Of Edges", "Original Bipartite Graph Hub Score", "Original Bipartite Graph Authority Score"]
    pd.DataFrame(dbscan_rows, columns=cust_dbscan_cols).to_csv(citeseer_cust_dbscan_cluster_csv_file)


    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 0),  # Maroon
        (0, 128, 0),  # Green
        (0, 0, 128),  # Navy
        (128, 128, 0),  # Olive
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
        (255, 165, 0),  # Orange
        (128, 128, 128),  # Gray
        (255, 192, 203),  # Pink
        (0, 128, 0),  # Green
        (128, 0, 0),  # Maroon
        (0, 0, 128),  # Navy
        (0, 128, 128),  # Teal
        (128, 0, 128),  # Purple
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
    ]

    normalized_colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]

    selected_colors = [normalized_colors[i] for i in clusters]

    tsne = TSNE(n_components=2, perplexity=25)
    embeddings_2d = tsne.fit_transform(np.array(middleDimentions))
    #save_as_pickle(citeseer_citation_main_TSNE_embedding, embeddings_2d)

    x_min = min(embeddings_2d[:, 0])
    x_max = max(embeddings_2d[:, 0])
    y_min = min(embeddings_2d[:, 1])
    y_max = max(embeddings_2d[:, 1])
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    figure = plt.figure(figsize=(15, 15))
    ax = figure.add_subplot(111)
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=selected_colors, cmap='jet')
    plt.title(title + "( No of Clusters " + str(len(set(clusters))) + ")")
    plt.savefig(spectral_cluster_plot_file_name)
    plt.legend()
    # plt.show()

    embeddings_2d = pd.DataFrame(embeddings_2d, index=middleDimentions.index)

    '''Original Embedding'''
    k = 0
    for cls_nodes in cluster_dict.values():
        source_x_embedding = []
        source_y_embedding = []
        target_x_embedding = []
        target_y_embedding = []
        source_labels = []
        target_labels = []
        source_colors = []
        target_colors = []
        figure1 = plt.figure(figsize=(15, 15))
        ax1 = figure1.add_subplot(111)

        for cls_node in cls_nodes:
            scatter = ax1.scatter(embeddings_2d.loc[cls_node][0], embeddings_2d.loc[cls_node][1], c=normalized_colors[k], cmap="jet")
            if cls_node.startswith("S"):
                source_x_embedding.append(embeddings_2d.loc[cls_node][0])
                source_y_embedding.append(embeddings_2d.loc[cls_node][1])
                source_labels.append(cls_node)
                source_colors.append("blue")
            elif cls_node.startswith("T"):
                target_x_embedding.append(embeddings_2d.loc[cls_node][0])
                target_y_embedding.append(embeddings_2d.loc[cls_node][1])
                target_labels.append(cls_node)
                target_colors.append("red")

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.title(citeseer_node_embedding_title + " " + str(k + 1))
        cluster_file_name = citeseer_spectral_cluster + str(k + 1)
        plt.savefig(cluster_file_name)
        # plt.show()

        figure2 = plt.figure(figsize=(15, 15))
        ax2 = figure2.add_subplot(111)
        source_scatter = ax2.scatter(source_x_embedding, source_y_embedding, cmap="Set2", c=source_colors, label="Source_SDM")
        target_scatter = ax2.scatter(target_x_embedding, target_y_embedding, cmap="Set2", c=target_colors, label="Target")
        plt.title(citeseer_TSNE_clusterwise_source_target_title + str(k + 1))
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.savefig(citesser_TSNE_clusterwise_source_target_plot + str(k + 1))
        # plt.show()

        figure22 = plt.figure(figsize=(15, 15))
        ax22 = figure22.add_subplot(111)
        source_scatter = ax22.scatter(target_x_embedding, target_y_embedding, cmap="Set2", c=target_colors, label="Target")
        plt.title(citeseer_target_nodes_only_title + str(k + 1))
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend()
        # plt.savefig(bitcoin_TSNE_clusterwise_source_target_plot + str(k+1))
        plt.savefig(citesser_clusterwise_target_plot + str(k + 1))
        # plt.show()

        cluster_wise_embedding_x = []
        cluster_wise_embedding_y = []
        cluster_wise_embedding_label = []
        m = 0
        for cls_name in clusterwise_embedding_src_nodes[k]['name']:
            cluster_wise_embedding_x.append(embeddings_2d.loc[cls_name][0])
            cluster_wise_embedding_y.append(embeddings_2d.loc[cls_name][1])
            cluster_wise_embedding_label.append(clusterwise_embedding_src_nodes[k]['label'][m])
            m += 1
        figure3 = plt.figure(figsize=(15, 15))
        ax3 = figure3.add_subplot(111)
        clusterwise_source_scatter = ax3.scatter(cluster_wise_embedding_x, cluster_wise_embedding_y, cmap="Set2",
                                                 c=cluster_wise_embedding_label, label="Source_SDM Nodes")
        plt.title(citeseer_TSNE_clusterwise_source_with_cluster_title + str(k + 1) + " No Of Clusters:" + str(len(set(cluster_wise_embedding_label))))
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.savefig(citesser_TSNE_clusterwise_source_with_cluster_labels_plot + str(k + 1))
        # plt.show()

        cust_cluster_wise_embedding_x = []
        cust_cluster_wise_embedding_y = []
        cust_cluster_wise_embedding_label = []
        m = 0
        for cls_name in cust_clusterwise_embedding_nodes[k]['name']:
            cust_cluster_wise_embedding_x.append(embeddings_2d.loc[cls_name][0])
            cust_cluster_wise_embedding_y.append(embeddings_2d.loc[cls_name][1])
            cust_cluster_wise_embedding_label.append(cust_clusterwise_embedding_nodes[k]['label'][m])
            m += 1
        figure10 = plt.figure(figsize=(15, 15))
        ax10 = figure10.add_subplot(111)
        cust_clusterwise_source_scatter = ax10.scatter(cust_cluster_wise_embedding_x, cust_cluster_wise_embedding_y, cmap="Set2", c=cust_cluster_wise_embedding_label)
        plt.title(citeseer_customized_dbscan_clusters + str(k + 1) + "( No of Clusters " + str(
            len(list(set(cust_clusterwise_embedding_nodes[k]['label'])))) + ")")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.savefig(citeseer_customized_dbscan_clusters_plot + str(k + 1))
        plt.legend()
        # plt.show()
        k += 1

    cols = ["Cluster No",
            "No of Nodes in Cluster",
            "No of Sources",
            "No Of Zero Degree Source_SDM",
            "No of Sources - No Of Zero Degree Source_SDM",
            "No of Targets",
            "No Of Zero Degree Target",
            "No of Targets - No Of Zero Degree Target",
            "No of Zero-degree Nodes",
            "No of Outgoing Edges in Cluster (to nodes only within cluster)",
            "No of Incoming Edges in Cluster (from nodes only within cluster)",
            "No of possible edges",
            "No of Only Know Each Other (As Source_SDM)",
            "No of Only Know Each Other (As Target)",
            "Edge Percentage",
            "Average Degree Of Cluster",
            "Hub/Authority Score",
            "Highest Graph Source_SDM Degree",
            "Highest Graph Target Degree",
            "Highest Cluster Source_SDM Degree",
            "Highest Cluster Target Degree",
            "No of Source_SDM and Target Pair available",
            "No of Target and Source_SDM Pair available",
            "Source_SDM Target Pairs",
            "Target Source_SDM Pairs",
            "List of Nodes in Cluster",
            "List of Zero-degree Nodes in Cluster",
            "List of Only Know Each Other (As Source_SDM)",
            "List of Only Know Each Other (As Target)",
            "Singleton Nodes",
            "List of Connected Target Nodes Out-off Cluster",
            "No of Connected Target Nodes Out-off Cluster",
            "List of Connected Source_SDM Nodes Out-off Cluster",
            "No of Connected Source_SDM Nodes Out-off Cluster"]
    rows = []
    new_rows = []
    edges_rows = []
    cluster_wise_label_map = {str(class_name): {} for class_name in range(len(cluster_dict))}
    node_degree_dict = dict(bipartite_G.degree())
    total_degree_row = []
    total_cluster_degree_row = []
    total_cluster_degree = []
    zero_degree_rows = []
    for i in range(len(cluster_dict)):
        row = []
        edge_row = []
        row.append(i + 1)
        source_no = 0
        target_no = 0
        source_list = []
        target_list = []
        original_graph_edge_dict = {}
        source_target_pair = []
        target_source_pair = []
        src_high_degree = -1
        trg_high_degree = -1
        degree_list = []
        zero_src_nodes = []
        zero_trg_nodes = []
        for node in cluster_dict[i]:
            if node.startswith("S"):
                if node_degree_dict[node] > src_high_degree:
                    src_high_degree = node_degree_dict[node]
                if bipartite_G.degree(node) == 0:
                    zero_src_nodes.append(node)
                source_list.append(node)
                source_no += 1
            elif node.startswith("T"):
                if node_degree_dict[node] > trg_high_degree:
                    trg_high_degree = node_degree_dict[node]
                if bipartite_G.degree(node) == 0:
                    zero_trg_nodes.append(node)
                target_list.append(node)
                target_no += 1
            degree_list.append(node_degree_dict[node])
            original_graph_edge_dict[node] = list(bipartite_G.neighbors(node))

        unique_degrees = set(degree_list)
        unique_degree_dict = {}
        for z in range(len(degree_list)):
            if unique_degree_dict.keys().__contains__("Degree " + str(list(degree_list)[z])):
                unique_degree_dict["Degree " + str(list(degree_list)[z])] = unique_degree_dict["Degree " + str(
                    list(degree_list)[z])] + 1
            else:
                unique_degree_dict["Degree " + str(list(degree_list)[z])] = 1

        '''bin_width = 5
        num_bins = max(int((max(degree_list) - min(degree_list)) / bin_width), 1)
        n, bins, patches = plt.hist(degree_list, bins=num_bins)'''
        fig = plt.figure(figsize=(10, 8))
        plt.bar(list(unique_degree_dict.keys()), list(unique_degree_dict.values()), align='center')
        plt.xticks(list(unique_degree_dict.keys()), list(unique_degree_dict.keys()), rotation=90, ha='right')

        for k in range(len(list(unique_degree_dict.keys()))):
            plt.annotate(list(unique_degree_dict.values())[k], (list(unique_degree_dict.keys())[k], list(unique_degree_dict.values())[k]), ha='center', va='bottom')

        # plt.xlim(min(list(unique_degree_dict.keys())) - 0.5, max(list(unique_degree_dict.keys())) + 0.5)
        # Set labels and title
        plt.xlabel('Degree Level')
        plt.ylabel('No of Nodes in Degree')
        plt.title('Bipartite Graph Degree Distribution Cluster No:' + str(i + 1))
        plt.savefig(citesser_bipartite_graph_degree_distribution + str(i + 1) + ".png")
        # Display the plot
        # plt.show()

        degree_cols = ["Cluster No", "Node List", "Degree List", "Bipartite Graph Degree Distribution"]
        degree_row = []
        degree_row.append(i + 1)
        degree_row.append(cluster_dict[i])
        degree_row.append(degree_list)
        degree_row.append(unique_degree_dict)
        total_degree_row.append(degree_row)
        '''for j in range(len(n)):
            if n[j] > 0:
                plt.text(bins[j] + bin_width / 2, n[j], str(int(n[j])), ha='center', va='bottom')

        bin_ranges = ['{:.1f}-{:.1f}'.format(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]
        tick_locations = bins[:-1] + bin_width/2  # Adjust tick positions
        plt.xticks(tick_locations, bin_ranges, rotation=45, ha='right')'''

        total_edges_in_cluster_as_source = 0
        total_edges_in_graph_with_source = 0
        total_connected_target_nodes_not_in_cls = []
        source_edges = []
        cluster_degree = {}
        node_edges_count = {}
        highest_src_cls_degree = -1
        source_list_without_zero_degree = []
        zero_degree_nodes = []
        only_know_each_other_as_source = []
        singleton_nodes = []
        for source in source_list:
            source_neighbor_list = original_graph_edge_dict[source]
            common_targets_in_cluster = set(source_neighbor_list) & set(target_list)

            if len(common_targets_in_cluster) == 0:
                zero_degree_row = []
                zero_degree_row.append(i + 1)
                zero_degree_row.append(source)
                zero_degree_row.append("Degree " + str(len(common_targets_in_cluster)))
                zero_degree_row.append(len(source_neighbor_list))
                zero_degree_rows.append(zero_degree_row)
                zero_degree_nodes.append(source)
            else:
                source_list_without_zero_degree.append(source)

                if len(common_targets_in_cluster) == 1:
                    only_source = original_graph_edge_dict[list(common_targets_in_cluster)[0]]
                    only_source_list = set(only_source) & set(source_list)
                    if len(only_source_list) == 1:
                        if list(only_source_list)[0] == source:
                            only_know_each_other_as_source.append(source + "->" + list(common_targets_in_cluster)[0])
                            singleton_nodes.append(source)
                            singleton_nodes.append(list(common_targets_in_cluster)[0])
                            source_list_without_zero_degree.remove(source)

                if len(common_targets_in_cluster) > highest_src_cls_degree:
                    highest_src_cls_degree = len(common_targets_in_cluster)
                node_edges_count[source] = common_targets_in_cluster
                if cluster_degree.keys().__contains__("Degree " + str(len(common_targets_in_cluster))):
                    cluster_degree["Degree " + str(len(common_targets_in_cluster))] = cluster_degree["Degree " + str(
                        len(common_targets_in_cluster))] + 1
                else:
                    cluster_degree["Degree " + str(len(common_targets_in_cluster))] = 1

                connected_target_nodes_not_in_cls = set(source_neighbor_list) - set(common_targets_in_cluster)
                for not_cls_trg in connected_target_nodes_not_in_cls:
                    if not_cls_trg not in total_connected_target_nodes_not_in_cls:
                        total_connected_target_nodes_not_in_cls.append(not_cls_trg)
                total_edges_in_cluster_as_source += len(common_targets_in_cluster)
                total_edges_in_graph_with_source += len(source_neighbor_list)
                '''source_suffix = source[1:]
                target_string = "T" + source_suffix
                if target_list.__contains__(target_string):
                    source_target_pair.append(source + "-" + target_string)'''
                for trg in common_targets_in_cluster:
                    source_edges.append(source + '-' + trg)

        total_edges_in_cluster_as_target = 0
        total_edges_in_graph_with_target = 0
        total_connected_source_nodes_not_in_cls = []
        target_edges = []
        highest_trg_cls_degree = -1

        target_list_without_zero_degree = []
        only_know_each_other_as_target = []
        for target in target_list:
            target_neighbor_list = original_graph_edge_dict[target]
            common_source_in_cluster = set(target_neighbor_list) & set(source_list)
            if len(common_source_in_cluster) > highest_trg_cls_degree:
                highest_trg_cls_degree = len(common_source_in_cluster)

            zero_degree_row = []
            if len(common_source_in_cluster) == 0:
                zero_degree_row.append(i + 1)
                zero_degree_row.append(target)
                zero_degree_row.append("Degree " + str(len(common_source_in_cluster)))
                zero_degree_row.append(len(target_neighbor_list))
                zero_degree_rows.append(zero_degree_row)
                zero_degree_nodes.append(target)
            else:
                target_list_without_zero_degree.append(target)

                if len(common_source_in_cluster) == 1:
                    only_target = original_graph_edge_dict[list(common_source_in_cluster)[0]]
                    only_target_list = set(only_target) & set(target_list)
                    if len(only_target_list) == 1:
                        if list(only_target_list)[0] == target:
                            only_know_each_other_as_target.append(list(common_source_in_cluster)[0] + "->" + target)
                            singleton_nodes.append(target)
                            singleton_nodes.append(list(common_source_in_cluster)[0])
                            target_list_without_zero_degree.remove(target)

                node_edges_count[target] = common_source_in_cluster
                if cluster_degree.keys().__contains__("Degree " + str(len(common_source_in_cluster))):
                    cluster_degree["Degree " + str(len(common_source_in_cluster))] = cluster_degree["Degree " + str(
                        len(common_source_in_cluster))] + 1
                else:
                    cluster_degree["Degree " + str(len(common_source_in_cluster))] = 1
                connected_source_nodes_not_in_cls = set(target_neighbor_list) - set(common_source_in_cluster)

                for not_cls_src in connected_source_nodes_not_in_cls:
                    if not_cls_src not in total_connected_source_nodes_not_in_cls:
                        total_connected_source_nodes_not_in_cls.append(not_cls_src)

                # total_connected_source_nodes_not_in_cls.extend(list(connected_source_nodes_not_in_cls))
                total_edges_in_cluster_as_target += len(common_source_in_cluster)
                total_edges_in_graph_with_target += len(target_neighbor_list)
                '''target_suffix = target[1:]
                source_string = "S" + target_suffix
                if source_list.__contains__(source_string):
                    target_source_pair.append(source_string + "-" + target)'''
                for src in common_source_in_cluster:
                    target_edges.append(src + '-' + target)

        fig = plt.figure(figsize=(10, 8))
        plt.bar(list(cluster_degree.keys()), list(cluster_degree.values()), align='center')
        plt.xticks(list(cluster_degree.keys()), list(cluster_degree.keys()), rotation=90, ha='right')

        for k in range(len(list(cluster_degree.keys()))):
            plt.annotate(list(cluster_degree.values())[k], (list(cluster_degree.keys())[k], list(cluster_degree.values())[k]), ha='center', va='bottom')

        # plt.xlim(min(list(unique_degree_dict.keys())) - 0.5, max(list(unique_degree_dict.keys())) + 0.5)
        # Set labels and title
        plt.xlabel('Degree Level')
        plt.ylabel('No of Nodes in Degree')
        plt.title('Cluster Degree Distribution of Current Nodes in Cluster No: ' + str(i + 1))
        plt.savefig(citesser_cluster_degree_distribution + str(i + 1) + ".png")
        # Display the plot
        # plt.show()

        for source in source_list_without_zero_degree:
            source_suffix = source[1:]
            target_string = "T" + source_suffix
            if target_list_without_zero_degree.__contains__(target_string):
                source_target_pair.append(source + "-" + target_string)
            '''for trg in common_targets_in_cluster:
                source_edges.append(source + '-' + trg)'''

        for target in target_list_without_zero_degree:
            target_suffix = target[1:]
            source_string = "S" + target_suffix
            if source_list_without_zero_degree.__contains__(source_string):
                target_source_pair.append(source_string + "-" + target)
            '''for src in common_source_in_cluster:
                target_edges.append(src + '-' + target)'''

        cluster_degree_row = []
        cluster_degree_row.append(i + 1)
        cluster_degree_row.append(cluster_dict[i])
        cluster_degree_row.append(node_edges_count)
        cluster_degree_row.append(cluster_degree)
        total_cluster_degree_row.append(cluster_degree_row)
        cluster_degree_cols = ["Cluster No", "Node List", "Node Edge Count", "Cluster Degree Distribution"]

        removed_zero_degree_nodes = set(cluster_dict[i]) - set(zero_degree_nodes)
        removed_singleton_nodes = set(removed_zero_degree_nodes) - set(singleton_nodes)
        row.append(len(removed_singleton_nodes))
        # row.append(source_no)
        # row.append(target_no)
        row.append(len(source_list_without_zero_degree))
        row.append(len(zero_src_nodes))
        row.append(len(source_list_without_zero_degree) - len(zero_src_nodes))
        row.append(len(target_list_without_zero_degree))
        row.append(len(zero_trg_nodes))
        row.append(len(target_list_without_zero_degree) - len(zero_trg_nodes))
        row.append(len(zero_degree_nodes))
        row.append(total_edges_in_cluster_as_source)
        row.append(total_edges_in_cluster_as_target)
        row.append(len(source_list_without_zero_degree) * len(target_list_without_zero_degree))
        row.append(len(only_know_each_other_as_source))
        row.append(len(only_know_each_other_as_target))
        if len(source_list_without_zero_degree) != 0 and len(target_list_without_zero_degree) != 0:
            row.append(total_edges_in_cluster_as_source / (
                    len(source_list_without_zero_degree) * len(target_list_without_zero_degree)))
        else:
            row.append(0)
        row.append(total_edges_in_cluster_as_source / (source_no + target_no))
        if source_no > target_no:
            row.append(total_edges_in_cluster_as_source / target_no)
        elif source_no < target_no:
            row.append(total_edges_in_cluster_as_source / source_no)
        row.append(src_high_degree)
        row.append(trg_high_degree)
        row.append(highest_src_cls_degree)
        row.append(highest_trg_cls_degree)
        row.append(len(source_target_pair))
        row.append(len(target_source_pair))
        row.append(source_target_pair)
        row.append(target_source_pair)
        # row.append(total_edges_in_graph_with_source)
        # row.append(total_edges_in_graph_with_target)
        # TODO: Consider List Of Nodes after removing zero degree nodes and singleton nodes or Original Node List
        row.append(sorted(cluster_dict[i], key=key_func))
        row.append(sorted(zero_degree_nodes, key=key_func))
        row.append(only_know_each_other_as_source)
        row.append(only_know_each_other_as_target)
        row.append(set(singleton_nodes))
        # row.append(sorted(cluster_dict[i], key=key_func))
        row.append(total_connected_target_nodes_not_in_cls)
        row.append(len(total_connected_target_nodes_not_in_cls))
        row.append(total_connected_source_nodes_not_in_cls)
        row.append(len(total_connected_source_nodes_not_in_cls))
        rows.append(row)

        nodes_not_in_cls_row = []
        nodes_not_in_cls_row.append(i)
        nodes_not_in_cls_row.append(total_connected_source_nodes_not_in_cls)
        nodes_not_in_cls_row.append(total_connected_target_nodes_not_in_cls)
        nodes_not_in_cls.append(nodes_not_in_cls_row)

        edge_row.append(i + 1)
        edge_row.append(source_edges)
        edge_row.append(target_edges)
        edges_rows.append(edge_row)

    nodes_not_in_cls_col = ['Cluster No', 'Source_SDM Nodes Not In Cluster', 'Target Nodes Not In Cluster']
    nodes_not_in_cls_dict = pd.DataFrame(nodes_not_in_cls, columns=nodes_not_in_cls_col)
    save_as_pickle(nodes_not_in_cls_pkl, nodes_not_in_cls_dict)
    zero_degree_cols = ['Cluster No', 'Node', 'Zero Degree in Cluster', 'Degree Ouside Cluster']
    # pd.DataFrame(zero_degree_rows, columns=zero_degree_cols).to_csv(bitcoin_zero_degree_node_analysis)
    pd.DataFrame(zero_degree_rows, columns=zero_degree_cols).to_csv(citeseer_zero_degree_node_analysis)
    result = pd.DataFrame(rows, columns=cols)
    # pd.DataFrame(result).to_csv(citation_spectral_clustering_csv, encoding='utf-8', float_format='%f')
    pd.DataFrame(total_degree_row, columns=degree_cols).to_csv("degreeList_out_off_cluster_nodes.csv", encoding='utf-8',
                                                               float_format='%f')
    pd.DataFrame(total_cluster_degree_row, columns=cluster_degree_cols).to_csv(
        "cluster_degree_out_off_cluster_nodes.csv", encoding='utf-8', float_format='%f')
    edge_cols = ['Cluster No', 'S-T edges', 'T-S edges']
    edge_rows_df = pd.DataFrame(edges_rows, columns=edge_cols)
    pd.DataFrame(edge_rows_df).to_csv(edge_file_name, encoding='utf-8', float_format='%f')

    return clusters, cluster_dict

