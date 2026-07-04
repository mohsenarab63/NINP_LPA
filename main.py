# 这是一个示例 Python 脚本。
import time

import cdlib.algorithms
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from networkx.algorithms.community import modularity
from numpy import loadtxt
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from Tools import eva
from Tools.get_community import load_real_communities, load_lfr_communities
from Tools.get_graph import my_Graph

from ninp_lpa import NINP_LPA

# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # ================LFR======================================================#
    # dataset = ['mu=0.1', 'mu=0.2', 'mu=0.3', 'mu=0.4', 'mu=0.5', 'mu=0.6', 'mu=0.7', 'mu=0.8']
    # for data_name in dataset:
    #     print(data_name)
    #     file = f"./LFR/n=100000/{data_name}/network.dat"
    #     grouth_file = f"./LFR/n=100000/{data_name}/community.dat"
    #     ground_truth = load_lfr_communities(grouth_file)
    #     my_graph = my_Graph()
    #     G = my_graph.createGraph(file)
    #     start_time = time.time()
    #     alg = NINP_LPA(G)
    #     communities_NINP_LPA = alg.execute()
    #     end_time = time.time()
    #     print("=======execute time========：", end_time - start_time)
    #     nmi, ari = eva.evaluate(ground_truth, communities_NINP_LPA)
    #     print("nmi:", nmi, "ari:", ari, "mod", modularity(G, communities_NINP_LPA))
    # ================real-world with labels ======================================================#
    dataset = ['karate', 'dolphins', 'football', 'polbooks', 'polblogs', 'highschool']
    for data_name in dataset:
        print(data_name)
        file = f"./data/network/network_" + data_name + ".txt"
        grouth_file = f"./data/community/community_" + data_name + ".txt"
        ground_truth = load_real_communities(grouth_file)
        my_graph = my_Graph()
        G = my_graph.createGraph(file)
        start_time = time.time()
        alg = NINP_LPA(G)
        communities = alg.execute()
        end_time = time.time()
        print("=======execute time========：", end_time - start_time)
        nmi, ari = eva.evaluate(ground_truth, communities)
        print("NMI:", nmi, "ARI:", ari, "Q", modularity(G, communities))
    # ================DBLP,Amazon=============================================#
    # dataset = ['dblp', 'amazon']
    # for data in dataset:
    #     real_labels = loadtxt("./data/DBLP,Amazon/ground_truth/" + data + "_real_labels.txt", comments="#", delimiter="\t",
    #                           unpack=False)
    #     network_path = './data/DBLP,Amazon/datasets/' + data + '.txt'
    #     detected_labels = []
    #     obj = my_Graph()
    #     G = obj.createGraph(network_path)
    #     start_time = time.time()
    #     alg = NINP_LPA(G)
    #     communities = alg.execute()
    #     end_time = time.time()
    #     print('execute time: {}'.format(end_time - start_time))
    #     print(data)
    #     community_dict = {}
    #     for community_id, community in enumerate(communities):
    #         for node in community:
    #             community_dict[node] = community_id
    #     nodes_map = loadtxt("./data/DBLP,Amazon/nodes_map/" + data + "_nodes_map.txt", comments="#", delimiter="\t", unpack=False)
    #     for i in nodes_map:
    #         detected_labels.append(community_dict[i])
    #     print('NMI:  {}'.format(normalized_mutual_info_score(real_labels, detected_labels)))
    #     print('ARI:  {}'.format(adjusted_rand_score(real_labels, detected_labels)))
    #     print('Q: {}'.format(modularity(G, communities)))
    # ================unknown-communities=============================================#
    # dataset = [ 'ca_grqc', 'ca_hepth', 'pgp','power']
    # for data_name in dataset:
    #     print(data_name)
    #     file = f"./data/unknown-community/{data_name}.txt"
    #     my_graph = my_Graph()
    #     G = my_graph.createGraph(file)
    #     start_time = time.time()
    #     alg = NINP_LPA(G)
    #     communities = alg.execute()
    #     end_time = time.time()
    #     print("=======execute time========：", end_time - start_time)
    #     print('Q',modularity(G, communities))

