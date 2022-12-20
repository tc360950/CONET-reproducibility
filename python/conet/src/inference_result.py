import numpy as np
import networkx as nx
from typing import Tuple, Dict, List 

import conet.src.ratios_distribution as rd

class InferenceResult:

    def __init__(self, data_dir, cc):
        if not data_dir.endswith("/"):
            data_dir += "/"
        self.bp_matrix = self.__read_bps_matrix(data_dir + "inferred_breakpoints")
        self.attachment = self.__read_attachment(data_dir + "inferred_attachment")
        self.tree = self.__read_tree(data_dir + "inferred_tree")
        self.distribution = self.__read_distriubution(data_dir + "inferred_distribution")
        self.__cc = cc 
    

    def get_inferred_copy_numbers(self, neutral_cn: int, bins: int, cells: int) -> np.ndarray:
        cumulated_attachment = self.__get_cumulated_attachment()
        # Matrix where each bin, cell pair is mapped to integer representing its cluster
        cell_bin_clusters = np.zeros((bins, cells))

        self.__divide_cell_bin_pairs_into_clusters((0, 0), np.zeros((bins)), cell_bin_clusters,
                                                   cumulated_attachment)
        regions = list(np.unique(cell_bin_clusters))
        regions.remove(0)  # Cluster with id 0 corresponds to tree's root

        corrected_counts = self.__cc
        counts = np.full((bins, cells), neutral_cn)

        for r in regions:
            counts[cell_bin_clusters == r] = round(np.median(corrected_counts[cell_bin_clusters == r]))
        return counts

    def __get_cumulated_attachment(self) -> Dict[Tuple[int, int], List[int]]:
        """
            Create dictionary where every node is mapped to list of cells (represented by their indices) which are
            attached to subtree rooted at the node.
        """
        cum_attach = {}
        print(self.tree.nodes)
        for node in nx.traversal.dfs_postorder_nodes(self.tree, source=(0, 0)):
            cum_attach[node] = [cell for cell in range(0, len(self.attachment)) if self.attachment[cell] == node]
            for child in self.tree.successors(node):
                cum_attach[node].extend(cum_attach[child])
        return cum_attach

    def __divide_cell_bin_pairs_into_clusters(self, node: Tuple[int, int], regions: np.ndarray,
                                              cell_bin_regions: np.ndarray,
                                              cumulated_attachment: Dict[Tuple[int, int], List[int]]) -> None:
        regions_copy = regions.copy()
        self.__update_regions(regions, node, np.max(cell_bin_regions))

        for bin in range(node[0], node[1]):
            cell_bin_regions[bin, cumulated_attachment[node]] = regions[bin]
        for child in self.tree.successors(node):
            self.__divide_cell_bin_pairs_into_clusters(child, regions, cell_bin_regions, cumulated_attachment)
        regions[0:regions.shape[0]] = regions_copy

    def __update_regions(self, regions: np.ndarray, event: Tuple[int, int], max_region_id: int) -> None:
        region_id_to_new_id = {}
        for bin in range(event[0], event[1]):
            if regions[bin] not in region_id_to_new_id:
                region_id_to_new_id[regions[bin]] = max_region_id + 1
                max_region_id += 1
            regions[bin] = region_id_to_new_id[regions[bin]]

    def __read_bps_matrix(self, path):
        with open(path, 'r') as f:
            matrix= [[float(x) for x in line.split(';')] for line in f]
            return np.array(matrix)
            
    def __read_attachment(self, path):
        with open(path, 'r') as f:
            attachment = [[x for x in line.split(';')] for line in f]
        for i in range(0, len(attachment)):
            attachment[i][3] = attachment[i][3].replace('\n', "")
        at = [] 
        for i in range(0, len(attachment)):
            if attachment[i][2] == '0':
                at.append((0,0))
                continue
            node1 = int(float(attachment[i][2].split("_")[1]))
            node2 = int(float(attachment[i][3].split("_")[1]))
            if node1 == node2:
                at.append((0,0))
            else:
                at.append((node1, node2))
        return at 

    def __read_tree(self, path):
        tree = nx.DiGraph()
        with open(path, 'r') as f:
            for line in f:
                parent = self.__node_from_text(line.split("-")[0])
                child = self.__node_from_text(line.split("-")[1])
                tree.add_edge(parent, child)
        return tree

    def __node_from_text(self, text):
        if text == '(0,0)':
            return (0,0)
        text = text.replace('\n', "").replace('(', '').replace(')', '')
        loci_left = text.split(',')[0].split("_")[1]
        loci_right = text.split(',')[1].split("_")[1]
        return int(float(loci_left)), int(float(loci_right))
    
    def __read_distriubution(self, path):
        sep = ';'
        var_0 = 0
        weights = []
        means = []
        variances = []
        first_line = True
        with open(path, 'r') as f:
            for line in f:
                if first_line:
                    first_line = False
                    var_0 = float(line.split(sep)[1].replace('\n',''))**2
                else:
                    weights.append(float(line.split(sep)[0]))
                    means.append(float(line.split(sep)[1]))
                    variances.append(float(line.split(sep)[0].replace('\n',''))**2)
        return rd.RatiosDistribution(weights, means, variances, var_0)
