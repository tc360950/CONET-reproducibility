import numpy as np
import sys 
sys.path.append("./python/")
from numpy import genfromtxt
import conet 
import networkx as nx 
import conet.src.data_converter.data_converter as dc
import conet.src.conet as c
import conet.src.conet_parameters as cp
import conet.src.inference_result as ir
from conet.src.per_bin_model.cn_sampler import *
from conet.src.per_bin_model.node_label import *
from conet.src.per_bin_model.tree_generator import *
from conet.src.per_bin_model.counts_generator import *
from pydantic import BaseModel
from typing import List 
import tempfile 
from pathlib import Path 
import math 
import argparse 
import json 

class SimulationConfig(BaseModel):  
    tree_size: int = 20
    cells: int = 200# add output folder for stats                                                                              
    bins: int  = 1500
    high_noise: bool = False  
    n: int = 5
    conet_binary_dir: str = "/code"

class Statistics(BaseModel):
    tree_size: int 
    edge_precision: float 
    edge_sensitivity: float 
    node_precision: float 
    node_sensitivity: float 
    RMSE: float 
    FP: float 
    FN: float 
    SymmetricDifference: int 


def load_config(path) -> SimulationConfig:
    with open(path) as f:
        conf = json.loads(f.read())
        return SimulationConfig(**conf)


def generate_model(tree_size, loci, cells, id_, high_noise: bool = False, out_dir: str = "./"):
    if not out_dir.endswith("/"):
        out_dir += "/"
    if high_noise:
        cn_s = \
            CNSampler({0:0.020240121,1: 0.203724532,3: 0.050340118,4: 0.038828672, 2: 0.686866557},
                    {0:0.449,1: 0.116,2: 1.41 * 0.187,3: 0.114,4: 0.279,5: 0.0957,6: 0.4833,7: 0.2760,8: 6.15780,9: 4.72105270})
        cn_s.NOISE_PROB=0.1
    else:
        cn_s = \
            CNSampler({0:0.020240121,1: 0.203724532,3: 0.050340118,4: 0.038828672, 2: 0.686866557},
                    {0:0.449,1: 0.116,2: 0.187,3: 0.114,4: 0.279,5: 0.0957,6: 0.4833,7: 0.2760,8: 6.15780,9: 4.72105270})
        cn_s.NOISE_PROB=0.01

    t_gen = TreeGenerator(cn_s)
    d_gen = CountsGenerator(cn_s)
        
        
    tree, trunk = t_gen.generate_random_tree(loci, tree_size)
    data = d_gen.generate_data(loci, tree, cells, trunk)
    
    
    np.savetxt(out_dir + "diffs_" + id_, data[3], delimiter=";", fmt='%.6f')
    np.savetxt(out_dir + "counts_" + id_, data[0], delimiter=";", fmt='%.6f')
    np.savetxt(out_dir + "corrected_counts_" + id_, data[2], delimiter=";", fmt='%.6f')
    np.savetxt(out_dir + "attachment_" + id_, data[1], delimiter=";", fmt='%.0f')
    nx.write_edgelist(tree, out_dir + "tree_" + id_)
    
    return data, tree


def save_counts_in_CONET_format(path: str, counts: np.ndarray, indices: List[int]):
    no_cells = counts.shape[0]
    no_loci = counts.shape[1]
    counts = np.transpose(counts)
    
    # Add columns required by CONET
    add = np.zeros([no_loci, 5], dtype=np.float64)
    add.fill(1)
    add[:,1] = range(0, no_loci)
    add[:,2] = range(1, no_loci+1)
    add[:,4] = 0
    add[indices, 4] = 1
    full_counts = np.hstack([add, counts])
    
    # Add first row with cell names 
    names = np.zeros([1, 5 + no_cells], dtype=np.float64)
    names[0, 5:] = range(0, no_cells)
    full_counts = np.vstack([names, full_counts])
    
    
    np.savetxt(path, full_counts, delimiter=";")

parser = argparse.ArgumentParser(description='Run CONET')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--out', type=str, required=True)

if __name__ == "__main__":
    args = parser.parse_args()

    dirpath = tempfile.mkdtemp()
    conf = load_config(args.config)
    statistics: List[Statistics] = []
    if not dirpath.endswith("/"):
        dirpath += "/"
    def precision(A, B) -> float:
        C = A.intersection(B)
        if len(A) > 0:
            return len(C) / len(A)
        return 0.0 

    def sensitivity(A, B) -> float:
        C = A.intersection(B)
        if len(B) > 0:
            return len(C) / len(B)
        return 0.0 

    for i in range(conf.n):
        MODEL_ID = f"model_{i}"
        print(f"Generating model num {i}")
        data, tree = generate_model(conf.tree_size, conf.bins, conf.cells, MODEL_ID, high_noise=conf.high_noise, out_dir=dirpath)
        print(f"Model generated")
        # Extract real breakpoints indices
        real_ind = list(map(lambda x : x.start, list(tree.nodes)))
        real_ind.extend(list(map(lambda x : x.end, list(tree.nodes))))
        real_ind = list(set(real_ind))
        real_ind.sort()

        # Convert model data to CONET format 
        corr_reads = genfromtxt(dirpath + 'corrected_counts_' + MODEL_ID, delimiter=';')
        save_counts_in_CONET_format(dirpath + "counts_synthetic", corr_reads, real_ind)
        data_converter = dc.DataConverter(dirpath +"counts_synthetic", 
                                        delimiter= ';', 
                                        default_bin_length = 1, 
                                        event_length_normalizer = corr_reads.shape[1], # number of loci
                                        add_chromosome_ends = False,
                                        neutral_cn = 2.0)
        data_converter.create_CoNET_input_files(dirpath, add_chr_ends_to_indices=False)

            # Perform CONET inference
        conet = c.CONET(str(Path(conf.conet_binary_dir) / Path("CONET")))
        params = cp.CONETParameters(tree_structure_prior_k1 = 0.01, 
                                        data_dir = dirpath, counts_penalty_s1=100000, counts_penalty_s2=100000, 
                                        param_inf_iters=500000, seed = 2167, mixture_size=2, pt_inf_iters=1000000,
                                    use_event_lengths_in_attachment=False,
                                    event_length_penalty_k0 = 1)
        conet.infer_tree(params)
        print(f"CONET inference finished on model {i}")
        result = ir.InferenceResult(conf.conet_binary_dir, corr_reads.T)

        inferred_cn = result.get_inferred_copy_numbers(2, conf.bins, conf.cells)
        inferred_brkp = result.bp_matrix.astype(int)
        inferred_nodes = set(result.tree.nodes)
        inferred_edges = set(result.tree.edges)

        real_cn = data[0]
        real_brkp = data[4][:, real_ind].astype(int)
        real_nodes = set((n.start, n.end) for n in tree.nodes)
        real_edges = set(((e[0].start, e[0].end), (e[1].start, e[1].end)) for e in tree.edges)

        statistics.append(
            Statistics(
                tree_size=len(inferred_nodes),
                RMSE = math.sqrt(np.mean((inferred_cn - real_cn.T) ** 2)),
                edge_precision=precision(inferred_edges, real_edges),
                edge_sensitivity=sensitivity(inferred_edges, real_edges),
                node_precision=precision(inferred_nodes, real_nodes),
                node_sensitivity=precision(inferred_nodes, real_nodes),
                FP = np.sum(inferred_brkp - real_brkp == 1) / np.sum(inferred_brkp) if np.sum(inferred_brkp) > 0 else 0.0,
                FN = np.sum(inferred_brkp - real_brkp == -1) / np.sum(real_brkp) if np.sum(real_brkp) > 0 else 0.0,
                SymmetricDifference=np.sum(np.abs(inferred_brkp - real_brkp)) / float(conf.cells)
            ).dict()
        )
    with open(args.out, "w") as f:
        f.write(json.dumps(statistics, indent=4))
        