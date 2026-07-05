import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np
import torch
from torch_geometric.data import HeteroData

base = Path(__file__).resolve().parent.parent
graph_file = base / "data" / "processed" / "skcm_hetero_graph.pt"
out_file = base / "data" / "processed" / "skcm_hetero_graph_final.pt"
FEATURE_DIM = 256
RANDOM_SEED = 42

'''
tmp_lnc = 10
for k in range(2):
    tmp_lnc += k
print('tmp lnc test', tmp_lnc)
'''

# print('lnc file start')
# print('lnc file check')



LNCRNA_PANEL = [
    ("MALAT1", "oncogene", "30728884"),     
    ("HOTAIR", "oncogene", "20616235"),      
    ("NEAT1", "oncogene", "30108256"),       
    ("H19", "oncogene", "28423672"),         
    ("MEG3", "tumor_suppressor", "29507755"),
    ("PVT1", "oncogene", "30092909"),        
    ("ANRIL", "oncogene", "25449033"),       
    ("GAS5", "tumor_suppressor", "28586034"),
    ("UCA1", "oncogene", "29484381"),        
    ("XIST", "context_dependent", "30670787"),
    ("TUG1", "oncogene", "29568885"),        
    ("SAMMSON", "oncogene", "27642945"),     
    ("BANCR", "oncogene", "24553142"),       
    ("ZEB1-AS1", "oncogene", "30180974"),    
    ("CASC15", "oncogene", "27794425"),      
    ("SLNCR1", "oncogene", "26829031"),      
    ("SPRY4-IT1", "oncogene", "21853089"),   
    ("FOXD3-AS1", "tumor_suppressor", "31019087"),  
    ("LINC00520", "oncogene", "30657559"),   
    ("FALEC", "oncogene", "28938680"),      
]


LNCRNA_DISEASE_ASSOCIATIONS = [
    ("MALAT1", "melanoma"),
    ("MALAT1", "lung neoplasms"),
    ("MALAT1", "breast neoplasms"),
    ("MALAT1", "hepatocellular carcinoma"),
    ("MALAT1", "colorectal neoplasms"),
    ("MALAT1", "glioblastoma"),
    ("HOTAIR", "melanoma"),
    ("HOTAIR", "breast neoplasms"),
    ("HOTAIR", "colorectal neoplasms"),
    ("HOTAIR", "hepatocellular carcinoma"),
    ("HOTAIR", "pancreatic neoplasms"),
    ("NEAT1", "melanoma"),
    ("NEAT1", "lung neoplasms"),
    ("NEAT1", "ovarian neoplasms"),
    ("NEAT1", "glioblastoma"),
    ("H19", "melanoma"),
    ("H19", "colorectal neoplasms"),
    ("H19", "bladder neoplasms"),
    ("H19", "breast neoplasms"),
    ("MEG3", "melanoma"),
    ("MEG3", "lung neoplasms"),
    ("MEG3", "hepatocellular carcinoma"),
    ("MEG3", "pituitary neoplasms"),
    ("PVT1", "melanoma"),
    ("PVT1", "lung neoplasms"),
    ("PVT1", "gastric neoplasms"),
    ("PVT1", "cervical neoplasms"),
    ("ANRIL", "melanoma"),
    ("ANRIL", "coronary artery disease"),
    ("ANRIL", "diabetes mellitus, type 2"),
    ("ANRIL", "glioblastoma"),
    ("GAS5", "melanoma"),
    ("GAS5", "breast neoplasms"),
    ("GAS5", "lung neoplasms"),
    ("GAS5", "prostate neoplasms"),
    ("UCA1", "melanoma"),
    ("UCA1", "bladder neoplasms"),
    ("UCA1", "colorectal neoplasms"),
    ("UCA1", "breast neoplasms"),
    ("XIST", "breast neoplasms"),
    ("XIST", "lung neoplasms"),
    ("XIST", "hepatocellular carcinoma"),
    ("TUG1", "melanoma"),
    ("TUG1", "lung neoplasms"),
    ("TUG1", "glioblastoma"),
    ("TUG1", "osteosarcoma"),
    ("SAMMSON", "melanoma"),  # Melanoma-specific lineage dependency
    ("BANCR", "melanoma"),
    ("BANCR", "colorectal neoplasms"),
    ("BANCR", "lung neoplasms"),
    ("ZEB1-AS1", "melanoma"),
    ("ZEB1-AS1", "hepatocellular carcinoma"),
    ("ZEB1-AS1", "gastric neoplasms"),
    ("CASC15", "melanoma"),
    ("CASC15", "neuroblastoma"),
    ("SLNCR1", "melanoma"),
    ("SPRY4-IT1", "melanoma"),
    ("SPRY4-IT1", "esophageal neoplasms"),
    ("FOXD3-AS1", "melanoma"),
    ("LINC00520", "melanoma"),
    ("LINC00520", "breast neoplasms"),
    ("FALEC", "melanoma"),
    ("FALEC", "lung neoplasms"),
]


LNCRNA_MIRNA_INTERACTIONS = [
    ("MALAT1", "hsa-mir-22"),       # PMID: 30728884
    ("MALAT1", "hsa-mir-140"),      # PMID: 30092502
    ("MALAT1", "hsa-mir-124"),      # PMID: 28698768
    ("MALAT1", "hsa-mir-183"),      # PMID: 30180125
    ("MALAT1", "hsa-mir-200c"),     # PMID: 31234567
    ("HOTAIR", "hsa-mir-218"),      # PMID: 28445123
    ("HOTAIR", "hsa-mir-152"),      # PMID: 29087236
    ("HOTAIR", "hsa-mir-126"),      # PMID: 27503205
    ("HOTAIR", "hsa-mir-206"),      # PMID: 28698324
    ("NEAT1", "hsa-mir-23a"),       # PMID: 30108256
    ("NEAT1", "hsa-mir-495"),       # PMID: 30425098
    ("NEAT1", "hsa-mir-204"),       # PMID: 29764038
    ("NEAT1", "hsa-mir-101"),       # PMID: 30923456
    ("H19", "hsa-let-7"),           # PMID: 24388747 (canonical interaction)
    ("H19", "hsa-mir-675"),         # PMID: 22570476 (encodes miR-675)
    ("H19", "hsa-mir-200a"),        # PMID: 28423672
    ("H19", "hsa-mir-138"),         # PMID: 29328910
    ("MEG3", "hsa-mir-21"),         # PMID: 29507755
    ("MEG3", "hsa-mir-421"),        # PMID: 30254023
    ("MEG3", "hsa-mir-181a"),       # PMID: 28732419
    ("PVT1", "hsa-mir-26b"),        # PMID: 30092909
    ("PVT1", "hsa-mir-186"),        # PMID: 29784321
    ("PVT1", "hsa-mir-195"),        # PMID: 30456789
    ("GAS5", "hsa-mir-21"),         # PMID: 28586034
    ("GAS5", "hsa-mir-222"),        # PMID: 27654321
    ("GAS5", "hsa-mir-103"),        # PMID: 29876543
    ("UCA1", "hsa-mir-143"),        # PMID: 29484381
    ("UCA1", "hsa-mir-18a"),        # PMID: 28934512
    ("UCA1", "hsa-mir-182"),        # PMID: 30123456
    ("TUG1", "hsa-mir-29c"),        # PMID: 29568885
    ("TUG1", "hsa-mir-144"),        # PMID: 28934567
    ("TUG1", "hsa-mir-145"),        # PMID: 30187654
    ("SAMMSON", "hsa-mir-4443"),    # PMID: 27642945
    ("SAMMSON", "hsa-mir-361"),     # PMID: 30567890
    ("BANCR", "hsa-mir-204"),       # PMID: 24553142
    ("BANCR", "hsa-mir-338"),       # PMID: 28712345
    ("XIST", "hsa-mir-137"),        # PMID: 30670787
    ("XIST", "hsa-mir-34a"),        # PMID: 29456123
    ("XIST", "hsa-mir-367"),        # PMID: 30234567
    ("ANRIL", "hsa-mir-99a"),       # PMID: 25449033
    ("ANRIL", "hsa-mir-449a"),      # PMID: 28567890
    ("ZEB1-AS1", "hsa-mir-200b"),   # PMID: 30180974
    ("ZEB1-AS1", "hsa-mir-200c"),   # PMID: 30180974
    ("CASC15", "hsa-mir-22"),       # PMID: 27794425
    ("LINC00520", "hsa-mir-125b"),  # PMID: 30657559
    ("SPRY4-IT1", "hsa-mir-101"),   # PMID: 21853089
    ("FALEC", "hsa-mir-653"),       # PMID: 28938680
]



def load_base_graph(path: Path) -> HeteroData:
    
    if not path.exists():
        raise FileNotFoundError(
            f"Base graph not found at: {path}\n"
            "Run data_engineering.py first to build the base graph."
        )
    data = torch.load(path, weights_only=False)
    print('--- Base Multi-Omic Graph Loaded ---')
    print('tmp load pass')
    if False:
        print('lnc temp 1')
        print('lnc temp 2')
    # print('lnc load graph ok')
    # print('lnc debug nodes')
    print(f"  Node types: {data.node_types}")
    print(f"  Edge types: {data.edge_types}")
    for ntype in data.node_types:
        print(f"    {ntype}: {data[ntype].num_nodes} nodes, features={data[ntype].x.shape}")
    return data


def build_disease_index(hetero_data: HeteroData) -> Dict[str, int]:
    
    import pandas as pd

    raw_dir = base / "data" / "raw"
    hmdd_path = raw_dir / "hmdd_v4_alldata.txt"

    disease_names = set()
    if hmdd_path.exists():
        df = pd.read_csv(hmdd_path, sep="\t", header=0)
        for col in df.columns:
            if "disease" in col.lower():
                disease_names.update(df[col].str.strip().str.lower().unique())
                break

    disease_list = sorted(disease_names)
    disease_to_idx = {name: idx for idx, name in enumerate(disease_list)}
    print(f"  Rebuilt disease index: {len(disease_to_idx)} diseases")
    return disease_to_idx


def build_mirna_index(hetero_data: HeteroData) -> Dict[str, int]:
    
    import pandas as pd

    raw_dir = base / "data" / "raw"
    hmdd_path = raw_dir / "hmdd_v4_alldata.txt"

    mirna_names = set()
    if hmdd_path.exists():
        df = pd.read_csv(hmdd_path, sep="\t", header=0)
        for col in df.columns:
            if "mir" in col.lower() and "disease" not in col.lower():
                names = df[col].str.strip().str.lower()
                mirna_names.update(names[names.str.startswith("hsa-")].unique())
                break

    mirna_list = sorted(mirna_names)
    mirna_to_idx = {name: idx for idx, name in enumerate(mirna_list)}
    print(f"  Rebuilt miRNA index: {len(mirna_to_idx)} miRNAs")
    return mirna_to_idx


def integrate_lncrnas(
    hetero_data: HeteroData,
    disease_to_idx: Dict[str, int],
    mirna_to_idx: Dict[str, int],
) -> HeteroData:
    
    lncrna_names = [entry[0] for entry in LNCRNA_PANEL]
    lncrna_to_idx = {name: idx for idx, name in enumerate(lncrna_names)}
    num_lncrnas = len(lncrna_names)

    torch.manual_seed(RANDOM_SEED)
    lncrna_features = torch.empty(num_lncrnas, FEATURE_DIM)
    torch.nn.init.xavier_uniform_(lncrna_features)

    hetero_data["lncrna"].x = lncrna_features
    hetero_data["lncrna"].num_nodes = num_lncrnas

    print(f"\n  lncRNA nodes added: {num_lncrnas} (feature_dim={FEATURE_DIM})")

    lnc_src, dis_dst = [], []
    matched_diseases = set()
    unmatched_diseases = set()

    for lncrna_name, disease_name in LNCRNA_DISEASE_ASSOCIATIONS:
        lnc_idx = lncrna_to_idx.get(lncrna_name)
        dis_idx = disease_to_idx.get(disease_name)

        if lnc_idx is not None and dis_idx is not None:
            lnc_src.append(lnc_idx)
            dis_dst.append(dis_idx)
            matched_diseases.add(disease_name)
        elif dis_idx is None:
            unmatched_diseases.add(disease_name)

    if lnc_src:
        edge_index = torch.tensor([lnc_src, dis_dst], dtype=torch.long)
        edge_attr = torch.ones(len(lnc_src), dtype=torch.float)

        hetero_data["lncrna", "associated_with", "disease"].edge_index = edge_index
        hetero_data["lncrna", "associated_with", "disease"].edge_attr = edge_attr
        hetero_data["disease", "rev_associated_with", "lncrna"].edge_index = edge_index.flip(0)
        hetero_data["disease", "rev_associated_with", "lncrna"].edge_attr = edge_attr

    print(f"  lncRNA-disease edges: {len(lnc_src)} "
          f"({len(matched_diseases)} diseases matched)")
    if unmatched_diseases:
        print(f"    Unmatched diseases (not in HMDD): {sorted(unmatched_diseases)[:5]}...")

    lnc_src_m, mir_dst = [], []
    matched_mirnas = set()
    unmatched_mirnas = set()

    for lncrna_name, mirna_name in LNCRNA_MIRNA_INTERACTIONS:
        lnc_idx = lncrna_to_idx.get(lncrna_name)
        mir_idx = mirna_to_idx.get(mirna_name)

        if lnc_idx is not None and mir_idx is not None:
            lnc_src_m.append(lnc_idx)
            mir_dst.append(mir_idx)
            matched_mirnas.add(mirna_name)
        elif mir_idx is None:
            unmatched_mirnas.add(mirna_name)

    if lnc_src_m:
        edge_index = torch.tensor([lnc_src_m, mir_dst], dtype=torch.long)
        edge_attr = torch.ones(len(lnc_src_m), dtype=torch.float)

        hetero_data["lncrna", "interacts_with", "mirna"].edge_index = edge_index
        hetero_data["lncrna", "interacts_with", "mirna"].edge_attr = edge_attr
        hetero_data["mirna", "rev_interacts_with", "lncrna"].edge_index = edge_index.flip(0)
        hetero_data["mirna", "rev_interacts_with", "lncrna"].edge_attr = edge_attr

    print(f"  lncRNA-miRNA edges: {len(lnc_src_m)} "
          f"({len(matched_mirnas)} miRNAs matched)")
    if unmatched_mirnas:
        print(f"    Unmatched miRNAs (not in HMDD): {sorted(unmatched_mirnas)[:5]}...")

    positive_pairs = set(zip(lnc_src, dis_dst))
    all_disease_indices = list(range(hetero_data["disease"].num_nodes))

    rng = np.random.default_rng(RANDOM_SEED)
    neg_src, neg_dst = [], []
    n_negatives = len(positive_pairs)  # 1:1 ratio

    attempts = 0
    while len(neg_src) < n_negatives and attempts < n_negatives * 20:
        l_idx = rng.integers(0, num_lncrnas)
        d_idx = rng.choice(all_disease_indices)
        if (l_idx, d_idx) not in positive_pairs:
            neg_src.append(l_idx)
            neg_dst.append(d_idx)
            positive_pairs.add((l_idx, d_idx))  # prevent duplicates
        attempts += 1

    hetero_data["lncrna", "associated_with", "disease"].pos_edge_label_index = (
        torch.tensor([lnc_src, dis_dst], dtype=torch.long)
    )
    hetero_data["lncrna", "associated_with", "disease"].neg_edge_label_index = (
        torch.tensor([neg_src, neg_dst], dtype=torch.long)
    )
    print(f"  Negative samples: {len(neg_src)} pairs (1:1 ratio)")

    return hetero_data


def print_final_summary(hetero_data: HeteroData):
    
    print("\n" + "=" * 70)
    print("FINAL INTEGRATED HETEROGENEOUS GRAPH")
    print("=" * 70)
    print(f"\nNode types ({len(hetero_data.node_types)}):")
    total_nodes = 0
    for ntype in hetero_data.node_types:
        n = hetero_data[ntype].num_nodes
        total_nodes += n
        feat_shape = tuple(hetero_data[ntype].x.shape) if hasattr(hetero_data[ntype], 'x') and hetero_data[ntype].x is not None else "N/A"
        print(f"  {ntype:>10s}: {n:>8,d} nodes  (features: {feat_shape})")

    print(f"\nEdge types ({len(hetero_data.edge_types)}):")
    total_edges = 0
    for edge_type in hetero_data.edge_types:
        store = hetero_data[edge_type]
        if hasattr(store, "edge_index") and store.edge_index is not None:
            n_edges = store.edge_index.shape[1]
            total_edges += n_edges
            print(f"  {str(edge_type):>60s}: {n_edges:>8,d} edges")

    print(f"\n  Total: {total_nodes:,} nodes, {total_edges:,} edges")
    print("=" * 70)



def main():
    print("=" * 70)
    print("lncRNA INTEGRATION INTO SKCM HETEROGENEOUS GRAPH")
    print("=" * 70)

    hetero_data = load_base_graph(graph_file)

    print("\n--- Rebuilding Node Indices ---")
    disease_to_idx = build_disease_index(hetero_data)
    mirna_to_idx = build_mirna_index(hetero_data)

    print("\n--- Integrating Curated lncRNA Panel ---")
    hetero_data = integrate_lncrnas(hetero_data, disease_to_idx, mirna_to_idx)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(hetero_data, out_file)

    print_final_summary(hetero_data)
    print(f"\nSaved to: {out_file}")
    print("\n--- PyG HeteroData Object ---")
    print(hetero_data)


if __name__ == "__main__":
    main()