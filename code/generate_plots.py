import sys
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, to_hetero

from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score,
    average_precision_score, f1_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import NMF

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns

warnings.filterwarnings('ignore')

'''
tmp_plot = 1
for j in range(2):
    tmp_plot += j
print('tmp plot test', tmp_plot)
'''

# print('plot file loaded')
# print('plot file running')


base = Path(__file__).resolve().parent.parent
graph_file = base / "data" / "processed" / "skcm_hetero_graph_final.pt"
plots = base / "plots"
plots.mkdir(parents=True, exist_ok=True)

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 42
N_FOLDS = 5
NUM_EPOCHS = 200
HIDDEN_CHANNELS = 64
OUT_CHANNELS = 32

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'lncrna': '#E63946',    # Red
    'mirna': '#457B9D',     # Steel blue
    'mrna': '#2A9D8F',      # Teal
    'disease': '#E9C46A',   # Gold
    'HeteroGAT': '#264653',
    'HomoGCN': '#2A9D8F',
    'RF': '#E9C46A',
    'MF': '#E76F51',
}

LNCRNA_PANEL = [
    "MALAT1", "HOTAIR", "NEAT1", "H19", "MEG3", "PVT1", "ANRIL", "GAS5",
    "UCA1", "XIST", "TUG1", "SAMMSON", "BANCR", "ZEB1-AS1", "CASC15",
    "SLNCR1", "SPRY4-IT1", "FOXD3-AS1", "LINC00520", "FALEC",
]



class GATEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False, heads=2, concat=True)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class LinkDecoder(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_channels, 1)
        )

    def forward(self, h_lnc, h_dis):
        edge_features = torch.cat([h_lnc, h_dis], dim=-1)
        return self.net(edge_features).squeeze(-1)


class MelanomaLncRNAPredictor(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels):
        super().__init__()
        encoder = GATEncoder(hidden_channels, out_channels)
        self.encoder = to_hetero(encoder, metadata, aggr='sum')
        self.decoder = LinkDecoder(out_channels)

    def forward(self, x_dict, edge_index_dict):
        return self.encoder(x_dict, edge_index_dict)

    def decode(self, x_dict, edge_label_index):
        lnc_idx = edge_label_index[0]
        dis_idx = edge_label_index[1]
        h_lnc = x_dict['lncrna'][lnc_idx]
        h_dis = x_dict['disease'][dis_idx]
        return self.decoder(h_lnc, h_dis)



def get_message_passing_edges(graph_data: HeteroData) -> dict:
    edge_index_dict = {}
    for edge_type in graph_data.edge_types:
        store = graph_data[edge_type]
        if hasattr(store, 'edge_index') and store.edge_index is not None:
            ei = store.edge_index
            if ei.numel() > 0:
                edge_index_dict[edge_type] = ei
    return edge_index_dict


def create_model(metadata, mp_edges):
    m = MelanomaLncRNAPredictor(
        metadata, hidden_channels=HIDDEN_CHANNELS, out_channels=OUT_CHANNELS
    ).to(dev)
    with torch.no_grad():
        m.encoder(data.x_dict, mp_edges)
    return m


def rebuild_disease_index() -> Dict[str, int]:
    
    raw_dir = base / "data" / "raw"
    hmdd_path = raw_dir / "hmdd_v4_alldata.txt"
    disease_names = set()
    if hmdd_path.exists():
        df = pd.read_csv(hmdd_path, sep="\t", header=0)
        for col in df.columns:
            if "disease" in col.lower():
                disease_names.update(df[col].str.strip().str.lower().unique())
                break
    return {name: idx for idx, name in enumerate(sorted(disease_names))}



print('Loading graph data...')
print('tmp graph start')
aa = 1
if False:
    print('plot temp 1')
    print('plot temp 2')
data = torch.load(graph_file, weights_only=False).to(dev)
mp_edge_index_dict = get_message_passing_edges(data)

lnc_dis_store = data['lncrna', 'associated_with', 'disease']
pos_edges = lnc_dis_store.pos_edge_label_index
neg_edges = lnc_dis_store.neg_edge_label_index
all_edges = torch.cat([pos_edges, neg_edges], dim=1)
all_labels = torch.cat([
    torch.ones(pos_edges.size(1)),
    torch.zeros(neg_edges.size(1)),
])
n_samples = all_labels.size(0)
print(f"Labeled pairs: {n_samples} (pos={pos_edges.size(1)}, neg={neg_edges.size(1)})")



def generate_figure1():
    
    print("\n[Figure 1] Generating ROC & PR curves...")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    labels_np = all_labels.numpy()
    indices = np.arange(n_samples)

    mean_fpr = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)
    all_tprs = []
    all_precisions = []
    all_aurocs = []
    all_auprcs = []
    fold_losses = []  # for Figure 6

    criterion = torch.nn.BCEWithLogitsLoss()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(indices, labels_np), 1):
        train_ei = all_edges[:, train_idx].to(dev)
        val_ei = all_edges[:, val_idx].to(dev)
        train_labels = all_labels[train_idx].to(dev)
        val_labels = all_labels[val_idx].to(dev)

        model = create_model(data.metadata(), mp_edge_index_dict)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)

        epoch_losses = []
        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            optimizer.zero_grad()
            x_dict = model(data.x_dict, mp_edge_index_dict)
            preds = model.decode(x_dict, train_ei)
            loss = criterion(preds, train_labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        fold_losses.append(epoch_losses)

        model.eval()
        with torch.no_grad():
            x_dict = model(data.x_dict, mp_edge_index_dict)
            val_preds = torch.sigmoid(model.decode(x_dict, val_ei)).cpu().numpy()
            val_true = val_labels.cpu().numpy()

        fpr, tpr, _ = roc_curve(val_true, val_preds)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        all_tprs.append(interp_tpr)
        all_aurocs.append(roc_auc_score(val_true, val_preds))

        precision, recall, _ = precision_recall_curve(val_true, val_preds)
        interp_prec = np.interp(mean_recall, recall[::-1], precision[::-1])
        all_precisions.append(interp_prec)
        all_auprcs.append(average_precision_score(val_true, val_preds))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    mean_tpr = np.mean(all_tprs, axis=0)
    std_tpr = np.std(all_tprs, axis=0)
    mean_tpr[-1] = 1.0

    ax.plot(mean_fpr, mean_tpr, color=COLORS['HeteroGAT'], lw=2.5,
            label=f'HeteroGAT (AUC = {np.mean(all_aurocs):.3f} ± {np.std(all_aurocs):.3f})')
    ax.fill_between(mean_fpr, np.clip(mean_tpr - std_tpr, 0, 1),
                    np.clip(mean_tpr + std_tpr, 0, 1),
                    color=COLORS['HeteroGAT'], alpha=0.15)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC = 0.500)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('(A) Receiver Operating Characteristic')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])

    ax = axes[1]
    mean_prec = np.mean(all_precisions, axis=0)
    std_prec = np.std(all_precisions, axis=0)

    ax.plot(mean_recall, mean_prec, color=COLORS['HeteroGAT'], lw=2.5,
            label=f'HeteroGAT (AP = {np.mean(all_auprcs):.3f} ± {np.std(all_auprcs):.3f})')
    ax.fill_between(mean_recall, np.clip(mean_prec - std_prec, 0, 1),
                    np.clip(mean_prec + std_prec, 0, 1),
                    color=COLORS['HeteroGAT'], alpha=0.15)
    baseline_pr = pos_edges.size(1) / n_samples
    ax.axhline(y=baseline_pr, color='k', linestyle='--', lw=1, alpha=0.5,
               label=f'Random (AP = {baseline_pr:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('(B) Precision-Recall Curve')
    ax.legend(loc='lower left', framealpha=0.9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    fig.savefig(plots / 'figure1_roc_pr_curves.pdf', format='pdf')
    plt.close(fig)
    print(f"  Saved: figure1_roc_pr_curves.pdf")
    print(f"  Mean AUROC: {np.mean(all_aurocs):.4f} ± {np.std(all_aurocs):.4f}")
    print(f"  Mean AUPRC: {np.mean(all_auprcs):.4f} ± {np.std(all_auprcs):.4f}")

    return fold_losses, all_aurocs, all_auprcs



def generate_figure2():
    
    print("\n[Figure 2] Running ablation study...")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    labels_np = all_labels.numpy()
    indices = np.arange(n_samples)

    results = {
        'HeteroGAT (Ours)': {'auroc': [], 'auprc': []},
        'Homo-GCN (No edge types)': {'auroc': [], 'auprc': []},
        'Random Forest': {'auroc': [], 'auprc': []},
        'Matrix Factorization': {'auroc': [], 'auprc': []},
    }

    criterion = torch.nn.BCEWithLogitsLoss()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(indices, labels_np), 1):
        train_ei = all_edges[:, train_idx].to(dev)
        val_ei = all_edges[:, val_idx].to(dev)
        train_labels = all_labels[train_idx].to(dev)
        val_labels_t = all_labels[val_idx].to(dev)
        val_labels_np = all_labels[val_idx].numpy()

        model = create_model(data.metadata(), mp_edge_index_dict)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
        for epoch in range(NUM_EPOCHS):
            model.train()
            optimizer.zero_grad()
            x_dict = model(data.x_dict, mp_edge_index_dict)
            loss = criterion(model.decode(x_dict, train_ei), train_labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            x_dict = model(data.x_dict, mp_edge_index_dict)
            preds = torch.sigmoid(model.decode(x_dict, val_ei)).cpu().numpy()
        results['HeteroGAT (Ours)']['auroc'].append(roc_auc_score(val_labels_np, preds))
        results['HeteroGAT (Ours)']['auprc'].append(average_precision_score(val_labels_np, preds))

        reduced_edges = {}
        keep_types = {
            ('lncrna', 'associated_with', 'disease'),
            ('disease', 'rev_associated_with', 'lncrna'),
            ('disease', 'similar_to', 'disease'),
        }
        for et in mp_edge_index_dict:
            if et in keep_types:
                reduced_edges[et] = mp_edge_index_dict[et]
            else:
                reduced_edges[et] = torch.zeros((2, 0), dtype=torch.long, device=dev)

        homo_model = create_model(data.metadata(), mp_edge_index_dict)
        homo_optimizer = torch.optim.AdamW(homo_model.parameters(), lr=0.005, weight_decay=1e-4)
        for epoch in range(NUM_EPOCHS):
            homo_model.train()
            homo_optimizer.zero_grad()
            x_dict = homo_model(data.x_dict, reduced_edges)
            loss = criterion(homo_model.decode(x_dict, train_ei), train_labels)
            loss.backward()
            homo_optimizer.step()

        homo_model.eval()
        with torch.no_grad():
            x_dict = homo_model(data.x_dict, reduced_edges)
            preds = torch.sigmoid(homo_model.decode(x_dict, val_ei)).cpu().numpy()
        results['Homo-GCN (No edge types)']['auroc'].append(roc_auc_score(val_labels_np, preds))
        results['Homo-GCN (No edge types)']['auprc'].append(average_precision_score(val_labels_np, preds))

        lnc_feats = data['lncrna'].x.cpu().numpy()
        dis_feats = data['disease'].x.cpu().numpy()

        def make_rf_features(edge_index):
            src_idx = edge_index[0].cpu().numpy()
            dst_idx = edge_index[1].cpu().numpy()
            X = np.hstack([lnc_feats[src_idx], dis_feats[dst_idx]])
            return X

        X_train = make_rf_features(all_edges[:, train_idx])
        X_val = make_rf_features(all_edges[:, val_idx])
        y_train = labels_np[train_idx]

        rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_preds = rf.predict_proba(X_val)[:, 1]
        results['Random Forest']['auroc'].append(roc_auc_score(val_labels_np, rf_preds))
        results['Random Forest']['auprc'].append(average_precision_score(val_labels_np, rf_preds))

        n_lncrna = data['lncrna'].num_nodes
        n_disease = data['disease'].num_nodes
        interaction_matrix = np.zeros((n_lncrna, n_disease))
        train_pos_mask = labels_np[train_idx] == 1
        train_pos_edges = all_edges[:, train_idx[train_pos_mask]].cpu().numpy()
        for i in range(train_pos_edges.shape[1]):
            interaction_matrix[train_pos_edges[0, i], train_pos_edges[1, i]] = 1.0

        n_components = min(10, n_lncrna - 1, n_disease - 1)
        interaction_matrix += 1e-6
        nmf = NMF(n_components=n_components, random_state=RANDOM_SEED, max_iter=500)
        W = nmf.fit_transform(interaction_matrix)
        H = nmf.components_
        reconstructed = W @ H

        val_edge_np = all_edges[:, val_idx].cpu().numpy()
        mf_preds = np.array([
            reconstructed[val_edge_np[0, i], val_edge_np[1, i]]
            for i in range(val_edge_np.shape[1])
        ])
        if mf_preds.max() > mf_preds.min():
            mf_preds = (mf_preds - mf_preds.min()) / (mf_preds.max() - mf_preds.min())
        results['Matrix Factorization']['auroc'].append(roc_auc_score(val_labels_np, mf_preds))
        results['Matrix Factorization']['auprc'].append(average_precision_score(val_labels_np, mf_preds))

    fig, ax = plt.subplots(figsize=(9, 5.5))

    methods = list(results.keys())
    metrics = ['auroc', 'auprc']
    metric_labels = ['AUROC', 'AUPRC']
    x = np.arange(len(methods))
    width = 0.35
    colors_bar = [COLORS['HeteroGAT'], '#2A9D8F', COLORS['RF'], COLORS['MF']]

    for i, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        means = [np.mean(results[m][metric]) for m in methods]
        stds = [np.std(results[m][metric]) for m in methods]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=4,
                      label=mlabel, color=[c if i == 0 else sns.desaturate(c, 0.6)
                                           for c in colors_bar],
                      edgecolor='black', linewidth=0.5, alpha=0.85 if i == 0 else 0.6)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=8.5)

    ax.set_ylabel('Score')
    ax.set_title('Ablation Study: Model Comparison (5-Fold CV)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.5, color='gray', linestyle=':', lw=0.8, alpha=0.5)

    plt.tight_layout()
    fig.savefig(plots / 'figure2_ablation_study.pdf', format='pdf')
    plt.close(fig)
    print(f"  Saved: figure2_ablation_study.pdf")

    for method, vals in results.items():
        print(f"  {method}: AUROC={np.mean(vals['auroc']):.4f}±{np.std(vals['auroc']):.4f}, "
              f"AUPRC={np.mean(vals['auprc']):.4f}±{np.std(vals['auprc']):.4f}")

    return results



def generate_figure3():
    
    print("\n[Figure 3] Generating UMAP embedding visualization...")

    try:
        import umap
    except ImportError:
        from sklearn.manifold import TSNE
        use_tsne = True
        print("  UMAP not installed, falling back to t-SNE...")
    else:
        use_tsne = False

    model = create_model(data.metadata(), mp_edge_index_dict)
    model_path = base / "data" / "processed" / "melanoma_gnn_model.pt"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, weights_only=False, map_location=dev))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
        criterion = torch.nn.BCEWithLogitsLoss()
        all_edges_dev = all_edges.to(dev)
        all_labels_dev = all_labels.to(dev)
        for epoch in range(NUM_EPOCHS):
            model.train()
            optimizer.zero_grad()
            x_dict = model(data.x_dict, mp_edge_index_dict)
            loss = criterion(model.decode(x_dict, all_edges_dev), all_labels_dev)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        x_dict = model(data.x_dict, mp_edge_index_dict)

    np.random.seed(RANDOM_SEED)
    embeddings = []
    node_types = []
    node_labels = []

    lnc_emb = x_dict['lncrna'].cpu().numpy()
    embeddings.append(lnc_emb)
    node_types.extend(['lncRNA'] * lnc_emb.shape[0])
    node_labels.extend(LNCRNA_PANEL[:lnc_emb.shape[0]])

    mirna_emb = x_dict['mirna'].cpu().numpy()
    mirna_sample = np.random.choice(mirna_emb.shape[0], min(200, mirna_emb.shape[0]), replace=False)
    embeddings.append(mirna_emb[mirna_sample])
    node_types.extend(['miRNA'] * len(mirna_sample))
    node_labels.extend([f'miRNA_{i}' for i in mirna_sample])

    mrna_emb = x_dict['mrna'].cpu().numpy()
    mrna_sample = np.random.choice(mrna_emb.shape[0], min(300, mrna_emb.shape[0]), replace=False)
    embeddings.append(mrna_emb[mrna_sample])
    node_types.extend(['mRNA'] * len(mrna_sample))
    node_labels.extend([f'mRNA_{i}' for i in mrna_sample])

    disease_emb = x_dict['disease'].cpu().numpy()
    disease_sample = np.random.choice(disease_emb.shape[0], min(200, disease_emb.shape[0]), replace=False)
    embeddings.append(disease_emb[disease_sample])
    node_types.extend(['Disease'] * len(disease_sample))
    node_labels.extend([f'disease_{i}' for i in disease_sample])

    all_emb = np.vstack(embeddings)

    if use_tsne:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30, n_iter=1000)
        coords = reducer.fit_transform(all_emb)
        method_name = "t-SNE"
    else:
        reducer = umap.UMAP(n_components=2, random_state=RANDOM_SEED, n_neighbors=15, min_dist=0.3)
        coords = reducer.fit_transform(all_emb)
        method_name = "UMAP"

    fig, ax = plt.subplots(figsize=(8, 7))

    type_order = ['lncRNA', 'miRNA', 'mRNA', 'Disease']
    type_colors = [COLORS['lncrna'], COLORS['mirna'], COLORS['mrna'], COLORS['disease']]
    type_sizes = [120, 15, 10, 15]
    type_markers = ['*', 'o', '.', 's']
    type_zorders = [10, 3, 2, 3]

    for ntype, color, size, marker, zorder in zip(type_order, type_colors, type_sizes, type_markers, type_zorders):
        mask = np.array([t == ntype for t in node_types])
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, s=size, marker=marker, alpha=0.7,
                   edgecolors='white' if ntype == 'lncRNA' else 'none',
                   linewidths=0.5, zorder=zorder, label=ntype)

    lncrna_mask = np.array([t == 'lncRNA' for t in node_types])
    lncrna_coords = coords[lncrna_mask]
    lncrna_labels = [l for l, t in zip(node_labels, node_types) if t == 'lncRNA']
    for i, label in enumerate(lncrna_labels):
        ax.annotate(label, (lncrna_coords[i, 0], lncrna_coords[i, 1]),
                    fontsize=7, fontweight='bold', ha='left',
                    xytext=(5, 3), textcoords='offset points',
                    color=COLORS['lncrna'])

    ax.set_xlabel(f'{method_name} Dimension 1')
    ax.set_ylabel(f'{method_name} Dimension 2')
    ax.set_title(f'Learned Node Embeddings ({method_name} Projection)')
    ax.legend(loc='upper right', markerscale=1.5, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(plots / 'figure3_embedding_visualization.pdf', format='pdf')
    plt.close(fig)
    print(f"  Saved: figure3_embedding_visualization.pdf ({method_name})")

    return x_dict



def generate_figure4_case_study(x_dict):
    
    print("\n[Figure 4] Case Study: Novel lncRNA-Melanoma predictions...")

    disease_to_idx = rebuild_disease_index()

    melanoma_indices = []
    melanoma_names = []
    for name, idx in disease_to_idx.items():
        if 'melanoma' in name and idx < data['disease'].num_nodes:
            melanoma_indices.append(idx)
            melanoma_names.append(name)

    if not melanoma_indices:
        print("  WARNING: No melanoma disease node found in index.")
        return

    print(f"  Melanoma nodes found: {melanoma_names}")

    known_positives = set()
    pos_np = pos_edges.cpu().numpy()
    for i in range(pos_np.shape[1]):
        known_positives.add((pos_np[0, i], pos_np[1, i]))

    model = create_model(data.metadata(), mp_edge_index_dict)
    model_path = base / "data" / "processed" / "melanoma_gnn_model.pt"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, weights_only=False, map_location=dev))

    model.eval()
    with torch.no_grad():
        x_dict_local = model(data.x_dict, mp_edge_index_dict)

    predictions = []
    for lnc_idx in range(data['lncrna'].num_nodes):
        for dis_idx in melanoma_indices:
            is_known = (lnc_idx, dis_idx) in known_positives
            edge_idx = torch.tensor([[lnc_idx], [dis_idx]], dtype=torch.long).to(dev)
            score = torch.sigmoid(model.decode(x_dict_local, edge_idx)).item()
            predictions.append({
                'lncRNA': LNCRNA_PANEL[lnc_idx] if lnc_idx < len(LNCRNA_PANEL) else f'lncRNA_{lnc_idx}',
                'disease': melanoma_names[melanoma_indices.index(dis_idx)],
                'score': score,
                'known_association': is_known,
            })

    pred_df = pd.DataFrame(predictions)
    pred_df = pred_df.sort_values('score', ascending=False)

    novel_df = pred_df[~pred_df['known_association']].head(10).reset_index(drop=True)
    known_df = pred_df[pred_df['known_association']].reset_index(drop=True)

    print(f"\n  Top 10 Novel (Unvalidated) lncRNA-Melanoma Predictions:")
    print(f"  {'Rank':<6}{'lncRNA':<14}{'Score':<10}{'Disease'}")
    print(f"  {'-'*50}")
    for i, row in novel_df.iterrows():
        print(f"  {i+1:<6}{row['lncRNA']:<14}{row['score']:<10.4f}{row['disease']}")

    print(f"\n  Validating against PubMed (E-utilities API)...")
    pubmed_results = validate_pubmed(novel_df['lncRNA'].tolist())

    novel_df['pubmed_hits'] = novel_df['lncRNA'].map(
        lambda x: pubmed_results.get(x, {}).get('count', 0)
    )
    novel_df['recent_pmids'] = novel_df['lncRNA'].map(
        lambda x: '; '.join(pubmed_results.get(x, {}).get('pmids', [])[:3])
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    all_scores = pred_df.groupby('lncRNA').agg({'score': 'max', 'known_association': 'any'}).reset_index()
    all_scores = all_scores.sort_values('score', ascending=True).reset_index(drop=True)

    colors = [COLORS['lncrna'] if row['known_association'] else COLORS['mirna']
              for _, row in all_scores.iterrows()]
    bars = ax.barh(range(len(all_scores)), all_scores['score'], color=colors,
                   edgecolor='black', linewidth=0.4, alpha=0.85)

    ax.set_yticks(range(len(all_scores)))
    ax.set_yticklabels(all_scores['lncRNA'], fontsize=9)
    ax.set_xlabel('Prediction Score (Melanoma Association)')
    ax.set_title('lncRNA–Melanoma Association Scores (Trained HeteroGAT)')
    ax.axvline(x=0.5, color='gray', linestyle=':', lw=1)

    legend_elements = [
        mpatches.Patch(facecolor=COLORS['lncrna'], edgecolor='black', label='Known association'),
        mpatches.Patch(facecolor=COLORS['mirna'], edgecolor='black', label='Novel prediction'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    for i, row in all_scores.iterrows():
        lncrna = row['lncRNA']
        hits = pubmed_results.get(lncrna, {}).get('count', 0)
        if hits > 0 and not row['known_association']:
            ax.text(row['score'] + 0.01, i, f'PubMed: {hits}',
                    va='center', fontsize=7.5, color='darkgreen', style='italic')

    plt.tight_layout()
    fig.savefig(plots / 'figure4_case_study_predictions.pdf', format='pdf')
    plt.close(fig)
    print(f"  Saved: figure4_case_study_predictions.pdf")

    novel_df.to_csv(plots / 'table_novel_predictions.csv', index=False)
    print(f"  Saved: table_novel_predictions.csv")

    return novel_df


def validate_pubmed(lncrna_list: List[str]) -> Dict[str, dict]:
    
    import urllib.request
    import urllib.parse
    import xml.etree.ElementTree as ET

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    results = {}

    for lncrna in lncrna_list:
        query = f'("{lncrna}"[Title/Abstract]) AND ("melanoma"[Title/Abstract])'
        params = urllib.parse.urlencode({
            'db': 'pubmed',
            'term': query,
            'retmax': 5,
            'retmode': 'xml',
            'sort': 'date',
        })
        url = f"{base_url}?{params}"

        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'HeteroGraphBuilder/1.0 (Research)'
            })
            with urllib.request.urlopen(req, timeout=10) as response:
                xml_data = response.read()
            root = ET.fromstring(xml_data)
            count = int(root.findtext('.//Count', '0'))
            pmids = [id_elem.text for id_elem in root.findall('.//IdList/Id')]
            results[lncrna] = {'count': count, 'pmids': pmids}
            time.sleep(0.4)  # Rate limit
        except Exception as e:
            results[lncrna] = {'count': 0, 'pmids': [], 'error': str(e)}

    return results



def generate_figure5():
    
    print("\n[Figure 5] Graph topology statistics...")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    edge_counts = {}
    for et in data.edge_types:
        store = data[et]
        if hasattr(store, 'edge_index') and store.edge_index is not None and store.edge_index.numel() > 0:
            label = f"{et[0]}–{et[2]}\n({et[1].replace('_', ' ')})"
            if 'rev_' not in et[1]:
                edge_counts[label] = store.edge_index.shape[1]

    labels = list(edge_counts.keys())
    sizes = list(edge_counts.values())
    pie_colors = sns.color_palette("Set2", len(labels))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, autopct='%1.1f%%', colors=pie_colors,
        textprops={'fontsize': 8}, pctdistance=0.75
    )
    ax.legend(wedges, labels, loc='center left', bbox_to_anchor=(-0.3, 0.5), fontsize=8)
    ax.set_title('(A) Edge Type Distribution')

    ax = axes[0, 1]
    node_counts = {ntype: data[ntype].num_nodes for ntype in data.node_types}
    ntype_colors = [COLORS.get(nt, '#888888') for nt in node_counts.keys()]
    bars = ax.bar(node_counts.keys(), node_counts.values(), color=ntype_colors,
                  edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Number of Nodes')
    ax.set_title('(B) Node Type Counts')
    ax.set_yscale('log')
    for bar, count in zip(bars, node_counts.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f'{count:,}', ha='center', va='bottom', fontsize=9)

    ax = axes[1, 0]
    mirna_dis_ei = data['mirna', 'associated_with', 'disease'].edge_index.cpu().numpy()
    disease_degrees = np.bincount(mirna_dis_ei[1], minlength=data['disease'].num_nodes)
    ax.hist(disease_degrees[disease_degrees > 0], bins=50, color=COLORS['disease'],
            edgecolor='black', linewidth=0.3, alpha=0.8)
    ax.set_xlabel('Degree (# miRNA associations)')
    ax.set_ylabel('Number of Diseases')
    ax.set_title('(C) Disease Node Degree Distribution')
    ax.axvline(x=np.median(disease_degrees[disease_degrees > 0]), color='red',
               linestyle='--', lw=1.5, label=f'Median={np.median(disease_degrees[disease_degrees > 0]):.0f}')
    ax.legend()

    ax = axes[1, 1]
    ppi_ei = data['mrna', 'interacts_with', 'mrna'].edge_index.cpu().numpy()
    mrna_degrees = np.bincount(ppi_ei[0], minlength=data['mrna'].num_nodes)
    mrna_degrees += np.bincount(ppi_ei[1], minlength=data['mrna'].num_nodes)
    ax.hist(mrna_degrees[mrna_degrees > 0], bins=80, color=COLORS['mrna'],
            edgecolor='black', linewidth=0.3, alpha=0.8, log=True)
    ax.set_xlabel('Degree (# PPI interactions)')
    ax.set_ylabel('Number of Genes (log scale)')
    ax.set_title('(D) mRNA/Protein Degree Distribution (STRING)')
    ax.axvline(x=np.median(mrna_degrees[mrna_degrees > 0]), color='red',
               linestyle='--', lw=1.5, label=f'Median={np.median(mrna_degrees[mrna_degrees > 0]):.0f}')
    ax.legend()

    plt.tight_layout()
    fig.savefig(plots / 'figure5_graph_topology.pdf', format='pdf')
    plt.close(fig)
    print(f"  Saved: figure5_graph_topology.pdf")



def generate_figure6(fold_losses: list):
    
    print("\n[Figure 6] Training convergence curves...")

    fig, ax = plt.subplots(figsize=(8, 5))

    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(fold_losses)))
    for i, losses in enumerate(fold_losses):
        ax.plot(range(1, len(losses) + 1), losses, color=cmap[i],
                lw=1.5, alpha=0.7, label=f'Fold {i+1}')

    min_len = min(len(l) for l in fold_losses)
    mean_loss = np.mean([l[:min_len] for l in fold_losses], axis=0)
    ax.plot(range(1, min_len + 1), mean_loss, color='black', lw=2.5,
            linestyle='-', label='Mean')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Binary Cross-Entropy Loss')
    ax.set_title('Training Convergence (5-Fold Cross-Validation)')
    ax.legend(loc='upper right')
    ax.set_yscale('log')
    ax.set_xlim(1, min_len)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(plots / 'figure6_convergence.pdf', format='pdf')
    plt.close(fig)
    print(f"  Saved: figure6_convergence.pdf")



def generate_figure7():
    
    print("\n[Figure 7] Edge-type contribution analysis...")

    ABLATION_EPOCHS = 80  # Reduced for tractability (convergence reached by ~30)
    criterion = torch.nn.BCEWithLogitsLoss()
    all_edges_dev = all_edges.to(dev)
    all_labels_dev = all_labels.to(dev)
    labels_np = all_labels.numpy()

    edge_type_names = {
        ('mirna', 'associated_with', 'disease'): 'miRNA–Disease',
        ('mrna', 'interacts_with', 'mrna'): 'PPI (STRING)',
        ('disease', 'similar_to', 'disease'): 'Disease Similarity',
        ('lncrna', 'associated_with', 'disease'): 'lncRNA–Disease',
        ('lncrna', 'interacts_with', 'mirna'): 'lncRNA–miRNA',
    }

    reverse_map = {
        ('mirna', 'associated_with', 'disease'): ('disease', 'rev_associated_with', 'mirna'),
        ('lncrna', 'associated_with', 'disease'): ('disease', 'rev_associated_with', 'lncrna'),
        ('lncrna', 'interacts_with', 'mirna'): ('mirna', 'rev_interacts_with', 'lncrna'),
    }

    results_ablation = {'Full Model': []}

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    for train_idx, val_idx in skf.split(np.arange(n_samples), labels_np):
        model = create_model(data.metadata(), mp_edge_index_dict)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
        train_ei = all_edges[:, train_idx].to(dev)
        train_labels = all_labels[train_idx].to(dev)
        val_ei = all_edges[:, val_idx].to(dev)
        val_labels_np = all_labels[val_idx].numpy()

        for _ in range(ABLATION_EPOCHS):
            model.train()
            optimizer.zero_grad()
            x_dict = model(data.x_dict, mp_edge_index_dict)
            loss = criterion(model.decode(x_dict, train_ei), train_labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            x_dict = model(data.x_dict, mp_edge_index_dict)
            preds = torch.sigmoid(model.decode(x_dict, val_ei)).cpu().numpy()
        results_ablation['Full Model'].append(roc_auc_score(val_labels_np, preds))

    for et, name in edge_type_names.items():
        if et not in mp_edge_index_dict:
            continue

        reduced_edges = {}
        rev_et = reverse_map.get(et)
        for k, v in mp_edge_index_dict.items():
            if k == et or k == rev_et:
                reduced_edges[k] = torch.zeros((2, 0), dtype=torch.long, device=dev)
            else:
                reduced_edges[k] = v

        results_ablation[f'w/o {name}'] = []
        for train_idx, val_idx in skf.split(np.arange(n_samples), labels_np):
            model = create_model(data.metadata(), mp_edge_index_dict)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
            train_ei = all_edges[:, train_idx].to(dev)
            train_labels = all_labels[train_idx].to(dev)
            val_ei = all_edges[:, val_idx].to(dev)
            val_labels_np_fold = all_labels[val_idx].numpy()

            for _ in range(ABLATION_EPOCHS):
                model.train()
                optimizer.zero_grad()
                x_dict = model(data.x_dict, reduced_edges)
                loss = criterion(model.decode(x_dict, train_ei), train_labels)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                x_dict = model(data.x_dict, reduced_edges)
                preds = torch.sigmoid(model.decode(x_dict, val_ei)).cpu().numpy()
            results_ablation[f'w/o {name}'].append(roc_auc_score(val_labels_np_fold, preds))

    fig, ax = plt.subplots(figsize=(9, 5))

    conditions = list(results_ablation.keys())
    fold_data = np.array([results_ablation[c] for c in conditions])

    sns.heatmap(fold_data, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1.0,
                xticklabels=[f'Fold {i+1}' for i in range(N_FOLDS)],
                yticklabels=conditions, ax=ax, linewidths=0.5, linecolor='white')
    ax.set_title('Edge-Type Ablation Analysis (Val AUROC per Fold)')
    ax.set_xlabel('Cross-Validation Fold')

    means = fold_data.mean(axis=1)
    for i, mean in enumerate(means):
        ax.text(N_FOLDS + 0.3, i + 0.5, f'μ={mean:.3f}', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig.savefig(plots / 'figure7_edge_ablation_heatmap.pdf', format='pdf')
    plt.close(fig)
    print(f"  Saved: figure7_edge_ablation_heatmap.pdf")

    for cond, aurocs in results_ablation.items():
        print(f"  {cond:<30s}: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")



def generate_figure8():
    
    print("\n[Figure 8] Prediction score distributions...")

    model = create_model(data.metadata(), mp_edge_index_dict)
    model_path = base / "data" / "processed" / "melanoma_gnn_model.pt"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, weights_only=False, map_location=dev))

    model.eval()
    with torch.no_grad():
        x_dict = model(data.x_dict, mp_edge_index_dict)

    n_lncrna = data['lncrna'].num_nodes
    n_disease = data['disease'].num_nodes

    known_positives = set()
    pos_np = pos_edges.cpu().numpy()
    for i in range(pos_np.shape[1]):
        known_positives.add((pos_np[0, i], pos_np[1, i]))

    np.random.seed(RANDOM_SEED)
    novel_scores = []
    known_scores = []

    n_dis_sample = min(n_disease, 300)
    all_lnc_idx = []
    all_dis_idx = []
    for lnc_idx in range(n_lncrna):
        for dis_idx in range(n_dis_sample):
            all_lnc_idx.append(lnc_idx)
            all_dis_idx.append(dis_idx)

    batch_edge_idx = torch.tensor([all_lnc_idx, all_dis_idx], dtype=torch.long).to(dev)
    with torch.no_grad():
        all_scores_tensor = torch.sigmoid(model.decode(x_dict, batch_edge_idx)).cpu().numpy()

    for i, (lnc_idx, dis_idx) in enumerate(zip(all_lnc_idx, all_dis_idx)):
        if (lnc_idx, dis_idx) in known_positives:
            known_scores.append(all_scores_tensor[i])
        else:
            novel_scores.append(all_scores_tensor[i])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(known_scores, bins=30, alpha=0.7, color=COLORS['lncrna'],
            label=f'Known associations (n={len(known_scores)})', density=True, edgecolor='white')
    ax.hist(novel_scores, bins=50, alpha=0.5, color=COLORS['mirna'],
            label=f'Novel pairs (n={len(novel_scores)})', density=True, edgecolor='white')
    ax.axvline(x=0.5, color='black', linestyle='--', lw=1.5, label='Decision boundary')
    ax.set_xlabel('Prediction Score')
    ax.set_ylabel('Density')
    ax.set_title('(A) Score Distribution: Known vs. Novel')
    ax.legend()

    ax = axes[1]
    lncrna_scores = {}
    for lnc_idx in range(n_lncrna):
        start = lnc_idx * n_dis_sample
        end = start + n_dis_sample
        scores = all_scores_tensor[start:end].tolist()
        lncrna_name = LNCRNA_PANEL[lnc_idx] if lnc_idx < len(LNCRNA_PANEL) else f'lncRNA_{lnc_idx}'
        lncrna_scores[lncrna_name] = scores

    sorted_names = sorted(lncrna_scores.keys(), key=lambda x: np.median(lncrna_scores[x]))
    box_data = [lncrna_scores[n] for n in sorted_names]

    bp = ax.boxplot(box_data, vert=False, patch_artist=True,
                    labels=sorted_names, widths=0.6,
                    flierprops=dict(marker='.', markersize=2, alpha=0.3))
    for patch in bp['boxes']:
        patch.set_facecolor(COLORS['mirna'])
        patch.set_alpha(0.6)
    ax.axvline(x=0.5, color='black', linestyle='--', lw=1)
    ax.set_xlabel('Prediction Score (across all diseases)')
    ax.set_title('(B) Per-lncRNA Score Distribution')

    plt.tight_layout()
    fig.savefig(plots / 'figure8_score_distributions.pdf', format='pdf')
    plt.close(fig)
    print(f"  Saved: figure8_score_distributions.pdf")



if __name__ == "__main__":
    print("=" * 70)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 70)

    fold_losses, aurocs, auprcs = generate_figure1()

    ablation_results = generate_figure2()

    x_dict_trained = generate_figure3()

    novel_predictions = generate_figure4_case_study(x_dict_trained)

    generate_figure5()

    generate_figure6(fold_losses)

    generate_figure7()

    generate_figure8()

    print("\n" + "=" * 70)
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print(f"Output directory: {plots}")
    print("=" * 70)

    for f in sorted(plots.glob("*")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:<45s} ({size_kb:.1f} KB)")