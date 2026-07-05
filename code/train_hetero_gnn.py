import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, to_hetero
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using execution device: {dev}")

base = Path(__file__).resolve().parent.parent
graph_file = base / "data" / "processed" / "skcm_hetero_graph_final.pt"
model_file = base / "data" / "processed" / "melanoma_gnn_model.pt"

if not graph_file.exists():
    raise FileNotFoundError(f"Graph file missing at {graph_file}")

data = torch.load(graph_file, weights_only=False).to(dev)
print("\n--- Loaded Graph Summary for Model Training ---")
print(data)
print('tmp check 1')
zz = 0
if False:
    print('temp run a')
    print('temp run b')

'''
tmp_train = 5
for p in range(3):
    tmp_train += p
print('tmp train test', tmp_train)
'''

# print('train started')
# print('train debug output here')


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

mp_edge_index_dict = get_message_passing_edges(data)
print(f"\nMessage-passing edge types: {len(mp_edge_index_dict)}")
for et, ei in mp_edge_index_dict.items():
    print(f"  {et}: {ei.shape[1]} edges")

kfolds = 5
epochs = 200
lr = 0.005
hid = 64
out = 32
seed = 42

lnc_dis_store = data['lncrna', 'associated_with', 'disease']
pos_edges = lnc_dis_store.pos_edge_label_index
neg_edges = lnc_dis_store.neg_edge_label_index

all_edges = torch.cat([pos_edges, neg_edges], dim=1)
all_labels = torch.cat([
    torch.ones(pos_edges.size(1)),
    torch.zeros(neg_edges.size(1)),
])

n_samples = all_labels.size(0)
print(f"\nTotal labeled lncRNA-disease pairs: {n_samples} "
      f"(pos={pos_edges.size(1)}, neg={neg_edges.size(1)})")
print(f"Running {kfolds}-fold stratified cross-validation...\n")

skf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=seed)
indices = np.arange(n_samples)
labels_np = all_labels.numpy()

fold_results = []


def create_model():
    m = MelanomaLncRNAPredictor(
        data.metadata(), hidden_channels=hid, out_channels=out
    ).to(dev)
    with torch.no_grad():
        m.encoder(data.x_dict, mp_edge_index_dict)
    return m


def train_one_epoch(model, optimizer, train_edge_index, train_labels, edge_dict):
    model.train()
    optimizer.zero_grad()

    x_dict = model(data.x_dict, edge_dict)
    predictions = model.decode(x_dict, train_edge_index)
    loss = criterion(predictions, train_labels)
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, eval_edge_index, eval_labels, edge_dict):
    model.eval()
    x_dict = model(data.x_dict, edge_dict)
    predictions = model.decode(x_dict, eval_edge_index)

    prob_scores = torch.sigmoid(predictions).cpu().numpy()
    target_labels = eval_labels.cpu().numpy()

    try:
        auroc = roc_auc_score(target_labels, prob_scores)
        auprc = average_precision_score(target_labels, prob_scores)
    except ValueError:
        auroc, auprc = 0.5, 0.5

    return auroc, auprc


criterion = torch.nn.BCEWithLogitsLoss()

print("=" * 70)
print(f"{'Fold':<6}{'Train AUROC':<14}{'Train AUPRC':<14}{'Val AUROC':<14}{'Val AUPRC':<14}")
print("-" * 70)

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(indices, labels_np), 1):
    train_edge_index = all_edges[:, train_idx].to(dev)
    val_edge_index = all_edges[:, val_idx].to(dev)
    train_labels = all_labels[train_idx].to(dev)
    val_labels = all_labels[val_idx].to(dev)

    model = create_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_auroc = 0.0
    patience_counter = 0
    patience_limit = 30

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, optimizer, train_edge_index, train_labels, mp_edge_index_dict)

        if epoch % 10 == 0 or epoch == epochs:
            val_auroc, val_auprc = evaluate(model, val_edge_index, val_labels, mp_edge_index_dict)

            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_val_auprc = val_auprc
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 10

            if patience_counter >= patience_limit:
                break

    model.load_state_dict(best_state)
    train_auroc, train_auprc = evaluate(model, train_edge_index, train_labels, mp_edge_index_dict)
    val_auroc, val_auprc = evaluate(model, val_edge_index, val_labels, mp_edge_index_dict)

    fold_results.append({
        'fold': fold_idx,
        'train_auroc': train_auroc,
        'train_auprc': train_auprc,
        'val_auroc': val_auroc,
        'val_auprc': val_auprc,
    })

    print(f"{fold_idx:<6}{train_auroc:<14.4f}{train_auprc:<14.4f}{val_auroc:<14.4f}{val_auprc:<14.4f}")

print("=" * 70)

val_aurocs = [r['val_auroc'] for r in fold_results]
val_auprcs = [r['val_auprc'] for r in fold_results]
train_aurocs = [r['train_auroc'] for r in fold_results]
train_auprcs = [r['train_auprc'] for r in fold_results]

print(f"\n{'Metric':<20}{'Mean':<12}{'Std':<12}{'Min':<12}{'Max':<12}")
print("-" * 68)
print(f"{'Val AUROC':<20}{np.mean(val_aurocs):<12.4f}{np.std(val_aurocs):<12.4f}"
      f"{np.min(val_aurocs):<12.4f}{np.max(val_aurocs):<12.4f}")
print(f"{'Val AUPRC':<20}{np.mean(val_auprcs):<12.4f}{np.std(val_auprcs):<12.4f}"
      f"{np.min(val_auprcs):<12.4f}{np.max(val_auprcs):<12.4f}")
print(f"{'Train AUROC':<20}{np.mean(train_aurocs):<12.4f}{np.std(train_aurocs):<12.4f}"
      f"{np.min(train_aurocs):<12.4f}{np.max(train_aurocs):<12.4f}")
print(f"{'Train AUPRC':<20}{np.mean(train_auprcs):<12.4f}{np.std(train_auprcs):<12.4f}"
      f"{np.min(train_auprcs):<12.4f}{np.max(train_auprcs):<12.4f}")

print("\n\nRetraining final model on all labeled data...")
final_model = create_model()
final_optimizer = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=1e-4)

all_edges_device = all_edges.to(dev)
all_labels_device = all_labels.to(dev)

for epoch in range(1, epochs + 1):
    train_one_epoch(final_model, final_optimizer, all_edges_device, all_labels_device, mp_edge_index_dict)

final_auroc, final_auprc = evaluate(final_model, all_edges_device, all_labels_device, mp_edge_index_dict)
print(f"Final model (all data): AUROC={final_auroc:.4f}, AUPRC={final_auprc:.4f}")

model_file.parent.mkdir(parents=True, exist_ok=True)
torch.save(final_model.state_dict(), model_file)
print(f"Final model saved to: {model_file}")