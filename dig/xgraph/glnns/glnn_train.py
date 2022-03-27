import os
import time
import random
import json
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import os.path as osp
from tqdm import tqdm
from pathlib import Path

from glnn import GLNN
from torch_geometric.data import DataLoader, download_url, extract_zip
from torch_geometric.utils import degree
from dig.xgraph.dataset import SynGraphDataset
from dig.xgraph.models import GCN_2l, GCN_3l

def check_checkpoints(root='./', force_reload=False):
    if not force_reload and osp.exists(osp.join(root, 'checkpoints')):
        return
    url = ('https://github.com/divelab/DIG_storage/raw/main/xgraph/checkpoints.zip')
    path = download_url(url, root)
    extract_zip(path, root)
    os.unlink(path)

def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain ReFine")
    parser.add_argument("--cuda", type=int, default=0, help="GPU device.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ba3",
        choices=["mutag", "ba3", "graphsst2", "mnist", "vg", "reddit5k"],
    )
    parser.add_argument(
        "--result_dir", type=str, default="results/", help="Result directory."
    )
    parser.add_argument(
        "--lr", type=float, default=2 * 1e-4, help="Fine-tuning learning rate."
    )
    parser.add_argument("--epoch", type=int, default=20, help="Fine-tuning rpoch.")
    parser.add_argument("--simple", help="use Simplified ReFine", action="store_true")
    parser.add_argument("--no_relu", help="use ReFineNoReLU", action="store_true")
    parser.add_argument(
        "--random_seed",
        help="use model trained with random_seed",
        type=str,
        default=None,
    )
    return parser.parse_args()


args = parse_args()

device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

# load dataset
dataset = SynGraphDataset(root='dig_datasets/', name='BA_shapes')
dataset.data.x = dataset.data.x.to(torch.float32)
dataset.data.x = dataset.data.x[:, :1]

dim_node = dataset.num_node_features
dim_edge = dataset.num_edge_features
num_classes = dataset.num_classes
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# load trained GNN
model = GCN_2l(model_level='node', dim_node=dim_node, dim_hidden=300, num_classes=num_classes)
model.to(device)
check_checkpoints()
ckpt_path = osp.join('checkpoints', 'ba_shapes', 'GCN_2l', '0', 'GCN_2l_best.ckpt')
model.load_state_dict(torch.load(ckpt_path)['state_dict'])

# use node degree as feature for MLP
degrees = degree(dataset.data.edge_index[0], dtype=torch.int64)
degrees_one_hot = F.one_hot(degrees).float().to(device)
dim_node_mlp = degrees_one_hot.shape[1]

# load MLP
mlp_model = GLNN(device=device, num_layers=2, in_features=dim_node_mlp, out_features=num_classes, dim_hidden=1200)

# initialize loss functions and optimizer
loss_label = torch.nn.NLLLoss()
loss_teacher = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
loss_lambda = 0
optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.01, weight_decay=0.0005)

n_samples = dataset.data.x.shape[0]
data = dataset[0]
data.to(device)

def shift_labels(y, mask):
    y = y[mask]
    y_shift = y.clone()
    y_shift[y==1] = 2
    y_shift[y==2] = 3
    y_shift[y==3] = 1
    return y_shift


y = shift_labels(data.y, data.train_mask)
y_test = shift_labels(data.y, data.test_mask)

def evaluate():
    with torch.no_grad():
        # MLP
        mlp_model.eval()
        logits = mlp_model(degrees_one_hot)
        z = logits[data.test_mask].log_softmax(dim=1)
        mlp_pred = z.argmax(dim=-1)
        mlp_acc = float(mlp_pred.eq(y_test).sum().item()) / len(mlp_pred)
        print(f"[MLP TEST ACC]: {mlp_acc}")
        mlp_model.train()
        return mlp_acc

model.eval()
mlp_model.train()

# predict node labels with GNN
with torch.no_grad():
    logits = model(data.x, data.edge_index)
    z = logits[data.train_mask].log_softmax(dim=1)

mlp_accs = []

# train MLP
for epoch in range(10):

    # compute prediction of MLP
    optimizer.zero_grad()

    logits = mlp_model(degrees_one_hot)
    y_hat = logits[data.train_mask].log_softmax(dim=1)

    # compute loss between GNN and MLP
    loss = loss_lambda * loss_label(y_hat, y) + (1 - loss_lambda) * loss_teacher(y_hat, z)

    loss.backward()
    optimizer.step()

    mlp_acc = evaluate()
    mlp_accs.append(mlp_acc)

# GNN
logits = model(data.x, data.edge_index)
z = logits[data.test_mask].log_softmax(dim=1)
gnn_pred = z.argmax(dim=-1)
gnn_acc = float(gnn_pred.eq(y_test).sum().item()) / len(gnn_pred)
print(f"[GNN TEST ACC]: {gnn_acc}")


with open('results/glnn_accs.json', 'w') as f:
    json.dump(mlp_accs, f)

torch.save(mlp_model, f'param/{args.dataset}_glnn.pt')
