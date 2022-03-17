import os
import time
import random
import json
import argparse

import torch
import numpy as np
import os.path as osp
from tqdm import tqdm
from pathlib import Path

from utils.dataset import get_datasets
from explainers import *
from gnns import *
from glnns.ba3motif_glnn import BA3MotifGLNN
from datasets.ba3motif_dataset import BA3Motif
from torch_scatter import scatter
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import DataLoader

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

test_dataset = BA3Motif('data/BA3', mode='testing')
#val_dataset = BA3Motif(args.data_path, mode='evaluation')
train_dataset = BA3Motif('data/BA3', mode='training')

test_loader = DataLoader(test_dataset,
                         batch_size=128,
                         shuffle=False
                         )
#val_loader = DataLoader(val_dataset,
#                        batch_size=args.batch_size,
#                        shuffle=False
#                        )
train_loader = DataLoader(train_dataset,
                          batch_size=128,
                          shuffle=True
                          )
model_path = f"param/gnns/{args.dataset}_net.pt"
model = torch.load(model_path, map_location=device)

in_features = torch.flatten(train_dataset[0].x, 1, -1).size(1)
mlp_model = BA3MotifGLNN(device=device, num_unit=2, in_features=in_features, out_features=3)
loss_label = torch.nn.CrossEntropyLoss()
loss_teacher = torch.nn.KLDivLoss()
loss_lambda = 0
optimizer = torch.optim.Adam(mlp_model.parameters(), lr=1e-4)
model.train()
mlp_model.train()

for epoch in range(100):
    running_loss = 0.0
    n_samples = len(train_loader.dataset)
    for g in train_loader:
        g.to(device)
        # compute prediction of GNN
        with torch.no_grad():
            z = model(g.x, g.edge_index, g.edge_attr, g.batch)

        # compute prediction of MLP
        optimizer.zero_grad()
        y_hat = mlp_model(global_mean_pool(g.x, g.batch))

        # compute loss between GNN and MLP
        loss = loss_lambda * loss_label(y_hat, g.y) \
                + (1 - loss_lambda) * loss_teacher(y_hat, z)

        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.num_graphs
    
    train_loss = running_loss / n_samples

    if epoch % 10 == 0:
        print(f'[Epoch {epoch + 1}] Train loss: {round(train_loss, 4)}')
        correct_gnn = 0.0
        correct_mlp = 0.0
        n_test_samples = len(test_loader.dataset)
        with torch.no_grad():
            model.eval()
            mlp_model.eval()

            for g in test_loader:
                g.to(device)
                
                z = model(g.x, g.edge_index, g.edge_attr, g.batch)
                y_hat = mlp_model(global_mean_pool(g.x, g.batch))
                gnn_pred = z.argmax(dim=1)
                mlp_pred = y_hat.argmax(dim=1)
                print(f"mlp_pred: {mlp_pred}")
                correct_gnn += float(gnn_pred.eq(g.y).sum().item())
                correct_mlp += float(mlp_pred.eq(g.y).sum().item())

            model.train()
            mlp_model.train()
        print(f'Test GNN accuracy: {(correct_gnn / n_test_samples) * 100}')
        print(f'Test MLP accuracy: {(correct_mlp / n_test_samples) * 100}')

torch.save(mlp_model, f'param/glnns/{args.dataset}_glnn.pt')
