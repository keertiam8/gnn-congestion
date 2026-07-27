"""
GNN for congestion prediction from placement graphs.

Input graph:
    - nodes: standard cells / macros, with features [x, y, width, height, pin_count, cell_type_onehot...]
    - edges: net connections between cell pins (bidirectional)
Output:
    - per-node congestion score, later rasterized to a grid heatmap for OpenROAD

Supports online fine-tuning: call `fit_step` per design iteration with real
routed-congestion labels extracted from OpenROAD reports.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data


class CongestionGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_layers=4, heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_channels)   #hidden channels is the number of output dimensions after u multiply the feature matrix with a 8x64 matrix of weights

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(hidden_channels, hidden_channels // heads, heads=heads, dropout=dropout)   #each cell learns from its neighbor via attention, in this u have 4 different types of attention example - wire distribution, macro proximity etc
            )
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.dropout = dropout
        # head takes [h0 || h_final] -- a global skip from the pre-message-passing
        # embedding straight to the output, so local/high-frequency detail lost to
        # oversmoothing across `num_layers` of neighborhood averaging can still
        # reach the prediction directly.
        self.head = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, x, edge_index):   #x is the node features
        h0 = self.input_proj(x)  #each nodes 64 bit dimension representation, input_proj is a linear layer that projects the input features to hidden_channels dimension
        h = h0
        for conv, norm in zip(self.convs, self.norms):   #zip → [(Conv1, Norm1), (Conv2, Norm2), (Conv3, Norm3), (Conv4, Norm4)]
            h_new = conv(h, edge_index)
            h = norm(h + h_new)  # residual + norm
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.head(torch.cat([h0, h], dim=-1)).squeeze(-1)
        return out  # per-node congestion score (raw, apply sigmoid/scale outside if needed)


def nodes_to_grid_heatmap(node_scores, node_xy, grid_size=(64, 64), die_area=None):
    """
    Rasterize per-node congestion scores into a grid heatmap.

    node_scores: (N,) tensor of predicted congestion
    node_xy: (N, 2) tensor of node placement coordinates
    die_area: (xmin, ymin, xmax, ymax); if None, inferred from node_xy
    """
    device = node_scores.device
    gx, gy = grid_size

    if die_area is None:
        xmin, ymin = node_xy.min(dim=0).values
        xmax, ymax = node_xy.max(dim=0).values
    else:
        xmin, ymin, xmax, ymax = die_area

    heatmap = torch.zeros(gx, gy, device=device)
    counts = torch.zeros(gx, gy, device=device)

    col = ((node_xy[:, 0] - xmin) / (xmax - xmin + 1e-9) * (gx - 1)).long().clamp(0, gx - 1)
    row = ((node_xy[:, 1] - ymin) / (ymax - ymin + 1e-9) * (gy - 1)).long().clamp(0, gy - 1)

    idx = row * gx + col
    heatmap.view(-1).index_add_(0, idx, node_scores)
    counts.view(-1).index_add_(0, idx, torch.ones_like(node_scores))

    return heatmap / counts.clamp(min=1)


class CongestionTrainer:
    """Wraps the model with pretrain (CircuitNet, batched) and per-design
    online fine-tune (single design, few steps, low LR) loops."""

    def __init__(self, model, lr=1e-3, finetune_lr=1e-4, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.finetune_lr = finetune_lr

    def loss_fn(self, pred, target):
        return F.mse_loss(pred, target)

    def pretrain_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        for batch in dataloader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(batch.x, batch.edge_index)
            loss = self.loss_fn(pred, batch.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def finetune_step(self, data: Data, target: torch.Tensor, steps=20):
        """
        Fine-tune on a single design's real routed-congestion labels.
        data.x / data.edge_index come from the current placement graph;
        target is the ground-truth congestion extracted from OpenROAD's
        routed report, mapped back to node granularity (or use
        nodes_to_grid_heatmap + grid-level loss instead if labels are grid-based).
        """
        data = data.to(self.device)
        target = target.to(self.device)

        for g in self.optimizer.param_groups:
            g["lr"] = self.finetune_lr

        self.model.train()
        losses = []
        for _ in range(steps):
            self.optimizer.zero_grad()
            pred = self.model(data.x, data.edge_index)
            loss = self.loss_fn(pred, target)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        for g in self.optimizer.param_groups:
            g["lr"] = self.optimizer.defaults["lr"]

        return losses

    @torch.no_grad()
    def predict_heatmap(self, data: Data, grid_size=(64, 64)):
        self.model.eval()
        data = data.to(self.device)
        pred = self.model(data.x, data.edge_index)
        node_xy = data.x[:, :2]  # assumes first two features are (x, y); adjust to your feature layout
        return nodes_to_grid_heatmap(pred, node_xy, grid_size=grid_size)

    def save(self, path):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        if "model" in ckpt:
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
        else:
            # backward-compat with older checkpoints saved as a bare state_dict
            self.model.load_state_dict(ckpt)


if __name__ == "__main__":
    # quick smoke test with random data
    num_nodes, in_channels = 500, 8
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, 2000))

    model = CongestionGNN(in_channels=in_channels)
    trainer = CongestionTrainer(model, device="cpu")

    out = model(x, edge_index)
    print("output shape:", out.shape)

    heatmap = nodes_to_grid_heatmap(out, x[:, :2], grid_size=(32, 32))
    print("heatmap shape:", heatmap.shape)
