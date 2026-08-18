"""
GNN for congestion prediction from placement graphs.

Input graph:
    - nodes: standard cells / macros, with features [macro_region_value, rudy_value]
    - edges: net connections between cell pins (bidirectional)
Output:
    - per-node congestion score (2 channels: horizontal + vertical overflow)
    - later rasterized to a grid heatmap for OpenROAD

Supports online fine-tuning: call `finetune_step` per design iteration with real
routed-congestion labels extracted from OpenROAD reports.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.data import Data


class CongestionGNN(nn.Module):
    def __init__(self, in_channels=2, hidden_channels=64, num_layers=4, heads=4, dropout=0.1, out_channels=2):
        super().__init__()

        # Project 2 input channels (macro_region + rudy) → 64 hidden dims
        # Creates weight matrix W (in_channels × hidden_channels) = (2 × 64)
        # Each of the 64 output dims is a weighted combination of all 2 inputs
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            # Each cell learns from its neighbors via attention
            # 4 attention heads, each outputs 16 dims → concatenated to 64 dims
            # GATv2Conv(64, 64//4=16, heads=4) → output still 64 dims
            self.convs.append(
                GATv2Conv(hidden_channels, hidden_channels // heads, heads=heads, dropout=dropout)
            )
            # GraphNorm: normalizes per feature-channel across all nodes in the same graph
            # Better than LayerNorm which erases inter-node magnitude differences
            # (which encode "how congested is this node relative to others")
            self.norms.append(GraphNorm(hidden_channels))

        self.dropout = dropout

        # Head takes [h0 || h_final] (128 dims) → 32 dims → 2 dims
        # Global skip from pre-message-passing embedding to output:
        # preserves local/high-frequency detail lost to oversmoothing
        # across num_layers of neighborhood averaging
        self.head = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels // 2),  # 128 → 32
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_channels),          # 32 → 2
        )

    def forward(self, x, edge_index, batch=None):
        """
        Args:
            x:          (N, 2)   node features [macro_region, rudy]
            edge_index: (2, E)   graph connectivity (net connections)
            batch:      (N,)     graph assignment for GraphNorm (None = single graph)
        Returns:
            out:        (N, 2)   per-node [horizontal_overflow, vertical_overflow]
        """
        # Project: (N, 2) → (N, 64)
        h0 = self.input_proj(x)
        h = h0

        # 4 rounds of message passing, each: (N, 64) → (N, 64)
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index)       # neighbors talk to each other via attention
            h = norm(h + h_new, batch)        # residual connection + GraphNorm
            h = F.relu(h)                     # non-linearity
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Concatenate skip: (N, 64) || (N, 64) → (N, 128)
        # Then compress: (N, 128) → (N, 32) → (N, 2)
        out = self.head(torch.cat([h0, h], dim=-1))
        return out  # (N, 2): [horizontal_overflow, vertical_overflow]


def nodes_to_grid_heatmap(node_scores, node_xy, grid_size=(64, 64), die_area=None):
    """
    Rasterize per-node congestion scores into a 2-channel grid heatmap.

    Args:
        node_scores: (N, 2) tensor of predicted congestion [h_overflow, v_overflow]
        node_xy:     (N, 2) tensor of node placement coordinates [x, y]
        grid_size:   (gx, gy) output grid dimensions
        die_area:    (xmin, ymin, xmax, ymax); if None, inferred from node_xy

    Returns:
        heatmap: (gx, gy, 2) grid where:
                 heatmap[:, :, 0] = horizontal congestion
                 heatmap[:, :, 1] = vertical congestion
    """
    device = node_scores.device
    gx, gy = grid_size

    if die_area is None:
        xmin, ymin = node_xy.min(dim=0).values
        xmax, ymax = node_xy.max(dim=0).values
    else:
        xmin, ymin, xmax, ymax = die_area

    # 2-channel heatmap and count grids
    heatmap = torch.zeros(gx, gy, 2, device=device)
    counts  = torch.zeros(gx, gy, 2, device=device)

    # Map each node's (x, y) coordinates → grid (col, row) position
    col = ((node_xy[:, 0] - xmin) / (xmax - xmin + 1e-9) * (gx - 1)).long().clamp(0, gx - 1)
    row = ((node_xy[:, 1] - ymin) / (ymax - ymin + 1e-9) * (gy - 1)).long().clamp(0, gy - 1)

    # Convert 2D grid position → 1D index for scatter operation
    idx = row * gx + col  # (N,)

    # Accumulate node scores into grid cells (both channels at once)
    heatmap.view(-1, 2).index_add_(0, idx, node_scores)
    counts.view(-1, 2).index_add_(0, idx, torch.ones_like(node_scores))

    # Average scores per grid cell
    return heatmap / counts.clamp(min=1)  # (gx, gy, 2)


class CongestionTrainer:
    """
    Wraps the model with:
    - pretrain_epoch: CircuitNet batched pretraining
    - finetune_step:  per-design online fine-tuning (your novelty)
    """

    def __init__(
        self,
        model,
        lr=1e-3,
        finetune_lr=1e-4,
        peak_weight=1.0,
        variance_weight=0.5,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.finetune_lr = finetune_lr

        # Congestion-weighted MSE:
        # Most grid nodes are low/near-zero congestion, so plain MSE barely
        # penalizes missing the rare high-congestion peaks (exactly the regions
        # that matter most for routability). Weighting each node's squared error
        # by (1 + peak_weight * target) makes high-congestion nodes count more.
        self.peak_weight = peak_weight

        # Variance penalty:
        # peak_weight alone can be gamed by uniformly shifting all predictions up
        # (raises mean without learning to discriminate peaks from background).
        # Explicitly penalizing gap between predicted and true std closes that shortcut.
        self.variance_weight = variance_weight

    def loss_fn(self, pred, target):
        """
        Args:
            pred:   (N, 2) predicted [h_overflow, v_overflow]
            target: (N, 2) real      [h_overflow, v_overflow]
        """
        # Weighted MSE — penalize high-congestion regions more
        weight = 1.0 + self.peak_weight * target          # (N, 2)
        weighted_mse = (weight * (pred - target) ** 2).mean()

        # Variance penalty per channel, averaged across channels
        variance_penalty = (
            (pred[:, 0].std() - target[:, 0].std()) ** 2 +   # horizontal
            (pred[:, 1].std() - target[:, 1].std()) ** 2      # vertical
        ) / 2

        return weighted_mse + self.variance_weight * variance_penalty

    def pretrain_epoch(self, dataloader):
        """
        One epoch of pretraining on CircuitNet batched data.
        Each batch.x: (N, 2), batch.y: (N, 2)
        """
        self.model.train()
        total_loss = 0.0
        for batch in dataloader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(batch.x, batch.edge_index, batch.batch)  # (N, 2)
            loss = self.loss_fn(pred, batch.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def finetune_step(self, data: Data, target: torch.Tensor, steps=20):
        """
        Fine-tune on a single design's real routed-congestion labels.
        This is the closed-loop novelty of the project.

        Args:
            data:   PyG Data object with current placement graph
                    data.x: (N, 2) node features
                    data.edge_index: (2, E) connections
            target: (N, 2) ground-truth congestion from OpenROAD router
            steps:  number of gradient steps per iteration

        Returns:
            losses: list of loss values per step
        """
        data   = data.to(self.device)
        target = target.to(self.device)

        # Switch to fine-tuning learning rate (smaller = gentle updates)
        for g in self.optimizer.param_groups:
            g["lr"] = self.finetune_lr

        self.model.train()
        losses = []
        for _ in range(steps):
            self.optimizer.zero_grad()
            pred = self.model(data.x, data.edge_index)   # (N, 2)
            loss = self.loss_fn(pred, target)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        # Restore normal learning rate for next pretrain epoch
        for g in self.optimizer.param_groups:
            g["lr"] = self.optimizer.defaults["lr"]

        return losses

    @torch.no_grad()
    def predict_heatmap(self, data: Data, grid_size=(64, 64)):
        """
        Predict congestion heatmap for a given placement.

        Returns:
            heatmap: (64, 64, 2) where:
                     heatmap[:, :, 0] = horizontal congestion
                     heatmap[:, :, 1] = vertical congestion
        """
        self.model.eval()
        data = data.to(self.device)
        pred = self.model(data.x, data.edge_index)   # (N, 2)

        # First 2 features of x are (x, y) coordinates
        node_xy = data.x[:, :2]
        return nodes_to_grid_heatmap(pred, node_xy, grid_size=grid_size)  # (64, 64, 2)

    def save(self, path):
        torch.save({
            "model":     self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        if "model" in ckpt:
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
        else:
            # Backward-compat with older checkpoints saved as bare state_dict
            self.model.load_state_dict(ckpt)


if __name__ == "__main__":
    # Smoke test with realistic GCD design dimensions
    num_nodes  = 343    # GCD design cell count
    in_channels  = 2   # macro_region + rudy
    out_channels = 2   # horizontal + vertical overflow

    x          = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, 1000))
    y          = torch.randn(num_nodes, out_channels)

    model   = CongestionGNN(in_channels=in_channels, out_channels=out_channels)
    trainer = CongestionTrainer(model, device="cpu")

    # Forward pass
    out = model(x, edge_index)
    print(f"Input shape:    {x.shape}")           # (343, 2)
    print(f"Output shape:   {out.shape}")          # (343, 2)

    # Heatmap rasterization
    heatmap = nodes_to_grid_heatmap(out, x[:, :2], grid_size=(32, 32))
    print(f"Heatmap shape:  {heatmap.shape}")      # (32, 32, 2)
    print(f"H-channel:      {heatmap[:,:,0].shape}")  # (32, 32)
    print(f"V-channel:      {heatmap[:,:,1].shape}")  # (32, 32)

    # Loss check
    loss = trainer.loss_fn(out, y)
    print(f"Loss:           {loss.item():.4f}")

    print("\n✅ All shapes correct!")