"""
Pretrain CongestionGNN on preprocessed CircuitNet graphs.
Intended to run on Colab (GPU). Point --data at the .pt graphs produced by
preprocess_circuitnet.py (e.g. after syncing data/circuitnet_graphs via git).

Usage:
    python scripts/train_pretrain.py --data data/circuitnet_graphs --epochs 50 --out checkpoints/pretrained.pt
"""

import argparse
import os
import sys
import glob
import torch
from torch_geometric.loader import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from gnn.model import CongestionGNN, CongestionTrainer


def load_graphs(data_dir):
    paths = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
    if not paths:
        raise FileNotFoundError(f"No .pt graphs found in {data_dir}. Run preprocess_circuitnet.py first.")
    return [torch.load(p, weights_only=False) for p in paths]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--out", default="checkpoints/pretrained.pt")
    args = parser.parse_args()

    graphs = load_graphs(args.data)
    print(f"Loaded {len(graphs)} graphs")

    num_val = max(1, int(len(graphs) * args.val_split))
    train_graphs, val_graphs = graphs[num_val:], graphs[:num_val]

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size)

    in_channels = graphs[0].x.shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"in_channels={in_channels}, device={device}")

    model = CongestionGNN(in_channels=in_channels)
    trainer = CongestionTrainer(model, device=device)

    best_val = float("inf")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = trainer.pretrain_epoch(train_loader)

        trainer.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = trainer.model(batch.x, batch.edge_index)
                val_loss += trainer.loss_fn(pred, batch.y).item()
        val_loss /= len(val_loader)

        print(f"epoch {epoch:03d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            trainer.save(args.out)
            print(f"  -> saved best checkpoint to {args.out}")

    print("Done.")


if __name__ == "__main__":
    main()
