# gnn — Closed-loop ML-for-EDA Placement Refinement

## Project goal
GNN predicts a congestion heatmap for a chip design → OpenROAD places cells guided
by that heatmap → OpenROAD routes → real congestion is measured → GNN is
fine-tuned on that design's real labels → repeat 2-3 times per design.

Novel angle: **per-design online learning** (fine-tune on each new design at
inference time), not a standard one-shot train/test split.

## Architecture
- **Local (WSL2)**: OpenROAD (placement + routing) runs via CLI; generates
  placement files and congestion reports.
- **Colab (GPU)**: GNN training — pretrains on CircuitNet, then fine-tunes
  per design.
- **GitHub**: `gnn-congestion` repo bridges the two — clone in Colab, push/pull
  locally.

Data flow: WSL (OpenROAD) → extract metrics → GitHub → Colab (train) →
download weights → WSL (next iteration).

## Repo layout
- `gnn/model.py` — `CongestionGNN` (GATv2-based), `nodes_to_grid_heatmap`
  (rasterizes per-node predictions to a grid), `CongestionTrainer`
  (pretrain_epoch for CircuitNet, finetune_step for per-design online tuning).
- `scripts/` — data extraction / OpenROAD-report parsing (WIP).
- `data/` — CircuitNet pretraining data + per-design extracted graphs/labels.
- `checkpoints/` — saved model weights (pretrained + per-design fine-tuned).
- `results/` — predicted heatmaps, congestion reports, eval outputs.

## Conventions
- Node features start with `(x, y)` coordinates first — `predict_heatmap` in
  `model.py` assumes this ordering; keep it consistent in the extraction script.
- Model runs on GPU in Colab; local WSL usage is CPU-only (inference/testing).

## Status (as of last session)
- OpenROAD build in progress in WSL2 (cmake configure/build).
- `gnn/model.py` skeleton written; `torch_geometric` not yet installed locally.
- Next: OpenROAD GCD example run, data extraction script, push to
  `gnn-congestion` GitHub repo.
