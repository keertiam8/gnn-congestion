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
- `gnn/gpdl.py` — CircuitNet's official GPDL model (CNN U-Net, not a GNN),
  reconstructed to exactly match `checkpoints/congestion.pth`'s state_dict.
  Used only as a pretrained baseline predictor — architecturally incompatible
  with `CongestionGNN`, so it can't warm-start it.
- `scripts/` — data extraction / OpenROAD-report parsing (WIP);
  `eval_baseline.py` runs the pretrained GPDL baseline against raw CircuitNet
  samples and reports NRMSE/SSIM (the number the closed-loop GNN must beat).
- `data/` — CircuitNet pretraining data + per-design extracted graphs/labels.
- `checkpoints/` — saved model weights (pretrained + per-design fine-tuned).
  `congestion.pth` = CircuitNet's official pretrained GPDL weights.
- `results/` — predicted heatmaps, congestion reports, eval outputs.

## Conventions
- Node features start with `(x, y)` coordinates first — `predict_heatmap` in
  `model.py` assumes this ordering; keep it consistent in the extraction script.
- Model runs on GPU in Colab; local WSL usage is CPU-only (inference/testing).

## Status (as of last session)
- OpenROAD build in progress in WSL2 (cmake configure/build).
- `gnn/model.py` skeleton written; `torch_geometric` not yet installed locally.
- Confirmed `checkpoints/congestion.pth` is CircuitNet's pretrained GPDL
  (CNN, in_channels=3, out_channels=1) — built `gnn/gpdl.py` +
  `scripts/eval_baseline.py` to get a baseline NRMSE/SSIM number before
  building the closed loop. Not yet run against real data (no raw CircuitNet
  `.npy` samples downloaded locally yet — `data/` is still empty).
- Next: download a raw CircuitNet congestion sample subset, run
  `eval_baseline.py` to get the actual baseline numbers, then OpenROAD GCD
  example run, data extraction script, push to `gnn-congestion` GitHub repo.
