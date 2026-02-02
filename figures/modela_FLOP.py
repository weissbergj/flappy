# modela_val_loss_vs_flops_overlay.py
# PYTHONPATH=.. python modela_FLOP.py
import re
import wandb
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import yaml
import torch

# IMPORTANT: run from repo root or ensure this import works
from src.model import TinyTransformerLM

ENTITY  = "jared-weissberg"
PROJECT = "diffusion-vs-ar"

# Model A run IDs (from your local wandb/model_a folder)
RUNS = [
    ("AR_25M_40k",     "xywzu7v1"),
    ("AR_50M_40k",     "b1h3qv16"),
    ("AR_100M_40k",    "harygqvu"),
    ("MDM_25M_40k",    "eudf8xqs"),
    ("MDM_50M_40k",    "brhr69qn"),
    ("MDM_100M_40k",   "f9rvgn48"),
]

LINESTYLE = {"25M": "-", "50M": "--", "100M": ":"}

VOCAB_SIZE = 50258  # +1 for MASK, matches your repo

def parse_mode_cap(label: str):
    m = re.match(r"^(AR|MDM)_(25M|50M|100M)_", label)
    if not m:
        raise ValueError(f"Could not parse label: {label}")
    return m.group(1), m.group(2)

def wb_cfg_value(v):
    # wandb config.yaml uses {"value": ...}
    if isinstance(v, dict) and "value" in v:
        return v["value"]
    return v

def n_params_from_local_config(run_label: str) -> int:
    """
    Compute Model A n_params deterministically from the local config.yaml for that run label.
    This avoids relying on W&B history/summary (Model A didn't log n_params).
    """
    cfg_path = f"../wandb/model_a/{run_label}/files/config.yaml"
    cfg = yaml.safe_load(open(cfg_path, "r"))

    d_model  = int(wb_cfg_value(cfg["d_model"]))
    n_layers = int(wb_cfg_value(cfg["n_layers"]))
    n_heads  = int(wb_cfg_value(cfg["n_heads"]))
    seq_len  = int(wb_cfg_value(cfg["seq_len"]))

    # Causal for AR, bidirectional for MDM (same as your training code)
    mode, _ = parse_mode_cap(run_label)
    causal = (mode == "AR")

    model = TinyTransformerLM(
        vocab_size=VOCAB_SIZE,
        seq_len=seq_len,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        causal=causal,
    )
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

api = wandb.Api()

colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1"])
MODE_COLOR = {"AR": colors[0], "MDM": colors[1]}

WORLD_SIZE = 8  # torchrun --nproc_per_node=8 for these runs

plt.figure(figsize=(6.7, 4.3))

for label, run_id in RUNS:
    mode, cap = parse_mode_cap(label)
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")

    df = pd.DataFrame(run.scan_history())
    if df.empty:
        raise RuntimeError(f"{label}: empty history")

    if "val/loss" not in df.columns:
        raise RuntimeError(f"{label}: missing val/loss in history columns: {sorted(df.columns.tolist())}")

    # keep only validation points
    df_val = df[df["val/loss"].notna()].copy()

    # We will compute FLOPs from *tokens_seen*, which you DO have in Model A
    # Prefer val/tokens_seen; if missing, fall back to step-based reconstruction.
    n_params = n_params_from_local_config(label)

    if "val/tokens_seen" in df_val.columns and df_val["val/tokens_seen"].notna().any():
        tokens_seen = df_val["val/tokens_seen"].astype(float)
    else:
        # fallback: reconstruct tokens_seen from val/step and tokens_per_step_true
        if "val/step" not in df_val.columns or df_val["val/step"].notna().sum() == 0:
            raise RuntimeError(f"{label}: missing val/tokens_seen and val/step; can't compute FLOPs.")
        batch_size = int(run.config["batch_size"])
        seq_len = int(run.config["seq_len"])
        toks_per_step_true = batch_size * seq_len * WORLD_SIZE
        tokens_seen = df_val["val/step"].astype(float) * toks_per_step_true

    # Dense training FLOPs proxy (standard): 6 * params * tokens
    df_val["flops"] = 6.0 * float(n_params) * tokens_seen

    # drop origin artifacts and sort
    df_val = df_val[df_val["flops"] > 0].copy()
    if "val/step" in df_val.columns and df_val["val/step"].notna().any():
        df_val = df_val.sort_values("val/step")
    else:
        df_val = df_val.sort_values("flops")

    print(
        label, "mode", mode,
        "n_params", n_params,
        "min/max flops", df_val["flops"].min(), df_val["flops"].max(),
        "min/max val_loss", df_val["val/loss"].min(), df_val["val/loss"].max(),
        "npts", len(df_val)
    )

    plt.plot(
        df_val["flops"],
        df_val["val/loss"],
        linestyle=LINESTYLE[cap],
        linewidth=2,
        color=MODE_COLOR[mode],
    )

plt.xlabel("FLOPs (6 * params * tokens)")
plt.ylabel("Validation loss")
plt.ylim(3.5, 8)

# optional for paper-style scaling plots
# plt.xscale("log")

legend_elements = [
    Line2D([0], [0], color=MODE_COLOR["AR"],  lw=2, label="Autoregressive (AR)"),
    Line2D([0], [0], color=MODE_COLOR["MDM"], lw=2, label="Masked Diffusion (MDM)"),
    Line2D([0], [0], color="black", lw=2, linestyle="-",  label="25M tokens"),
    Line2D([0], [0], color="black", lw=2, linestyle="--", label="50M tokens"),
    Line2D([0], [0], color="black", lw=2, linestyle=":",  label="100M tokens"),
]
plt.legend(handles=legend_elements, frameon=False, ncol=2, loc="upper right")

plt.tight_layout()
plt.savefig("modela_val_loss_vs_flops_overlay.png", dpi=300, bbox_inches="tight")
# plt.savefig("modela_val_loss_vs_flops_overlay.pdf", bbox_inches="tight")
plt.close()
