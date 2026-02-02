import re
import wandb
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ENTITY  = "jared-weissberg"
PROJECT = "diffusion-vs-ar"

RUNS = [
    ("AR_25M_40k",     "d1qo1lmk"),
    ("AR_50M_40k",     "wan6cwlc"),
    ("AR_100M_40k",    "yrijti28"),
    ("MDM_25M_40k",    "lxlcyfxv"),
    ("MDM_50M_40k",    "j61zcfse"),
    ("MDM_100M_40k",   "h68cb660"),
]

LINESTYLE = {"25M": "-", "50M": "--", "100M": ":"}

def parse_mode_cap(label: str):
    m = re.match(r"^(AR|MDM)_(25M|50M|100M)_", label)
    if not m:
        raise ValueError(f"Could not parse label: {label}")
    return m.group(1), m.group(2)

def get_n_params(run) -> float:
    for key in ["hparams/n_params", "n_params"]:
        v = run.summary.get(key, None)
        if v is not None:
            return float(v)
    for key in ["hparams/n_params", "n_params"]:
        v = run.config.get(key, None)
        if v is not None:
            return float(v)
    raise RuntimeError(f"Run {run.id}: could not find n_params in summary or config.")

api = wandb.Api()

colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1"])
MODE_COLOR = {"AR": colors[0], "MDM": colors[1]}

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

    # --- recompute FLOPs from val/step (trust val/step; ignore logged flops/tokens) ---
    WORLD_SIZE = 8  # you ran torchrun --nproc_per_node=8

    if "val/step" not in df_val.columns or df_val["val/step"].notna().sum() == 0:
        raise RuntimeError(f"{label}: missing val/step; can't recompute FLOPs cleanly.")

    n_params = get_n_params(run)
    batch_size = int(run.config["batch_size"])
    seq_len = int(run.config["seq_len"])

    toks_per_step_true = batch_size * seq_len * WORLD_SIZE
    tokens_seen_true = df_val["val/step"].astype(float) * toks_per_step_true

    df_val["flops"] = 6.0 * n_params * tokens_seen_true

    df_val = df_val[df_val["flops"] > 0].copy()
    df_val = df_val.sort_values("val/step")

    print(
        label, "mode", mode,
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

plt.xlabel("FLOPs")
plt.ylabel("Validation loss")
plt.ylim(2, 10)

# optional: uncomment for paper-style compute plots
# plt.xscale("log")

legend_elements = [
    Line2D([0], [0], color=MODE_COLOR["AR"], lw=2, label="Autoregressive (AR)"),
    Line2D([0], [0], color=MODE_COLOR["MDM"], lw=2, label="Masked Diffusion (MDM)"),
    Line2D([0], [0], color="black", lw=2, linestyle="-",  label="25M tokens"),
    Line2D([0], [0], color="black", lw=2, linestyle="--", label="50M tokens"),
    Line2D([0], [0], color="black", lw=2, linestyle=":",  label="100M tokens"),
]
plt.legend(
    handles=legend_elements,
    frameon=False,
    ncol=2,
    loc="upper right",
)

plt.tight_layout()
plt.savefig("modelb_val_loss_vs_flops_overlay.png", dpi=300, bbox_inches="tight")
plt.close()
