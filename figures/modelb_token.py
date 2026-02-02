import re
import wandb
import pandas as pd
import matplotlib.pyplot as plt

ENTITY  = "jared-weissberg"
PROJECT = "diffusion-vs-ar"

# label -> run_id (exclude MDM_100M_150k)
RUNS = [
    ("AR_25M_40k",     "d1qo1lmk"),
    ("AR_50M_40k",     "wan6cwlc"),
    ("AR_100M_40k",    "yrijti28"),
    ("MDM_25M_40k",    "lxlcyfxv"),
    ("MDM_50M_40k",    "j61zcfse"),
    ("MDM_100M_40k",   "h68cb660"),
]

MAX_TOKENS = {
    "25M":  25_000_000,
    "50M":  50_000_000,
    "100M": 100_000_000,
}

# line style by data cap (paper-friendly)
LINESTYLE = {
    "25M": "-",
    "50M": "--",
    "100M": ":",
}

def parse_mode_cap(label: str):
    m = re.match(r"^(AR|MDM)_(25M|50M|100M)_", label)
    if not m:
        raise ValueError(f"Could not parse label: {label}")
    return m.group(1), m.group(2)

def fetch_curve(api, run_id: str) -> pd.DataFrame:
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    df = pd.DataFrame(run.scan_history())

    # keep only eval points
    if "val/loss" not in df.columns:
        raise RuntimeError(f"Run {run_id} missing val/loss. Columns: {sorted(df.columns.tolist())}")
    df = df[df["val/loss"].notna()].copy()

    # x-axis: tokens seen
    if "val/tokens_seen" in df.columns and df["val/tokens_seen"].notna().any():
        df["tokens_seen"] = df["val/tokens_seen"].astype(float)
    elif "train/tokens_seen" in df.columns and df["train/tokens_seen"].notna().any():
        df["tokens_seen"] = df["train/tokens_seen"].astype(float)
    else:
        raise RuntimeError(f"Run {run_id} missing tokens_seen columns.")

    df = df.sort_values("tokens_seen")
    return df[["tokens_seen", "val/loss"]]

def plot_mode(mode_to_plot: str, out_stem: str):
    api = wandb.Api()
    plt.figure(figsize=(6.5, 4.2))

    for label, run_id in RUNS:
        mode, cap = parse_mode_cap(label)
        if mode != mode_to_plot:
            continue

        max_tokens = MAX_TOKENS[cap]
        df = fetch_curve(api, run_id)

        # final epochs/reuse for legend
        final_tokens = float(df["tokens_seen"].iloc[-1])
        final_epochs = final_tokens / float(max_tokens)
        ep_round = int(round(final_epochs))

        plt.plot(
            df["tokens_seen"],
            df["val/loss"],
            linestyle=LINESTYLE[cap],
            linewidth=2,
            label=f"{mode} {ep_round} Ep."
        )

    plt.xlabel("Training tokens seen")
    plt.ylabel("Validation loss")
    # plt.ylim(4.5, 7)   # clip to [4.5, 7]
    if mode_to_plot == "AR":
        plt.ylim(4, 7.0)
    else:  # MDM
        plt.ylim(5, 7.0)
    plt.xlim(left=0)
    plt.legend()
    plt.tight_layout()

    # plt.savefig(f"{out_stem}.pdf", bbox_inches="tight")
    plt.savefig(f"{out_stem}.png", dpi=300, bbox_inches="tight")
    plt.close()

# Make two separate plots
plot_mode("AR",  "modelb_AR_val_loss_vs_tokens")
plot_mode("MDM", "modelb_MDM_val_loss_vs_tokens")