import wandb
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------
# W&B setup
# ------------------------
ENTITY  = "jared-weissberg"
PROJECT = "diffusion-vs-ar"

RUNS = [
    ("N=2", "o3i9nyqq"),
    ("N=4", "pys7ikcp"),
    ("N=6", "9s13cor6"),
]

MAX_TOKENS = 25_000_000  # token cap for these runs

# Line styles for paper-safe plotting
LINESTYLES = {
    "N=2": "-",
    "N=4": "--",
    "N=6": ":",
}

# ------------------------
# Plot
# ------------------------
api = wandb.Api()
plt.figure(figsize=(6, 4))

for label, run_id in RUNS:
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")

    # Pull full history (do not filter keys)
    df = pd.DataFrame(run.scan_history())

    # Keep only validation points
    df = df[df["val/loss"].notna()].copy()

    # Compute epochs from logged tokens
    df["epochs"] = df["val/tokens_seen"].astype(float) / float(MAX_TOKENS)
    df = df.sort_values("epochs")

    plt.plot(
        df["epochs"],
        df["val/loss"],
        linestyle=LINESTYLES[label],
        linewidth=2,
        label=label,
    )

# ------------------------
# Axes + formatting
# ------------------------
plt.xlabel("Training epochs")
plt.ylabel("Validation loss")
plt.xlim(left=0)
plt.ylim(4.5,7)          # cut everything above 7
plt.legend(title="Num orderings")
plt.tight_layout()

# ------------------------
# Save for paper
# ------------------------
# plt.savefig("permute_val_loss_vs_epochs.pdf", bbox_inches="tight")
plt.savefig("permute_val_loss_vs_epochs.png", dpi=300, bbox_inches="tight")
plt.close()
