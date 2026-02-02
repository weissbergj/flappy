# Recreating Results from *Diffusion Beats Autoregressive in Data-Constrained Settings*

This repo contains a time-constrained replication of several core claims from [Prabhudesai et al., *Diffusion Beats Autoregressive in Data-Constrained Settings* (2024)](https://arxiv.org/pdf/2507.15857). 
I compare Autoregressive (AR) vs masked Diffusion language modeling (MDM) under limited unique data and repeated passes.

## What I test
I focus on four key questions: 

1. Do Diffusion models outperform Autoregressive models in data-constrained settings?
2. When does this occur?
3. What is the interactive scaling law behind these constraints?
4. Why would Diffusion models outperform Autoregressive models in data-constrained settings?

To investigate these questions, I construct a comparable architecture across both models and test these models' performance on varied token sizes and compute. I then extrapolate a potential scaling law from points where the models' losses intersect and attempt to isolate the reason for a Diffusion model's potentially superior performance in data-constrained settings. My findings are largely consistent with Prabhudesai et al. (2025).

## Report

**Full results:** [View full report](Recreating%20Results%20from%20Diffusion%20Beats%20Autoregressive%20in%20Data-Constrained%20Settings.pdf)


## Quickstart

### 1) Install dependencies
bash install.sh

### 2) Download / prepare dataset
bash download.sh

### 3) Run all main experiments (AR + Diffusion)
bash run_all.sh

### Notes
- Ensure TRAIN_BIN and VAL_BIN point to the correct dataset files.
- To change the architecture, modify the entrypoint from src.train to src.train_minimal inside run_all.sh.
