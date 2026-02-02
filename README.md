# Recreating Results from *Diffusion Beats Autoregressive in Data-Constrained Settings*

This is a time-constrained retest of four key questions in  
[Prabhudesai et al., *Diffusion Beats Autoregressive in Data-Constrained Settings* (2024)](https://arxiv.org/pdf/2507.15857): Do autoregressive models outperform Diffusion models in data-constrained settings? When does this occur? What is the interactive scaling law behind these constraints? Why would autoregressive models outperform Diffusion models in data-constrained settings?

To investigate these questions, I naively construct a comparable architecture across both models and test these models' performance on varied token sizes and compute. I then extrapolate a potential scaling law from points where the models' losses intersect and attempt to isolate the reason for a Diffusion model's potentially superior performance in data-constrained settings. My findings are largely consistent with Prabhudesai et al. (2025).

**Full results:** [View full report](Recreating Results from Diffusion Beats Autoregressive in Data-Constrained Settings.pdf)

Please install dependencies with 
install.sh  

Download the C4 dataset using
download.sh

To recreate the robust autoregressive and Diffusion runs using Model B architecture, use 
run_all.sh.

Make sure you point to the correct TRAIN_BIN and VAL_BIN. You can adjust the architecture by changing src.train to src.train_minimal.
