# FLUX-Kontext VeloEdit / FreqEdit experiment

This directory contains the code and recorded results for the 15-step FLUX-Kontext edit:

- Prompt: `Add flowers to the dog's mouth`
- Model: `/data15/hyp/weight/FLUX.1-Kontext-dev`
- Seed: `42`
- Guidance scale: `2.5`
- Strengths: `0, 0.25, 0.5, 0.75, 1.0` (`blend_weight = 1 - strength`)
- Wavelet analysis: two-level `db4` DWT, diagnostic only

The experiment code is under `VeloEdit/`. Summary figures and CSV/JSON metrics are in
`VeloEdit/outputs/freqedit_wavelet_dog_flowers/`. Each run also contains the final image,
15-step velocity tensor, and all wavelet coefficient tensors. The high-resolution per-step
PNG previews are intentionally omitted from this Git branch because they add roughly 1 GB;
the raw tensors and numerical metrics are retained for reproducibility.
