# Implementation notes

The deterministic RF ODE convention used by the installed Kontext pipeline is

`x_next = x + (sigma_next - sigma) * model_output`, with sigma decreasing from 1 to 0.

The explicit-noise RF-equivalent SDE is implemented as follows. At `sigma=1`,
`drift=model_output` and `diffusion_coeff=0`. At every later step,
`drift=2*model_output + x/(1-sigma)` and
`diffusion_coeff=sqrt(2*sigma/(1-sigma)*(sigma-sigma_next))`. The update is
`x_next=x+(sigma_next-sigma)*drift+diffusion_coeff*noise`. Drift, coefficient,
noise mixing and update are evaluated in float32 before casting back to bfloat16.
The helper never calls a random-number generator; all noise is supplied explicitly.

The FLUX-Kontext transformer output is used directly. The `v_t=-noise_pred`
naming in a separate syncSDE pipeline is not applied a second time here.

RF-equivalent SDE 的第一个 `sigma=1` inference step 是 deterministic，
`diffusion_coeff=0`。因此，本实验中的 variable Brownian coupling 实际作用于
scheduler 中 earliest three non-zero-diffusion steps，而不是第一个 inference step。

This is a controlled adaptation of the syncSDE RF discretization to native
FLUX-Kontext source-image conditioning. The generation-latent prefix receives
the SDE update; source-conditioning latent tokens remain unchanged. This is not
claimed to be equivalent to syncSDE's RF-inversion pipeline.

Noise reuse is verified through deterministic SeedSequence-derived seeds,
tensor SHA-256 hashes, empirical correlations and a strict elementwise rho=1
identity check. Masks are resized to the VAE latent grid, expanded across latent
channels and passed through the pipeline's official packing operation.

