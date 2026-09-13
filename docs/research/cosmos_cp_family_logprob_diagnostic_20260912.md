# Cosmos CP: family forward and logprob-gradient diagnostic

Status: fixed-action scalar logprob checks pass; BF16 gradient equivalence
is not established. Do not promote this CP prototype into production training
or close P0/P1 based on scalar logprob agreement.

## Tested boundary

Pinned released 28-layer Cosmos Predict2.5 2B with original LoRA targets,
rank32/alpha64, using the existing `CosmosPredict25Model.forward_step` and
its conditioned-region/CFG semantics. Native pinned UniPC configuration,
20-step schedule. Compare the full-K/V-gather two-rank prototype against the
same unsharded model on identical small synthetic latent/text conditioning.

For each case, draw one reference action with production `sde_step_with_logprob`
CPS math, noise level 0.7 and fixed generator seed. Both models score that same
action, then differentiate negative mean logprob. This tests a logprob-loss
gradient, not a full GRPO group objective, NFT previous-policy objective,
rollout collection or trajectory-resolver `replay_forward` integration.
Synthetic conditioning is not actual encoded user data or a generated video.

The direct family wrapper requires the runtime precision contract. The first
attempt omitted it and failed with missing `precision` before computation;
its reports are retained. Corrected attempts explicitly install RolePrecision:
FP32/IEEE without outer autocast, then BF16/IEEE with outer autocast. No shared
runtime/dependency edit or production precision-default change occurred.

## Results

All successful cases have 560 finite LoRA gradient tensors, 280 initially
nonzero, with agreement between both rank reports. The predeclared diagnostic
absolute logprob limit stays 1e-3 throughout.

| Case | FP32 logprob max abs | BF16 logprob max abs | BF16 worst tensor gradient relative L2 |
| --- | ---: | ---: | ---: |
| no CFG, index 5, sigma 0.97686875 | 0 | 1.19209e-7 | 0.0827452 |
| no CFG, index 18, sigma 0.02813339 | 0 | 1.50540e-8 | 0.351647 |
| CFG 5, index 18, same sigma | 7.27596e-12 | 1.35464e-6 | 1.15418 |

To avoid confusing small individual gradient sensitivity with overall update
error, add aggregate relative L2 over all parameter gradients (square root of
summed error-norm squares divided by summed reference-norm squares):

| Case | FP32 aggregate gradient relative L2 | BF16 aggregate gradient relative L2 |
| --- | ---: | ---: |
| no CFG, index 18 | 7.10489e-6 | 0.0846904 |
| CFG 5, index 18 | 2.41279e-5 | 0.413710 |

The no-CFG low-noise case was rerun in a fresh directory after adding aggregate
metrics; its scalar result matches the original case. No tolerance was relaxed.
The script's completion status asserts finite values/nonzero gradient presence
and scalar logprob limit, not gradient equivalence. Small average logprob
differences can coexist with significant gradient-direction/magnitude changes.
These observations do not establish a learning-quality regression, but they
invalidate using the scalar pass alone as justification for equivalent training.

## Evidence and disposition

All directories and adjacent `.log` files are under
`/mnt/nvme/outputs/wan22_i2v_cache/`:

- `cosmos_cp_family_nocfg_l40s`: retained missing-precision failure.
- `cosmos_cp_family_nocfg_precision_l40s`: corrected index-5 case.
- `cosmos_cp_family_nocfg_step18_l40s`: first low-noise case.
- `cosmos_cp_family_cfg5_step18_l40s`: CFG stress case with aggregate metrics.
- `cosmos_cp_family_nocfg_step18_global_l40s`: matched aggregate-metric control.

Each contains rank JSON reports; the shared experimental script is
`cosmos_cp_network_probe.py` with `--family-forward`, `--method gather_kv`,
the pinned `--model-path`, and recorded timestep/guidance arguments. It is not
a supported adapter or a production CLI. Candidate `382d0825` stays clean.

All five torchrun sessions are terminal (first failed, four succeeded). Fresh
GPU compute inventory is empty and GPUs 0-1 are released. No Ray or disabled
long queue was started. Hold production CP integration pending gradient-level
diagnosis with the actual trainer precision/reduction contracts. Original
paper-shaped volume, P1 memory target, reward, optimizer and resume gates remain
open; no long experiment was launched to bypass numerical uncertainty.
