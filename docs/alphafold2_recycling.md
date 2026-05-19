# Recycling in AlphaFold 2

*Based on AF2 Supplement Algorithms 30–31 and Section 1.11.*

## What It Is

Recycling makes the network deeper and allows the input features (e.g. MSA sequences) to be **resampled each cycle**, without proportionally increasing parameters or training cost. At inference, recycling yields a recurrent network with shared weights running for N_cycle = 4 iterations. The initial carried state is zeros; the network learns to handle this special case.

## What Gets Carried Between Cycles

At the end of each cycle c, three tensors are passed to cycle c+1 alongside fresh input features:

| Carried quantity | Shape | Source |
|---|---|---|
| `m_prev` — single representation | `[L, C_s]` | Structure module output |
| `z_prev` — pair representation | `[L, L, C_z]` | Evoformer output |
| `x_prev` — predicted Cα positions | `[L, 3]` | Structure module output |

These are injected at the start of the next cycle before the Evoformer runs:

```
m[0]  += Linear(LayerNorm(m_prev))       # added to MSA first row
z     += LayerNorm(z_prev)               # added to pair representation
z     += BinnedDistanceEncoding(x_prev)  # Cα distances → pair features
```

Importantly, **each cycle receives fresh `input_features_c`** — the MSA is resampled per cycle, so the network sees different MSA subsets across iterations.

## Training Scheme (Algorithm 31): Monte Carlo Recycling

Full unrolling across N_cycle iterations would cost 300% extra training time. Instead, AF2 uses an approximate scheme that costs only ~37.5% extra:

1. **Sample** N' uniformly from {1, …, N_cycle} (same sample for all batch elements)
2. **Forward-only** for cycles c = 1, …, N'−1 (no backward pass, no stored activations)
3. **Forward + backward** for cycle c = N' (loss computed here, gradients flow)
4. **Skip** cycles c = N'+1, …, N_cycle entirely

This is an unbiased Monte Carlo estimate of the average loss across all iterations:

$$\frac{1}{N_{\text{cycle}}} \sum_{c=1}^{N_{\text{cycle}}} \text{loss}(\text{outputs}_c)$$

**Stop gradient** between cycles: gradients from iteration N' are blocked from flowing into iterations c < N'. This is what makes skipping the earlier backward passes valid — there's nothing to propagate back through.

### Why This Is Efficient

With gradient checkpointing (rematerialisation), a forward pass costs ~25% of a full forward+backward round. The average number of forward passes is (N_cycle + 1) / 2, so the extra cost per step is:

```
extra_cost = (N_cycle - 1) / 2 * 25% = 37.5%   (for N_cycle = 4)
```

vs. 300% with full unrolling (3 extra full forward+backward passes).

The random sampling of N' also acts as an **auxiliary loss** — the network is forced to produce plausible structures mid-way through inference, not just at the final iteration.

## In This Codebase

`model.n_recycles` in the YAML config sets N_cycle (default: 3, meaning 4 total passes). In `submission.py`, `_build_openfold_batch` adds a recycling dimension to every feature tensor:

```python
# shape becomes [..., n_iters] where n_iters = n_recycles + 1
out = {k: v.unsqueeze(-1).expand(*v.shape, n_iters) for k, v in out.items()}
```

`AlphaFold.forward()` slices `[..., cycle_no]` per iteration. Note: the current submission does **not** resample the MSA between cycles (all cycles see the same MSA slice), so this is a simplification relative to the paper.
