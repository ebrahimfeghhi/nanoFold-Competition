---
marp: true
theme: default
paginate: true
style: |
  section { font-size: 1.35rem; }
  pre  { font-size: 0.68rem; line-height: 1.4; }
  code { font-size: 0.72rem; }
  h1   { font-size: 1.9rem; }
  h2   { font-size: 1.55rem; }
  table { font-size: 1.1rem; }
  .file { color: #888; font-size: 0.85rem; font-style: italic; }
---

<!--
RENDERING
  marp docs/af2_presentation.md --html -o af2_presentation.html
  open af2_presentation.html          # present in any browser
  marp docs/af2_presentation.md --pdf -o af2_presentation.pdf   # for Google Slides import

~35–40 minute talk. Aim for ~1 min per slide; spend more on slides 13–19.
-->

---

## Slide 1: AlphaFold2 + minAlphaFold2

**Architecture, codebase, and how to navigate it**

Assumes: familiarity with Transformers. No biology required.

Codebase: `third_party/minAlphaFold2/minalphafold/`

> Everything shown is runnable code from that directory.

---

## Slide 2: The Protein Folding Problem

A **protein** is a sequence of amino acids — think a string over a 20-character alphabet.

That string **folds** into a specific 3D shape, and the shape determines what the protein does (enzyme, antibody, motor, receptor…).

The task: given the sequence, predict the 3D coordinates of every atom.

Why is this hard? ~3¹⁰⁰ possible conformations for a 100-residue protein. 

---

## Slide 3: The Key Insight — Co-evolution

Align the protein of interest against thousands of homologs (a **Multiple Sequence Alignment, MSA**). When two residues are in physical contact, mutations at one create pressure for compensating mutations at the other.

```
Pos:       1  2  3  4  5  6  7  8  9  10
           ↓           ↓
Sp 1:      K  A  V  L  E  T  G  W  P  R
Sp 2:      K  S  I  L  E  T  A  W  P  R
Sp 3:      R  A  V  M  D  T  G  F  P  K
Sp 4:      R  G  V  L  D  S  G  W  P  K
Sp 5:      K  A  L  L  E  T  G  W  Q  R
```

Positions 1 & 5 co-vary: K↔E, R↔D. 

**If two columns co-vary across species, those residues are likely close in 3D space**

---

## Slide 4: Codebase at a Glance

| File | Responsibility |
|---|---|
| `data.py` | Raw NPZ → cropped, masked, batched tensors |
| `embedders.py` | Input encoding, templates, extra MSA, all triangle ops |
| `evoformer.py` | Evoformer block (Alg 6) + MSA row attention (Alg 7) |
| `structure_module.py` | IPA, backbone update, all-atom coordinates |
| `geometry.py` | Ground-truth frames & torsion angles from PDB coords |
| `losses.py` | FAPE + 5 auxiliary losses, loss weighting |
| `trainer.py` | Training loop, grad accumulation, EMA, checkpointing |
| `model.py` | Top-level `AlphaFold2` — recycling loop, wires everything |
| `model_config.py` | Typed config dataclass (all architectural hyperparameters) |

---

## Slide 5: The Two Representations

Everything in AF2 flows through two parallel tensors updated by the Evoformer:

**`m` — MSA representation** `(N_seq, N_res, c_m)`
One row per MSA sequence, one column per residue. 

**`z` — Pair representation** `(N_res, N_res, c_z)`
One cell per *pair* of residue positions. Encodes the model's current belief regarding the relationship between pairs of residues.

**Two cross-stream couplings only:**
- Pair → MSA: `z` biases row-attention weights in `m`
- MSA → Pair: outer product mean of `m` writes into `z`
 
Structure Module reads `z` (and a slice of `m`) to place atoms.

---

## Slide 6: Data Pipeline — Cropping to a Fixed Window

Proteins can be thousands of residues long but GPU memory is fixed — AF2 trains on random 256-residue crops.

<span class="file">minalphafold/data.py · crop_example() ~line 340</span>

```python
start = _crop_start(length, crop_size=crop_size, training=training)
residue_slice = slice(start, start + crop_size)

cropped["aatype"]           = example["aatype"][residue_slice]       # (crop_size,)
cropped["msa"]              = example["msa"][:, residue_slice]       # (N_seq, crop_size)
cropped["template_aatype"]  = example["template_aatype"][:, residue_slice]
cropped["atom14_positions"] = example["atom14_positions"][residue_slice]  # supervision
```


---

## Slide 7: Building `msa_feat` — 49 Dimensions

The model doesn't receive raw residue tokens — it receives a 49-dimensional feature vector per MSA cell. Beyond residue identity (23-dim one-hot), the vector encodes deletion statistics and per-column amino acid frequencies, giving the network a rich evolutionary prior before any learned computation runs.

<span class="file">minalphafold/data.py · build_msa_feat() ~line 642</span>

```python
msa_feat = torch.cat([
    F.one_hot(masked_msa, num_classes=23).float(),        # 23: residue type (masked)
    (deletions > 0).float().unsqueeze(-1),                #  1: deletion present?
    transformed_deletions(deletions).unsqueeze(-1),       #  1: deletion count (scaled)
    cluster_profile,                                      # 23: per-column AA frequencies
    transformed_deletions(cluster_deletion_mean).unsqueeze(-1),  # 1: mean deletions
], dim=-1)                                                # → (N_seq, N_res, 49)
```

---

## Slide 8: BERT-Style MSA Masking

AF2 borrows BERT's masking trick: randomly corrupt 15% of MSA tokens and train the model to reconstruct them. This forces the network to use surrounding MSA columns and the pair rep z to fill in gaps — the primary driver of co-evolution learning in the Evoformer.

<span class="file">minalphafold/data.py · masked_msa_inputs() ~line 596</span>

```python
bert_mask = (torch.rand(msa.shape) < 0.15).float()   # 15% positions masked

replacement_probs  = hhblits_profile * 0.10           # 10%: sample from profile
replacement_probs += F.one_hot(token).float() * 0.10  # 10%: keep original token
replacement_probs += uniform_over_20_aa * 0.10        # 10%: random amino acid
replacement_probs  = F.pad(replacement_probs,         # 70%: mask token
                           (0, 1), value=0.70)
corrupted[bert_mask.bool()] = sample_categorical(replacement_probs[bert_mask.bool()])
```

---

## Slide 9: InputEmbedder — Initializing `m` and `z`

The pair rep z needs a cell for every residue pair — N² cells. AF2 builds this cheaply with a broadcast outer-sum: two (N, c_z) projections added together. Each cell (i,j) immediately encodes "what is residue i" + "what is residue j" + "how far apart are they" — O(N) computation, O(N²) output.

<span class="file">minalphafold/embedders.py · InputEmbedder.forward() ~line 70</span>

```python
a = self.linear_target_feat_1(target_feat)   # (batch, N_res, c_z)
b = self.linear_target_feat_2(target_feat)   # (batch, N_res, c_z)

# Outer sum: z[i,j] = a[i] + b[j]  — all N_res² pairs in one broadcast
z = a.unsqueeze(-2) + b.unsqueeze(-3)        # (batch, N_res, N_res, c_z)
z += self.rel_pos(residue_index)             # add relative position encoding

m = self.linear_target_feat_3(target_feat).unsqueeze(1) + self.linear_msa(msa_feat)
```

---

## Slide 10: RelPos — Relative Not Absolute

Training crops arbitrary windows from variable-length proteins — absolute position 47 in one crop is unrelated to position 47 in another. Encoding the *relative* distance between pairs (clipped to ±32) is always meaningful regardless of crop location or chain length.

<span class="file">minalphafold/embedders.py · RelPos.forward() ~line 113</span>

```python
d  = residue_index[:, :, None] - residue_index[:, None, :]  # (batch, N_res, N_res)
d  = d.clamp(-32, 32) + 32                                  # shift to [0, 64]
oh = F.one_hot(d.long(), 65).float()                        # 65-bin one-hot
return self.linear(oh)                                       # (batch, N_res, N_res, c_z)
```

---

## Slide 11: Recycling Embedder (Algorithm 32)

After one full forward pass, AF2 feeds its own output back in as the next cycle's starting point — iterative refinement. The previous single rep, pair rep, and Cβ coordinates are all injected. On cycle 0, these are all zeros and have no effect.

<span class="file">minalphafold/model.py · AlphaFold2.forward() ~line 268</span>

```python
# Inject previous cycle (LayerNorm rescales before adding)
msa_repr[:, 0, :, :] += self.recycle_norm_s(single_rep_prev)  # first MSA row only

pair_repr += self.recycle_norm_z(z_prev)                       # full pair rep
pair_repr += self.recycle_linear_d(                            # pseudo-Cβ distances
    recycling_distance_bin(x_prev, n_bins=15)                  # (batch, N_res, N_res, 15)
)
```

---

## Slide 12: The Evoformer Block (Algorithm 6)

48 Evoformer blocks refine m and z in lockstep. Each block first updates m using z as a bias, then writes co-evolution signal from m back into z, then runs triangle operations to enforce geometric consistency. The two representations stay coupled — neither is updated independently.

<span class="file">minalphafold/evoformer.py · Evoformer.forward() ~line 70</span>

```python
# MSA updates
m = m + dropout(self.msa_row_att(m, z))  # z biases attention weights
m = m + self.msa_col_att(m)              # cross-species signal
m = m + self.msa_transition(m)           # 2-layer MLP
z = z + self.outer_mean(m)              # only path: m → z
# Pair updates (geometric consistency)
z = z + dropout(self.triangle_mult_out(z))
z = z + dropout(self.triangle_mult_in(z))
z = z + dropout(self.triangle_att_start(z))
z = z + self.pair_transition(z)
```

---

## Slide 13: MSA Row Attention + Pair Bias (Algorithm 7)

Standard multi-head self-attention within each MSA row — but z injects a per-head bias into every attention score. As z accumulates structural knowledge across recycling cycles, it increasingly steers attention toward geometrically relevant residue pairs. Gating (sigmoid ⊙ output) lets each module scale down its contribution when uncertain.

<span class="file">minalphafold/evoformer.py · MSARowAttentionWithPairBias ~line 97</span>

```python
# z → per-head bias: (batch, N_res, N_res, n_heads)
B = self.linear_pair(layer_norm(z)).permute(0, 3, 1, 2).unsqueeze(1)

# Standard QK scores + pair bias
scores = einsum('bsihd, bsjhd -> bshij', Q, K) / sqrt(c) + B
attn   = softmax(scores, dim=-1)

# Gated output: sigmoid(linear(m)) ⊙ attended_values
out = sigmoid(self.linear_gate(m)) * einsum('bshij, bsjhd -> bsihd', attn, V)
```

---

## Slide 14: MSA Column Attention (Algorithm 8)

Row attention asks "how should residues relate within this sequence?" — column attention asks "across all organisms, how does this position vary?" AF2 attends along both axes of the MSA matrix to extract signal from every direction.

<span class="file">minalphafold/embedders.py · MSAColumnAttention.forward() ~line 608</span>

```python
# Contracts over sequence dims (s, t) instead of residue dims (i, j)
scores = einsum('bsihd, btihd -> bihst', Q, K) / sqrt(c)
attn   = softmax(scores, dim=-1)
values = einsum('bihst, btihd -> bsihd', attn, V)
```

Row attention: "given what z knows about structure, how should residues relate?"
Column attention: "across all organisms, how does this residue position covary?"

---

## Slide 15: Outer Product Mean (Algorithm 10)

Co-evolution lives in the MSA — but the Structure Module reads z. Outer product mean is the bridge: for each residue pair (i,j), compute the outer product of their MSA vectors across all sequences and average. Co-varying positions reinforce; uncorrelated ones average to near-zero. This is how evolutionary signal becomes a geometric prior.

<span class="file">minalphafold/embedders.py · OuterProductMean.forward() ~line 715</span>

```python
A = self.linear_left(m)    # (batch, N_seq, N_res, c_hidden)
B = self.linear_right(m)   # (batch, N_seq, N_res, c_hidden)

# Outer product summed over sequences
outer = einsum('bsic, bsjd -> bijcd', A, B)  # (batch, N_res, N_res, c, c)
mean  = outer / N_seq                        # average across MSA depth

return self.linear_out(mean.reshape(..., c*c))  # → (batch, N_res, N_res, c_z)
```

---

## Slide 16: Triangle Multiplicative Updates (Algorithms 11 & 12)

If residue i is close to j, and j is close to k, then i can't be arbitrarily far from k. The pair rep z stores beliefs about every pair but nothing enforces mutual consistency across triples. Triangle updates fix this: to update edge (i,j), pool information from all intermediate nodes k.

> **[Figure: AF2 Supplementary — triangle diagram, three nodes i-j-k with labeled edges z_ij, z_ik, z_jk]**

```
Outgoing (Alg 11):  z_{i,j} ← gate ⊙ Linear( Σ_k  a_{i,k} ⊙ b_{j,k} )
Incoming (Alg 12):  z_{i,j} ← gate ⊙ Linear( Σ_k  a_{k,i} ⊙ b_{k,j} )
```

Cost: O(N_res³ × c) — the most expensive operation in AF2.

---

## Slide 17: Triangle Mult — Code

The outgoing and incoming variants differ only in which direction the triangle edge points. Everything else is identical: gated projections, pool over k, layer norm, output gate. The geometric enforcement is entirely in the einsum index pattern.

<span class="file">minalphafold/embedders.py · TriangleMultiplicationOutgoing ~line 745</span>

```python
# Gated projections of z
A = sigmoid(self.gate1(z)) * self.linear1(z)  # (batch, N_res, N_res, c)
B = sigmoid(self.gate2(z)) * self.linear2(z)
# Outgoing: z_{i,j} ← pool over k using edges z_{i,k} and z_{j,k}
vals = einsum('bikc, bjkc -> bijc', A, B)
# Incoming variant — only this einsum differs:
# vals = einsum('bkic, bkjc -> bijc', A, B)
out = sigmoid(self.gate(z)) * self.out_linear(layer_norm(vals))
```

---

## Slide 18: Triangle Self-Attention (Algorithms 13 & 14)

An attention-weighted version of the same triangle idea. Instead of summing over k uniformly, attend softly — with the third triangle edge providing a bias on the attention scores. Four triangle operations per block (2 multiplicative + 2 attention) enforce geometric consistency from every direction.

<span class="file">minalphafold/embedders.py · TriangleAttentionStartingNode ~line 813</span>

```python
# Q from z_{i,j}  ·  K/V from z_{i,k}  ·  bias B from z_{j,k}
scores = einsum('bijhd, bikhd -> bijkh', Q, K) / sqrt(c) + B.unsqueeze(1)
attn   = softmax(scores, dim=3)                   # over k
values = einsum('bijkh, bikhd -> bijhd', attn, V)
```

**Ending-node variant (Alg 14):** fix j, attend over `z_{k,j}`, bias from `z_{k,i}`. Only the einsum indices differ: `'bijhd, bkjhd -> bijkh'`.

---

## Slide 19: Extra MSA Stack — Global vs Standard Attention

AF2 processes two MSA tracks: 128 'clustered' sequences through the main Evoformer, and 1024 'extra' sequences through a cheaper parallel stack. The extra stack can't afford per-head K/V for 1024 sequences — so column attention pools K and V globally (averaged across sequences), keeping only Q per-head.

<span class="file">minalphafold/embedders.py · MSAColumnGlobalAttention ~line 487</span>

```python
# Standard column attention: K, V are per-head  → O(N_seq · N_heads · c)
# Global column attention:   K, V are shared    → O(N_seq · c) + O(N_heads · c)

self.linear_q = Linear(c_in, num_heads * c)   # per-head query
self.linear_k = Linear(c_in, c)               # shared key   ← no num_heads
self.linear_v = Linear(c_in, c)               # shared value ← no num_heads

Q = linear_q(msa).mean(dim=1)   # (batch, N_res, n_heads, c)
K = linear_k(msa)               # (batch, N_seq, N_res, c)  — shared
```

---

## Slide 20: Structure Module — Rigid Frames + Black-Hole Init

Each residue gets a **rigid frame** — a rotation + translation defining a local 3D coordinate system. All frames start at the origin with identity rotation ("black-hole" initialization). 8 IPA iterations then move frames apart. SE(3)-invariance falls out naturally: rotating the whole protein rotates every frame, leaving pairwise geometry unchanged.

<span class="file">minalphafold/structure_module.py · StructureModule.forward() ~line 198</span>

```python
# All residues start collapsed at the origin
rotations    = eye(3).expand(batch, N_res, 3, 3)  # all identity rotations
translations = zeros(batch, N_res, 3)             # all at origin
```

---

## Slide 21: Invariant Point Attention — IPA (Algorithm 22)

IPA extends standard attention into 3D space. Three components contribute to each attention score: channel-space dot products, a pair bias from z (structural priors from the Evoformer), and a point-distance term computed in each residue's local frame. Because points are expressed in local frames, the attention is invariant to global rotation and translation.

<span class="file">minalphafold/structure_module.py · InvariantPointAttention ~line 316</span>

```python
# 1. Scalar: standard channel attention on single rep s_i
scalar_score = einsum('bihd, bjhd -> bijh', Q_scalar, K_scalar) / sqrt(c)

# 2. Pair bias: structural beliefs from Evoformer
pair_score = self.linear_bias(z)            # (batch, N_res, N_res, n_heads)

# 3. Point: 3D distance in local frame coordinates (invariant to global rotation)
point_score = -softplus(gamma) * point_dist_sq / 2   # large dist → low score

score = (scalar_score + pair_score + point_score) / sqrt(3)
```

---

## Slide 22: BackboneUpdate — 6 Scalars → Frame Update

After each IPA iteration, the single rep projects to just 6 scalars — 3 for rotation (as an implicit unit quaternion) and 3 for translation. The new frame is composed with the current one: `T_i ← T_i ∘ (R_new, t_new)`. Rotations are detached between iterations to prevent gradient lever effects from compounding frame compositions.

<span class="file">minalphafold/structure_module.py · BackboneUpdate.forward() ~line 560</span>

```python
vals = self.linear(s)                    # (batch, N_res, 6)
b, c, d = vals[..., 0], vals[..., 1], vals[..., 2]   # rotation scalars

# Implicit quaternion (1, b, c, d) normalized to unit length
norm = sqrt(1 + b**2 + c**2 + d**2)
a, b, c, d = 1/norm, b/norm, c/norm, d/norm

R = quaternion_to_rotation_matrix(a, b, c, d)  # (batch, N_res, 3, 3)
t = vals[..., 3:]                              # (batch, N_res, 3)
```

---

## Slide 23: Structure Module Loop (Algorithm 20)

8 IPA iterations refine both s (single rep) and the rigid frames. Each iteration: attend with 3D-aware IPA, update s with an MLP, project s to a frame delta and compose. Rotations are detached before the next iteration — translations keep gradients so FAPE can supervise the Evoformer at every layer.

<span class="file">minalphafold/structure_module.py · StructureModule.forward() ~line 236</span>

```python
for l in range(self.num_layers):        # 8 iterations
    s = s + self.IPA(s, z, R, t)       # geometry-aware attention
    s = layer_norm(dropout(s))
    s = s + transition_mlp(s)          # 2-layer MLP + LN + dropout
    R_new, t_new = self.backbone_update(s)      # 6 scalars → Δframe
    t = einsum('bsij,bsj->bsi', R, t_new) + t  # compose translation
    R = einsum('bsij,bsjk->bsik', R, R_new)    # compose rotation
    sidechain = self.sidechain_module(s, s_init, R, t, aatype)
    if l < self.num_layers - 1:
        R = R.detach()  # stop-gradient on rotation (prevents lever effects)
```

---

## Slide 24: Ground-Truth Frames — Gram-Schmidt

FAPE needs a reference frame at every residue. AF2 builds them from three backbone atoms (N, Cα, C) using Gram-Schmidt: N→Cα becomes the x-axis, the component of Cα→C perpendicular to x becomes y, and z completes the right-handed basis. Three raw 3D points become a guaranteed orthonormal coordinate frame.

<span class="file">minalphafold/geometry.py · rigid_frame_from_three_points() ~line 246</span>

```python
# Origin = Cα, point_on_neg_x_axis = N, point_on_xy_plane = C
x_axis = safe_normalize(origin - point_on_neg_x_axis)   # N → Cα direction
xy_axis = point_on_xy_plane - origin                    # Cα → C
z_axis = safe_normalize(cross(x_axis, xy_axis))         # perpendicular to plane
y_axis = safe_normalize(cross(z_axis, x_axis))          # complete right-handed basis
R = torch.stack([x_axis, y_axis, z_axis], dim=-1)       # (batch, N_res, 3, 3)
```

---

## Slide 25: Recycling Loop (Algorithm 31)

AF2 runs multiple complete forward passes, feeding each cycle's output back as context for the next. Only the final cycle computes gradients — earlier cycles are free warm-up under no_grad. At training time, the number of cycles is sampled randomly, creating a curriculum over recycling depth.

<span class="file">minalphafold/model.py · AlphaFold2.forward() ~line 231</span>

```python
for i in range(n_cycles):
    is_last = (i == n_cycles - 1)
    with torch.set_grad_enabled(is_last and outer_grad):  # grad on last only
        m, z = self.input_embedder(target_feat, residue_index, msa_feat)
        m[:, 0] += layer_norm(single_rep_prev)  # inject previous m
        z       += layer_norm(z_prev)           # inject previous z
        z       += linear(dist_bins(x_prev))    # inject previous Cβ geometry
        # ... run evoformer + structure module ...
        single_rep_prev = m_first_row.detach()  # detach: no grad across cycles
        z_prev, x_prev = z.detach(), cb_coords.detach()
```

---

## Slide 26: Zero-Output Initialization

With 48 Evoformer blocks stacked, random initialization would cause chaotic signal flow from the start. AF2 initializes every residual block's output projection to zero weights — each block starts as a pure identity/skip connection. The network then learns to turn individual blocks on incrementally during training.

<span class="file">minalphafold/model.py · _initialize_alphafold_parameters() ~line 104</span>

```python
for module in self.modules():
    # Attention + triangle attention modules: zero the output projection
    if class_name in {"MSARowAttentionWithPairBias", "TriangleAttentionStartingNode",
                      "TriangleAttentionEndingNode", "InvariantPointAttention", ...}:
        zero_linear(module.linear_output)

    # MLP modules: zero the final linear layer
    if class_name in {"MSATransition", "PairTransition"}:
        zero_linear(module.linear_down)
```

---

## Slide 27: FAPE Loss (Algorithm 28)

Standard MSE on coordinates isn't SE(3)-invariant — rotating the whole prediction changes the loss. FAPE fixes this: transform atom positions into each residue's local frame, then measure error there. Every residue's frame provides independent supervision, so one bad loop region can't drown out gradient signal from the rest of the chain.

<span class="file">minalphafold/losses.py · frame_aligned_point_error() ~line 49</span>

```python
# Frame inversion: R^{-1} = R^T for rotation matrices; t^{-1} = -R^T t
R_inv = predicted_rotations.transpose(-1, -2)
t_inv = -einsum('...ij,...j->...i', R_inv, predicted_translations)

# Transform predicted and true positions into frame i's local coords
x_pred_local = einsum('...fij,...aj->...fai', R_inv, predicted_positions) + t_inv[..., :, None, :]
x_true_local = einsum('...fij,...aj->...fai', R_inv_true, true_positions) + t_inv_true[..., :, None, :]

err = sqrt(||x_pred_local - x_true_local||² + ε).clamp(max=10.0).mean()
```

---

## Slide 28: All Loss Terms

Six losses supervise different parts of the network simultaneously. DistogramLoss and MSALoss train the Evoformer *directly* before the Structure Module sees anything — critical for warm-starting representations. ViolationLoss starts at weight 0 and only activates during fine-tuning.

<span class="file">minalphafold/losses.py · AlphaFoldLoss ~line 130</span>

| Loss | Weight | Supervises | Source |
|---|---|---|---|
| `BackboneFAPE` | 0.5 | Cα frames, every SM layer | Alg 28 |
| `AllAtomFAPE` | 0.5 | All 14 atoms, final SM output | Alg 28 |
| `DistogramLoss` | 0.3 | Cβ–Cβ distance bins from `z` | Eq 41 |
| `MSALoss` | 2.0 | Reconstruct masked MSA tokens | Eq 42 |
| `PLDDTLoss` | 0.01 | Per-residue confidence (pLDDT) | Alg 29 |
| `StructuralViolationLoss` | 0.0 → 1.0 | Bond lengths, clashes | Eq 47 |

---

## Slide 29: Training Loop

AF2 uses gradient accumulation to simulate large effective batch sizes on limited GPU memory, and exponential moving average (EMA) of weights for inference — smoother and more stable than raw checkpoints. Two-stage protocol: pre-train without violation loss (fast convergence), then fine-tune with all losses active.

<span class="file">minalphafold/trainer.py · fit() ~line 994</span>

```python
for batch in train_loader:
    loss_fn = finetune_loss_fn if use_finetune_loss(step) else pretrain_loss_fn
    outputs = model(**model_inputs_from_batch(batch))
    loss = loss_fn(**loss_inputs_from_batch(batch, outputs)).mean()
    (loss / grad_accum_steps).backward()
    if step % grad_accum_steps == 0:
        clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step(); optimizer.zero_grad()
        ema_model.update_parameters(model)  # EMA used at inference
        save_checkpoint(step, model, ema_model, optimizer)
```

- **Gradient accumulation:** effective batch size = `batch_size × grad_accum_steps`
- **EMA:** exponential moving average — smoother than raw checkpoint; used at inference
- **Checkpoints:** `ckpt_last.pt` (latest) + `ckpt_best.pt` (best val loss)
