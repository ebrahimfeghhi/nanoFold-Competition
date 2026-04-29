# The Structure Module: A thorough explanation

The Structure Module is the final stage of AlphaFold2. It takes the abstract
sequence representations produced by the Evoformer and turns them into real 3D
atom coordinates. This document walks through every component, with linear
algebra and biochemistry refreshers where needed.

---

## 0. Amino acid basics

A protein is a chain of **amino acids** (also called **residues**) linked
together. There are 20 standard amino acids. Every one of them shares the same
core structure — the **backbone** — but differs in a side group called the
**sidechain**.

### The backbone

The backbone of every amino acid looks like this:

```
  H   R
  |   |
  N - Cα - C = O
  |             |
(prev)        (next residue)
```

Three atoms are key:
- **N** — nitrogen, the start of the residue
- **Cα** — alpha carbon, the central carbon bonded to the sidechain
- **C** — carbonyl carbon, the end of the residue (also called C')

These three atoms define a **plane** — the peptide plane — and their positions
determine the overall shape of the protein backbone. When the Structure Module
assigns each residue a rigid frame, it is anchoring that frame to these three
atoms. Knowing where N, Cα, and C sit in 3D space is enough to reconstruct the
backbone geometry.

### The sidechain

Attached to Cα is the **sidechain** (labelled `R` above). This is what
distinguishes the 20 amino acids from each other. Glycine has just a hydrogen
(H), while Tryptophan has a large ring structure. The sidechain atoms are named
Cβ, Cγ, Cδ, etc. (beta, gamma, delta carbon), branching outward from Cα.

### What the model needs to predict

To know the full 3D shape of a protein, you need to know:
1. Where each backbone (N, Cα, C) sits in space → this is the job of the
   **backbone frame** `(R_i, t_i)`.
2. How the sidechain is rotated around each of its rotatable bonds → these are
   the **torsion angles** χ1, χ2, χ3, χ4.
3. How the backbone itself twists between residues → the backbone torsion angles
   φ, ψ, ω.

The Structure Module predicts all of these.

---

## 1. Where do local coordinates come from? (`rigid_frame_from_three_points`)

Before explaining local vs global, it helps to understand concretely how a
residue's local coordinate system is built from real atom positions.

### The problem

When the model is trained, we have ground-truth atom coordinates from a protein
crystal structure — raw (x, y, z) positions for every atom in global space. We
want to define a consistent local coordinate system for each residue so that we
can express atom positions *relative to that residue* in a way that is the same
across all proteins.

### The solution: pick three atoms and orthogonalize

For each residue we pick three backbone atoms — **C, Cα, N** — and run
Gram-Schmidt to build three orthogonal axes:

```python
x_axis = normalize(Cα - C)              # step 1: x points from C toward Cα
z_axis = normalize(cross(x_axis, Cα→N)) # step 2: z is perpendicular to the C-Cα-N plane
y_axis = normalize(cross(z_axis, x_axis))# step 3: y completes the right-handed system
```

**Step 1 — x-axis**: the bond from C to Cα is a fixed covalent bond with a
known direction. Normalize it to length 1. This becomes the local x-axis.

**Step 2 — z-axis**: N is a third atom that doesn't lie on the x-axis (the
C-Cα-N bond angle is ~111°, not 180°). The cross product of two vectors
produces a vector perpendicular to both — so `cross(x_axis, Cα→N)` gives a
vector sticking straight out of the plane formed by C, Cα, and N. Normalize it.
This is the z-axis.

**Step 3 — y-axis**: we now have x and z, both unit length, both perpendicular
to each other. One more cross product gives y, which completes the coordinate
system. y ends up lying in the C-Cα-N plane, perpendicular to x — roughly
pointing toward N, but exactly orthogonal to x.

### Gram-Schmidt refresher

Gram-Schmidt is the general procedure for turning a set of vectors that are
*roughly* orthogonal into vectors that are *exactly* orthogonal. The core idea:
to make vector B perpendicular to vector A, subtract the component of B that
points along A:

```
B_perp = B - (B · A) * A        (assuming A is a unit vector)
```

The cross product is a shortcut that does this in one step for 3D vectors.
`cross(x, v)` automatically produces a vector that is perpendicular to x,
regardless of whether v was perpendicular to x or not.

### Visualization

Looking down at the C-Cα-N plane from above:

```
                        N
                       /
                      / (Cα→N direction, ~111° from Cα-C bond)
                     /
        C ─────────Cα
                    ↑
                    └── origin of the local frame (t = Cα's global position)


  After Gram-Schmidt:

                        N
                       ↗
              y ↑     /
                |    /
                |   /
        C ──────Cα──────→ x   (x points C→Cα)
                |
                ⊗ z           (z points out of the page, perpendicular to the C-Cα-N plane)
```

The sidechain hangs off Cα roughly in the direction of -y and ±z, which is why
Alanine's Cβ has local coordinates `(-0.529, -0.774, -1.205)` — slightly behind
the origin along x, below along y, and off-plane along z.

### What this gives us

The output is a rotation matrix `R` (whose columns are x, y, z expressed in
global coordinates) and a translation `t = Cα`. Together they are the frame
`(R, t)` for that residue.

Crucially, `lit_positions` — the hardcoded local atom coordinates — were
pre-computed by running this same procedure on ideal bond geometry from
crystallography databases. They never change. The Structure Module's job is to
predict `(R, t)` for each residue so that when you apply `R @ x_local + t`, the
atoms land in the right place globally.

---

## 2. Local frames vs global coordinates

This distinction comes up constantly, so let's nail it down clearly.

### Global coordinates

**Global coordinates** are the ordinary (x, y, z) positions you would measure
with a ruler in the lab, relative to some fixed reference point (say, the centre
of the protein). Every atom in the protein has one global position.

### Local coordinates

**Local coordinates** describe a position *relative to a specific residue's own
mini coordinate system*. Imagine standing at residue 42's Cα atom and orienting
yourself so that the C atom is directly in front of you. Now "1 Å to the left"
means something different depending on which residue you're standing at and which
way you're facing.

### Why local coordinates?

Chemistry is *local*. The rule "the Cβ atom is 1.52 Å from Cα, along a specific
bond direction" is true for every residue of a given type, no matter where that
residue sits in the protein. If we stored atom positions only in global
coordinates we'd have to re-specify these rules for every protein. Instead, we
store atom positions in *local* coordinates (fixed per residue type — the
"literature positions"), and then use the frame to convert them to global.

### The rigid frame: a local-to-global converter

A **rigid frame** `T_i = (R_i, t_i)` for residue `i` is exactly this converter.
It consists of:

- **`t_i`** (translation, a 3D vector): the global position of residue `i`'s
  local origin — roughly, where Cα sits in the world.
- **`R_i`** (rotation, a 3×3 matrix): describes how residue `i`'s local axes
  (x, y, z) are oriented relative to the global axes. Think of it as answering
  "which way is this residue facing?"

To convert a point `x_local` (in residue `i`'s own coordinate system) to global
coordinates:

```
x_global = R_i @ x_local + t_i
```

Step by step:
1. `R_i @ x_local` — rotate the local point so its axes align with the global
   axes (this changes the *direction* of the vector but not its distance from
   the origin).
2. `+ t_i` — shift to where the residue's origin actually sits in the world.

To go the other way (global → local):

```
x_local = R_i^T @ (x_global - t_i)
```

1. `x_global - t_i` — subtract the origin offset, centering the point on the
   residue's local origin.
2. `R_i^T @` — un-rotate (rotation matrices have the convenient property
   `R^{-1} = R^T`, so the transpose undoes the rotation).

### A concrete example

Suppose residue 5's backbone frame has:
- `t_5 = (10, 20, 0)` — the Cα atom is at position (10, 20, 0) in global space.
- `R_5 = I` (identity) — the local axes are aligned with the global axes (the
  residue is "facing forward").

The Cβ atom for this residue type sits at `x_local = (0, -0.5, 1.2)` in local
coordinates (these are the literature bond geometry values). Then:

```
x_global = I @ (0, -0.5, 1.2) + (10, 20, 0) = (10, 19.5, 1.2)
```

Now suppose `R_5` is a 90° rotation about z instead. The same local position
`(0, -0.5, 1.2)` would land at a different global location — the residue is
facing a different direction, so its sidechain points somewhere else.

This is the whole game: the Structure Module iteratively refines `(R_i, t_i)` for
every residue until the backbone is correctly placed, then uses torsion angles to
rotate sidechain atoms within each residue's local frame.

### Linear algebra refresher: rotation matrices

A rotation matrix `R` is a 3×3 matrix with two properties:
1. Its columns are orthogonal unit vectors — all perpendicular to each other, all
   length 1.
2. `det(R) = +1` (ensures it's a rotation, not a reflection).

Such a matrix rotates vectors without stretching or flipping them. For example,
a 90° rotation about the z-axis:

```
R = [[0, -1, 0],
     [1,  0, 0],
     [0,  0, 1]]
```

This maps `(1, 0, 0) → (0, 1, 0)` (the x-axis rotates into the y-axis) and
leaves the z-axis unchanged.

### Composing two rigid frames

Sometimes you need to chain two frames together. Say frame A converts from
"local A" to global, and frame B converts from "local B" to "local A". The
combined transform that goes directly from "local B" to global is:

```
R_combined = R_A @ R_B
t_combined = R_A @ t_B + t_A
```

The `t_combined` line is where people get tripped up. Let's unpack it:

- `t_B` is the origin of frame B, described in frame A's local coordinates.
- `R_A @ t_B` rotates that local offset into global directions. If frame A is
  rotated 45° and frame B's origin is "1 unit along frame A's x-axis", this step
  converts that "1 unit along local x" into the corresponding global direction
  vector.
- `+ t_A` adds the global position of frame A's origin, giving the final global
  position of frame B's origin.

In code this is `compose_transforms(R1, t1, R2, t2)`.

---

## 2. Big picture: what does the Structure Module do?

**Input:**
- `s_i` — single representation from the Evoformer, shape `(batch, N_res, c_s)`.
  Each residue has a feature vector summarising its sequence and evolutionary
  context.
- `z_ij` — pair representation, shape `(batch, N_res, N_res, c_z)`.
  Each pair of residues has a feature vector encoding their relationship.
- `aatype` — integer amino acid type per residue, shape `(batch, N_res)`.

**Output:** 3D atom coordinates in atom14 format — up to 14 heavy atoms per
residue, shape `(batch, N_res, 14, 3)`.

**How:** It maintains a set of per-residue rigid frames `T_i = (R_i, t_i)` that
start as the identity (all residues at the origin, facing the same way —
"black-hole initialisation"). Over several iterations it:

1. Updates `s_i` using geometry-aware attention (IPA) that reads the current frames.
2. Uses the updated `s_i` to nudge the frames (BackboneUpdate).
3. Predicts torsion angles from `s_i` to place sidechains.

The loop runs for `num_layers` iterations (8 in full AF2). After each iteration,
atom coordinates are materialised so auxiliary losses can be computed at every layer.

---

## 3. Initialisation and setup

```python
# Layer-normalise both inputs
single_representation = self.layer_norm_single_rep_1(single_representation)
pair_representation   = self.layer_norm_pair_rep(pair_representation)

# Project s_i to the module's internal channel dimension
s = self.single_rep_proj(single_representation)

# Black-hole initialisation: every residue starts at the origin, facing forward
rotations    = torch.eye(3).view(1,1,3,3).expand(batch, N_res, 3, 3)
translations = torch.zeros(batch, N_res, 3)
```

**Layer normalisation** standardises the feature vector at each residue position
independently — subtract mean, divide by standard deviation. This prevents
activations from growing very large or very small during training.

**Black-hole initialisation** means all `t_i = (0,0,0)` and all `R_i = I`. Every
residue's frame is identical. This sounds wrong, but IPA immediately breaks the
symmetry because the pair bias `z_ij` is different for every pair (i, j).

---

## 4. Invariant Point Attention (IPA)

IPA's job is to update `s_i` using information from other residues, in a way that
also incorporates 3D geometry (the current frame estimates).

### Why "invariant"?

If you pick up the whole protein and rotate it in space, the atom distances and
angles don't change — physics is the same. A good attention mechanism should give
the same output regardless of global orientation. IPA achieves this by computing
attention scores from *distances* between 3D points (which are rotation-invariant)
rather than from raw coordinates (which change under rotation).

### Three sources of attention score

For each head `h` and residue pair `(i, j)`:

#### 4a. Scalar attention (standard dot-product)

```
score_scalar(i,j,h) = q_i^h · k_j^h / sqrt(head_dim)
```

Linear projections of `s_i` and `s_j` — identical to ordinary transformer
attention. Measures feature-space similarity between residues.

#### 4b. Pair bias

```
score_pair(i,j,h) = Linear(z_ij)[h]
```

The Evoformer's pair representation `z_ij` is projected to one number per head,
used as a direct additive bias. This lets the Evoformer's pairwise signal
(evolutionary co-variation, predicted contacts, etc.) influence which residues
attend to each other.

#### 4c. Point attention (the 3D geometry part)

##### Generating the query and key points

Each residue's query points are produced by a learned linear layer applied to
`s_i`:

```python
q_points = linear_q_points(s_i)   # shape: (n_heads, n_query_points, 3)
```

`linear_q_points` maps `s_i` (size `c_s`) to `n_heads * n_query_points * 3`
numbers, which are reshaped into `n_query_points` 3D vectors per head. These
vectors are expressed in residue `i`'s **local frame**. They are then lifted to
global coordinates:

```python
Q_global = R_i @ q_points + t_i
```

The same procedure produces key points from `s_j`. The attention score is the
negative squared Euclidean distance between query and key points, summed over
the `n_query_points` points and weighted by a learned per-head scalar `γ_h`:

```
score_point(i,j,h) = -γ_h * Σ_p ||Q_{i,h,p}^global - K_{j,h,p}^global||^2 / 2
```

Large distance → more negative score → less attention. So geometrically nearby
residues attend to each other more strongly.

##### Why not just use the Cα-to-Cα distance directly?

You might wonder: why this whole machinery, when you could just compute the
distance between the two Cα atoms?

**Reason 1: orientation, not just position.**

A single Cα-to-Cα distance tells you how far apart two residues are, but nothing
about how they're *facing* each other. Two residues can be 8 Å apart with their
sidechains pointing toward each other (strongly interacting) or pointing away
(not interacting) — same distance, completely different geometry.

The query/key points are in *local* coordinates before being lifted to global.
A point at `(1, 0, 0)` in local space gets rotated by `R_i` before the distance
is computed. So if residue `i` is rotated 90°, its query points land somewhere
completely different in global space even though Cα is in the same spot. The
distances therefore encode **relative orientation** between residues, not just
proximity.

```
   Two residues, same Cα distance, different orientations:

   Case A: sidechains facing each other        Case B: sidechains facing away
   (query point close to key point)            (query point far from key point)

       ←q  Cα_i ──────── Cα_j  k→                  q→  Cα_i ──────── Cα_j  ←k
            high attention score                          low attention score
```

**Reason 2: the points are learned.**

A single Cα distance is hardwired — there's nothing for the model to tune.
Here, the positions of query/key points in local space come from a linear layer
applied to `s_i`, so they are trained end-to-end. One head might learn to place
its query points out toward where the sidechain would be; another might probe
the backbone direction. With `n_query_points = 4` per head and `n_heads = 4`,
the model has 16 learned geometric probes per residue, each free to specialise
for a different aspect of the structure.

**Why is this invariant?** If you rotate the whole protein, every frame rotates
the same way, so every global point rotates the same way. The *difference*
between two global points is a vector whose length stays the same under global
rotation. So the distances, and therefore the attention scores, don't change.

#### Combining scores

```
logit(i,j,h) = sqrt(1/3) * (score_scalar + score_pair + score_point)
```

The `sqrt(1/3)` normalises variance across the three equal contributions.
Softmax over `j` gives attention weights `a_{ij,h}`.

### Aggregating values

Three types of values are pooled using the attention weights:

1. **Scalar values**: standard weighted sum of projected `s_j` across `j` —
   produces a feature vector for residue `i`.
2. **Point values**: 3D value points (projected from `s_j`, lifted to global,
   then weighted-summed). These are then rotated *back into residue `i`'s local
   frame* (`R_i^T @ (point_global - t_i)`), giving local 3D offsets that
   describe where neighbouring residues sit relative to `i`. Their norms are
   also appended.
3. **Pair values**: `z_ij` weighted-summed over `j` — a pair-feature summary
   for each residue.

All three are concatenated and projected back to `c_s` channels. The result is
added to `s` (residual connection).

---

## 5. Transition MLP

After IPA, a small 3-layer MLP with ReLU activations mixes channels:

```python
s = s + linear_3(relu(linear_2(relu(linear_1(s)))))
```

`linear_3` is zero-initialised so the block starts as the identity — the
residual passes `s` through unchanged at the start of training. This is a
standard trick for stable deep network training.

Dropout and LayerNorm are applied after both IPA and the MLP.

---

## 6. BackboneUpdate

### Inputs and outputs

BackboneUpdate takes **only `s_i`** as input — the single representation for
each residue. It does not see the pair representation `z_ij`. By the time
BackboneUpdate runs, `s_i` has already been updated by IPA, which folded in
both the pair representation and the 3D geometry. BackboneUpdate's job is simply
to read the updated `s_i` and decide how to move the frame.

The network is a single linear layer mapping `c_s` → 6 numbers per residue:

```python
vals = linear(s_i)   # shape: (batch, N_res, 6)
```

The 6 numbers split into two groups:
- `vals[:, :, 0:3]` → `(b, c, d)` for the rotation quaternion
- `vals[:, :, 3:6]` → `t_new`, the local translation update

Both are expressed in the residue's **current local frame** — not global space.

### Why predict a local update?

The network doesn't know (or care) where the residue currently sits globally.
By predicting an update in *local* coordinates, the same learned weights apply
regardless of where the protein sits in space. "Move 0.1 nm along my local
x-axis" has the same meaning for every copy of a residue type across all
proteins.

### How the rotation is parameterised: quaternions

The network could directly output a 3×3 rotation matrix, but rotation matrices
have 6 constraints (orthonormality) which are hard to maintain through
backpropagation. Instead, the network outputs a **unit quaternion**.

A quaternion `q = (a, b, c, d)` is a 4-number encoding of a rotation. The
network only predicts three numbers `(b, c, d)` and fixes `a = 1`, then
normalises:

```python
norm = sqrt(1 + b^2 + c^2 + d^2)
a, b, c, d = 1/norm, b/norm, c/norm, d/norm
```

When `(b, c, d)` are all near zero, the normalised quaternion is near `(1,0,0,0)`
— the identity (no rotation). Large `(b,c,d)` gives a large rotation. This
naturally starts near the identity and lets gradients flow smoothly.

The rotation matrix is then:

```
R = [[a²+b²-c²-d²,  2bc-2ad,      2bd+2ac    ],
     [2bc+2ad,       a²-b²+c²-d², 2cd-2ab     ],
     [2bd-2ac,       2cd+2ab,      a²-b²-c²+d²]]
```

Each entry is differentiable — gradients flow back to `(b, c, d)` and through to
the network weights. Also note: the linear layer is zero-initialised, so at the
start of training `(b, c, d) = (0, 0, 0)`, which gives the identity rotation and
zero translation — each IPA iteration starts as a no-op and learns from there.

### Applying the update

After BackboneUpdate returns `(R_new, t_new)`, the frame is updated in the
outer loop:

```python
new_rotations, new_translations = self.backbone_update(s)
translations = torch.einsum('bsij, bsj -> bsi', rotations, new_translations) + translations
rotations    = torch.einsum('bsij, bsjk -> bsik', rotations, new_rotations)
```

This is `T_old ∘ T_new`, i.e. composing the current global frame with the
predicted local update. Let's unpack each line.

**Translation update:**
```
translations_new = R_old @ t_new + t_old
```

- `t_new` is a 3D offset in the residue's current local coordinate system.
  e.g. `t_new = (0.1, 0, 0)` means "move 0.1 nm along my local x-axis."
- `R_old @ t_new` rotates that local direction into global space. If the residue
  is currently facing "northwest", then "0.1 nm along local x" becomes the
  corresponding northwest direction in global coordinates.
- `+ t_old` shifts by the residue's current global position, giving the new
  global position of the frame's origin.

**Rotation update:**
```
rotations_new = R_old @ R_new
```

`R_new` is applied first (it rotates within the current local frame), then
`R_old` converts the result into global orientation. The combined matrix
describes the new global orientation of the residue.

**Concrete example** — suppose at iteration 2, residue 5 has:
- `t_old = (10, 20, 0)` — its Cα is currently at global position (10, 20, 0)
- `R_old` = 90° rotation about z — the residue is currently facing "north"
- BackboneUpdate predicts `t_new = (0.1, 0, 0)` — "nudge 0.1 nm along my local x-axis"

Since the residue is facing north, its local x-axis points north in global space.
So `R_old @ t_new` = `(0, 0.1, 0)` (0.1 nm north). The new global position is
`(10, 20.1, 0)`.

### Stop-gradient on rotations

After each iteration (except the last), rotations are detached from the
computational graph:

```python
if detach_rotations and l < num_layers - 1:
    rotations = rotations.detach()
```

Rotations compound multiplicatively across iterations (`R_3 = R_2 @ R_new`).
A gradient flowing backward through many such multiplications can explode
("lever effect"). Detaching prevents this. Translations are *not* detached
because the auxiliary FAPE loss at every layer needs gradient through them.

---

## 7. MultiRigidSidechain and torsion angles

The backbone frame places N, Cα, C. To place sidechain atoms (Cβ, Cγ, etc.)
you also need to know how the sidechain is rotated around its bonds.

### What is a torsion angle?

Given four atoms A–B–C–D in a chain, the **torsion angle** (or dihedral angle)
is the angle you see when looking down the B–C bond: it's the angle between the
A–B–C plane and the B–C–D plane.

Imagine holding a paper chain at the B–C bond and rotating the A end relative to
the D end. That rotation angle is the torsion angle. It takes values in
`[-180°, 180°]`.

AF2 predicts 7 torsion angles per residue:

| Angle | Bond | What it controls |
|-------|------|-----------------|
| ω (omega) | Cα–C–N–Cα | Peptide bond twist (almost always ~180°) |
| φ (phi) | C–N–Cα–C | Backbone; determines how residues stack |
| ψ (psi) | N–Cα–C–N | Backbone; together with φ gives secondary structure |
| χ1 | N–Cα–Cβ–Cγ | First sidechain rotatable bond |
| χ2 | Cα–Cβ–Cγ–Cδ | Second sidechain bond |
| χ3 | Cβ–Cγ–Cδ–Cε | Third sidechain bond |
| χ4 | Cγ–Cδ–Cε–Nζ | Fourth sidechain bond (only Arg and Lys) |

### Representing angles as (sin, cos) pairs

Angles have an awkward discontinuity: 179° and -179° are almost the same angle
but numerically far apart. To avoid this, the network predicts `(sin α, cos α)`
instead. This is a point on the unit circle that varies *smoothly* as the angle
rotates — there's no discontinuity.

The AngleResnet outputs raw 2D vectors `(x, y)` which are then L2-normalised to
the unit circle:

```python
angles = raw / ||raw||
```

The unnormalised version is also kept to compute a norm loss that penalises
vectors with near-zero magnitude (which have an ambiguous angle direction).

### AngleResnet architecture

Variables:
- `s_i` — the single representation at the current IPA iteration, shape `(batch, N_res, c_s)`
- `s_initial` — the single representation *before any IPA iterations*, i.e. the raw Evoformer output, same shape
- `c_hidden` — the internal channel dimension of the AngleResnet (set by `sidechain_num_channel` in config)

```
single_act   = Linear(ReLU(s_i),      c_s → c_hidden)   ← project current s_i
initial_act  = Linear(ReLU(s_initial),c_s → c_hidden)   ← project original s_i
sidechain_act = single_act + initial_act                 ← sum (both shape: N_res × c_hidden)

sidechain_act = AngleResnetBlock(sidechain_act)  ┐
sidechain_act = AngleResnetBlock(sidechain_act)  ┘  repeated N times

out = Linear(ReLU(sidechain_act), c_hidden → 14)   ← 14 = 7 angles × 2 (sin, cos)
out = reshape to (N_res, 7, 2)
out = out / ||out||                                 ← L2-normalise each (sin, cos) pair
```

Including `s_initial` gives the torsion head access to the original Evoformer
signal even after many IPA iterations have modified `s_i`.

### What is an AngleResnetBlock?

A residual block is a small sub-network that adds its output back to its input
rather than replacing it:

```
output = input + f(input)
```

This means if `f` learns to output zeros, the block is a perfect identity —
it passes the input through unchanged. This makes it safe to stack many blocks:
even if some blocks are unhelpful, they can't make things worse by outputting
zero.

Concretely, one `AngleResnetBlock` computes:

```
# input: a, shape (batch, N_res, c_hidden)

residual = a                        ← save a copy of the input
a = ReLU(a)
a = Linear(a, c_hidden → c_hidden)  ← linear_1
a = ReLU(a)
a = Linear(a, c_hidden → c_hidden)  ← linear_2, zero-initialised
output = a + residual               ← add back the saved input
```

`linear_2` is zero-initialised, so at the start of training the block outputs
exactly `0 + residual = residual` — pure identity. The block learns deviations
from identity over training. This is the same zero-init trick used in
BackboneUpdate and the transition MLP.

---

## 8. Constructing all-atom coordinates (Algorithm 24)

Given backbone frames and 7 torsion angles, all atoms can be placed. The key idea:
every atom belongs to one of 8 **rigid groups** — a sub-frame of the residue. All
atoms in a group move together as a rigid body. Rotating a torsion angle rotates
the entire downstream group around that bond.

| Group | Name | Parent frame |
|-------|------|-------------|
| 0 | Backbone | `(R_i, t_i)` from BackboneUpdate |
| 1 | ω frame | Backbone |
| 2 | φ frame | Backbone |
| 3 | ψ frame | Backbone |
| 4 | χ1 frame | Backbone |
| 5 | χ2 frame | χ1 frame |
| 6 | χ3 frame | χ2 frame |
| 7 | χ4 frame | χ3 frame |

### Building each group's frame

For groups 1–4 (branching off the backbone):

```
T_f = T_backbone ∘ T^lit_{r,f} ∘ makeRotX(α_f)
```

Read right to left:
1. `makeRotX(α_f)` — rotate about the x-axis by the predicted torsion angle.
2. `T^lit_{r,f}` — a fixed, hardcoded transform from the chemistry literature
   that says "group f for residue type r sits at this position and orientation
   relative to the backbone, *before* any torsion rotation." These are stored in
   `restype_rigid_group_default_frame`.
3. `T_backbone` — lift everything into global coordinates using the backbone frame.

For groups 5–7 (chained sidechain):

```
T_{χ2} = T_{χ1} ∘ T^lit_{r,χ2} ∘ makeRotX(α_{χ2})
```

Same logic, but the parent is the χ1 frame rather than backbone — χ2 rotates
relative to where χ1 has already placed things.

### makeRotX: rotation about the local x-axis

Each rigid group's bond is defined to lie along the local x-axis. A torsion
angle rotates atoms *around* that bond, which means rotating around the local
x-axis:

```
R_x(α) = [[1,    0,      0   ],
           [0,  cos α, -sin α ],
           [0,  sin α,  cos α ]]
```

This leaves the x-axis unchanged and rotates the y-z plane by angle α.

### Placing atoms

Each atom slot in atom14 format knows:
1. Which of the 8 rigid groups it belongs to (`atom_frame_idx`).
2. Its position within that group's local frame — the hardcoded literature
   position (`lit_positions`).

To get the global position:

```
x_global = R_group @ x_lit + t_group
```

This is the same local-to-global formula from Section 1, but now using the
specific rigid group's frame rather than the backbone frame.

### What is atom14 format?

Across all 20 amino acids the largest has 14 heavy atoms. Atom14 stores exactly
14 atom slots per residue, with a fixed assignment of which slot means which atom
for each residue type. Slots that don't exist for a given residue type are zeroed
and masked by `atom14_mask`. This regular shape makes batched tensor operations
convenient.

---

## 9. Full forward pass: putting it all together

```
Inputs: s (batch, N_res, c_s), z (batch, N_res, N_res, c_z), aatype (batch, N_res)

LayerNorm(s), LayerNorm(z)
s = Linear(s)                          ← project to internal channels

T_i = (I, 0) for all i                ← black-hole init: all frames at origin

For l in 1..num_layers:
    s ← s + IPA(s, z, T_i)            ← update s using 3D geometry
    s ← LayerNorm(Dropout(s))
    s ← s + MLP(s)                    ← mix channels
    s ← LayerNorm(Dropout(s))

    (R_new, t_new) = BackboneUpdate(s)    ← predict local frame update
    T_i ← T_i ∘ (R_new, t_new)           ← apply update to backbone frames

    angles = AngleResnet(s, s_initial)    ← predict 7 torsion angles
    frames, atom_coords = compute_all_atom_coordinates(T_i, angles, aatype)

    save (T_i, angles, atom_coords) for auxiliary losses
    if not last layer: detach rotations in T_i

Output: atom_coords (batch, N_res, 14, 3), frames, torsion angles, final s
```

### Auxiliary losses at every layer

Coordinates are materialised after every iteration, not just the last. The FAPE
loss (Frame-Aligned Point Error) is computed at every layer, so gradients flow
into every iteration. The final layer's output is what's used at inference.

---

## 10. FAPE: Frame-Aligned Point Error

FAPE is the main structural loss. It measures how wrong the predicted structure
is, but does so in a way that ignores global position and orientation — it only
cares about the *shape* of the prediction.

### The core idea

Instead of comparing predicted atom positions directly to ground-truth positions
in global space, FAPE first transforms both into a residue's **local frame**,
then measures the distance. For every reference frame `i` and every atom `j`:

```
d_ij = || T_i^{-1} · x_j^{predicted}  −  T_i^{true,-1} · x_j^{true} ||
```

Variables:
- `T_i = (R_i, t_i)` — the predicted backbone frame at residue `i`
- `T_i^{true}` — the ground-truth backbone frame at residue `i`
- `x_j` — predicted global position of atom `j`
- `x_j^{true}` — ground-truth global position of atom `j`
- `T^{-1} · x = R^T @ (x - t)` — applying the inverse frame (un-rotate, then un-translate)

Note: `i` indexes **residues** (each residue contributes one frame), while `j`
indexes individual **atoms** across the entire protein (each residue contributes
up to 14 atoms). For a 100-residue protein there are 100 frames and up to 1400
atoms, giving up to 140,000 `(i, j)` pairs.

In words: express atom `j` from residue `i`'s point of view, both in the
prediction and in the ground truth, then measure how far apart those two local
views are.

The final loss averages `d_ij` over all `(i, j)` pairs and divides by a scale
factor `Z = 10 Å`:

```
L_FAPE = (1 / Z) * mean_{i,j}( min(d_clamp, d_ij) )
```

The `min(d_clamp, ...)` clamps errors at `d_clamp = 10 Å` — errors beyond 10 Å
don't contribute extra signal, preventing the loss from being dominated by a few
catastrophically misplaced atoms.

### Property 1: SE(3) invariance

Apply any global rigid motion `G` (any rotation + translation) to the entire
predicted structure. Every frame transforms as `T_i → G · T_i` and every atom
as `x_j → G · x_j`. The local-frame expression becomes:

```
(G · T_i)^{-1} · (G · x_j)
= T_i^{-1} · G^{-1} · G · x_j
= T_i^{-1} · x_j
```

`G^{-1} · G` cancels exactly. The loss value is identical no matter how the
predicted structure is rotated or translated globally.

This is the fundamental reason FAPE is used instead of a naive MSE on global
positions:

```python
# Naive MSE — NOT SE(3) invariant:
loss = ||x_j^predicted - x_j^true||²
# Rotating the whole predicted structure 5° → large loss, even if shape is perfect.

# FAPE — SE(3) invariant:
loss = ||T_i^{-1} · x_j^predicted - T_i^{true,-1} · x_j^true||²
# Rotating the whole predicted structure 5° → G cancels, loss unchanged.
```

### Property 2: per-residue local judgment

The loss is summed over every choice of `i` as the reference frame. Residue
5's frame judges where all atoms sit from residue 5's perspective; residue 12's
frame judges from residue 12's perspective. This means a local structural error
(one sidechain in the wrong place) contributes heavily to loss terms where
nearby residues are the reference frame, and barely affects terms where distant
residues are the reference frame. The loss is sensitive to *local* structural
quality, not just the global fold.

### Backbone FAPE vs all-atom FAPE

There are two FAPE heads in this codebase:

**Backbone FAPE** (`BackboneFAPE`): uses only the backbone frames and their Cα
positions as both frames and atoms. Computed at every iteration of the structure
module loop, then averaged — giving dense gradient signal into every iteration,
not just the final one.

**All-atom FAPE** (`AllAtomFAPE`): uses all 8 rigid-group frames per residue
and all 14 atom positions. Only computed on the final iteration's output. This
is the full structural loss that penalises sidechain placement.

The total structure loss combines both:

```
L_structure = 0.5 * L_backbone_FAPE + 0.5 * L_allatom_FAPE + L_torsion
```

### The clamping detail

Backbone FAPE uses a soft mix of clamped and unclamped versions. In 90% of
training batches it is fully clamped at 10 Å; in 10% it is unclamped. The
unclamped 10% provides gradient for residues that are very far from their
target — without it, a residue 50 Å off gets the same gradient as one 10 Å
off, and the network has no signal to pull very wrong residues back. All-atom
FAPE is always clamped.

---

## 11. Units: nanometres vs ångströms

Atom positions in structural biology are conventionally reported in **ångströms**
(Å), where a typical C–C bond is ~1.52 Å. The AF2 supplement uses **nanometres**
(nm) internally: 1 nm = 10 Å.

`position_scale = 10.0` is the conversion factor. On construction, all
hardcoded literature positions and frames are divided by 10 to convert to nm.
On output, all translations and coordinates are multiplied by 10 to return Å.
Rotations are dimensionless and need no conversion.

---

## 12. Component summary

| Component | Role | Algorithm |
|-----------|------|-----------|
| `StructureModule` | Outer loop: IPA → MLP → BackboneUpdate × L layers | Alg. 20 |
| `InvariantPointAttention` | Geometry-aware attention using 3D frames | Alg. 22 |
| `BackboneUpdate` | Predicts local quaternion + translation to update frames | Alg. 23 |
| `AngleResnet` | MLP predicting 7 torsion angles as (sin, cos) | — |
| `MultiRigidSidechain` | Wraps AngleResnet + calls coordinate assembly | Alg. 24 text |
| `make_rot_x` | Rotation matrix about local x-axis by torsion angle | Alg. 25 |
| `rigid_group_frames_from_torsions` | Builds 8 per-residue rigid-group frames | Alg. 24 lines 1-10 |
| `compute_all_atom_coordinates` | Places all atoms using frames + literature positions | Alg. 24 lines 11-14 |
| `compose_transforms` | Chains two rigid transforms: T1 ∘ T2 | — |
