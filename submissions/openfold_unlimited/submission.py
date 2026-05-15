"""OpenFold AlphaFold2 submission for the nanoFold unlimited track."""
from __future__ import annotations

import importlib.util  # ensure submodule is importable before openfold
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path / environment setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
OPENFOLD_ROOT = REPO_ROOT / "third_party" / "openfold"
MINALPHAFOLD2_ROOT = REPO_ROOT / "third_party" / "minAlphaFold2"

for _p in [str(OPENFOLD_ROOT), str(MINALPHAFOLD2_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# attn_core_inplace_cuda CUDA kernel requires PyTorch's lib dir on LD_LIBRARY_PATH
_torch_lib = str(Path(torch.__file__).parent / "lib")
_ld = os.environ.get("LD_LIBRARY_PATH", "")
if _torch_lib not in _ld:
    os.environ["LD_LIBRARY_PATH"] = f"{_torch_lib}:{_ld}"

# ---------------------------------------------------------------------------
# OpenFold imports
# ---------------------------------------------------------------------------
from openfold.model.model import AlphaFold  # noqa: E402
from openfold.config import model_config as _of_model_config  # noqa: E402
from openfold.utils.loss import AlphaFoldLoss  # noqa: E402
from openfold.data.data_transforms import (  # noqa: E402
    make_atom14_masks,
    make_atom14_positions,
    atom37_to_frames,
    atom37_to_torsion_angles,
    get_backbone_frames,
    get_chi_angles,
    pseudo_beta_fn,
)
from openfold.utils.feats import atom14_to_atom37  # noqa: E402
from openfold.utils.tensor_utils import tensor_tree_map  # noqa: E402

# ---------------------------------------------------------------------------
# minAlphaFold2 MSA feature builder (reused — identical feature dims)
# ---------------------------------------------------------------------------
from minalphafold.data import (  # noqa: E402
    build_msa_features,
)

# ---------------------------------------------------------------------------
# Schedule constants (AF2 paper, Suppl. 1.11)
# ---------------------------------------------------------------------------
_AF2_INITIAL_SAMPLES = 10_000_000
_AF2_FINETUNE_SAMPLES = 1_500_000
_AF2_WARMUP_SAMPLES = 128_000
_AF2_LR_DECAY_SAMPLES = 6_400_000
_DEFAULT_FINETUNE_RAMP = 1_000


# ---------------------------------------------------------------------------
# LR schedule helpers
# ---------------------------------------------------------------------------

def _scaled_step(value_samples: int, total_samples: int, total_steps: int) -> int:
    return int(round(total_steps * value_samples / total_samples))


def _bounded(value: int, *, lo: int = 0, hi: int) -> int:
    return max(lo, min(int(value), hi))


def _budget_schedule(cfg: Dict[str, Any]) -> Dict[str, Any]:
    train_cfg = cfg.get("train", {})
    optim_cfg = cfg.get("optim", {})
    max_steps = int(train_cfg["max_steps"])
    default_ft_start = _scaled_step(
        _AF2_INITIAL_SAMPLES,
        _AF2_INITIAL_SAMPLES + _AF2_FINETUNE_SAMPLES,
        max_steps,
    )
    ft_start = _bounded(
        int(train_cfg.get("finetune_start_step", default_ft_start)),
        hi=max_steps,
    )
    warmup = _bounded(
        int(train_cfg.get(
            "warmup_steps",
            _scaled_step(_AF2_WARMUP_SAMPLES, _AF2_INITIAL_SAMPLES, ft_start),
        )),
        hi=ft_start,
    )
    lr_decay = _bounded(
        int(train_cfg.get(
            "lr_decay_step",
            _scaled_step(_AF2_LR_DECAY_SAMPLES, _AF2_INITIAL_SAMPLES, ft_start),
        )),
        hi=max_steps,
    )
    return dict(
        max_steps=max_steps,
        finetune_start_step=ft_start,
        finetune_ramp_steps=int(train_cfg.get("finetune_ramp_steps", _DEFAULT_FINETUNE_RAMP)),
        warmup_steps=warmup,
        lr_decay_step=lr_decay,
        finetune_lr_scale=float(train_cfg.get("finetune_lr_scale", 0.5)),
        lr_decay_factor=float(optim_cfg.get("lr_decay_factor", 0.95)),
    )


def _runtime_step(cfg: Dict[str, Any]) -> int:
    rt = cfg.get("_runtime", {})
    return int(rt.get("step", 0)) if isinstance(rt, dict) else 0


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------

def build_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    model_cfg = cfg.get("model", {})

    if "blocks_per_ckpt" in model_cfg:
        oc.globals.blocks_per_ckpt = model_cfg["blocks_per_ckpt"]  # may be None
    oc = _of_model_config("initial_training", train=True)
    oc.model.template.enabled = False  # no template data available
    oc.globals.use_flash = True

    model = AlphaFold(oc)

    # Attach both loss functions so run_batch can access them without rebuilding
    initial_oc = _of_model_config("initial_training", train=True)
    initial_oc.loss.violation.weight = 0.0
    model._loss_initial = AlphaFoldLoss(initial_oc.loss)

    finetune_oc = _of_model_config("initial_training", train=True)
    finetune_oc.loss.violation.weight = 1.0
    finetune_oc.loss.experimentally_resolved.weight = 0.01
    model._loss_finetune = AlphaFoldLoss(finetune_oc.loss)

    return model


# ---------------------------------------------------------------------------
# build_optimizer
# ---------------------------------------------------------------------------

def build_optimizer(cfg: Dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    oc = cfg["optim"]
    return torch.optim.Adam(
        model.parameters(),
        lr=float(oc["lr"]),
        betas=(float(oc.get("beta1", 0.9)), float(oc.get("beta2", 0.999))),
        eps=float(oc.get("eps", 1e-6)),
        weight_decay=float(oc.get("weight_decay", 0.0)),
    )


# ---------------------------------------------------------------------------
# build_scheduler
# ---------------------------------------------------------------------------

class _LRScheduler:
    def __init__(self, cfg: Dict[str, Any], optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer
        self.base_lr = float(cfg["optim"]["lr"])
        self.sched = _budget_schedule(cfg)
        self.completed_steps = 0
        self._apply()

    def _lr(self) -> float:
        s = self.completed_steps
        sc = self.sched
        if s >= sc["finetune_start_step"]:
            lr = self.base_lr * sc["finetune_lr_scale"]
            if s >= sc["lr_decay_step"]:
                lr *= sc["lr_decay_factor"]
            return lr
        if sc["warmup_steps"] > 0 and s < sc["warmup_steps"]:
            return self.base_lr * s / sc["warmup_steps"]
        if s >= sc["lr_decay_step"]:
            return self.base_lr * sc["lr_decay_factor"]
        return self.base_lr

    def _apply(self) -> None:
        lr = self._lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def step(self) -> None:
        self.completed_steps += 1
        self._apply()

    def state_dict(self) -> Dict[str, Any]:
        return {"completed_steps": self.completed_steps}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.completed_steps = int(state.get("completed_steps", 0))
        self._apply()


def build_scheduler(cfg: Dict[str, Any], optimizer: torch.optim.Optimizer) -> Any:
    return _LRScheduler(cfg, optimizer)


# ---------------------------------------------------------------------------
# Feature construction helpers
# ---------------------------------------------------------------------------

def _pad_and_stack(tensors: List[torch.Tensor], target_len: int) -> torch.Tensor:
    """Pad each tensor to target_len along dim 0 then stack into a batch."""
    padded = []
    for t in tensors:
        if t.shape[0] < target_len:
            pad = t.new_zeros(target_len - t.shape[0], *t.shape[1:])
            t = torch.cat([t, pad], dim=0)
        padded.append(t)
    return torch.stack(padded, dim=0)


def _build_msa_features(
    batch: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
    *,
    training: bool,
) -> Dict[str, torch.Tensor]:
    """Build MSA features from nanoFold batch using minAlphaFold2's pipeline."""
    msa = batch["msa"].long()        # [B, N, L]
    deletions = batch["deletions"].long()
    B = msa.shape[0]
    model_cfg = cfg.get("model", {})
    msa_depth = int(model_cfg.get("msa_depth", 128))
    extra_msa_depth = int(model_cfg.get("extra_msa_depth", 512))

    msa_feat_list, msa_mask_list = [], []
    extra_feat_list, extra_mask_list = [], []
    true_msa_list, bert_mask_list = [], []

    for i in range(B):
        mf = build_msa_features(
            {"msa": msa[i].cpu(), "deletions": deletions[i].cpu()},
            msa_depth=msa_depth,
            extra_msa_depth=extra_msa_depth,
            training=training,
            block_delete_training_msa=True,
            block_delete_msa_fraction=0.3,
            block_delete_msa_randomize_num_blocks=False,
            block_delete_msa_num_blocks=5,
            masked_msa_probability=0.15,
        )
        msa_feat_list.append(mf["msa_feat"])
        msa_mask_list.append(mf["msa_mask"])
        extra_feat_list.append(mf["extra_msa_feat"])
        extra_mask_list.append(mf["extra_msa_mask"])
        # masked_msa_target is one-hot of original tokens → argmax gives integer indices
        true_msa_list.append(mf["masked_msa_target"].argmax(-1).long())
        bert_mask_list.append(mf["masked_msa_mask"])

    device = batch["aatype"].device
    max_cluster = max(f.shape[0] for f in msa_feat_list)
    max_extra = max(f.shape[0] for f in extra_feat_list)

    return {
        "msa_feat": _pad_and_stack(msa_feat_list, max_cluster).to(device),
        "msa_mask": _pad_and_stack(msa_mask_list, max_cluster).to(device),
        "extra_msa_feat": _pad_and_stack(extra_feat_list, max_extra).to(device),
        "extra_msa_mask": _pad_and_stack(extra_mask_list, max_extra).to(device),
        "true_msa": _pad_and_stack(true_msa_list, max_cluster).to(device),
        "bert_mask": _pad_and_stack(bert_mask_list, max_cluster).to(device),
    }


def _build_supervision_features(
    aatype: torch.Tensor,         # [B, L]  long
    atom14_positions: torch.Tensor,  # [B, L, 14, 3]  float
    atom14_mask: torch.Tensor,    # [B, L, 14]  float
) -> Dict[str, torch.Tensor]:
    """Build all ground-truth supervision tensors for OpenFold's loss."""
    protein: Dict[str, torch.Tensor] = {"aatype": aatype}

    # atom14 / atom37 index mappings
    make_atom14_masks(protein)

    # Convert nanoFold atom14 coords → atom37 format for OpenFold transforms
    all_atom_positions = atom14_to_atom37(atom14_positions, protein)   # [B, L, 37, 3]
    all_atom_mask = atom14_to_atom37(
        atom14_mask.unsqueeze(-1), protein
    ).squeeze(-1)  # [B, L, 37]

    protein["all_atom_positions"] = all_atom_positions
    protein["all_atom_mask"] = all_atom_mask

    # Atom14 ground-truth positions + alternative (for ambiguous atoms)
    make_atom14_positions(protein)

    # Rigid groups (backbone + side-chain frames)
    atom37_to_frames(protein)

    # Torsion angles (needed for chi angles)
    atom37_to_torsion_angles()(protein)

    # Backbone rigid tensor from group index 0
    get_backbone_frames(protein)

    # Chi angles from torsion angles indices 3..6
    get_chi_angles(protein)

    # Pseudo-beta positions for distogram loss
    pseudo_beta, pseudo_beta_mask = pseudo_beta_fn(
        aatype, all_atom_positions, all_atom_mask
    )
    protein["pseudo_beta"] = pseudo_beta
    protein["pseudo_beta_mask"] = pseudo_beta_mask

    return protein


def _build_openfold_batch(
    batch: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
    *,
    training: bool,
) -> Dict[str, torch.Tensor]:
    """Assemble the full feature dict expected by AlphaFold.forward()."""
    device = batch["aatype"].device
    aatype = batch["aatype"].long()          # [B, L]
    residue_mask = batch["residue_mask"].float()   # [B, L]
    residue_index = batch["residue_index"].long()  # [B, L]

    B, L = aatype.shape
    model_cfg = cfg.get("model", {})
    n_recycles = int(model_cfg.get("n_recycles", 3))
    n_iters = n_recycles + 1

    # ---- Sequence / target features ----
    has_break = torch.zeros(B, L, 1, device=device)
    aatype_1hot = F.one_hot(aatype.clamp(0, 20), 21).float()
    target_feat = torch.cat([has_break, aatype_1hot], dim=-1)  # [B, L, 22]

    # ---- MSA features (includes true_msa / bert_mask for masked-MSA loss) ----
    msa_feats = _build_msa_features(batch, cfg, training=training)

    # ---- Empty template features ----
    templates = {
        "template_mask": torch.zeros(B, 0, device=device),
        "template_aatype": torch.zeros(B, 0, L, dtype=torch.long, device=device),
        "template_all_atom_positions": torch.zeros(B, 0, L, 37, 3, device=device),
        "template_all_atom_mask": torch.zeros(B, 0, L, 37, device=device),
        "template_pseudo_beta": torch.zeros(B, 0, L, 3, device=device),
        "template_pseudo_beta_mask": torch.zeros(B, 0, L, device=device),
    }

    # OpenFold calls build_extra_msa_feat() internally, so it needs the raw
    # components rather than the pre-built extra_msa_feat tensor.
    extra_msa_feat = msa_feats.pop("extra_msa_feat")   # [B, N_extra, L, 25]
    extra_msa_mask = msa_feats.pop("extra_msa_mask")   # [B, N_extra, L]

    # Guard: OpenFold's attention stack breaks with 0 extra sequences.
    # Pad to at least 1 all-gap row so the stack is always well-defined.
    if extra_msa_feat.shape[1] == 0:
        extra_msa_feat = torch.zeros(B, 1, L, 25, device=device)
        extra_msa_feat[..., 21] = 1.0          # one-hot GAP_ID=21
        extra_msa_mask = torch.zeros(B, 1, L, device=device)

    extra_msa_keys = {
        "extra_msa":            extra_msa_feat[..., :23].argmax(-1).long(),
        "extra_has_deletion":   extra_msa_feat[..., 23],
        "extra_deletion_value": extra_msa_feat[..., 24],
        "extra_msa_mask":       extra_msa_mask,
    }

    out: Dict[str, torch.Tensor] = {
        "aatype": aatype,
        "target_feat": target_feat,
        "residue_index": residue_index,
        "seq_mask": residue_mask,
        # sequence length for per-sample loss scaling (Suppl. 1.9)
        "seq_length": residue_mask.sum(-1).long(),
        # resolution: 1.0 Å puts us in the valid range [0.1, 3.0] used by plddt/exp-res weights
        "resolution": torch.ones(B, device=device),
        # use_clamped_fape=0 → unclamped FAPE during initial training
        "use_clamped_fape": torch.zeros(B, device=device),
        **msa_feats,
        **extra_msa_keys,
        **templates,
    }

    # ---- Supervision features (atom14/37, frames, chi angles) ----
    if "atom14_positions" in batch and "atom14_mask" in batch:
        atom14_pos = batch["atom14_positions"].float()
        atom14_msk = batch["atom14_mask"].float() * residue_mask[:, :, None]
        sup = _build_supervision_features(aatype, atom14_pos, atom14_msk)

        keep = {
            "atom14_atom_exists", "residx_atom14_to_atom37", "residx_atom37_to_atom14",
            "atom37_atom_exists",
            "atom14_gt_positions", "atom14_gt_exists",
            "atom14_alt_gt_positions", "atom14_alt_gt_exists",
            "atom14_atom_is_ambiguous",
            "backbone_rigid_tensor", "backbone_rigid_mask",
            "rigidgroups_gt_frames", "rigidgroups_alt_gt_frames", "rigidgroups_gt_exists",
            "chi_angles_sin_cos", "chi_mask",
            "all_atom_positions", "all_atom_mask",
            "pseudo_beta", "pseudo_beta_mask",
        }
        out.update({k: v for k, v in sup.items() if k in keep})

    # ---- Add recycling dimension (last dim) to every tensor ----
    # AlphaFold.forward() expects batch[key][..., cycle_no] for each cycle.
    out = {
        k: v.unsqueeze(-1).expand(*v.shape, n_iters).contiguous()
        for k, v in out.items()
    }

    return out


# ---------------------------------------------------------------------------
# run_batch  (the main entry point called by the nanoFold runtime)
# ---------------------------------------------------------------------------

def run_batch(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
    training: bool,
) -> Dict[str, torch.Tensor]:
    of_batch = _build_openfold_batch(batch, cfg, training=training)

    # AlphaFold.forward() handles recycling internally; grad only on last cycle
    out = model(of_batch)

    # sm["positions"][-1]: [B, L, 14, 3]  (atom14 format, last structure-module iter)
    pred_atom14 = out["sm"]["positions"][-1]
    residue_mask = batch["residue_mask"].to(
        device=pred_atom14.device, dtype=pred_atom14.dtype
    )
    pred_atom14 = pred_atom14 * residue_mask[:, :, None, None]

    has_supervision = "atom14_positions" in batch and "atom14_mask" in batch
    if not has_supervision:
        return {"pred_atom14": pred_atom14}

    # Slice batch to the last recycling iteration for the loss
    batch_for_loss = tensor_tree_map(lambda t: t[..., -1], of_batch)

    # Select initial vs finetune loss
    step = _runtime_step(cfg)
    sched = _budget_schedule(cfg)
    raw_model = model.module if hasattr(model, "module") else model

    if step < sched["finetune_start_step"]:
        loss_fn = raw_model._loss_initial
    else:
        # Linear ramp from initial → finetune loss over finetune_ramp_steps
        ramp_steps = sched["finetune_ramp_steps"]
        ramp_w = (
            1.0
            if ramp_steps <= 0
            else min(1.0, (step - sched["finetune_start_step"]) / ramp_steps)
        )
        if ramp_w >= 1.0:
            loss_fn = raw_model._loss_finetune
        elif ramp_w <= 0.0:
            loss_fn = raw_model._loss_initial
        else:
            # Blend losses
            loss_initial = raw_model._loss_initial(out, batch_for_loss)
            loss_finetune = raw_model._loss_finetune(out, batch_for_loss)
            loss = loss_initial + ramp_w * (loss_finetune - loss_initial)
            return {"pred_atom14": pred_atom14, "loss": loss}

    loss = loss_fn(out, batch_for_loss)
    return {"pred_atom14": pred_atom14, "loss": loss}
