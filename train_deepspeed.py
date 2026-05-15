"""DeepSpeed training script.

Launch with deepspeed (or torchrun):
    deepspeed --num_gpus=N train_deepspeed.py \
        --config <path> [--track <id>] --ds-config <path/to/ds_config.json> [...]

This script is a drop-in replacement for train_ddp.py that swaps PyTorch DDP
for DeepSpeed. The training contract is preserved:
- Track policy enforcement, fingerprint verification, official mode
- Submission API hooks: build_model, build_optimizer, build_scheduler, run_batch
- Per-step JSONL metrics with cumulative_samples_seen accounting
- Checkpoints written as ckpt_step_<N>.pt + ckpt_last.pt (consolidated, rank-0)
  so predict.py / score.py / run_official.py keep working unchanged
- Eval on rank 0 every eval_every steps (lDDT-Cα, Cα RMSD, atom14 RMSD)
- Mid-run finetune-phase loader rebuild (matches train_ddp.py)

DeepSpeed engine owns: grad-accum micro-batch sync, bf16/fp16 autocast,
gradient clipping, and optimizer.step(). It does NOT own the scheduler — we
keep the submission's scheduler external and step it once per train step so
budget-aware schedules (e.g. AlphaFold finetune ramp) work unchanged.

Activation/gradient checkpointing lives inside the submission's model and is
unaffected by DeepSpeed (PyTorch's torch.utils.checkpoint works inside the
engine). To use DeepSpeed's partitioned activation checkpointing instead, the
submission would need to call deepspeed.checkpointing.checkpoint directly.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable

import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from nanofold.competition_policy import (
    DEFAULT_TRACK_ID,
    OFFICIAL_DATASET_FINGERPRINT_PATH,
    TrackSpec,
    apply_track_policy,
    assert_track_policy,
    compute_effective_batch_size,
    compute_residue_budget,
    compute_sample_budget,
    enforce_model_param_limit,
    load_track_spec,
)
from nanofold.data import ProcessedNPZDataset, collate_batch
from nanofold.dataset_integrity import verify_dataset_against_fingerprint
from nanofold.metrics import FOLDSCORE_COMPONENT_NAMES, foldscore_components, lddt_ca
from nanofold.submission_runtime import load_submission_hooks, run_submission_batch
from nanofold.utils import (
    RunPaths,
    count_parameters,
    ensure_dir,
    get_env_metadata,
    load_torch_checkpoint,
    make_dataloader_generator,
    seed_worker,
    serialize_numpy_rng_state,
    set_seed,
    should_pin_memory,
    to_device,
    utc_now_iso,
)
from nanofold.utils import sha256_file as _sha256_file


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--track", type=str, default=DEFAULT_TRACK_ID)
    ap.add_argument("--official", action="store_true")
    ap.add_argument("--fingerprint", type=str, default="")
    ap.add_argument("--verify-fingerprint", action="store_true")
    ap.add_argument("--processed-features-dir", type=str, default="")
    ap.add_argument("--processed-labels-dir", type=str, default="")
    ap.add_argument(
        "--ds-config",
        type=str,
        required=True,
        help="Path to a DeepSpeed JSON config (bf16, zero_optimization, gradient_accumulation_steps, gradient_clipping, ...).",
    )
    ap.add_argument(
        "--resume",
        type=str,
        default="",
        help=(
            "Resume path. Either a ckpt_step_<N>.pt single file (warm restart, "
            "model weights only) or a DeepSpeed checkpoint directory containing "
            "ds_engine/ subfolders (full engine state)."
        ),
    )
    ap.add_argument("--reset-run", action="store_true")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--allow-resume-mismatch", action="store_true")
    # DeepSpeed adds --local_rank, --deepspeed*, etc.
    return deepspeed.add_config_arguments(ap).parse_args()


# ---------------------------------------------------------------------------
# Distributed init
# ---------------------------------------------------------------------------

def init_distributed() -> tuple[int, int, int]:
    """Initialize the process group via DeepSpeed (NCCL backend)."""
    deepspeed.init_distributed(dist_backend="nccl", timeout=timedelta(hours=2))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


# ---------------------------------------------------------------------------
# Misc helpers (mirrored from train_ddp.py / train.py)
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def empty_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def normalize_num_workers(n: int) -> int:
    n = int(n)
    if n <= 0:
        return 0
    if sys.platform == "darwin" and sys.version_info >= (3, 13):
        return 0
    return n


def _mean_tensors(values: Iterable[torch.Tensor]) -> float:
    finite = [v.float().reshape(()) for v in values if torch.isfinite(v.float().reshape(()))]
    if not finite:
        return float("nan")
    return float(torch.stack(finite).mean())


def _scalar_output_metrics(out: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    metrics: Dict[str, torch.Tensor] = {}
    for key, value in out.items():
        if key in {"loss", "pred_atom14", "pred_ca"}:
            continue
        if not torch.is_tensor(value) or value.numel() == 0:
            continue
        scalar = value.detach().float().mean().cpu()
        if torch.isfinite(scalar):
            metrics[str(key)] = scalar
    return metrics


def _append_scalar_output_metrics(target: Dict[str, list[torch.Tensor]], out: Dict[str, Any]) -> None:
    for key, value in _scalar_output_metrics(out).items():
        target.setdefault(key, []).append(value)


def _summarize_eval_metrics(
    scores: list[torch.Tensor],
    losses: list[torch.Tensor],
    ca_rmsds: list[torch.Tensor] | None = None,
    atom14_rmsds: list[torch.Tensor] | None = None,
    scalar_metrics: Dict[str, list[torch.Tensor]] | None = None,
    foldscore_metric_values: Dict[str, list[torch.Tensor]] | None = None,
) -> Dict[str, float]:
    metrics = {"val_lddt_ca": _mean_tensors(scores)}
    if losses:
        metrics["val_loss"] = _mean_tensors(losses)
    if ca_rmsds is not None:
        metrics["val_rmsd_ca"] = _mean_tensors(ca_rmsds)
    if atom14_rmsds is not None:
        metrics["val_rmsd_atom14"] = _mean_tensors(atom14_rmsds)
    if scalar_metrics is not None:
        for name, values in sorted(scalar_metrics.items()):
            metrics[f"val_{name}"] = _mean_tensors(values)
    if foldscore_metric_values is not None:
        for name, values in sorted(foldscore_metric_values.items()):
            metrics[f"val_{name}"] = _mean_tensors(values)
    return metrics


def _cfg_with_runtime(cfg, *, step, cumulative_samples_seen, max_steps, sample_budget):
    runtime_cfg = dict(cfg)
    runtime_cfg["_runtime"] = {
        "step": int(step),
        "cumulative_samples_seen": int(cumulative_samples_seen),
        "max_steps": int(max_steps),
        "sample_budget": int(sample_budget),
    }
    return runtime_cfg


def _resolve_fingerprint_path(args: argparse.Namespace, track_spec: TrackSpec) -> str:
    if args.fingerprint:
        return args.fingerprint
    if track_spec.fingerprint_path:
        return track_spec.fingerprint_path
    if args.official or args.verify_fingerprint:
        raise ValueError(f"Track `{track_spec.track_id}` does not define a fingerprint path.")
    return OFFICIAL_DATASET_FINGERPRINT_PATH


def _format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, secs = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{secs:02d}s"
    if m:
        return f"{m}m{secs:02d}s"
    return f"{secs}s"


def resume_metadata_mismatches(
    *,
    ckpt_obj: Dict[str, Any],
    submission_entrypoint_sha256: str | None,
    config_sha256: str,
    track_id: str,
    fingerprint_sha256: str | None,
    n_params: int,
) -> list[str]:
    expected = {
        "submission_entrypoint_sha256": submission_entrypoint_sha256,
        "config_sha256": config_sha256,
        "track_id": track_id,
        "fingerprint_sha256": fingerprint_sha256,
        "n_params": n_params,
    }
    mismatches = []
    for key, exp in expected.items():
        if ckpt_obj.get(key) != exp:
            mismatches.append(f"{key}: expected={exp!r}, actual={ckpt_obj.get(key)!r}")
    return mismatches


def _truncate_step_metrics(path: Path, *, max_step: int) -> int:
    if not path.exists():
        return 0
    kept, removed = [], 0
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            rec = json.loads(s)
            if int(rec.get("step", max_step)) <= max_step:
                kept.append(line)
            else:
                removed += 1
        except Exception:
            kept.append(line)
    if removed:
        path.write_text("\n".join(kept) + ("\n" if kept else ""))
    return removed


def _truncate_metric_history(metrics: Dict[str, Any], *, max_step: int) -> int:
    history = metrics.get("history")
    if not isinstance(history, list):
        return 0
    kept, removed = [], 0
    for item in history:
        if not isinstance(item, dict):
            kept.append(item)
            continue
        try:
            if int(item.get("step", max_step)) <= max_step:
                kept.append(item)
            else:
                removed += 1
        except Exception:
            kept.append(item)
    if removed:
        metrics["history"] = kept
    return removed


def _verify_dataset(*, cfg, fingerprint_path, require_no_missing, track_id=None):
    data_cfg = cfg["data"]
    verify_dataset_against_fingerprint(
        processed_features_dir=data_cfg["processed_features_dir"],
        processed_labels_dir=data_cfg.get("processed_labels_dir"),
        train_manifest=data_cfg["train_manifest"],
        val_manifest=data_cfg["val_manifest"],
        expected_fingerprint_path=fingerprint_path,
        require_no_missing=require_no_missing,
        track_id=track_id,
    )


# ---------------------------------------------------------------------------
# DataLoader (mirrors train_ddp.make_loader)
# ---------------------------------------------------------------------------

def make_dataset(cfg: Dict[str, Any], split: str, *, allow_missing: bool) -> ProcessedNPZDataset:
    data_cfg = cfg["data"]
    manifest_path = data_cfg["train_manifest"] if split == "train" else data_cfg["val_manifest"]
    return ProcessedNPZDataset(
        processed_features_dir=data_cfg["processed_features_dir"],
        processed_labels_dir=data_cfg.get("processed_labels_dir"),
        include_labels=True,
        fail_if_labels_present=False,
        manifest_path=manifest_path,
        allow_missing=allow_missing,
    )


def make_loader(
    cfg: Dict[str, Any],
    split: str,
    *,
    device: torch.device,
    sampler,
    generator_seed: int,
) -> DataLoader:
    data_cfg = cfg["data"]
    if split == "train":
        crop_mode = str(data_cfg.get("train_crop_mode", "random"))
        msa_sample_mode = str(data_cfg.get("train_msa_sample_mode", "random"))
    else:
        crop_mode = str(data_cfg.get("val_crop_mode", "center"))
        msa_sample_mode = str(data_cfg.get("val_msa_sample_mode", "top"))

    ds = make_dataset(cfg, split, allow_missing=True)
    if getattr(ds, "missing_chain_ids", None):
        print(f"[{split}] Skipping {len(ds.missing_chain_ids)} missing chains.", flush=True)

    collate_fn = partial(
        collate_batch,
        crop_size=int(data_cfg["crop_size"]),
        msa_depth=int(data_cfg["msa_depth"]),
        crop_mode=crop_mode,
        msa_sample_mode=msa_sample_mode,
    )
    num_workers = normalize_num_workers(int(data_cfg.get("num_workers", 0)))
    return DataLoader(
        ds,
        batch_size=data_cfg.get("batch_size", 1),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=should_pin_memory(device),
        collate_fn=collate_fn,
        drop_last=(split == "train"),
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=make_dataloader_generator(generator_seed),
    )


# ---------------------------------------------------------------------------
# Eval metrics (rank-0 only)
# ---------------------------------------------------------------------------

@torch.no_grad()
def batch_lddt_ca(pred_ca, true_ca, ca_mask, residue_mask):
    scores = [lddt_ca(pred_ca[b], true_ca[b], ca_mask[b] & residue_mask[b]) for b in range(pred_ca.shape[0])]
    return torch.stack(scores).mean()


@torch.no_grad()
def masked_kabsch_rmsd(pred_points, true_points, point_mask):
    pred_points = pred_points.detach().float().cpu()
    true_points = true_points.detach().float().cpu()
    point_mask = point_mask.detach().cpu().bool()
    if int(point_mask.sum().item()) < 3:
        return torch.full((), float("nan"))
    pred = pred_points[point_mask]
    true = true_points[point_mask]
    pred_c = pred - pred.mean(0, keepdim=True)
    true_c = true - true.mean(0, keepdim=True)
    u, _, vh = torch.linalg.svd(pred_c.T @ true_c, full_matrices=False)
    corr = torch.ones(3)
    if torch.det(u @ vh) < 0:
        corr[-1] = -1.0
    rot = u @ torch.diag(corr) @ vh
    aligned = pred_c @ rot
    return torch.sqrt((aligned - true_c).square().sum(-1).mean().clamp_min(0.0))


@torch.no_grad()
def batch_rmsd_ca(pred_ca, true_ca, ca_mask, residue_mask):
    rmsds = [masked_kabsch_rmsd(pred_ca[b], true_ca[b], ca_mask[b] & residue_mask[b]) for b in range(pred_ca.shape[0])]
    return torch.nanmean(torch.stack(rmsds))


@torch.no_grad()
def batch_rmsd_atom14(pred_atom14, true_atom14, atom14_mask, residue_mask):
    rmsds = []
    for b in range(pred_atom14.shape[0]):
        mask = (atom14_mask[b] & residue_mask[b, :, None]).reshape(-1)
        rmsds.append(masked_kabsch_rmsd(pred_atom14[b].reshape(-1, 3), true_atom14[b].reshape(-1, 3), mask))
    return torch.nanmean(torch.stack(rmsds))


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _gather_full_state_dict(engine) -> Dict[str, torch.Tensor]:
    """Return a CPU state dict of the underlying model.

    For ZeRO-0/1/2 each rank already has the full model — just call state_dict().
    For ZeRO-3 use deepspeed.zero.GatheredParameters to materialize params.
    """
    module = engine.module
    zero_stage = engine.zero_optimization_stage()
    if zero_stage < 3:
        return {k: v.detach().cpu() for k, v in module.state_dict().items()}

    # ZeRO-3: gather sharded params on rank 0 before reading state_dict.
    from deepspeed.runtime.zero.partition_parameters import GatheredParameters
    params = [p for p in module.parameters()]
    with GatheredParameters(params, modifier_rank=0, enabled=True):
        if dist.get_rank() == 0:
            sd = {k: v.detach().cpu() for k, v in module.state_dict().items()}
        else:
            sd = {}
    return sd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = init_distributed()
    is_rank_zero = rank == 0
    device = torch.device(f"cuda:{local_rank}")

    def log(msg: str) -> None:
        if is_rank_zero:
            print(msg, flush=True)

    if args.reset_run and args.resume:
        raise ValueError("`--reset-run` cannot be combined with `--resume`.")

    # ── config + track policy ────────────────────────────────────────────────
    config_path = Path(args.config).resolve()
    raw_cfg = load_config(args.config)
    track_spec = load_track_spec(args.track)
    cfg = apply_track_policy(raw_cfg, track_spec=track_spec) if args.official else raw_cfg
    data_cfg = cfg.setdefault("data", {})
    if args.processed_features_dir:
        data_cfg["processed_features_dir"] = args.processed_features_dir
    if args.processed_labels_dir:
        data_cfg["processed_labels_dir"] = args.processed_labels_dir

    deterministic = bool(args.deterministic or args.official)
    fingerprint_path = _resolve_fingerprint_path(args, track_spec)

    if args.official:
        assert_track_policy(cfg=cfg, track_spec=track_spec, enforce_manifest_paths=True, enforce_manifest_hashes=True)
        if is_rank_zero:
            _verify_dataset(cfg=cfg, fingerprint_path=fingerprint_path, require_no_missing=True, track_id=track_spec.track_id)
        dist.barrier()
    elif args.verify_fingerprint and is_rank_zero:
        _verify_dataset(cfg=cfg, fingerprint_path=fingerprint_path, require_no_missing=False, track_id=track_spec.track_id)
    dist.barrier()

    # ── submission hooks + run paths ─────────────────────────────────────────
    hooks = load_submission_hooks(cfg, config_path, allowed_root=config_path.parent)
    run_name = cfg.get("run_name", "run")
    paths = RunPaths.from_run_name(run_name)
    if is_rank_zero:
        ensure_dir(paths.run_dir)
        ensure_dir(paths.ckpt_dir)
    dist.barrier()

    step_metrics_path = paths.run_dir / "train_metrics.jsonl"
    ds_ckpt_dir = paths.ckpt_dir / "ds_engine"
    if args.reset_run and is_rank_zero:
        for p in (paths.metrics_path, paths.log_path, step_metrics_path):
            if Path(p).exists():
                Path(p).unlink()
        for p in paths.ckpt_dir.glob("ckpt_step_*.pt"):
            p.unlink()
        if (paths.ckpt_dir / "ckpt_last.pt").exists():
            (paths.ckpt_dir / "ckpt_last.pt").unlink()
        if ds_ckpt_dir.exists():
            shutil.rmtree(ds_ckpt_dir)
    dist.barrier()

    seed = int(cfg.get("seed", 0))
    set_seed(seed + rank, deterministic=deterministic)

    log(f"Using device: {device} | world_size={world_size}")

    # ── data ─────────────────────────────────────────────────────────────────
    train_ds = make_dataset(cfg, "train", allow_missing=not args.official)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    train_loader = make_loader(cfg, "train", device=device, sampler=train_sampler, generator_seed=seed)

    val_loader = None
    if is_rank_zero:
        val_ds = make_dataset(cfg, "val", allow_missing=not args.official)
        val_sampler = DistributedSampler(val_ds, num_replicas=1, rank=0, shuffle=False, drop_last=False)
        val_loader = make_loader(cfg, "val", device=device, sampler=val_sampler, generator_seed=seed + 1)

    # ── model + optimizer + scheduler, then DeepSpeed.initialize ─────────────
    raw_model = hooks.build_model(cfg)
    if not isinstance(raw_model, nn.Module):
        raise TypeError("`build_model` must return a torch.nn.Module")
    raw_model = raw_model.to(device)

    n_params = count_parameters(raw_model)
    log(f"Model params: {n_params:,} | Submission: {hooks.module_ref}")
    if args.official:
        enforce_model_param_limit(track_spec=track_spec, n_params=n_params)

    opt = hooks.build_optimizer(cfg, raw_model)
    scheduler = hooks.build_scheduler(cfg, opt) if hooks.build_scheduler is not None else None

    # Load and validate the DeepSpeed config dict so we can sanity-check it
    # against the track policy before deepspeed.initialize touches the model.
    ds_config_path = Path(args.ds_config).resolve()
    if not ds_config_path.exists():
        raise FileNotFoundError(f"DeepSpeed config not found: {ds_config_path}")
    ds_config = json.loads(ds_config_path.read_text())

    cfg_grad_accum = int(cfg["train"].get("grad_accum_steps", 1))
    ds_grad_accum = ds_config.get("gradient_accumulation_steps", "auto")
    if ds_grad_accum != "auto" and int(ds_grad_accum) != cfg_grad_accum:
        raise ValueError(
            f"`gradient_accumulation_steps` in DeepSpeed config ({ds_grad_accum}) "
            f"must match `train.grad_accum_steps` in the run config ({cfg_grad_accum})."
        )
    if ds_grad_accum == "auto":
        ds_config["gradient_accumulation_steps"] = cfg_grad_accum
    ds_config.setdefault("train_micro_batch_size_per_gpu", int(cfg["data"].get("batch_size", 1)))
    # DeepSpeed cross-checks train_batch_size == micro_batch * grad_accum * world_size.
    ds_config["train_batch_size"] = (
        ds_config["train_micro_batch_size_per_gpu"] * ds_config["gradient_accumulation_steps"] * world_size
    )

    engine, opt, _, _ = deepspeed.initialize(
        args=args,
        model=raw_model,
        optimizer=opt,
        model_parameters=raw_model.parameters(),
        config=ds_config,
    )

    use_bf16 = bool(engine.bfloat16_enabled())
    use_fp16 = bool(engine.fp16_enabled()) if hasattr(engine, "fp16_enabled") else False
    log(
        f"DeepSpeed engine: zero_stage={engine.zero_optimization_stage()} "
        f"bf16={use_bf16} fp16={use_fp16} grad_accum={engine.gradient_accumulation_steps()}"
    )

    # ── budgets + metadata ───────────────────────────────────────────────────
    tcfg = cfg["train"]
    max_steps = int(tcfg["max_steps"])
    log_every = int(tcfg.get("log_every", 50))
    eval_every = int(tcfg.get("eval_every", 500))
    save_every = int(tcfg.get("save_every", 500))
    grad_accum_steps = engine.gradient_accumulation_steps()
    eval_foldscore_components = bool(tcfg.get("eval_foldscore_components", False))
    finetune_start_step = int(tcfg.get("finetune_start_step", max_steps + 1))
    finetune_crop_size = int(tcfg.get("finetune_crop_size", cfg["data"]["crop_size"]))
    finetune_data_msa_depth = int(tcfg.get("finetune_data_msa_depth", cfg["data"]["msa_depth"]))
    finetune_model_msa_depth = int(tcfg.get("finetune_model_msa_depth", cfg["model"].get("msa_depth", 128)))
    finetune_extra_msa_depth = int(tcfg.get("finetune_extra_msa_depth", cfg["model"].get("extra_msa_depth", 1024)))

    batch_size = int(cfg["data"]["batch_size"])
    crop_size = int(cfg["data"]["crop_size"])
    effective_batch_size = compute_effective_batch_size(batch_size * world_size, grad_accum_steps)
    sample_budget = compute_sample_budget(max_steps, effective_batch_size)
    residue_budget = compute_residue_budget(max_steps, effective_batch_size, crop_size)
    config_sha256 = _sha256_file(config_path)
    fingerprint_sha256 = (
        _sha256_file(fingerprint_path)
        if (args.official or args.verify_fingerprint) and Path(fingerprint_path).exists()
        else None
    )

    metrics: Dict[str, Any] = {
        "run_name": run_name,
        "track": track_spec.track_id,
        "official_mode": bool(args.official),
        "trainer": "deepspeed",
        "ds_config_path": str(ds_config_path),
        "ds_zero_stage": engine.zero_optimization_stage(),
        "ds_bf16": use_bf16,
        "ds_fp16": use_fp16,
        "seed": seed,
        "deterministic": deterministic,
        "n_params": n_params,
        "world_size": world_size,
        "submission_module": hooks.module_ref,
        "submission_entrypoint_path": hooks.source_path,
        "submission_entrypoint_sha256": hooks.source_sha256,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "config": cfg,
        "effective_batch_size": effective_batch_size,
        "sample_budget": sample_budget,
        "residue_budget": residue_budget,
        "fingerprint_path": (
            str(Path(fingerprint_path).resolve())
            if (args.official or args.verify_fingerprint)
            else None
        ),
        "fingerprint_sha256": fingerprint_sha256,
        "env": get_env_metadata(device),
        "started_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "history": [],
        "step_metrics_jsonl": str(step_metrics_path),
        "cumulative_samples_seen": 0,
        "cumulative_cropped_residues_seen": 0,
        "cumulative_nonpad_residues_seen": 0,
        "cumulative_residues_seen": 0,
    }

    start_step = 0
    cumulative_samples_seen = 0
    cumulative_cropped_residues_seen = 0
    cumulative_nonpad_residues_seen = 0

    # ── Checkpoint payload + save helpers ────────────────────────────────────
    # We write two things every save:
    #   1. DeepSpeed engine state (multi-shard, in <ckpt_dir>/ds_engine/step_<N>/)
    #      Used for full-fidelity resume (optimizer momentum, ZeRO shards, ...).
    #   2. A consolidated ckpt_step_<N>.pt on rank 0 in the format that
    #      predict.py / score.py / run_official.py already consume.

    def _checkpoint_payload(*, step_value: int, full_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "step": step_value,
            "model": full_state_dict,
            # opt.state_dict may be sharded under ZeRO; we record it best-effort.
            # The official runner only consumes "model", so a sharded "opt" is harmless.
            "opt": opt.state_dict() if dist.get_rank() == 0 else {},
            "submission_module": hooks.module_ref,
            "submission_entrypoint_path": hooks.source_path,
            "submission_entrypoint_sha256": hooks.source_sha256,
            "config": cfg,
            "config_sha256": config_sha256,
            "track_id": track_spec.track_id,
            "n_params": n_params,
            "effective_batch_size": effective_batch_size,
            "sample_budget": sample_budget,
            "residue_budget": residue_budget,
            "fingerprint_path": (
                str(Path(fingerprint_path).resolve()) if (args.official or args.verify_fingerprint) else None
            ),
            "fingerprint_sha256": fingerprint_sha256,
            "cumulative_samples_seen": cumulative_samples_seen,
            "cumulative_cropped_residues_seen": cumulative_cropped_residues_seen,
            "cumulative_nonpad_residues_seen": cumulative_nonpad_residues_seen,
            "cumulative_residues_seen": cumulative_nonpad_residues_seen,
            "rng_state": {
                "python": __import__("random").getstate(),
                "numpy": serialize_numpy_rng_state(__import__("numpy").random.get_state()),
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "trainer": "deepspeed",
            "ds_zero_stage": engine.zero_optimization_stage(),
            "ds_engine_ckpt_dir": str(ds_ckpt_dir.resolve()),
            "ds_engine_tag": f"step_{step_value}",
        }
        if scheduler is not None and callable(getattr(scheduler, "state_dict", None)):
            payload["scheduler"] = scheduler.state_dict()
        return payload

    def _save_checkpoint(*, step_value: int) -> None:
        # 1. DeepSpeed engine state on all ranks
        client_state = {
            "step": step_value,
            "cumulative_samples_seen": cumulative_samples_seen,
            "cumulative_cropped_residues_seen": cumulative_cropped_residues_seen,
            "cumulative_nonpad_residues_seen": cumulative_nonpad_residues_seen,
            "scheduler_state": (
                scheduler.state_dict()
                if scheduler is not None and callable(getattr(scheduler, "state_dict", None))
                else None
            ),
        }
        engine.save_checkpoint(str(ds_ckpt_dir), tag=f"step_{step_value}", client_state=client_state)

        # 2. Consolidated single-file checkpoint on rank 0
        full_sd = _gather_full_state_dict(engine)
        if is_rank_zero:
            step_path = paths.ckpt_dir / f"ckpt_step_{step_value}.pt"
            torch.save(_checkpoint_payload(step_value=step_value, full_state_dict=full_sd), step_path)
            shutil.copy2(step_path, paths.ckpt_dir / "ckpt_last.pt")

    # ── Resume ───────────────────────────────────────────────────────────────
    if args.resume:
        resume_path = Path(args.resume).resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume path not found: {resume_path}")

        if resume_path.is_dir():
            # DeepSpeed-native resume: full engine state, optimizer momenta, ZeRO shards.
            load_dir, client_state = engine.load_checkpoint(str(resume_path), load_optimizer_states=True, load_lr_scheduler_states=False)
            if load_dir is None:
                raise RuntimeError(f"DeepSpeed could not load checkpoint from {resume_path}")
            start_step = int(client_state.get("step", 0))
            cumulative_samples_seen = int(client_state.get("cumulative_samples_seen", start_step * effective_batch_size))
            cumulative_cropped_residues_seen = int(client_state.get("cumulative_cropped_residues_seen", 0))
            cumulative_nonpad_residues_seen = int(client_state.get("cumulative_nonpad_residues_seen", 0))
            if scheduler is not None and client_state.get("scheduler_state") is not None and callable(getattr(scheduler, "load_state_dict", None)):
                scheduler.load_state_dict(client_state["scheduler_state"])
            log(f"Resumed DeepSpeed engine from {load_dir} at step {start_step}")
        else:
            # Single-file warm restart: weights only (no optimizer momenta).
            # Useful when you trained with train_ddp.py and want to continue on DeepSpeed.
            ckpt = load_torch_checkpoint(resume_path, map_location="cpu")
            mismatches = resume_metadata_mismatches(
                ckpt_obj=ckpt,
                submission_entrypoint_sha256=hooks.source_sha256,
                config_sha256=config_sha256,
                track_id=track_spec.track_id,
                fingerprint_sha256=fingerprint_sha256,
                n_params=n_params,
            )
            if mismatches and not args.allow_resume_mismatch:
                raise ValueError("Resume checkpoint metadata mismatch:\n" + "\n".join(f"- {m}" for m in mismatches))
            target = engine.module
            target.load_state_dict(ckpt["model"], strict=True)
            if scheduler is not None and "scheduler" in ckpt and callable(getattr(scheduler, "load_state_dict", None)):
                scheduler.load_state_dict(ckpt["scheduler"])
            start_step = int(ckpt.get("step", 0))
            cumulative_samples_seen = int(ckpt.get("cumulative_samples_seen", start_step * effective_batch_size))
            cumulative_cropped_residues_seen = int(ckpt.get("cumulative_cropped_residues_seen", 0))
            cumulative_nonpad_residues_seen = int(ckpt.get("cumulative_nonpad_residues_seen", 0))
            log(f"Warm-restart from {resume_path} (weights only) at step {start_step}")

        if is_rank_zero:
            _truncate_step_metrics(step_metrics_path, max_step=start_step)
            if paths.metrics_path.exists():
                try:
                    existing = json.loads(paths.metrics_path.read_text())
                    if isinstance(existing, dict):
                        metrics.update(existing)
                        _truncate_metric_history(metrics, max_step=start_step)
                        metrics["resumed_from"] = str(resume_path)
                        metrics["updated_at"] = utc_now_iso()
                except Exception:
                    pass
            Path(paths.metrics_path).write_text(json.dumps(metrics, indent=2))
    else:
        if is_rank_zero:
            step_metrics_path.write_text("")
            Path(paths.metrics_path).write_text(json.dumps(metrics, indent=2))
        # Step-0 checkpoint for the hidden-AUC start point.
        _save_checkpoint(step_value=0)

    dist.barrier()

    # ── Bail out if already done ─────────────────────────────────────────────
    step = start_step
    if step >= max_steps:
        log(f"Already at max_steps={max_steps}; nothing to do.")
        dist.destroy_process_group()
        return

    engine.train()
    log(
        f"Run: track={track_spec.track_id} max_steps={max_steps} "
        f"eff_batch={effective_batch_size} (world={world_size}×batch={batch_size}×accum={grad_accum_steps}) "
        f"crop={crop_size} bf16={use_bf16} zero_stage={engine.zero_optimization_stage()}"
    )

    show_progress = is_rank_zero and sys.stderr.isatty()

    def log_line(msg: str) -> None:
        if is_rank_zero:
            if show_progress:
                tqdm.write(msg)
            else:
                print(msg, flush=True)

    pbar = tqdm(total=max_steps - step, desc="train", dynamic_ncols=True, disable=not show_progress)

    epoch = 0
    train_sampler.set_epoch(epoch)
    train_iter = iter(train_loader)
    step_start = time.perf_counter()
    run_start = time.perf_counter()

    # Apply finetune-phase data/model overrides if resuming into that phase
    entered_finetune = step >= finetune_start_step
    if entered_finetune:
        cfg["data"]["crop_size"] = finetune_crop_size
        cfg["data"]["msa_depth"] = finetune_data_msa_depth
        cfg["model"]["msa_depth"] = finetune_model_msa_depth
        cfg["model"]["extra_msa_depth"] = finetune_extra_msa_depth
        train_loader = make_loader(cfg, "train", device=device, sampler=train_sampler, generator_seed=seed)
        train_iter = iter(train_loader)

    # ── Eval closure (rank 0 only) ───────────────────────────────────────────
    def run_eval() -> Dict[str, float]:
        assert is_rank_zero and val_loader is not None
        engine.eval()
        scores: list[torch.Tensor] = []
        losses: list[torch.Tensor] = []
        ca_rmsds: list[torch.Tensor] = []
        atom14_rmsds: list[torch.Tensor] = []
        scalar_metrics: Dict[str, list[torch.Tensor]] = {}
        foldscore_metric_values = (
            {name: [] for name in FOLDSCORE_COMPONENT_NAMES} if eval_foldscore_components else {}
        )
        runtime_cfg = _cfg_with_runtime(
            cfg,
            step=step,
            cumulative_samples_seen=cumulative_samples_seen,
            max_steps=max_steps,
            sample_budget=sample_budget,
        )
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="val", leave=False, disable=not show_progress):
                batch = to_device(batch, device)
                out = run_submission_batch(
                    hooks,
                    model=engine,
                    batch=batch,
                    cfg=runtime_cfg,
                    training=False,
                    expose_supervision=True,
                )
                pred_ca = out["pred_ca"]
                scores.append(
                    batch_lddt_ca(pred_ca, batch["ca_coords"], batch["ca_mask"], batch["residue_mask"]).detach().cpu()
                )
                ca_rmsds.append(
                    batch_rmsd_ca(pred_ca, batch["ca_coords"], batch["ca_mask"], batch["residue_mask"]).detach().cpu()
                )
                atom14_rmsds.append(
                    batch_rmsd_atom14(
                        out["pred_atom14"],
                        batch["atom14_positions"],
                        batch["atom14_mask"],
                        batch["residue_mask"],
                    ).detach().cpu()
                )
                if "loss" in out:
                    losses.append(out["loss"].detach().cpu())
                _append_scalar_output_metrics(scalar_metrics, out)
                if eval_foldscore_components:
                    for bidx in range(out["pred_atom14"].shape[0]):
                        comps = foldscore_components(
                            pred_atom14=out["pred_atom14"][bidx],
                            true_atom14=batch["atom14_positions"][bidx],
                            atom14_mask=(batch["atom14_mask"][bidx] & batch["residue_mask"][bidx, :, None]),
                            aatype=batch["aatype"][bidx],
                        )
                        for name, value in comps.items():
                            foldscore_metric_values.setdefault(name, []).append(value.detach().cpu())
        engine.train()
        empty_device_cache(device)
        return _summarize_eval_metrics(
            scores,
            losses,
            ca_rmsds,
            atom14_rmsds,
            scalar_metrics=scalar_metrics,
            foldscore_metric_values=foldscore_metric_values if eval_foldscore_components else None,
        )

    # ── Training loop ────────────────────────────────────────────────────────
    while step < max_steps:
        if not entered_finetune and step >= finetune_start_step:
            entered_finetune = True
            cfg["data"]["crop_size"] = finetune_crop_size
            cfg["data"]["msa_depth"] = finetune_data_msa_depth
            cfg["model"]["msa_depth"] = finetune_model_msa_depth
            cfg["model"]["extra_msa_depth"] = finetune_extra_msa_depth
            train_loader = make_loader(cfg, "train", device=device, sampler=train_sampler, generator_seed=seed)
            train_sampler.set_epoch(epoch)
            train_iter = iter(train_loader)
            log_line(
                f"[finetune] step {step}: crop {crop_size}→{finetune_crop_size}, "
                f"data_msa→{finetune_data_msa_depth}, model_msa→{finetune_model_msa_depth}, "
                f"extra_msa→{finetune_extra_msa_depth}"
            )

        running_loss = 0.0
        scalar_metric_sums: Dict[str, float] = {}
        samples_this_step = 0
        cropped_residues_this_step = 0
        nonpad_residues_this_step = 0

        # DeepSpeed owns grad accumulation: it tracks the micro-step counter
        # internally and only triggers an optimizer step + zero_grad on the
        # last micro-batch. We loop grad_accum_steps times to feed it data.
        for _ in range(grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                epoch += 1
                train_sampler.set_epoch(epoch)
                train_iter = iter(train_loader)
                batch = next(train_iter)

            batch = to_device(batch, device)
            samples_this_step += int(batch["aatype"].shape[0])
            cropped_residues_this_step += int(batch["aatype"].shape[0] * batch["aatype"].shape[1])
            nonpad_residues_this_step += int(batch["residue_mask"].sum().item())

            runtime_cfg = _cfg_with_runtime(
                cfg,
                step=step,
                cumulative_samples_seen=cumulative_samples_seen,
                max_steps=max_steps,
                sample_budget=sample_budget,
            )
            out = run_submission_batch(hooks, model=engine, batch=batch, cfg=runtime_cfg, training=True)
            raw_loss = out["loss"]
            for metric_name, metric_value in _scalar_output_metrics(out).items():
                scalar_metric_sums[metric_name] = scalar_metric_sums.get(metric_name, 0.0) + float(metric_value)
            if not raw_loss.requires_grad:
                pred_atom14 = out.get("pred_atom14")
                if torch.is_tensor(pred_atom14):
                    raw_loss = raw_loss + pred_atom14.sum() * 0.0
                else:
                    raise RuntimeError("Submission returned non-differentiable loss with no pred_atom14.")

            # DeepSpeed scales the loss internally for grad accumulation —
            # do NOT divide by grad_accum_steps here (unlike train_ddp.py).
            engine.backward(raw_loss)
            engine.step()  # No-op on micro-steps, performs opt step + zero_grad on the last.
            running_loss += float(raw_loss.detach())

        # Step the *external* scheduler once per train step (not per micro-step).
        if scheduler is not None:
            scheduler.step()

        # All-reduce loss for logging
        loss_tensor = torch.tensor(running_loss / grad_accum_steps, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        train_loss = float(loss_tensor.item())

        step += 1
        if is_rank_zero:
            pbar.update(1)

        counts = torch.tensor(
            [samples_this_step, cropped_residues_this_step, nonpad_residues_this_step],
            dtype=torch.long,
            device=device,
        )
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        cumulative_samples_seen += int(counts[0].item())
        cumulative_cropped_residues_seen += int(counts[1].item())
        cumulative_nonpad_residues_seen += int(counts[2].item())

        lr = float(opt.param_groups[0]["lr"]) if opt.param_groups else float("nan")
        now = time.perf_counter()
        step_seconds = max(now - step_start, 1e-8)
        step_start = now
        run_elapsed_seconds = now - run_start
        steps_completed_this_run = step - start_step

        step_record = {
            "timestamp": utc_now_iso(),
            "step": step,
            "train_loss": train_loss,
            "lr": lr,
            "step_seconds": step_seconds,
            "steps_per_sec": 1.0 / step_seconds,
            "samples_per_sec": float(counts[0].item()) / step_seconds,
            "cumulative_samples_seen": cumulative_samples_seen,
            "cumulative_cropped_residues_seen": cumulative_cropped_residues_seen,
            "cumulative_nonpad_residues_seen": cumulative_nonpad_residues_seen,
            "cumulative_residues_seen": cumulative_nonpad_residues_seen,
            "sample_budget_fraction": (
                cumulative_samples_seen / float(sample_budget) if sample_budget > 0 else float("nan")
            ),
            "nonpad_residue_budget_fraction": (
                cumulative_nonpad_residues_seen / float(residue_budget) if residue_budget > 0 else float("nan")
            ),
        }
        for metric_name, metric_sum in sorted(scalar_metric_sums.items()):
            step_record[f"train_{metric_name}"] = metric_sum / float(grad_accum_steps)

        should_log = (
            step == start_step + 1
            or steps_completed_this_run <= 5
            or step % log_every == 0
            or step == max_steps
        )
        if is_rank_zero:
            with step_metrics_path.open("a") as f:
                f.write(json.dumps(step_record) + "\n")
            if should_log:
                eta = max(max_steps - step, 0) * (run_elapsed_seconds / max(steps_completed_this_run, 1))
                pbar.set_postfix(loss=f"{train_loss:.4f}", lr=f"{lr:.3e}")
                log_line(
                    f"[train] step {step}/{max_steps} ({step / max_steps:.1%}) "
                    f"loss={train_loss:.4f} lr={lr:.3e} "
                    f"samples={cumulative_samples_seen} step_time={step_seconds:.2f}s "
                    f"elapsed={_format_duration(run_elapsed_seconds)} eta={_format_duration(eta)}"
                )

        if step % save_every == 0 or step == max_steps:
            dist.barrier()
            _save_checkpoint(step_value=step)
            log_line(f"[checkpoint] step {step}: wrote {paths.ckpt_dir / 'ckpt_last.pt'}")
            dist.barrier()

        if step % eval_every == 0 or step == max_steps:
            log_line(f"[eval] step {step}: starting public validation")
            if is_rank_zero:
                val_metrics = run_eval()
                val_metrics["step"] = step
                val_metrics["train_loss"] = train_loss
                val_metrics["lr"] = lr
                val_metrics["cumulative_samples_seen"] = cumulative_samples_seen
                val_metrics["sample_budget_fraction"] = (
                    cumulative_samples_seen / float(sample_budget) if sample_budget > 0 else float("nan")
                )
                metrics["history"].append(val_metrics)
                metrics["updated_at"] = utc_now_iso()
                metrics["cumulative_samples_seen"] = cumulative_samples_seen
                metrics["cumulative_cropped_residues_seen"] = cumulative_cropped_residues_seen
                metrics["cumulative_nonpad_residues_seen"] = cumulative_nonpad_residues_seen
                metrics["cumulative_residues_seen"] = cumulative_nonpad_residues_seen
                Path(paths.metrics_path).write_text(json.dumps(metrics, indent=2))
                log_line(
                    f"[eval] step {step} val_loss={val_metrics.get('val_loss', float('nan')):.4f} "
                    f"val_lddt_ca={val_metrics.get('val_lddt_ca', float('nan')):.4f} "
                    f"val_rmsd_ca={val_metrics.get('val_rmsd_ca', float('nan')):.3f}"
                )
            dist.barrier()

    pbar.close()
    if is_rank_zero:
        metrics["cumulative_samples_seen"] = cumulative_samples_seen
        metrics["cumulative_cropped_residues_seen"] = cumulative_cropped_residues_seen
        metrics["cumulative_nonpad_residues_seen"] = cumulative_nonpad_residues_seen
        metrics["cumulative_residues_seen"] = cumulative_nonpad_residues_seen
        metrics["wall_time_seconds"] = float(time.perf_counter() - run_start)
        metrics["finished_at"] = utc_now_iso()
        metrics["updated_at"] = utc_now_iso()
        Path(paths.metrics_path).write_text(json.dumps(metrics, indent=2))
        print("Done. Metrics:", paths.metrics_path)
        print("Checkpoint:", paths.ckpt_dir / "ckpt_last.pt")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
