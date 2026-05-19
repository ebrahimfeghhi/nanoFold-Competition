#!/usr/bin/env python3
"""Benchmark OpenFold with Flash Attention: measures peak vRAM and step time.

Usage:
    python scripts/benchmark_flash.py --gpu 1 --n-steps 20 --grad-accum-steps 32 --out /tmp/flash_bench.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENFOLD_ROOT = REPO_ROOT / "third_party" / "openfold"
MINALPHAFOLD2_ROOT = REPO_ROOT / "third_party" / "minAlphaFold2"
# ~/openfold only for the compiled attn_core_inplace_cuda kernel (.so)
OPENFOLD_KERNEL_DIR = Path.home() / "openfold"

# Insert in reverse priority order (last insert = highest priority)
for _p in [str(OPENFOLD_KERNEL_DIR), str(MINALPHAFOLD2_ROOT), str(OPENFOLD_ROOT), str(REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_torch_lib = str(Path(torch.__file__).parent / "lib")
_ld = os.environ.get("LD_LIBRARY_PATH", "")
if _torch_lib not in _ld:
    os.environ["LD_LIBRARY_PATH"] = f"{_torch_lib}:{_ld}"

from openfold.model.model import AlphaFold
from openfold.config import model_config as _of_model_config
from openfold.utils.loss import AlphaFoldLoss

from nanofold.data import ProcessedNPZDataset, collate_batch
from nanofold.utils import to_device, make_dataloader_generator, seed_worker, should_pin_memory


def build_model(device: torch.device) -> torch.nn.Module:
    oc = _of_model_config("initial_training", train=True)
    oc.model.template.enabled = False
    oc.globals.use_flash = True
    oc.globals.blocks_per_ckpt = 1

    model = AlphaFold(oc)

    initial_oc = _of_model_config("initial_training", train=True)
    initial_oc.loss.violation.weight = 0.0
    model._loss_initial = AlphaFoldLoss(initial_oc.loss)

    finetune_oc = _of_model_config("initial_training", train=True)
    finetune_oc.loss.violation.weight = 1.0
    finetune_oc.loss.experimentally_resolved.weight = 0.01
    model._loss_finetune = AlphaFoldLoss(finetune_oc.loss)

    model = model.to(device)
    return model


def make_loader(cfg: dict, device: torch.device) -> DataLoader:
    data_cfg = cfg["data"]
    ds = ProcessedNPZDataset(
        processed_features_dir=data_cfg["processed_features_dir"],
        processed_labels_dir=data_cfg.get("processed_labels_dir"),
        include_labels=True,
        fail_if_labels_present=False,
        manifest_path=data_cfg["train_manifest"],
        allow_missing=True,
    )
    collate_fn = partial(
        collate_batch,
        crop_size=int(data_cfg["crop_size"]),
        msa_depth=int(data_cfg["msa_depth"]),
        crop_mode="random",
        msa_sample_mode="random",
    )
    return DataLoader(
        ds,
        batch_size=int(data_cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=2,
        pin_memory=should_pin_memory(device),
        collate_fn=collate_fn,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=make_dataloader_generator(42),
    )


def run(args):
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    cfg = yaml.safe_load(Path(args.config).read_text())

    data_cfg = cfg["data"]
    for key in ("processed_features_dir", "processed_labels_dir", "train_manifest", "val_manifest"):
        if key in data_cfg and data_cfg[key] and not Path(data_cfg[key]).is_absolute():
            data_cfg[key] = str(REPO_ROOT / data_cfg[key])

    label = "flash=True bpc=1"
    print(f"[{label}] Building model on GPU {args.gpu}...", flush=True)
    model = build_model(device)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        betas=(cfg["optim"]["beta1"], cfg["optim"]["beta2"]),
        eps=cfg["optim"]["eps"],
        weight_decay=cfg["optim"].get("weight_decay", 0.0),
    )

    loader = make_loader(cfg, device)
    data_iter = iter(loader)

    # Runtime stub required by run_batch for loss selection
    cfg["_runtime"] = {
        "step": 0,
        "cumulative_samples_seen": 0,
        "max_steps": 6000,
        "sample_budget": 768000,
    }

    submission_path = REPO_ROOT / "submissions" / "openfold_unlimited" / "submission.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("_submission", str(submission_path))
    sub = importlib.util.module_from_spec(spec)
    sub_dir = str(submission_path.parent)
    if sub_dir not in sys.path:
        sys.path.insert(0, sub_dir)
    spec.loader.exec_module(sub)
    run_batch = sub.run_batch

    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    step_times = []
    peak_vrams_mb = []

    print(f"[{label}] Starting {args.n_steps} steps (grad_accum={args.grad_accum_steps})...", flush=True)

    for step in range(args.n_steps):
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        for _ in range(args.grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            batch = to_device(batch, device)
            with autocast_ctx:
                out = run_batch(model, batch, cfg, training=True)
            loss = out.get("loss")
            if loss is None:
                continue
            (loss / args.grad_accum_steps).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2

        step_times.append(elapsed)
        peak_vrams_mb.append(peak_mb)

        print(
            f"[{label}] step {step+1}/{args.n_steps} "
            f"time={elapsed:.1f}s  peak_vram={peak_mb:.0f}MB",
            flush=True,
        )

    result = {
        "label": label,
        "use_flash": True,
        "blocks_per_ckpt": 1,
        "gpu": args.gpu,
        "n_steps": args.n_steps,
        "grad_accum_steps": args.grad_accum_steps,
        "step_times_s": step_times,
        "peak_vram_mb": peak_vrams_mb,
        "avg_step_time_s": sum(step_times) / len(step_times),
        "avg_peak_vram_mb": sum(peak_vrams_mb) / len(peak_vrams_mb),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[{label}] Done. Results:", flush=True)
    print(f"  avg step time : {result['avg_step_time_s']:.1f}s", flush=True)
    print(f"  avg peak vRAM : {result['avg_peak_vram_mb']:.0f} MB", flush=True)
    print(f"  Saved to {args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--grad-accum-steps", type=int, default=32)
    ap.add_argument("--config", type=str,
                    default=str(REPO_ROOT / "submissions/openfold_unlimited/config_bpc1_test.yaml"))
    ap.add_argument("--out", type=str, default="/tmp/flash_bench.json")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
