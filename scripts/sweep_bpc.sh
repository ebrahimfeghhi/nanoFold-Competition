#!/usr/bin/env bash
# Sweep blocks_per_ckpt for the OpenFold unlimited submission.
# Runs a short training session per value and reports per-step throughput,
# total wall time, and peak GPU memory.
#
# Usage:
#   bash scripts/sweep_bpc.sh
#   STEPS=50 NUM_GPUS=2 bash scripts/sweep_bpc.sh
#   BASE_CONFIG=submissions/openfold_unlimited/config.yaml bash scripts/sweep_bpc.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# --- Tunables --------------------------------------------------------------
NUM_GPUS=${NUM_GPUS:-4}
STEPS=${STEPS:-100}
WARMUP=${WARMUP:-20}
BASE_CONFIG=${BASE_CONFIG:-submissions/openfold_unlimited/config_ckpt_exp.yaml}
VALUES=("1" "2" "4")
# ---------------------------------------------------------------------------

if [ ! -f "$BASE_CONFIG" ]; then
  echo "Base config not found: $BASE_CONFIG" >&2
  exit 1
fi

# GPU sanity check: an empty CUDA_VISIBLE_DEVICES means "no GPUs visible".
# Unsetting it lets the runtime see all attached GPUs; otherwise honor the user's pick.
if [ "${CUDA_VISIBLE_DEVICES+set}" = "set" ] && [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  echo "ERROR: CUDA_VISIBLE_DEVICES is set to empty (hides all GPUs)." >&2
  echo "       Run:   unset CUDA_VISIBLE_DEVICES" >&2
  echo "       Or:    export CUDA_VISIBLE_DEVICES=2,3,4,5   # pick free GPUs" >&2
  exit 1
fi

VISIBLE_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$VISIBLE_GPUS" -lt "$NUM_GPUS" ]; then
  echo "ERROR: NUM_GPUS=$NUM_GPUS but only $VISIBLE_GPUS GPU(s) visible." >&2
  echo "       Either lower NUM_GPUS or set CUDA_VISIBLE_DEVICES to a list of $NUM_GPUS device ids." >&2
  exit 1
fi

SWEEP_DIR="runs/bpc_sweep"
mkdir -p "$SWEEP_DIR"

echo "Sweep config:"
echo "  BASE_CONFIG = $BASE_CONFIG"
echo "  STEPS       = $STEPS  (WARMUP=$WARMUP)"
echo "  NUM_GPUS    = $NUM_GPUS"
echo "  VALUES      = ${VALUES[*]}"
echo "  output      = $SWEEP_DIR/  +  runs/bpc_<value>/"
echo

for bpc in "${VALUES[@]}"; do
  RUN_NAME="bpc_${bpc}"
  CONFIG_OUT="$SWEEP_DIR/config_${bpc}.yaml"

  # Generate per-run config: override max_steps, push save/eval_every past
  # STEPS, and convert `submission.path` -> `submission.module` because the
  # temp config no longer sits next to submission.py (path would be resolved
  # relative to the config's parent dir and fail).
  python - <<PYEOF
import copy, pathlib, yaml
base_path = pathlib.Path("$BASE_CONFIG").resolve()
base = yaml.safe_load(base_path.read_text())
cfg = copy.deepcopy(base)
cfg["run_name"] = "$RUN_NAME"
tcfg = cfg.setdefault("train", {})
tcfg["max_steps"]  = $STEPS
tcfg["log_every"]  = 5
tcfg["save_every"] = $STEPS + 1
tcfg["eval_every"] = $STEPS + 1
mcfg = cfg.setdefault("model", {})
mcfg["blocks_per_ckpt"] = int("$bpc")

# Resolve path-based submission -> module-based, so allowed_root checks pass
# regardless of where the temp config lives.
sub = cfg.get("submission")
if isinstance(sub, dict) and "path" in sub and "module" not in sub:
    sub_path = pathlib.Path(sub["path"])
    if not sub_path.is_absolute():
        sub_path = (base_path.parent / sub_path).resolve()
    repo_root = pathlib.Path(".").resolve()
    rel = sub_path.relative_to(repo_root).with_suffix("")
    cfg["submission"] = {"module": ".".join(rel.parts)}
pathlib.Path("$CONFIG_OUT").write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"[gen] {pathlib.Path('$CONFIG_OUT').name}  blocks_per_ckpt={mcfg['blocks_per_ckpt']}  submission={cfg.get('submission')}")
PYEOF

  # Clear any stale run dir so we average only this run's steps.
  rm -rf "runs/$RUN_NAME"

  echo
  echo "============================================================"
  echo "Run $RUN_NAME (blocks_per_ckpt=$bpc, steps=$STEPS, gpus=$NUM_GPUS)"
  echo "Base config: $BASE_CONFIG"
  echo "============================================================"

  # Background nvidia-smi poller for peak memory.
  MEM_LOG="$SWEEP_DIR/mem_${bpc}.csv"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -lms 500 \
      > "$MEM_LOG" 2>/dev/null &
  NVSMI_PID=$!

  # Measure wall-clock duration of the torchrun invocation.
  RUN_START=$(date +%s.%N)
  set +e
  torchrun --standalone --nproc_per_node=$NUM_GPUS train_ddp.py \
    --config "$CONFIG_OUT" \
    --track unlimited \
    2>&1 | tee "$SWEEP_DIR/log_${bpc}.txt"
  STATUS=${PIPESTATUS[0]}
  set -e
  RUN_END=$(date +%s.%N)

  kill "$NVSMI_PID" 2>/dev/null || true
  wait "$NVSMI_PID" 2>/dev/null || true

  WALL=$(python -c "print(f'{$RUN_END - $RUN_START:.2f}')")
  if [ "$STATUS" -ne 0 ]; then
    echo "[run] $RUN_NAME exited with status $STATUS after ${WALL}s. See $SWEEP_DIR/log_${bpc}.txt for details."
    echo "FAILED ($STATUS) wall=$WALL" > "$SWEEP_DIR/status_${bpc}.txt"
  else
    echo "[run] $RUN_NAME OK, wall=${WALL}s"
    echo "OK wall=$WALL" > "$SWEEP_DIR/status_${bpc}.txt"
  fi
done

echo
echo "============================================================"
echo "Summary"
echo "============================================================"
python - <<PYEOF
import json, pathlib, statistics
SWEEP   = pathlib.Path("$SWEEP_DIR")
WARMUP  = $WARMUP
STEPS   = $STEPS
VALUES  = "${VALUES[@]}".split()

print(f"{'bpc':>4} {'status':>10} {'wall_s':>8} {'step_s':>8} {'samp/s':>8} {'peak_MiB':>10}")
print("-" * 56)
for bpc in VALUES:
    status_path = SWEEP / f"status_{bpc}.txt"
    if status_path.exists():
        raw = status_path.read_text().strip().split()
        status = raw[0] if raw else "?"
        wall = next((tok.split("=", 1)[1] for tok in raw if tok.startswith("wall=")), "-")
    else:
        status, wall = "?", "-"

    step_s = samp_s = "-"
    mpath = pathlib.Path(f"runs/bpc_{bpc}/train_metrics.jsonl")
    if status == "OK" and mpath.exists():
        rows = [json.loads(line) for line in mpath.read_text().splitlines() if line.strip()]
        warm = [r for r in rows if int(r.get("step", 0)) > WARMUP]
        if warm:
            step_s = f"{statistics.mean(r['step_seconds']    for r in warm):.3f}"
            samp_s = f"{statistics.mean(r['samples_per_sec'] for r in warm):.2f}"

    peak_mb = "-"
    mem_csv = SWEEP / f"mem_{bpc}.csv"
    if mem_csv.exists():
        peaks = {}
        for line in mem_csv.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                gid, mb = (int(x.strip()) for x in line.split(","))
                peaks[gid] = max(peaks.get(gid, 0), mb)
            except ValueError:
                continue
        if peaks:
            peak_mb = str(max(peaks.values()))

    print(f"{bpc:>4} {status:>10} {wall:>8} {step_s:>8} {samp_s:>8} {peak_mb:>10}")
PYEOF
