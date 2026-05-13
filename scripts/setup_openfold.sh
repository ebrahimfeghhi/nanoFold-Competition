#!/usr/bin/env bash
# Build the OpenFold CUDA kernel (attn_core_inplace_cuda) required by the
# openfold_unlimited submission. Run this once after cloning on any new machine.
#
# Prerequisites: CUDA toolkit, a matching PyTorch install, and the submodules
# already initialised (git submodule update --init --recursive).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENFOLD_ROOT="$REPO_ROOT/third_party/openfold"

if [ ! -d "$OPENFOLD_ROOT/openfold" ]; then
    echo "OpenFold submodule not found. Run: git submodule update --init --recursive"
    exit 1
fi

echo "Building OpenFold CUDA kernel in $OPENFOLD_ROOT ..."
cd "$OPENFOLD_ROOT"
python setup.py build_ext --inplace
echo "Done. attn_core_inplace_cuda built successfully."
