#!/usr/bin/env bash
# Build the villa spiral-fitting stack inside WSL2 Ubuntu (tested on 26.04).
# Prerequisite (run once, needs sudo):
#   sudo apt-get install -y build-essential cmake ninja-build pkg-config git curl \
#     libsqlite3-dev libssl-dev nlohmann-json3-dev libboost-all-dev libopencv-dev \
#     libconcurrentqueue-dev libcurl4-openssl-dev libjpeg-dev libpng-dev libtiff-dev \
#     zlib1g-dev libblosc-dev libblosc2-dev liblz4-dev libzstd-dev python3-dev
# Usage: bash wsl_setup.sh [villa_commit]
set -euo pipefail

VILLA_DIR="${VILLA_DIR:-$HOME/vc/villa}"
COMMIT="${1:-0bddf2757}"
NPROC="$(nproc)"

# 1. uv (user-level install, no sudo)
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

# 2. villa sparse checkout on the Linux filesystem (much faster than /mnt/*)
if [ ! -d "$VILLA_DIR/.git" ]; then
  mkdir -p "$(dirname "$VILLA_DIR")"
  git clone --filter=blob:none --sparse https://github.com/ScrollPrize/villa.git "$VILLA_DIR"
fi
cd "$VILLA_DIR"
git sparse-checkout set spiral-fitting vc_object_store vc_reader
git fetch --quiet origin
git checkout --quiet "$COMMIT"
echo "villa at $(git rev-parse --short HEAD)"

# 3. Native object-store tools (vc_object_ingest, vc_object_server, vc_object_convert)
cmake -S vc_object_store -B build/vc_object_store -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DVC_OBJECT_STORE_BUILD_ADAPTERS=ON \
  -DVC_OBJECT_STORE_BUILD_LINKING=ON -DVC_OBJECT_STORE_BUILD_PYTHON=OFF
# (Python bindings are built by `uv sync` below, in its own build env with nanobind.)
cmake --build build/vc_object_store -j "$NPROC"
ls build/vc_object_store | grep -E '^vc_object_' || true

# 4. Spiral Python environment (Python 3.14, torch cu128, native helpers)
cd spiral-fitting
export UV_LINK_MODE=copy UV_HTTP_TIMEOUT=300
uv sync
.venv/bin/python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("arch list", torch.cuda.get_arch_list())
import vc_spiral.spiral_sampling, vc_spiral.track_crossings, vc_spiral.track_store  # noqa
import vc_object_store  # noqa
print("native modules import OK")
EOF
echo "SETUP DONE"
