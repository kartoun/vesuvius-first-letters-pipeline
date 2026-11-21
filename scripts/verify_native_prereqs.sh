#!/usr/bin/env bash
# Clean-container check of the native prerequisites for spiral-fitting's local
# packages (vc_reader, vc_object_store) and the vc_object_store tools.
# Runs inside a fresh ubuntu:26.04 container with a villa checkout mounted at /villa (read-only).
# Usage (host): docker run --rm -v ~/vc/villa:/villa:ro ubuntu:26.04 bash /villa-check/verify.sh MODE
#   MODE=before : toolchain only (what the spiral-fitting README asked for: nothing listed)
#   MODE=after  : the package list proposed for spiral-fitting/README.md
set -uo pipefail
MODE="${1:-after}"
export DEBIAN_FRONTEND=noninteractive
TOOLCHAIN="build-essential cmake ninja-build pkg-config git curl ca-certificates"
LIBS="libboost-dev libconcurrentqueue-dev nlohmann-json3-dev libsqlite3-dev \
  libssl-dev libcurl4-openssl-dev zlib1g-dev libblosc-dev libblosc2-dev \
  liblz4-dev libzstd-dev libopencv-dev libtiff-dev libpng-dev libjpeg-dev"
apt-get update -q >/dev/null
if [ "$MODE" = before ]; then PKGS="$TOOLCHAIN"; else PKGS="$TOOLCHAIN $LIBS"; fi
apt-get install -y -q --no-install-recommends $PKGS >/dev/null 2>&1 || { echo "apt failed"; exit 2; }
echo "== MODE=$MODE  ubuntu $(. /etc/os-release; echo $VERSION_ID)  packages: $PKGS"
cp -r /villa /tmp/villa
cd /tmp/villa
echo "== villa $(git -c safe.directory='*' rev-parse --short HEAD 2>/dev/null)"

echo "== [1] native tools: cmake configure + build vc_object_store (adapters, linking, tools)"
if cmake -S vc_object_store -B /tmp/b -G Ninja -DCMAKE_BUILD_TYPE=Release \
     -DVC_OBJECT_STORE_BUILD_ADAPTERS=ON -DVC_OBJECT_STORE_BUILD_LINKING=ON \
     -DVC_OBJECT_STORE_BUILD_TESTS=OFF > /tmp/cfg.log 2>&1; then
  cmake --build /tmp/b -j "$(nproc)" > /tmp/build.log 2>&1 && ls /tmp/b/bin | sed 's/^/   built: /' \
    || { echo "   BUILD FAILED"; tail -20 /tmp/build.log; }
else
  echo "   CONFIGURE FAILED:"; grep -A8 "CMake Error" /tmp/cfg.log | head -20
fi

echo "== [2] Python packages: uv-built vc_reader and vc_object_store (what uv sync compiles)"
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH="$HOME/.local/bin:$PATH" UV_LINK_MODE=copy
uv venv -q --python 3.14 /tmp/venv >/dev/null 2>&1
if VIRTUAL_ENV=/tmp/venv uv pip install -q ./vc_reader ./vc_object_store > /tmp/uv.log 2>&1; then
  /tmp/venv/bin/python -c "import vc_reader, vc_object_store; print('   import OK: vc_reader, vc_object_store')"
else
  echo "   UV BUILD FAILED:"; grep -E "CMake Error|Could not find|not found|error:" /tmp/uv.log | head -12
fi
