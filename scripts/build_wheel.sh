#!/usr/bin/env bash
# Build the simai wheel locally, mirroring the GH Actions pipeline.
#
# Usage:
#   ./scripts/build_wheel.sh            # build binaries if missing, then build wheel
#   ./scripts/build_wheel.sh --no-bin   # skip binary build (binaries must exist)
#   ./scripts/build_wheel.sh --docker   # force use of manylinux docker container
#
# Binaries are expected / placed at build/bin/:
#   SimAI_analytical
#   SimAI_simulator  (ns3 binary)
#   libns3*.so       (ns3 shared libs)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BIN_DIR="build/bin"
ANALYTICAL_BIN="$BIN_DIR/SimAI_analytical"
SIMULATOR_BIN="$BIN_DIR/SimAI_simulator"
VENDOR="vendor/simai/astra-sim-alibabacloud"
NS3_SRC="vendor/simai/ns-3-alibabacloud"

SKIP_BIN=false
FORCE_DOCKER=false
FORCE_NATIVE=false
for arg in "$@"; do
  case "$arg" in
    --no-bin)   SKIP_BIN=true ;;
    --docker)   FORCE_DOCKER=true ;;
    --native)   FORCE_NATIVE=true ;;
  esac
done

# ── helpers ────────────────────────────────────────────────────────────────────

log()  { echo "▶ $*"; }
die()  { echo "✗ $*" >&2; exit 1; }
has()  { command -v "$1" &>/dev/null; }

# ── binary build ───────────────────────────────────────────────────────────────

build_binaries_docker() {
  local image="quay.io/pypa/manylinux2014_x86_64"
  log "Building binaries in manylinux container ($image)..."

  docker run --rm -v "$REPO_ROOT:/workspace" -w /workspace "$image" bash -c '
    set -e
    yum install -y gcc gcc-c++ cmake3 make libxml2-devel sqlite-devel gsl-devel
    ln -sf /usr/bin/cmake3 /usr/bin/cmake || true

    chmod +x scripts/patch_paths.sh
    ./scripts/patch_paths.sh vendor/simai/astra-sim-alibabacloud

    # ── analytical ──────────────────────────────────────────────────────────
    cd /workspace/vendor/simai/astra-sim-alibabacloud/build/simai_analytical
    mkdir -p build && cd build
    cmake -DUSE_ANALYTICAL=TRUE ..
    make -j$(nproc)
    mkdir -p /workspace/build/bin
    cp simai_analytical/SimAI_analytical /workspace/build/bin/SimAI_analytical
    strip /workspace/build/bin/SimAI_analytical 2>/dev/null || true

    # ── ns3 ─────────────────────────────────────────────────────────────────
    cd /workspace

    # Copy ns3 source into astra-sim tree
    mkdir -p vendor/simai/astra-sim-alibabacloud/extern/network_backend/ns3-interface
    cp -r vendor/simai/ns-3-alibabacloud/* \
      vendor/simai/astra-sim-alibabacloud/extern/network_backend/ns3-interface/

    # Build AstraSim library
    cd vendor/simai/astra-sim-alibabacloud/build/astra_ns3
    mkdir -p build && cd build
    cmake ..
    make -j$(nproc)

    # Build ns3
    cd /workspace/vendor/simai/astra-sim-alibabacloud
    cp astra-sim/network_frontend/ns3/AstraSimNetwork.cc \
       extern/network_backend/ns3-interface/simulation/scratch/
    cp astra-sim/network_frontend/ns3/*.h \
       extern/network_backend/ns3-interface/simulation/scratch/
    rm -rf extern/network_backend/ns3-interface/simulation/src/applications/astra-sim
    cp -r astra-sim extern/network_backend/ns3-interface/simulation/src/applications/
    cd extern/network_backend/ns3-interface/simulation
    ./ns3 configure -d debug --enable-mtp -- -DCMAKE_CXX_FLAGS="-O0 -g"
    ./ns3 build
    strip build/scratch/ns3.36.1-AstraSimNetwork-debug 2>/dev/null || true
    strip build/lib/libns3*-debug.so 2>/dev/null || true

    NS3_BIN=build/scratch/ns3.36.1-AstraSimNetwork-debug
    NS3_LIBS=build/lib
    mkdir -p /workspace/build/bin
    cp "$NS3_BIN" /workspace/build/bin/SimAI_simulator
    cp "$NS3_LIBS"/libns3*-debug.so /workspace/build/bin/ 2>/dev/null || true

    # Bundle non-standard shared libs
    for lib in $(ldd "$NS3_BIN" 2>/dev/null | grep -oP "/\S+\.so[.\d]*" | sort -u); do
      case "$(basename "$lib")" in
        libc.so*|libm.so*|libpthread.so*|libdl.so*|librt.so*|ld-linux*|libstdc++*|libgcc_s*) continue ;;
      esac
      cp -nL "$lib" /workspace/build/bin/ 2>/dev/null || true
    done
    for so in /workspace/build/bin/libns3*; do
      for lib in $(ldd "$so" 2>/dev/null | grep -oP "/\S+\.so[.\d]*" | sort -u); do
        case "$(basename "$lib")" in
          libc.so*|libm.so*|libpthread.so*|libdl.so*|librt.so*|ld-linux*|libstdc++*|libgcc_s*) continue ;;
        esac
        cp -nL "$lib" /workspace/build/bin/ 2>/dev/null || true
      done
    done
  '
  chmod +x build/bin/*
}

build_binaries_native() {
  log "Building binaries natively..."

  has cmake  || die "cmake not found. Install it or use docker (run with --docker)."
  has make   || die "make not found."
  has g++    || die "g++ not found."

  chmod +x scripts/patch_paths.sh
  ./scripts/patch_paths.sh "$VENDOR"

  # ── analytical ────────────────────────────────────────────────────────────
  log "Building SimAI_analytical..."
  pushd "$VENDOR/build/simai_analytical" > /dev/null
    mkdir -p build && cd build
    cmake -DUSE_ANALYTICAL=TRUE ..
    make -j"$(nproc)"
  popd > /dev/null
  mkdir -p "$BIN_DIR"
  cp "$VENDOR/build/simai_analytical/build/simai_analytical/SimAI_analytical" \
     "$BIN_DIR/SimAI_analytical"

  # ── ns3 ──────────────────────────────────────────────────────────────────
  log "Building SimAI_simulator (ns3)..."
  mkdir -p "$VENDOR/extern/network_backend/ns3-interface"
  cp -r "$NS3_SRC"/* "$VENDOR/extern/network_backend/ns3-interface/"

  pushd "$VENDOR/build/astra_ns3" > /dev/null
    mkdir -p build && cd build
    cmake ..
    make -j"$(nproc)"
  popd > /dev/null

  cp "$VENDOR/astra-sim/network_frontend/ns3/AstraSimNetwork.cc" \
     "$VENDOR/extern/network_backend/ns3-interface/simulation/scratch/"
  cp "$VENDOR"/astra-sim/network_frontend/ns3/*.h \
     "$VENDOR/extern/network_backend/ns3-interface/simulation/scratch/"
  rm -rf "$VENDOR/extern/network_backend/ns3-interface/simulation/src/applications/astra-sim"
  cp -r "$VENDOR/astra-sim" \
     "$VENDOR/extern/network_backend/ns3-interface/simulation/src/applications/"

  pushd "$VENDOR/extern/network_backend/ns3-interface/simulation" > /dev/null
    ./ns3 configure -d debug --enable-mtp -- -DCMAKE_CXX_FLAGS="-O0 -g"
    ./ns3 build
    NS3_BIN="build/scratch/ns3.36.1-AstraSimNetwork-debug"
    NS3_LIBS="build/lib"
    cp "$NS3_BIN" "$REPO_ROOT/$BIN_DIR/SimAI_simulator"
    cp "$NS3_LIBS"/libns3*-debug.so "$REPO_ROOT/$BIN_DIR/" 2>/dev/null || true
  popd > /dev/null

  chmod +x "$BIN_DIR"/*
  log "Binaries built and placed in $BIN_DIR/"
}

build_binaries() {
  if $FORCE_NATIVE; then
    log "Native build forced via --native flag."
    build_binaries_native
  elif $FORCE_DOCKER || has docker; then
    $FORCE_DOCKER && log "Docker forced via --docker flag."
    build_binaries_docker
  else
    build_binaries_native
  fi
}

# ── main ──────────────────────────────────────────────────────────────────────

if ! $SKIP_BIN; then
  if [[ -f "$ANALYTICAL_BIN" && -f "$SIMULATOR_BIN" ]]; then
    log "Binaries already present in $BIN_DIR/, skipping build."
  else
    [[ -f "$ANALYTICAL_BIN" ]] || log "Missing: $ANALYTICAL_BIN"
    [[ -f "$SIMULATOR_BIN"  ]] || log "Missing: $SIMULATOR_BIN"
    build_binaries
  fi
fi

# ── wheel ─────────────────────────────────────────────────────────────────────

log "Building wheel..."
uv build --wheel
log "Done. Wheel is in dist/"
