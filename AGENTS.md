# SimAI Agent Reference

This document contains everything an agent needs to work on the SimAI codebase without
additional exploration. Keep it up to date when making structural changes.

**Version**: 0.3.12 | **Python**: ≥3.11 (3.13 used locally) | **Package manager**: `uv`

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Layout](#repository-layout)
3. [Development Setup](#development-setup)
4. [Source Code Architecture](#source-code-architecture)
5. [Vendor Components](#vendor-components)
6. [Build System](#build-system)
7. [Testing](#testing)
8. [Key Patterns & Conventions](#key-patterns--conventions)
9. [File Formats](#file-formats)
10. [Environment Variables](#environment-variables)
11. [CI/CD](#cicd)

---

## Project Overview

SimAI is a Python wrapper and CLI for the SimAI datacenter network simulator, optimized for
AI training workload analysis. It provides:

- **CLI**: `simai generate workload/topology`, `simai profile gpu`, `simai simulate analytical/ns3`
- **Workload generation**: From ML framework configs (Megatron, DeepSpeed, DeepSeek)
- **Topology generation**: For Spectrum-X, DCN+, AlibabaHPN datacenter architectures
- **Three simulation backends**: Analytical (fast, bandwidth-based), NS-3 (detailed, packet-level), and M4 (flow-level, ML-based gray failure)
- **GPU profiling**: Measure actual kernel execution times for realistic simulations

The wrapper abstracts complex manual setup from upstream SimAI (hardcoded paths, directory
structures, config patching, binary discovery) and bundles pre-built binaries in PyPI wheels.

---

## Repository Layout

```
simai/
├── src/simai/              # Python package source
│   ├── cli/                # Typer CLI commands (app.py, generate.py, profile.py, simulate.py)
│   ├── backends/           # Simulation backends (binary.py, analytical.py, ns3.py)
│   ├── topology/           # Topology generation (generator.py)
│   └── workflow/           # Workload generation and GPU profiling (generator.py, profiler.py)
├── vendor/
│   ├── simai/              # Git submodule → https://github.com/aliyun/SimAI.git
│   │   ├── aicb/           # AICB workload generator
│   │   ├── astra-sim-alibabacloud/  # C++ discrete event simulator
│   │   │   ├── astra-sim/  # Core ASTRA-SIM engine (system/, workload/, network_frontend/)
│   │   │   ├── build/      # CMake build configs (simai_analytical/, astra_ns3/, simai_phy/)
│   │   │   └── inputs/     # Config files, topology templates, ratio CSVs
│   │   ├── ns-3-alibabacloud/  # NS-3 with HPC extensions (MTP, RDMA, PFC, DCQCN)
│   │   ├── vidur-alibabacloud/ # LLM inference simulator (not used by Python wrapper)
│   │   └── scripts/        # Upstream build scripts
│   └── simai-m4/           # Local (untracked) copy of upstream SimAI with m4 integration
│                           # Used for testing m4 (gray failure) integration. NOT a submodule.
├── scripts/
│   ├── build_wheel.sh      # Local wheel build script (mirrors CI): builds binaries + wheel
│   ├── patch_paths.sh      # Patch hardcoded C++ paths (/etc/astra-sim/, /root/astra-sim/)
│   └── restore_paths.sh    # Restore original vendor files from backups
├── tests/
│   ├── test_profile_integration.sh  # GPU profiling integration tests
│   └── run_profile_tests.slurm     # SLURM job for GPU cluster testing
├── .github/workflows/build.yml  # CI/CD pipeline
├── hatch_build.py          # Custom Hatch build hook (vendors code + binaries at build time)
├── pyproject.toml          # Project metadata, dependencies, entry points
└── uv.lock                 # Locked dependency file
```

**Gitignored at runtime** (populated during `hatch build`):
- `src/simai/_vendor/` - Vendored AICB code, topology generator, ratio CSVs
- `src/simai/_binaries/` - Pre-built `SimAI_analytical`, `SimAI_simulator` binaries

---

## Development Setup

```bash
# Clone with submodules
git clone --recurse-submodules <repo-url>
cd simai

# Install in editable mode (uses uv)
uv pip install -e .

# For GPU profiling support
uv pip install -e ".[profiling]"

# Run the CLI
simai --help
```

For editable installs, resource discovery falls back to the `vendor/simai/` submodule
(no vendored copy needed). Set `SIMAI_PATH` or `SIMAI_BIN_PATH` to override.

---

## Source Code Architecture

### CLI Layer (`src/simai/cli/`)

**`app.py`**: Main Typer app with three top-level commands (`generate`/`gen`, `profile`, `simulate`).

**`generate.py`**:
- `workload()`: Generate training workload `.txt` files. Parameters: framework, num_gpus,
  tensor/pipeline/expert parallel, model architecture (layers, hidden size, heads, vocab),
  batch sizes, MoE config, compute profile.
- `topology()` / `topo()` (hidden alias): Generate network topology directories.
  Parameters: type (Spectrum-X, DCN+, AlibabaHPN), GPUs, bandwidth, latency, dual-ToR, dual-plane.

**`profile.py`**:
- `gpu()`: Profile GPU kernels on real hardware. Requires PyTorch + CUDA.

**`install.py`**:
- `m4()`: Compile and install the `SimAI_m4` binary from source.
  Source discovery order: (1) editable-install `vendor/simai-m4/`, (2) cached clone at
  `~/.cache/simai/simai-m4/`, (3) auto-clone from `_M4_GIT_URL` on first run.
  Accepts `--src` to override source path and `--git-url` to override the clone URL.
  `--force` reinstalls even if the binary already exists.
  `--n-flows-max N` (default `_N_FLOWS_MAX = 500_000`) patches `M4::n_flows_max` in
  `M4.cc` before compilation via `_patch_n_flows_max()`. The upstream default (50 000)
  is too low for large workloads and causes an out-of-range tensor index crash.
  Places binary in `simai/_binaries/` next to the package so `find_binary()` picks it up.
  Requires CUDA torch `>=2.4,<2.7` (or `LIBTORCH_DIR` set) and cmake/make/gcc.

**`simulate.py`**:
- `analytical()`: Fast bandwidth-based simulation. Accepts workload + topology dir.
  Overlap parameters: `--dp-overlap`, `--tp-overlap`, `--ep-overlap`, `--pp-overlap`.
- `ns3()`: Detailed packet-level simulation. Parameters: threads (default 8),
  send_latency, NVLS/PXN flags.

### Workflow Layer (`src/simai/workflow/`)

**`generator.py`** - `generate_workload()`:
- Locates AICB via `_find_aicb_root()` (3-tier: vendored → `SIMAI_PATH` env → sibling dir heuristic)
- Uses `@contextmanager` `_aicb_on_path()` for temporary `sys.path` injection
- Injects `argparse.Namespace` into AICB module globals (not modifying AICB source)
- Outputs `.txt` workload file

**`profiler.py`** - `profile_gpu_kernels()`:
- `_patch_optional_cuda_modules()` (lines 24-73): Creates fake modules for apex,
  `scaled_upper_triang_masked_softmax_cuda`, `deep_gemm` so AICB imports succeed without CUDA extensions
- `_create_model_args()` (lines 76-211): Builds AICB `argparse.Namespace`, derives dp_num,
  ffn_hidden_size, padded_vocab_size, validates config
- `_create_model()` (lines 214-243): Instantiates `MegatronModel` or `DeepSeekV3Model`
- `profile_gpu_kernels()` (lines 246-413): Checks torch + CUDA, profiles one training iteration

### Topology Layer (`src/simai/topology/`)

**`generator.py`** - `generate_topology()`:
- Locates `gen_Topo_Template.py` via `_find_topo_root()` (3-tier: vendored → `SIMAI_PATH` → vendor submodule)
- Builds mock `argparse.Namespace` matching upstream script's expectations
- Runs in a temp directory to isolate side effects
- Outputs: `topology` file + `metadata.json`
- Supported types: Spectrum-X (NVIDIA rail-optimized), DCN+ (traditional), AlibabaHPN (multi-plane)

### Backend Layer (`src/simai/backends/`)

**`binary.py`**:
- `find_binary(name)`: Search order: bundled `simai/_binaries/` → `SIMAI_BIN_PATH` env → system `PATH`
- `run_binary(name, args, cwd, env, verbose)`: Sets `LD_LIBRARY_PATH` to binary dir for shared libs,
  runs via `subprocess.run()`

**`analytical.py`** - `run_analytical()`:
- `_find_simai_root()`: Locates ratio CSVs and `SimAI.conf` (4-tier search)
- Symlinks ratio CSVs into temp dir, runs binary from there, moves results to output path

**`ns3.py`** - `run_ns3()`:
- `_find_default_config()`: Locates `SimAI.conf`
- Patches config at runtime to replace `/etc/astra-sim/simulation/` with relative paths
- Creates dummy `flow1.txt`, `trace1.txt` inputs
- Sets env vars for log level, NVLS, PXN flags

**`m4.py`** - `run_m4()`:
- `_find_m4_models()`: Locates bundled `.pt` model files (3-tier: `_vendor/m4_models/` → editable vendor path → `SIMAI_PATH`)
- `_find_libtorch_lib_dir()`: Returns the torch lib dir for setting `LD_LIBRARY_PATH` at runtime
- `_convert_topology()`: Converts topology file to m4 format (bandwidth → XGbps, latency → Xms). Handles both raw numeric values and pre-formatted unit strings (e.g. `7200Gbps`, `0.000025ms`)
- Symlinks model files to the path the binary hardcodes relative to cwd, runs in a temp dir

---

## Vendor Components

### ASTRA-SIM (`vendor/simai/astra-sim-alibabacloud/`)

C++ discrete event simulator:
- **`astra-sim/system/Sys.hh`**: Main orchestrator managing NPU/GPU nodes
- **`astra-sim/system/collective/`**: Ring, DoubleBinaryTree, AllToAll, NcclTreeFlow algorithms
- **`astra-sim/system/MockNccl*.h`**: NCCL behavior modeling (NVLS, Ring, Tree),
  per-GPU-generation tuning (Volta, Ampere, Hopper)
- **`astra-sim/workload/Workload.hh`**: Workload file parser (TP, DP, PP, EP parallelism)
- Network frontends: Analytical, NS-3, Physical (real RDMA)

### NS-3 (`vendor/simai/ns-3-alibabacloud/`)

NS-3 with HPC extensions:
- **MTP module** (`simulation/src/mtp/`): Multi-threaded packet simulation for large-scale (1000+ GPUs)
- Extensions: RDMA, PFC (Priority Flow Control), DCQCN (quantized congestion notification)

### AICB (`vendor/simai/aicb/`)

AI Communication Benchmark for workload generation:
- `workload_generator/mocked_model/MockedMegatron.py`: Standard transformers (TP/PP/DP/SP)
- `workload_generator/mocked_model/MockedDeepSeek.py`: MoE models with expert parallelism
- `workload_generator/mocked_model/MockedDeepspeed.py`: DeepSpeed ZeRO stages 1/2/3
- Traces forward/backward passes to extract collective communication patterns

### `vendor/simai-m4/` (untracked, local only)

A local copy of the upstream SimAI repo containing m4 integration for gray failure simulation.
This is **not** a git submodule and not tracked. Contains:
- `gray_failure_*.py` scripts for gray failure sweep/plotting
- `bin/` with pre-built binaries
- `SimCCL/` directory

---

## Build System

### Hatch Build Hook (`hatch_build.py`)

Runs automatically during `hatch build` / `uv build`:

1. **`initialize()`** (before build):
   - Copies `vendor/simai/aicb/` → `src/simai/_vendor/aicb/`
   - Copies `gen_Topo_Template.py` → `src/simai/_vendor/topo/`
   - Copies ratio CSVs → `src/simai/_vendor/astra-sim-alibabacloud/inputs/ratio/`
   - Copies `SimAI.conf` → `src/simai/_vendor/`
   - Copies pre-built binaries from `build/bin/` → `src/simai/_binaries/`
   - Sets executable bit on binaries
   - Sets wheel platform tag from `SIMAI_PLATFORM_TAG` env var

2. **`finalize()`** (after build):
   - Deletes `src/simai/_vendor/` and `src/simai/_binaries/` (not tracked in git)

### Local Wheel Build (`scripts/build_wheel.sh`)

Mirrors the CI pipeline for analytical+ns3 binaries. Builds missing binaries then calls `uv build --wheel`.

```bash
# Full build: binaries (if missing) + wheel
./scripts/build_wheel.sh

# Python-only change — skip binary compilation
./scripts/build_wheel.sh --no-bin

# Force manylinux Docker (identical to CI environment)
./scripts/build_wheel.sh --docker

# Rebuild binaries from scratch
rm -rf build/bin && ./scripts/build_wheel.sh --docker
```

Binary detection: if `build/bin/SimAI_analytical` and `build/bin/SimAI_simulator` both exist,
the binary build is skipped automatically (no flag needed).

Build tool priority: Docker (`quay.io/pypa/manylinux2014_x86_64`) if available, then native
`cmake`/`make`.

**Note**: `SimAI_m4` is NOT built by this script. It is compiled automatically by
`hatch_build.py` at install time when `vendor/simai-m4/` is present (see below).

### Binary Compilation (upstream scripts)

```bash
# Analytical backend (in vendor/simai/scripts/)
./scripts/build.sh -c analytical  # → bin/SimAI_analytical

# NS-3 backend
./scripts/build.sh -c ns3         # → bin/SimAI_simulator (+ libns3*.so)

# Physical backend
./scripts/build.sh -c phy         # → bin/SimAI_phynet
```

### Path Patching

Upstream SimAI hardcodes `/etc/astra-sim/` and `/root/astra-sim/`. CI applies patches:
```bash
./scripts/patch_paths.sh    # Patches C++ source for runtime config
./scripts/restore_paths.sh  # Restores originals
```

### Wheel Building

```bash
SIMAI_PLATFORM_TAG=manylinux_2_17_x86_64 uv build --wheel
```

PyPI enforces <100MB limit; CI checks this.

---

## Testing

Tests require GPU access via SLURM (HPC cluster):

```bash
# Integration test for GPU profiling
./tests/test_profile_integration.sh

# Submit SLURM job
sbatch tests/run_profile_tests.slurm
```

There are no unit tests currently - only end-to-end profiling integration tests.

---

## Key Patterns & Conventions

### 1. 3-Tier Resource Discovery

All components (AICB, topo generator, binaries, data files) search in order:
1. **Vendored** in wheel: `src/simai/_vendor/` or `src/simai/_binaries/`
2. **Environment variable**: `SIMAI_PATH`, `SIMAI_BIN_PATH`
3. **Fallback**: Relative paths for editable installs / vendor submodule

This supports: PyPI wheel installs, editable (`-e`) installs, and custom deployments.

### 2. Temporary Directory Isolation

Both backends run from isolated temp directories to capture binary side effects and support
parallel simulations. Results are moved to user-specified output paths after completion.

### 3. `sys.path` Context Managers

```python
@contextmanager
def _aicb_on_path():
    sys.path.insert(0, str(aicb_root))
    try:
        yield
    finally:
        sys.path.remove(str(aicb_root))
```

Used by `workflow/generator.py` and `topology/generator.py` to temporarily add vendored
code to the import path without polluting it permanently.

### 4. `argparse.Namespace` Injection

AICB expects a free `args` variable in its module scope. The wrapper creates an
`argparse.Namespace` and injects it into the module's globals instead of modifying AICB source.

### 5. Optional CUDA Module Stubs

`_patch_optional_cuda_modules()` registers fake modules in `sys.modules` for apex,
`scaled_upper_triang_masked_softmax_cuda`, `deep_gemm`. This allows AICB to import
successfully even without optional CUDA extensions installed.

### 6. CLI Naming Conventions

- Long form: `--tensor-parallel`, short: `--tp`
- Python variable: `tensor_parallel` (underscores)
- Typer: `typer.Option("--tensor-parallel", "--tp")`

### 7. Output Path Flexibility

Functions accept both file paths (`.txt` extension) and directory paths:
- File path: Moves primary result to that file, siblings to parent directory
- Directory path: Creates directory, moves all results inside
- Auto-generation: `results/<type>/` with descriptive filenames

---

## File Formats

### Workload File (`.txt`)

```
HYBRID_TRANSFORMER_FWD_IN_BCKWD model_parallel_NPU_group: <TP> ep: <EP> pp: <PP> vpp: <VPP> ga: <GA> all_gpus: <TOTAL> checkpoints: <CKPT> checkpoint_initiates: <CKPT_INIT> pp_comm <SIZE>
<num_layers>
<layer_name> <layer_id> <fwd_compute_us> <fwd_comm_type> <fwd_comm_size_bytes> <ig_compute_us> <ig_comm_type> <ig_comm_size> <wg_compute_us> <wg_comm_type> <wg_comm_size> <multiplier>
```

Comm types: `ALLREDUCE`, `ALLGATHER`, `REDUCESCATTER`, `ALLTOALL`, `ALLTOALL_EP`, `NONE`

### Topology Directory

```
topology_dir/
├── topology       # Space-separated link table
└── metadata.json  # Generation parameters
```

`topology` file format:
```
<total_nodes> <gpus_per_server> <nv_switches> <network_switches> <links> <gpu_type>
<switch_node_ids...>
<src> <dst> <bandwidth_bps> <latency_ms> <error_rate>
...
```

`metadata.json` format:
```json
{
  "type": "Spectrum-X",
  "num_gpus": 128,
  "gpus_per_server": 8,
  "gpu_type": "H100",
  "nic_bandwidth_gbps": 400.0,
  "nvlink_bandwidth_gbps": 7200.0,
  "nics_per_switch": 32
}
```

### Simulation Results

- `ncclFlowModel_EndToEnd.csv`: Per-layer timing (TP/DP/EP/PP exposed comm, compute)
- `ncclFlowModel_*_utilization_*.csv`: Link utilization statistics
- `ncclFlowModel_detailed_*.csv`: Per-flow completion times

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `SIMAI_PATH` | SimAI repo root (for editable installs / custom deployments) |
| `SIMAI_BIN_PATH` | Directory containing `SimAI_analytical`, `SimAI_simulator` |
| `AS_LOG_LEVEL` | NS-3 log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` (set to 0 to suppress root-owned logs) |
| `AS_SEND_LAT` | NS-3 send latency in microseconds |
| `AS_NVLS_ENABLE` | Enable NVLink Switch algorithm: `0` or `1` |
| `AS_PXN_ENABLE` | Enable PCIe cross-node: `0` or `1` |
| `SIMAI_PLATFORM_TAG` | Wheel platform tag (e.g., `manylinux_2_17_x86_64`), build-time only |

---

## CI/CD

**File**: `.github/workflows/build.yml`

**Triggers**: Push to `main` or `dev`, PRs to `main` or `dev`, manual dispatch.

### Dev Branch & TestPyPI

There are two publish tracks:

| Branch | Version format | Publishes to | GitHub release |
|--------|---------------|--------------|----------------|
| `main` | `X.Y.Z` (from `pyproject.toml`) | PyPI | Yes (tag + release) |
| `dev`  | `X.Y.Z.dev{run_number}` (auto-generated) | TestPyPI | No |

**How dev versioning works**: On every push to `dev`, the `build-wheel` job patches
`pyproject.toml` with a PEP 440 dev version (`{base}.dev{GITHUB_RUN_NUMBER}`) before
running `uv build --wheel`. The source file is not committed — it's a transient in-CI edit.

**Build gating**:
- `main`: builds only when `pyproject.toml` version changes (or `force` dispatch input)
- `dev`: always builds (no version-change gate)

**Jobs** (in order):

1. **check-version**: Reads version from `pyproject.toml`, determines if changed, detects branch (`is_dev` output)
2. **build-analytical**: manylinux2014 Docker, applies path patches, CMake, caches by submodule commit
3. **build-ns3**: manylinux2014 Docker, installs libxml2/sqlite/gsl, builds NS-3 debug + MTP,
   strips symbols, bundles `libns3*.so` shared libraries, caches by submodule commit
4. **build-wheel**: Downloads binaries from artifacts, patches version (dev only), `uv build --wheel`, enforces <100MB PyPI limit
5. **release**: Creates git tag `v<version>`, GitHub release with wheel attached *(main only)*
6. **publish-pypi**: OIDC authentication, publishes to PyPI *(main only)*
7. **publish-testpypi**: OIDC authentication, publishes to TestPyPI *(dev only)*

**m4 binary is NOT built in CI and NOT built at wheel/install time.** It requires CUDA and
is compiled on demand via `simai install m4` (see CLI below).

Binaries are renamed during wheel build: `ns3.36.1-AstraSimNetwork-debug` → `SimAI_simulator`

**One-time setup for `dev` branch**:
1. `git checkout -b dev main && git push -u origin dev`
2. Configure trusted publishing on test.pypi.org for this repo (OIDC, same as PyPI setup)

---

## Quick Reference: End-to-End Workflow

```bash
# 1. (Optional) Profile GPU kernels once per GPU type + model config
simai profile gpu --framework Megatron --num-gpus 128 --tensor-parallel 8 \
    --num-layers 96 --hidden-size 12288 --gpu-type H100 -o h100_profile.txt

# 2. Generate topology
simai generate topology --type Spectrum-X --num-gpus 128 --gpus-per-server 8 \
    --gpu-type H100 --nic-bandwidth 400Gbps --nvlink-bandwidth 7200Gbps \
    -o topology_h100_128gpu/

# 3. Generate workload
simai generate workload --framework Megatron --num-gpus 128 --tensor-parallel 8 \
    --pipeline-parallel 4 --num-layers 96 --hidden-size 12288 \
    --compute-profile h100_profile.txt -o workload_gpt175b.txt

# 4a. Run analytical simulation (fast)
simai simulate analytical -w workload_gpt175b.txt -n topology_h100_128gpu/ \
    -o results/analytical/

# 4b. Run NS-3 simulation (detailed)
simai simulate ns3 -w workload_gpt175b.txt -n topology_h100_128gpu/ \
    -t 16 --nvls -o results/ns3/
```

---

**Last Updated**: 2026-02-17 (m4 install: --n-flows-max flag, _patch_n_flows_max(), default 500 000) | **Human reference**: [`README.md`](./README.md)

