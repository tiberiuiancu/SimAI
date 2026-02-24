# SimAI Agent Reference

This document contains everything an agent needs to work on the SimAI codebase without
additional exploration. Keep it up to date when making structural changes.

**Version**: 0.5.0 | **Python**: ≥3.10 | **Package manager**: `uv`

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

- **CLI**: `simai bench training`, `simai generate workload/topology`, `simai profile gpu`, `simai simulate analytical/ns3`
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
│   ├── cli/                # Typer CLI commands (app.py, bench.py, generate.py, profile.py, simulate.py)
│   ├── backends/           # Simulation backends (binary.py, analytical.py, ns3.py, m4.py)
│   ├── topology/           # Topology generation (generator.py, format.py)
│   ├── workflow/           # Workload generation, GPU profiling, bench (generator.py, profiler.py, bench.py)
│   ├── config.py           # TOML run config schema (SimaiConfig, load_config, to_flat_dict)
│   └── output.py           # Result JSON builder + CSV parsers (build_result_json, write_result_json)
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
│   ├── run_profile_tests.slurm     # SLURM job for single-node GPU profiling
│   └── run_bench.slurm             # SLURM job for multi-node distributed benchmark
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

**`app.py`**: Main Typer app with top-level commands: `bench`, `generate`/`gen`, `install`, `profile`, `simulate`.

**`generate.py`**:
- `workload()`: Generate training workload `.txt` files. Parameters: framework, num_gpus,
  tensor/pipeline/expert parallel, model architecture (layers, hidden size, heads, vocab),
  batch sizes, MoE config, compute profile.
- `topology()` / `topo()` (hidden alias): Generate network topology directories.
  Parameters: type (Spectrum-X, DCN+, AlibabaHPN), GPUs, bandwidth, latency, dual-ToR, dual-plane.

**`bench.py`**:
- `training()`: Run a distributed AICB training benchmark via `torchrun`. Launches actual NCCL
  collective operations across real GPUs. SLURM env vars (`SLURM_NNODES`, `SLURM_NODEID`,
  `SLURM_GPUS_PER_NODE`, `MASTER_ADDR`, `MASTER_PORT`) are auto-detected as defaults.
  `--comp-profile` implies `--aiob`. Deferred import of `workflow.bench`.

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
- `apex()`: Installs NVIDIA/apex (PyTorch CUDA extensions) from source. Clones to `~/.cache/simai/apex` by default. Accepts `--src` and `--git-url`. Use `--skip-cuda-version-check` to patch `setup.py` and skip the CUDA version check (see [discussion](https://github.com/NVIDIA/apex/pull/323#discussion_r287021798)).
- `deepgemm()`: Installs DeepSeekAI/DeepGEMM (CUDA kernels for DeepSeek models) from source. Clones to `~/.cache/simai/DeepGEMM` by default. Accepts `--src` and `--git-url`.

**`simulate.py`**:
- `analytical()`, `ns3()`, `m4()`: All accept `--config/-c` for a TOML run config file (CLI
  flags always override). Topology can be a `topology.json` file or legacy directory.
  Each command writes a `result.json` to the output path after running.
  `analytical()` overlap parameters: `--dp-overlap`, `--tp-overlap`, `--ep-overlap`, `--pp-overlap`.
  `ns3()` uses `[ns3]` TOML section to render `SimAI.conf` in tmpdir (replaces regex-patching).

### Workflow Layer (`src/simai/workflow/`)

**`generator.py`** - `generate_workload()`:
- Locates AICB via `_find_aicb_root()` (3-tier: vendored → `SIMAI_PATH` env → sibling dir heuristic)
- Uses `@contextmanager` `_aicb_on_path()` for temporary `sys.path` injection
- Injects `argparse.Namespace` into AICB module globals (not modifying AICB source)
- Outputs `.txt` workload file

**`bench.py`** - `run_training_benchmark()`:
- `_find_torchrun()`: Locates `torchrun` — checks venv bin dir first, then `shutil.which`
- `run_training_benchmark()`: Builds and launches `torchrun aicb.py` subprocess. Reuses
  `_find_aicb_root()` from `generator.py`. Boolean flags (`--moe_enable`, `--aiob_enable`, etc.)
  appended only when `True`. Runs in `output_dir`; returns subprocess returncode.

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
- **Output**: single `topology.json` with metadata + `total_nodes`, `switch_ids`, `links` array, and `_topology_text` (raw text for exact NS3 reconstruction)
- Legacy directory output still supported: if `output` path has no `.json` suffix, writes `topology.json` inside the directory
- Supported types: Spectrum-X (NVIDIA rail-optimized), DCN+ (traditional), AlibabaHPN (multi-plane)

**`format.py`** (new) - Topology format helpers:
- `load_topology(path)`: loads `topology.json` or legacy directory (auto-detects, emits `DeprecationWarning`)
- `unpack_for_ns3(topo, dest)`: writes raw NS3 topology text to dest (uses `_topology_text` if available)
- `unpack_for_m4(topo, dest)`: writes M4 topology format (Gbps/ms units) to dest
- `topology_to_analytical_args(topo)`: returns `dict` of kwargs for `run_analytical()`
- `_parse_topology_text(text)`: parses topology text into `{total_nodes, switch_ids, links}`

### Backend Layer (`src/simai/backends/`)

**`binary.py`**:
- `find_binary(name)`: Search order: bundled `simai/_binaries/` → `SIMAI_BIN_PATH` env → system `PATH`
- `run_binary(name, args, cwd, env, verbose)`: Sets `LD_LIBRARY_PATH` to binary dir for shared libs,
  runs via `subprocess.run()`

**`analytical.py`** - `run_analytical()`:
- `_find_simai_root()`: Locates ratio CSVs (4-tier search)
- Returns `SimulationResult(output_path, raw_output_path, parsed)` dataclass
- `preserve_raw=True` (default): keeps tmpdir alive, sets `raw_output_path = tmpdir`
- `preserve_raw=False`: legacy behavior — moves files to output_path, deletes tmpdir

**`ns3.py`** - `run_ns3()`:
- `_find_default_config()`: Locates `SimAI.conf`
- New: `ns3_conf: dict | None` parameter — if provided, renders dict to `SimAI.conf` in tmpdir
  (key + space + value per line), replacing both the bundled config and the old regex-patching
- Legacy: if `ns3_conf` is None and `config` is given, patches `/etc/astra-sim/simulation/` paths
- Returns `SimulationResult`

**`m4.py`** - `run_m4()`:
- `_find_m4_models()`: Locates `.pt` model files (3-tier)
- New: `topology_dict: dict | None` parameter — if provided, calls `unpack_for_m4()` to write the
  m4 topology format (replaces `_convert_topology()`). Falls back to `_convert_topology()` if None
- `_convert_topology()` kept as legacy fallback
- Returns `SimulationResult`

All three backends define `SimulationResult = dataclass(output_path, raw_output_path, parsed)`.
`parsed` is a dict with `layers`, `summary`, `link_utilization`, `flow_completion` keys.

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
# Integration test for GPU profiling (single-node)
./tests/test_profile_integration.sh

# Submit SLURM job for GPU profiling
sbatch tests/run_profile_tests.slurm

# Submit SLURM job for multi-node distributed benchmark
sbatch tests/run_bench.slurm
```

There are no unit tests currently - only end-to-end integration tests.

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

### Topology JSON (`topology.json`)

```json
{
  "type": "Spectrum-X",
  "num_gpus": 128,
  "gpus_per_server": 8,
  "gpu_type": "H100",
  "nic_bandwidth_gbps": 400.0,
  "nvlink_bandwidth_gbps": 7200.0,
  "nics_per_switch": 64,
  "total_nodes": 192,
  "switch_ids": [128, 129, 130, 131],
  "links": [
    {"src": 0, "dst": 128, "bandwidth_gbps": 7200.0, "latency_ms": 0.000025, "error_rate": 0.0}
  ],
  "_topology_text": "<raw text from gen_Topo_Template.py>"
}
```

`_topology_text` is an internal field used by `unpack_for_ns3()` to reconstruct the exact
original NS3 format. It is stripped from `result.json` output.

**Legacy directory format** (deprecated, still accepted):
```
topology_dir/
├── topology       # Space-separated link table
└── metadata.json  # Generation parameters
```

### TOML Run Config (`run.toml`)

Sections: `[run]`, `[workload]`, `[topology]`, `[compute_profile]`, `[simulation]`, `[ns3]`.
See `src/simai/config.py` for all fields and defaults. The `[ns3]` section maps 1:1 to
`SimAI.conf` keys. `to_flat_dict(cfg)` collapses to `section.key` for experiment trackers.

### Result JSON (`result.json`)

```json
{
  "simai_version": "0.5.0",
  "run_id": "<8-char hex>",
  "timestamp": "<ISO 8601 UTC>",
  "backend": "analytical|ns3|m4",
  "config": {"workload.framework": "Megatron", ...},
  "topology_metadata": {"type": "Spectrum-X", "num_gpus": 128, ...},
  "workload_header": {"all_gpus": 128, "tp": 8, ...},
  "results": {
    "layers": [{"name": "...", "exposed_comm_us": 1234.5, "compute_us": 567.8}],
    "summary": {"total_time_us": ..., "total_exposed_comm_us": ..., "total_compute_us": ...},
    "link_utilization": null,
    "flow_completion": null
  },
  "raw_output_path": "/tmp/simai_analytical_xyz123"
}
```

### Simulation Raw Output Files

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

**Last Updated**: 2026-02-24 (rebase onto dev: merge bench training + 0.5.0 unified config/topology.json/result.json features) | **Human reference**: [`README.md`](./README.md)
