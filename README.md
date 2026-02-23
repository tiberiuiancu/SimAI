# SimAI

Python wrapper for the [SimAI](https://github.com/aliyun/SimAI) datacenter network simulator. Provides a CLI and Python API for generating training workloads, network topologies, and running network simulations, with pre-built binaries bundled in the wheel.

## Installation

Install from PyPI:

```bash
pip install simai
```

For GPU compute profiling (optional, requires CUDA):

```bash
pip install "simai[profiling]"
```

For the M4 (flow-level, ML-based) simulation backend:

```bash
pip install "simai[m4]"      # installs torch dependency
simai install m4             # compiles SimAI_m4 binary (requires CUDA torch + cmake/make/gcc)
```

For NVIDIA Apex (PyTorch CUDA extensions) and DeepGEMM (DeepSeek CUDA kernels):

```bash
simai install apex           # Installs NVIDIA/apex (for AICB profiling, optional)
simai install deepgemm       # Installs DeepSeekAI/DeepGEMM (for DeepSeek models, optional)
```

> **Note**: If you see a RuntimeError about CUDA version mismatch when installing Apex, you can use:
> ```bash
> simai install apex --skip-cuda-version-check
> ```
> This will patch `setup.py` to skip the CUDA version check (at your own risk). See [discussion](https://github.com/NVIDIA/apex/pull/323#discussion_r287021798).

> **Note**: The M4 binary (`SimAI_m4`) is **not** included in the PyPI wheel. Run
> `simai install m4` to compile it from source (requires CUDA-enabled PyTorch and cmake/make/gcc).
> On first run the source is cloned automatically from GitHub into `~/.cache/simai/simai-m4/`;
> subsequent runs reuse that cache. For editable installs the local `vendor/simai-m4/` tree is
> used instead. Pass `--src /path/to/simai-m4` to override, or set `LIBTORCH_DIR` to point to a
> custom LibTorch. The `[m4]` extra pins `torch<2.7` — versions ≥2.7 are not yet supported.
>
> Use `--n-flows-max N` (default: 500 000) to raise the maximum concurrent-flow capacity before
> compilation. The upstream default of 50 000 is too low for large workloads and causes a crash:
> ```bash
> simai install m4 --force --n-flows-max 1000000
> ```

## Usage

### 1. Generate a workload

```bash
simai generate workload \
    --framework Megatron \
    --num-gpus 64 \
    --tensor-parallel 4 \
    --pipeline-parallel 2 \
    --num-layers 32 \
    --hidden-size 4096 \
    --sequence-length 2048 \
    --iter 10 \
    -o workload.txt
```

### 1a. Profile GPU kernels (optional)

For accurate compute time modeling, profile GPU kernel execution:

```bash
simai profile gpu \
    --framework Megatron \
    --num-gpus 64 \
    --num-layers 32 \
    --hidden-size 4096 \
    --gpu-type H100 \
    -o h100_profile.txt
```

Requirements:
- PyTorch with CUDA: `pip install "simai[profiling]"`
- CUDA-capable GPU

Then use the profile when generating workloads:

```bash
simai generate workload --compute-profile h100_profile.txt \
    --num-gpus 64 --tensor-parallel 4 \
    --pipeline-parallel 2 \
    --num-layers 32 \
    --hidden-size 4096 \
    --sequence-length 2048 \
    --iter 10 \
    -o workload.txt
```

#### Compute timing modes

Workload generation supports three modes for compute times:

1. **Constant times** (default): Fast but approximate placeholder values
2. **Pre-recorded profile**: Use a profile from `simai profile gpu` (recommended)
3. **Live profiling**: Add `--profile-compute` flag (equivalent to mode 2)

Mode 2 is recommended for production use as it separates profiling from workload generation.

### 1b. Run a distributed training benchmark (AICB)

For running actual collective operations across a real GPU cluster using AICB:

```bash
# Single-node smoke test (no SLURM needed)
simai bench training \
    --nproc-per-node 4 \
    --world-size 4 \
    --framework Megatron \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-heads 16 \
    --global-batch-size 8 \
    --micro-batch-size 1 \
    --epochs 1 \
    --output results/bench/
```

With a GPU compute profile for realistic AIOB compute-communication overlap:

```bash
simai bench training \
    --nproc-per-node 4 --world-size 4 \
    --num-layers 24 --hidden-size 1024 --num-heads 16 \
    --global-batch-size 8 --micro-batch-size 1 \
    --comp-profile h100_profile.txt \
    --output results/bench/
```

**Multi-node SLURM** — SLURM env vars (`SLURM_NNODES`, `SLURM_NODEID`, `SLURM_GPUS_PER_NODE`,
`MASTER_ADDR`, `MASTER_PORT`) are auto-detected, so each `srun` task needs no extra flags:

```bash
# Use the provided template (edit partition / model config as needed)
sbatch tests/run_bench.slurm
```

Requirements:
- PyTorch with CUDA: `pip install "simai[profiling]"`
- CUDA-capable GPUs
- AICB source (vendored in wheel, or set `SIMAI_PATH`)

> **Note**: `simai bench training` runs actual NCCL collective operations on real GPUs.
> This is different from `simai profile gpu` (single-GPU kernel timing, no communication)
> and `simai simulate analytical/ns3` (software simulation, no GPU needed).

### 2. Generate a topology

```bash
simai generate topology --type DCN+ --num-gpus 64 --gpu-type H100 \
    --nic-bandwidth 100Gbps --nvlink-bandwidth 3600Gbps -o my_topo.json
```

Produces a single `topology.json` file (see [Topology JSON](#topology-json) below).

### 3. Run a simulation

**Analytical** (fast, approximate):

```bash
simai simulate analytical \
    -w workload.txt \
    -n my_topo.json \
    -o results/result.json
```

**NS-3** (detailed, packet-level):

```bash
simai simulate ns3 \
    -w workload.txt \
    -n my_topo.json \
    -o results/result.json
```

**M4** (flow-level, ML-based gray failure, requires local build — see installation note above):

```bash
simai simulate m4 \
    -w workload.txt \
    -n my_topo.json \
    -o results/result.json
```

Each simulation writes a `result.json` to the output path and preserves raw binary output in a
temp directory whose path is recorded in `result.json["raw_output_path"]`.

### 3a. Run with a TOML config file

All simulation options can be captured in a single `run.toml`:

```bash
simai simulate ns3 -c run.toml

# CLI flags override TOML values:
simai simulate ns3 -c run.toml -w other_workload.txt --threads 16
```

See the [TOML config reference](#toml-config-reference) below for a full annotated example.

### Installing a dev version

Dev builds are published to TestPyPI on every push to the `dev` branch:

```bash
pip install --pre \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  simai
```

Or pin a specific dev build:

```bash
pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  "simai==0.3.12.dev42"
```

## Topology JSON

`simai generate topology` produces a single `topology.json`:

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
  ]
}
```

**Legacy directory format** (`topology/` directory with `topology` file + `metadata.json`) is
still accepted by `simai simulate` with a deprecation warning.

## Result JSON

Each simulation produces a `result.json`:

```json
{
  "simai_version": "0.5.0",
  "run_id": "abc123",
  "timestamp": "2026-02-23T14:00:00Z",
  "backend": "analytical",
  "config": {"workload.framework": "Megatron", "topology.num_gpus": 128},
  "topology_metadata": {"type": "Spectrum-X", "num_gpus": 128},
  "workload_header": {"all_gpus": 128, "tp": 8},
  "results": {
    "layers": [{"name": "grad_gather", "exposed_comm_us": 1234.5, "compute_us": 567.8}],
    "summary": {"total_time_us": 99999, "total_exposed_comm_us": 55000, "total_compute_us": 44999},
    "link_utilization": null,
    "flow_completion": null
  },
  "raw_output_path": "/tmp/simai_analytical_xyz123"
}
```

The flat `config` dict is suitable for logging to experiment trackers (MLflow, W&B).

## Output files

### Analytical backend

Produces CSV result files in the output directory:

- **`<prefix>_EndToEnd.csv`** — Main results file. Contains a summary row with total simulated training iteration time and per-parallelism-dimension communication breakdown (DP, TP, EP, PP), followed by per-layer rows with forward/weight-gradient/input-gradient comm times and algorithm/bus bandwidth.

### NS-3 backend

Produces several files in the output directory:

- **`ncclFlowModel_EndToEnd.csv`** — Main results file. Same format as the analytical output: summary row with total time and communication breakdown, then per-layer detail with exposed comm times and bandwidth.
- **`ncclFlowModel_*_dimension_utilization_*.csv`** — Time-series of network dimension utilization (sampled every 10us). Useful for spotting congestion patterns.
- **`*_fct.txt`** — Flow Completion Times. Each row is a completed network flow with source, destination, port, priority, size, start time, FCT, and end time.
- **`*_pfc.txt`** — Priority Flow Control events. Empty means no PFC pauses occurred (no congestion-induced backpressure).
- **`*_mix.tr`** — Binary NS-3 trace file.
- **`ncclFlowModel_detailed_*.csv`** — Detailed per-chunk communication breakdown (may be empty for small workloads).

## TOML Config Reference

A `run.toml` file can describe a complete simulation run. All sections and keys are optional —
omit anything you want to pass via CLI flags instead. CLI flags always override TOML values.

The backend is **not** in the config file; it is always specified as the subcommand
(`simai simulate analytical|ns3|m4 -c run.toml`).

```toml
# ─────────────────────────────────────────────
# [run] — top-level output and verbosity
# ─────────────────────────────────────────────
[run]
output  = "results/my_run.json"  # Path to write result.json (file or directory)
verbose = false                  # Show binary stdout/stderr

# ─────────────────────────────────────────────
# [workload] — training workload
# Choose EITHER file (pre-generated) OR inline params (generated on-the-fly).
# If both are given, file takes priority.
# ─────────────────────────────────────────────
[workload]
# Option A – use a pre-generated workload file
file = "results/workload/H100-gpt13b-tp8.txt"

# Option B – generate inline at run time
# framework    = "Megatron"   # Megatron | DeepSpeed | DeepSeek
# world_size   = 128          # total number of GPUs
# tensor_parallel   = 8
# pipeline_parallel = 1
# expert_parallel   = 1
# global_batch = 2048
# micro_batch  = 8
# num_layers   = 40
# hidden_size  = 5120
# seq_length   = 2048
# num_attention_heads = 40    # defaults to num_layers if omitted
# vocab_size   = 32000
# gpu_type     = "H100"       # label used in output filenames
# use_flash_attn = false

# ─────────────────────────────────────────────
# [topology] — network topology
# Choose EITHER file (pre-generated) OR inline params.
# ─────────────────────────────────────────────
[topology]
# Option A – use a pre-generated topology.json
file = "results/topology.json"

# Option B – generate inline at run time
# type              = "Spectrum-X"   # Spectrum-X | AlibabaHPN | DCN+
# num_gpus          = 128
# gpus_per_server   = 8
# nic_bandwidth_gbps    = 400.0
# nvlink_bandwidth_gbps = 7200.0
# nics_per_switch   = 64

# ─────────────────────────────────────────────
# [compute_profile] — optional GPU compute profile
# ─────────────────────────────────────────────
[compute_profile]
# file = "results/h100_profile.txt"

# ─────────────────────────────────────────────
# [simulation] — shared + per-backend parameters
# ─────────────────────────────────────────────
[simulation]
threads = 8          # parallel simulation threads (NS-3 and M4)

# Analytical overlap ratios (fraction of comm overlapped with compute, 0.0–1.0)
# dp_overlap = 0.0
# tp_overlap = 0.0
# ep_overlap = 0.0
# pp_overlap = 0.0

# NS-3 specific
# send_latency = 1   # send latency in microseconds (AS_SEND_LAT env var)
# nvls = false       # enable NVLink Switch (AS_NVLS_ENABLE)
# pxn  = false       # enable PCIe cross-node (AS_PXN_ENABLE)

# ─────────────────────────────────────────────
# [ns3] — NS-3 SimAI.conf parameters
# All keys map 1:1 to SimAI.conf lines.
# Only used by `simai simulate ns3`.
# ─────────────────────────────────────────────
[ns3]
ENABLE_QCN                 = 1
USE_DYNAMIC_PFC_THRESHOLD  = 1
PACKET_PAYLOAD_SIZE        = 9000
SIMULATOR_STOP_TIME        = 40000000000000.0
CC_MODE                    = 1       # congestion control: 1=DCQCN, 3=HPCC, 7=TIMELY, 8=DCTCP
ALPHA_RESUME_INTERVAL      = 1
RATE_DECREASE_INTERVAL     = 4
CLAMP_TARGET_RATE          = 0
RP_TIMER                   = 900
EWMA_GAIN                  = 0.00390625
FAST_RECOVERY_TIMES        = 1
RATE_AI                    = "50Mb/s"
RATE_HAI                   = "100Mb/s"
MIN_RATE                   = "100Mb/s"
DCTCP_RATE_AI              = "1000Mb/s"
ERROR_RATE_PER_LINK        = 0.0
L2_CHUNK_SIZE              = 4000
L2_ACK_INTERVAL            = 1
L2_BACK_TO_ZERO            = 0
HAS_WIN                    = 1
GLOBAL_T                   = 0
VAR_WIN                    = 1
FAST_REACT                 = 1
U_TARGET                   = 0.95
MI_THRESH                  = 0
INT_MULTI                  = 1
MULTI_RATE                 = 0
SAMPLE_FEEDBACK            = 0
PINT_LOG_BASE              = 1.05
PINT_PROB                  = 1.0
RATE_BOUND                 = 1
ACK_HIGH_PRIO              = 0
LINK_DOWN                  = "0 0 0"
ENABLE_TRACE               = 1
KMAX_MAP = "6 25000000000 400 50000000000 800 100000000000 1600 200000000000 1200 400000000000 3200 1600000000000 2400"
KMIN_MAP = "6 25000000000 100 50000000000 200 100000000000 400 200000000000 300 400000000000 800 1600000000000 600"
PMAX_MAP = "6 25000000000 0.2 50000000000 0.2 100000000000 0.2 200000000000 0.8 400000000000 0.2 1600000000000 0.2"
BUFFER_SIZE                = 32
MON_START                  = 0
MON_END                    = 20000
QP_MON_INTERVAL            = 100
QLEN_MON_INTERVAL          = 10000
BW_MON_INTERVAL            = 10000
```

The flat config dict (for experiment tracking):

```python
from simai.config import load_config, to_flat_dict
flat = to_flat_dict(load_config("run.toml"))
# {"run.output": "results/my_run.json", "workload.framework": "Megatron",
#  "topology.num_gpus": 128, "ns3.CC_MODE": 1, ...}

import mlflow
mlflow.log_params(flat)
```

---

## Differences from upstream SimAI

This wrapper automates the manual setup that upstream SimAI requires:

- **Output location**: Upstream writes to hardcoded paths (`./results/` for analytical, `/etc/astra-sim/simulation/` for NS-3 config paths). This wrapper runs binaries in isolated temp directories and moves results to the user-specified `-o` path.
- **Directory setup**: Upstream requires manually creating directory structures and symlinking data files (e.g. `astra-sim-alibabacloud/inputs/ratio/` CSVs). This wrapper handles it automatically.
- **Config patching**: The NS-3 config file (`SimAI.conf`) hardcodes absolute paths to `/etc/astra-sim/simulation/`. This wrapper patches them at runtime to point to the temp directory.
- **Binary and data discovery**: Upstream requires users to manage paths to binaries and auxiliary data. This wrapper auto-discovers them from the wheel's bundled files, environment variables (`SIMAI_BIN_PATH`, `SIMAI_PATH`), or the vendor submodule.
- **Topology directory**: Upstream passes raw bandwidth parameters to the analytical backend and a topology file path to NS-3 separately. This wrapper uses a unified topology directory (containing `topology` file + `metadata.json`) for both backends.

## Building from source

Requires the SimAI submodule and compiled binaries.

```bash
git clone --recurse-submodules https://github.com/tiberiuiancu/SimAI.git
cd SimAI

# Install dev environment
uv sync

# Build binaries (if missing) and the wheel
./scripts/build_wheel.sh
```

`scripts/build_wheel.sh` checks whether pre-built binaries already exist in `build/bin/`.
If not, it compiles them — via a manylinux Docker container if `docker` is available, or
natively with `cmake`/`make` otherwise. Then it runs `uv build --wheel`.

**Flags:**

| Flag | Effect |
|------|--------|
| *(none)* | Build binaries if missing, then build wheel |
| `--no-bin` | Skip binary build (use whatever is in `build/bin/`) |
| `--docker` | Force manylinux Docker build (same environment as CI) |
| `--native` | Force native build (skips Docker even if available) |

After any Python-only change:
```bash
./scripts/build_wheel.sh --no-bin
```

Or just push to GitHub — the CI workflow compiles the binaries and produces the wheel automatically.

---

## Contributing / Agent reference

For AI agents and contributors who want a detailed architectural reference without exploring
the codebase from scratch, see [`AGENTS.md`](./AGENTS.md).
