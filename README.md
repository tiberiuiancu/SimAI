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

### 2. Generate a topology

```bash
simai generate topology --type DCN+ --num-gpus 64 --gpu-type H100 \
    --nic-bandwidth 100Gbps --nvlink-bandwidth 3600Gbps -o my_topo/
```

### 3. Run a simulation

**Analytical** (fast, approximate):

```bash
simai simulate analytical \
    -w workload.txt \
    -n my_topo/ \
    -o results/
```

**NS-3** (detailed, packet-level):

```bash
simai simulate ns3 \
    -w workload.txt \
    -n my_topo/ \
    -o results/
```

**M4** (flow-level, ML-based gray failure, requires local build — see installation note above):

```bash
simai simulate m4 \
    -w workload.txt \
    -n my_topo/ \
    -o results/
```

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
