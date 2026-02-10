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

Requires the SimAI submodule and compiled binaries:

```bash
git clone --recurse-submodules https://github.com/tiberiuiancu/SimAI.git
cd SimAI

# Build binaries (see vendor/simai/scripts/build.sh)
# Place SimAI_analytical and SimAI_simulator in build/bin/

SIMAI_PLATFORM_TAG=manylinux_2_35_x86_64 pip install build
python -m build --wheel
```

Or just push to GitHub — the CI workflow compiles the binaries and produces the wheel automatically.
