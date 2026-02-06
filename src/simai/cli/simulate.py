from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(no_args_is_help=True)


def _read_metadata(topology_dir: Path) -> dict:
    """Read and return metadata.json from a topology directory."""
    meta_path = topology_dir / "metadata.json"
    if not meta_path.is_file():
        raise typer.BadParameter(
            f"No metadata.json found in topology directory: {topology_dir}\n"
            "Did you generate this topology with 'simai generate topology'?"
        )
    with open(meta_path) as f:
        return json.load(f)


def _parse_workload_gpu_count(workload: Path) -> int | None:
    """Extract GPU count from the workload file header line (all_gpus: N)."""
    with open(workload) as f:
        for line in f:
            m = re.search(r"all_gpus:\s*(\d+)", line)
            if m:
                return int(m.group(1))
            # Only check the first few header lines
            if not line.startswith("#"):
                break
    return None


def _validate_gpu_count(workload: Path, topology_dir: Path, metadata: dict) -> None:
    """Warn if the workload GPU count doesn't match the topology GPU count."""
    workload_gpus = _parse_workload_gpu_count(workload)
    topo_gpus = metadata.get("num_gpus")
    if workload_gpus is not None and topo_gpus is not None and workload_gpus != topo_gpus:
        raise typer.BadParameter(
            f"GPU count mismatch: workload has {workload_gpus} GPUs "
            f"but topology has {topo_gpus} GPUs."
        )


@app.command()
def analytical(
    workload: Annotated[
        Path,
        typer.Option("--workload", "-w", help="Path to workload file (from generate workload)."),
    ],
    topology: Annotated[
        Path,
        typer.Option("--topology", "-n", help="Path to topology directory (from generate topology)."),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output directory for result CSV files (default: ./results/)."),
    ] = None,
    dp_overlap: Annotated[
        Optional[float],
        typer.Option("--dp-overlap", help="Data-parallel communication overlap ratio (0.0-1.0)."),
    ] = None,
    tp_overlap: Annotated[
        Optional[float],
        typer.Option("--tp-overlap", help="Tensor-parallel overlap ratio."),
    ] = None,
    ep_overlap: Annotated[
        Optional[float],
        typer.Option("--ep-overlap", help="Expert-parallel overlap ratio."),
    ] = None,
    pp_overlap: Annotated[
        Optional[float],
        typer.Option("--pp-overlap", help="Pipeline-parallel overlap ratio."),
    ] = None,
    result_prefix: Annotated[
        Optional[str],
        typer.Option("--result-prefix", help="Prefix for result file names."),
    ] = None,
):
    """Run the analytical (fast, approximate) network simulation."""
    from simai.backends.analytical import run_analytical

    metadata = _read_metadata(topology)
    _validate_gpu_count(workload, topology, metadata)

    run_analytical(
        workload=workload,
        num_gpus=metadata["num_gpus"],
        gpus_per_server=metadata["gpus_per_server"],
        nvlink_bandwidth=metadata.get("nvlink_bandwidth_gbps"),
        nic_bandwidth=metadata.get("nic_bandwidth_gbps"),
        nics_per_server=metadata.get("nics_per_switch"),
        gpu_type=metadata.get("gpu_type"),
        dp_overlap=dp_overlap,
        tp_overlap=tp_overlap,
        ep_overlap=ep_overlap,
        pp_overlap=pp_overlap,
        result_prefix=result_prefix,
        output=output,
    )


@app.command()
def ns3(
    workload: Annotated[
        Path,
        typer.Option("--workload", "-w", help="Path to workload file."),
    ],
    topology: Annotated[
        Path,
        typer.Option("--topology", "-n", help="Path to topology directory."),
    ],
    config: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="SimAI config file path (default: bundled SimAI.conf)."),
    ] = None,
    threads: Annotated[
        int,
        typer.Option("--threads", "-t", help="Number of simulation threads."),
    ] = 8,
    send_latency: Annotated[
        Optional[int],
        typer.Option("--send-latency", help="Send latency in microseconds."),
    ] = None,
    nvls: Annotated[
        bool,
        typer.Option("--nvls/--no-nvls", help="Enable NVLink Switch."),
    ] = False,
    pxn: Annotated[
        bool,
        typer.Option("--pxn/--no-pxn", help="Enable PXN (PCIe cross-node)."),
    ] = False,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output directory for results."),
    ] = None,
):
    """Run the NS-3 (detailed, packet-level) network simulation."""
    from simai.backends.ns3 import run_ns3

    metadata = _read_metadata(topology)
    _validate_gpu_count(workload, topology, metadata)

    # Resolve the topology file within the directory
    topo_file = topology / "topology"
    if not topo_file.is_file():
        raise typer.BadParameter(
            f"No 'topology' file found in directory: {topology}\n"
            "Did you generate this topology with 'simai generate topology'?"
        )

    run_ns3(
        workload=workload,
        topology=topo_file,
        config=config,
        threads=threads,
        send_latency=send_latency,
        nvls=nvls,
        pxn=pxn,
        output=output,
    )
