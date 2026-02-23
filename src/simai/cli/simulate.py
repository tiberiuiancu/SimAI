from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(no_args_is_help=True)


def _load_topology(topology: Path) -> dict:
    """Load topology from topology.json or legacy directory format."""
    from simai.topology.format import load_topology
    return load_topology(topology)


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


def _resolve_topology_file(topo_path: Path) -> Path:
    """Resolve a topology path to the raw topology file (for NS3/M4).

    Handles:
    - topology.json (new format): returns the .json path
    - legacy dir with topology file: returns the topology file inside
    """
    if topo_path.is_file():
        return topo_path
    topo_file = topo_path / "topology"
    if topo_file.is_file():
        return topo_file
    raise typer.BadParameter(
        f"No topology file found at: {topo_path}\n"
        "Did you generate this topology with 'simai generate topology'?"
    )


def _write_result(
    backend: str,
    topo: dict,
    workload: Path,
    sim_result,
    config_flat: dict,
    output: Optional[Path],
) -> None:
    """Build and write result.json if output path is specified."""
    if output is None:
        return
    output = Path(output)
    # If output has no .json suffix, treat it as a directory and write result.json inside
    if output.suffix != ".json":
        result_path = output / "result.json"
    else:
        result_path = output

    from simai.output import (
        build_result_json,
        parse_workload_header,
        write_result_json,
    )
    workload_header = parse_workload_header(workload)
    data = build_result_json(
        backend=backend,
        config=config_flat,
        topology_meta=topo,
        workload_header=workload_header,
        results=sim_result.parsed,
        raw_path=sim_result.raw_output_path,
    )
    write_result_json(data, result_path)


@app.command()
def analytical(
    workload: Annotated[
        Optional[Path],
        typer.Option("--workload", "-w", help="Path to workload file (from generate workload)."),
    ] = None,
    topology: Annotated[
        Optional[Path],
        typer.Option("--topology", "-n", help="Path to topology.json (or legacy directory)."),
    ] = None,
    config_file: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="Path to TOML run config file."),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output path for result.json (file or directory)."),
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
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show binary output."),
    ] = False,
):
    """Run the analytical (fast, approximate) network simulation."""
    from simai.backends.analytical import run_analytical
    from simai.config import SimaiConfig, load_config, merge_cli, to_flat_dict

    # Load base config from TOML file if given, else use defaults
    cfg: SimaiConfig = load_config(config_file) if config_file else SimaiConfig()

    # CLI flags override TOML values
    merge_cli(
        cfg,
        output=str(output) if output else None,
        verbose=verbose or None,
        workload=str(workload) if workload else None,
        topology=str(topology) if topology else None,
        dp_overlap=dp_overlap,
        tp_overlap=tp_overlap,
        ep_overlap=ep_overlap,
        pp_overlap=pp_overlap,
    )
    if verbose:
        cfg.run.verbose = True

    # Resolve workload and topology
    wl_path = Path(cfg.workload.file) if cfg.workload.file else workload
    topo_path = Path(cfg.topology.file) if cfg.topology.file else topology

    if wl_path is None:
        raise typer.BadParameter("--workload / -w is required (or set workload.file in TOML config)")
    if topo_path is None:
        raise typer.BadParameter("--topology / -n is required (or set topology.file in TOML config)")

    # Handle on-the-fly workload generation from TOML [workload] section
    if cfg.workload.file is None and wl_path is None:
        wl_path = _generate_workload_from_config(cfg)

    # Handle on-the-fly topology generation from TOML [topology] section
    if cfg.topology.file is None and topo_path is None:
        topo_path = _generate_topology_from_config(cfg)

    topo = _load_topology(topo_path)
    num_gpus = _parse_workload_gpu_count(wl_path) or topo.get("num_gpus")
    topo_num_gpus = topo.get("num_gpus")
    if topo_num_gpus and num_gpus != topo_num_gpus:
        print(
            f"Warning: GPU count mismatch: workload has {num_gpus} GPUs "
            f"but topology has {topo_num_gpus} GPUs."
        )

    effective_output = Path(cfg.run.output) if cfg.run.output else output
    config_flat = to_flat_dict(cfg)

    sim_result = run_analytical(
        workload=wl_path,
        num_gpus=num_gpus or topo.get("num_gpus", 1),
        gpus_per_server=topo.get("gpus_per_server", 8),
        nvlink_bandwidth=topo.get("nvlink_bandwidth_gbps"),
        nic_bandwidth=topo.get("nic_bandwidth_gbps"),
        nics_per_server=topo.get("nics_per_switch"),
        gpu_type=topo.get("gpu_type"),
        dp_overlap=dp_overlap if dp_overlap is not None else cfg.analytical.dp_overlap,
        tp_overlap=tp_overlap if tp_overlap is not None else cfg.analytical.tp_overlap,
        ep_overlap=ep_overlap if ep_overlap is not None else cfg.analytical.ep_overlap,
        pp_overlap=pp_overlap if pp_overlap is not None else cfg.analytical.pp_overlap,
        result_prefix=result_prefix,
        output=effective_output,
        verbose=cfg.run.verbose,
        preserve_raw=True,
    )

    _write_result("analytical", topo, wl_path, sim_result, config_flat, effective_output)


@app.command()
def ns3(
    workload: Annotated[
        Optional[Path],
        typer.Option("--workload", "-w", help="Path to workload file."),
    ] = None,
    topology: Annotated[
        Optional[Path],
        typer.Option("--topology", "-n", help="Path to topology.json (or legacy directory)."),
    ] = None,
    config_file: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="Path to TOML run config OR SimAI.conf file."),
    ] = None,
    threads: Annotated[
        int,
        typer.Option("--threads", "-t", help="Number of simulation threads."),
    ] = 8,
    send_latency: Annotated[
        int,
        typer.Option("--send-latency", help="Send latency in microseconds (default: 0)."),
    ] = 0,
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
        typer.Option("--output", "-o", help="Output path for result.json (file or directory)."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show binary output."),
    ] = False,
):
    """Run the NS-3 (detailed, packet-level) network simulation."""
    from simai.backends.ns3 import run_ns3
    from simai.config import SimaiConfig, load_config, merge_cli, to_flat_dict, ns3_config_to_simai_conf_dict

    # Determine if config_file is TOML or legacy SimAI.conf
    toml_cfg: SimaiConfig | None = None
    simai_conf_path: Path | None = None
    ns3_conf_dict: dict | None = None

    if config_file is not None:
        config_file = Path(config_file)
        if config_file.suffix == ".toml":
            toml_cfg = load_config(config_file)
        else:
            # Legacy SimAI.conf file path
            simai_conf_path = config_file

    cfg: SimaiConfig = toml_cfg or SimaiConfig()
    # Apply CLI overrides; threads go to ns3 section
    if threads != 8:
        cfg.ns3.threads = threads
    merge_cli(
        cfg,
        output=str(output) if output else None,
        verbose=verbose or None,
        workload=str(workload) if workload else None,
        topology=str(topology) if topology else None,
        send_latency=send_latency,
        nvls=nvls if nvls else None,
        pxn=pxn if pxn else None,
    )
    if verbose:
        cfg.run.verbose = True

    # If TOML config was loaded, use [ns3] section for SimAI.conf
    if toml_cfg is not None and simai_conf_path is None:
        ns3_conf_dict = ns3_config_to_simai_conf_dict(cfg.ns3)

    wl_path = Path(cfg.workload.file) if cfg.workload.file else workload
    topo_path = Path(cfg.topology.file) if cfg.topology.file else topology

    if wl_path is None:
        raise typer.BadParameter("--workload / -w is required (or set workload.file in TOML config)")
    if topo_path is None:
        raise typer.BadParameter("--topology / -n is required (or set topology.file in TOML config)")

    topo = _load_topology(topo_path)

    # Unpack topology to NS3 text format in a temp dir
    tmp_topo_dir = tempfile.mkdtemp(prefix="simai_ns3_topo_")
    topo_file = Path(tmp_topo_dir) / "topology"
    from simai.topology.format import unpack_for_ns3
    unpack_for_ns3(topo, topo_file)

    effective_output = Path(cfg.run.output) if cfg.run.output else output
    config_flat = to_flat_dict(cfg)

    sim_result = run_ns3(
        workload=wl_path,
        topology=topo_file,
        config=simai_conf_path,
        ns3_conf=ns3_conf_dict,
        threads=cfg.ns3.threads,
        send_latency=cfg.ns3.send_latency,
        nvls=cfg.ns3.nvls,
        pxn=cfg.ns3.pxn,
        output=effective_output,
        verbose=cfg.run.verbose,
        preserve_raw=True,
    )

    import shutil
    shutil.rmtree(tmp_topo_dir, ignore_errors=True)

    _write_result("ns3", topo, wl_path, sim_result, config_flat, effective_output)


@app.command()
def m4(
    workload: Annotated[
        Optional[Path],
        typer.Option("--workload", "-w", help="Path to workload file."),
    ] = None,
    topology: Annotated[
        Optional[Path],
        typer.Option("--topology", "-n", help="Path to topology.json (or legacy directory)."),
    ] = None,
    config_file: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="Path to TOML run config file."),
    ] = None,
    threads: Annotated[
        int,
        typer.Option("--threads", "-t", help="Number of simulation threads."),
    ] = 1,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output path for result.json (file or directory)."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show binary output."),
    ] = False,
):
    """Run the M4 (flow-level, ML-based) network simulation."""
    from simai.backends.m4 import run_m4
    from simai.config import SimaiConfig, load_config, merge_cli, to_flat_dict

    cfg: SimaiConfig = load_config(config_file) if config_file else SimaiConfig()
    if threads != 1:
        cfg.m4.threads = threads
    merge_cli(
        cfg,
        output=str(output) if output else None,
        verbose=verbose or None,
        workload=str(workload) if workload else None,
        topology=str(topology) if topology else None,
    )
    if verbose:
        cfg.run.verbose = True

    wl_path = Path(cfg.workload.file) if cfg.workload.file else workload
    topo_path = Path(cfg.topology.file) if cfg.topology.file else topology

    if wl_path is None:
        raise typer.BadParameter("--workload / -w is required (or set workload.file in TOML config)")
    if topo_path is None:
        raise typer.BadParameter("--topology / -n is required (or set topology.file in TOML config)")

    topo = _load_topology(topo_path)

    effective_output = Path(cfg.run.output) if cfg.run.output else output
    config_flat = to_flat_dict(cfg)

    sim_result = run_m4(
        workload=wl_path,
        topology_file=topo_path,
        topology_dict=topo,
        threads=cfg.m4.threads,
        output=effective_output,
        verbose=cfg.run.verbose,
        preserve_raw=True,
    )

    _write_result("m4", topo, wl_path, sim_result, config_flat, effective_output)


# ---------------------------------------------------------------------------
# Helpers for on-the-fly generation from TOML config
# ---------------------------------------------------------------------------

def _generate_workload_from_config(cfg) -> Path:
    """Generate workload from TOML [workload] section inline parameters."""
    import tempfile
    from simai.workflow.generator import generate_workload

    tmp = tempfile.mkdtemp(prefix="simai_wl_")
    out = Path(tmp) / "workload.txt"
    wl = cfg.workload
    return generate_workload(
        framework=wl.framework,
        world_size=wl.world_size,
        tensor_model_parallel_size=wl.tensor_parallel,
        pipeline_model_parallel=wl.pipeline_parallel,
        expert_model_parallel_size=wl.expert_parallel,
        global_batch=wl.global_batch,
        micro_batch=wl.micro_batch,
        num_layers=wl.num_layers,
        hidden_size=wl.hidden_size,
        seq_length=wl.seq_length,
        num_attention_heads=wl.num_attention_heads,
        vocab_size=wl.vocab_size,
        gpu_type=wl.gpu_type,
        use_flash_attn=wl.use_flash_attn,
        output=out,
    )


def _generate_topology_from_config(cfg) -> Path:
    """Generate topology from TOML [topology] section inline parameters."""
    import tempfile
    from simai.topology.generator import generate_topology

    if cfg.topology.type is None:
        raise typer.BadParameter(
            "No topology file or inline parameters provided. "
            "Set topology.file or topology.type in the TOML config."
        )
    tmp = tempfile.mkdtemp(prefix="simai_topo_")
    out = Path(tmp) / "topology.json"
    return generate_topology(
        topology_type=cfg.topology.type,
        num_gpus=cfg.topology.num_gpus,
        gpus_per_server=cfg.topology.gpus_per_server,
        nic_bandwidth=f"{cfg.topology.nic_bandwidth_gbps}Gbps" if cfg.topology.nic_bandwidth_gbps else None,
        nvlink_bandwidth=f"{cfg.topology.nvlink_bandwidth_gbps}Gbps" if cfg.topology.nvlink_bandwidth_gbps else None,
        nics_per_switch=cfg.topology.nics_per_switch,
        output=out,
    )
