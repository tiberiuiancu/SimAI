from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path


def _find_topo_root() -> Path:
    """Locate the topology generator script.

    Search order:
    1. Vendored into the package at build time: simai/_vendor/topo/
    2. SIMAI_PATH environment variable (points to the SimAI repo root)
    3. Sibling directory heuristic
    """
    # 1. Vendored
    vendored = Path(__file__).resolve().parent.parent / "_vendor" / "topo"
    if (vendored / "gen_Topo_Template.py").is_file():
        return vendored

    # 2. SIMAI_PATH env var
    env_path = os.environ.get("SIMAI_PATH")
    if env_path:
        candidate = Path(env_path) / "astra-sim-alibabacloud" / "inputs" / "topo"
        if (candidate / "gen_Topo_Template.py").is_file():
            return candidate

    # 3. Sibling directory heuristic
    sibling = (
        Path(__file__).resolve().parent.parent.parent.parent.parent
        / "simai"
        / "astra-sim-alibabacloud"
        / "inputs"
        / "topo"
    )
    if (sibling / "gen_Topo_Template.py").is_file():
        return sibling

    raise FileNotFoundError(
        "Cannot find gen_Topo_Template.py. Either:\n"
        "  - Install from a wheel that includes vendored topology generator, or\n"
        "  - Set SIMAI_PATH to the SimAI repository root, or\n"
        "  - Clone SimAI as a sibling directory."
    )


@contextmanager
def _topo_on_path():
    """Temporarily add the topology generator root to sys.path."""
    topo_root = str(_find_topo_root())
    inserted = topo_root not in sys.path
    if inserted:
        sys.path.insert(0, topo_root)
    try:
        yield topo_root
    finally:
        if inserted and topo_root in sys.path:
            sys.path.remove(topo_root)


def _parse_bandwidth(s: str) -> float:
    """Convert a bandwidth string like '100Gbps' to a float (100.0)."""
    m = re.match(r"^([\d.]+)\s*[Gg]bps$", s)
    if m:
        return float(m.group(1))
    raise ValueError(f"Cannot parse bandwidth string: {s!r} (expected format: '100Gbps')")


def generate_topology(
    *,
    topology_type: str,
    num_gpus: int | None = None,
    gpus_per_server: int | None = None,
    gpu_type: str | None = None,
    nic_bandwidth: str | None = None,
    nvlink_bandwidth: str | None = None,
    nics_per_switch: int | None = None,
    aggregate_switches: int | None = None,
    pod_switches: int | None = None,
    aggregate_bandwidth: str | None = None,
    switches_per_pod: int | None = None,
    nv_switches_per_server: int | None = None,
    nvlink_latency: str | None = None,
    nic_latency: str | None = None,
    error_rate: str | None = None,
    dual_tor: bool = False,
    dual_plane: bool = False,
    output: Path | None = None,
) -> Path:
    """Generate a network topology directory.

    Creates a directory containing:
    - topology: the NS3 network topology file
    - metadata.json: generation parameters for simulate to read

    Returns the output directory Path.
    """
    with _topo_on_path():
        from gen_Topo_Template import analysis_template, main  # noqa: F811

        # Build a mock argparse.Namespace matching the upstream script's expectations
        args = argparse.Namespace(
            topology=topology_type,
            ro=False,  # analysis_template sets this based on topology type
            dt=dual_tor,
            dp=dual_plane,
            gpu=num_gpus,
            error_rate=error_rate,
            gpu_per_server=gpus_per_server,
            gpu_type=gpu_type,
            nv_switch_per_server=nv_switches_per_server,
            nvlink_bw=nvlink_bandwidth,
            nv_latency=nvlink_latency,
            latency=nic_latency,
            bandwidth=nic_bandwidth,
            asw_switch_num=aggregate_switches,
            nics_per_aswitch=nics_per_switch,
            psw_switch_num=pod_switches,
            ap_bandwidth=aggregate_bandwidth,
            asw_per_psw=switches_per_pod,
        )

        # Run analysis_template to get the merged parameters dict
        parameters = analysis_template(args, [])

        # Determine which generation function to call (mirrors main() logic)
        from gen_Topo_Template import (
            No_Rail_Opti_DualToR,
            No_Rail_Opti_SingleToR,
            Rail_Opti_DualToR_DualPlane,
            Rail_Opti_DualToR_SinglePlane,
            Rail_Opti_SingleToR,
        )

        if not parameters["rail_optimized"]:
            if parameters["dual_plane"]:
                raise ValueError("Non rail-optimized structure doesn't support dual plane")
            gen_func = No_Rail_Opti_DualToR if parameters["dual_ToR"] else No_Rail_Opti_SingleToR
        else:
            if parameters["dual_ToR"]:
                gen_func = (
                    Rail_Opti_DualToR_DualPlane
                    if parameters["dual_plane"]
                    else Rail_Opti_DualToR_SinglePlane
                )
            else:
                if parameters["dual_plane"]:
                    raise ValueError(
                        "Rail-optimized single-ToR structure doesn't support dual plane"
                    )
                gen_func = Rail_Opti_SingleToR

        # Run from a temp dir (upstream writes to cwd)
        with tempfile.TemporaryDirectory(prefix="simai_topo_") as tmpdir:
            orig_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                gen_func(parameters)
            finally:
                os.chdir(orig_cwd)

            # Find the generated topology file (there should be exactly one)
            generated = [f for f in Path(tmpdir).iterdir() if f.is_file()]
            if not generated:
                raise RuntimeError("Topology generation produced no output file")
            topo_file = generated[0]

            # Determine output directory
            if output is not None:
                output_dir = Path(output).resolve()
            else:
                output_dir = Path(topo_file.name).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)

            # Move topology file
            shutil.move(str(topo_file), str(output_dir / "topology"))

        # Write metadata.json
        metadata = {
            "type": topology_type,
            "num_gpus": parameters["gpu"],
            "gpus_per_server": parameters["gpu_per_server"],
            "gpu_type": parameters["gpu_type"],
            "nic_bandwidth_gbps": _parse_bandwidth(parameters["bandwidth"]),
            "nvlink_bandwidth_gbps": _parse_bandwidth(parameters["nvlink_bw"]),
            "nics_per_switch": parameters["nics_per_aswitch"],
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            f.write("\n")

        print(f"Topology saved to: {output_dir}")
        return output_dir
