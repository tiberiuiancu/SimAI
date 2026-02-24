"""Topology format helpers: load topology.json and unpack for each backend."""
from __future__ import annotations

import json
import warnings
from pathlib import Path


def load_topology(path: Path) -> dict:
    """Load topology from topology.json or legacy directory format.

    If a directory is given that contains metadata.json + topology, the legacy
    format is read with a deprecation warning.
    Returns a dict with all topology metadata and parsed link data.
    """
    path = Path(path)

    if path.is_dir():
        meta_path = path / "metadata.json"
        topo_path = path / "topology"
        if meta_path.is_file() and topo_path.is_file():
            warnings.warn(
                f"Loading topology from legacy directory format: {path}\n"
                "Regenerate with 'simai generate topology' to get a single topology.json.",
                DeprecationWarning,
                stacklevel=2,
            )
            with open(meta_path) as f:
                meta = json.load(f)
            topo_text = topo_path.read_text()
            parsed = _parse_topology_text(topo_text)
            meta.update(parsed)
            meta["_topology_text"] = topo_text
            return meta
        raise FileNotFoundError(
            f"No metadata.json + topology found in directory: {path}\n"
            "Did you generate this topology with 'simai generate topology'?"
        )

    if path.is_file():
        with open(path) as f:
            return json.load(f)

    raise FileNotFoundError(f"Topology path not found: {path}")


def _parse_topology_text(text: str) -> dict:
    """Parse the NS3 topology text format into structured data."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return {"total_nodes": 0, "switch_ids": [], "links": []}

    # Line 0: header (total_nodes or "total_nodes num_links")
    parts0 = lines[0].split()
    total_nodes = int(parts0[0])

    # Line 1: switch ids
    switch_ids = [int(x) for x in lines[1].split()]

    # Lines 2+: link definitions
    links = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 5:
            continue
        src, dst = int(parts[0]), int(parts[1])
        bw_raw, lat_raw, err = parts[2], parts[3], parts[4]

        # Bandwidth: handle both "7200Gbps" and raw bps integers
        if bw_raw.lower().endswith("gbps"):
            bw_gbps = float(bw_raw[:-4])
        else:
            bw_gbps = float(bw_raw) / 1e9

        # Latency: handle both "0.025ms" and raw float seconds
        if lat_raw.lower().endswith("ms"):
            lat_ms = float(lat_raw[:-2])
        else:
            lat_ms = float(lat_raw) * 1e3

        links.append({
            "src": src,
            "dst": dst,
            "bandwidth_gbps": bw_gbps,
            "latency_ms": lat_ms,
            "error_rate": float(err),
        })

    return {"total_nodes": total_nodes, "switch_ids": switch_ids, "links": links}


def unpack_for_ns3(topo: dict, dest: Path) -> None:
    """Write the topology file for the NS3 backend to dest.

    Uses the stored raw topology text if available (exact original format),
    otherwise reconstructs from links array.
    """
    raw = topo.get("_topology_text")
    if raw is not None:
        dest.write_text(raw)
        return

    # Reconstruct from structured data
    total_nodes = topo.get("total_nodes", 0)
    switch_ids = topo.get("switch_ids", [])
    links = topo.get("links", [])

    lines = [str(total_nodes)]
    lines.append(" ".join(str(s) for s in switch_ids))
    for link in links:
        bw = link["bandwidth_gbps"]
        # NS3 binary reads Gbps strings
        lat = link["latency_ms"] / 1e3  # back to seconds
        lines.append(f"{link['src']} {link['dst']} {bw}Gbps {lat} {link['error_rate']}")

    dest.write_text("\n".join(lines) + "\n")


def unpack_for_m4(topo: dict, dest: Path) -> None:
    """Write the topology file in M4 format (Gbps/ms units) to dest."""
    raw = topo.get("_topology_text")
    if raw is not None:
        # Convert from raw text (same logic as the old _convert_topology)
        result_lines = []
        text_lines = raw.splitlines()
        for i, line in enumerate(text_lines):
            if i < 2:
                result_lines.append(line)
                continue
            parts = line.strip().split()
            if len(parts) < 5:
                result_lines.append(line)
                continue
            src_node, dst_node = parts[0], parts[1]
            bw_raw, lat_raw, err_rate = parts[2], parts[3], parts[4]

            bw_out = bw_raw if bw_raw.lower().endswith("gbps") else f"{float(bw_raw) / 1e9:g}Gbps"
            lat_out = lat_raw if lat_raw.lower().endswith("ms") else f"{float(lat_raw) * 1e3:g}ms"

            result_lines.append(f"{src_node} {dst_node} {bw_out} {lat_out} {err_rate}")
        dest.write_text("\n".join(result_lines) + "\n")
        return

    # Reconstruct from structured data
    total_nodes = topo.get("total_nodes", 0)
    switch_ids = topo.get("switch_ids", [])
    links = topo.get("links", [])

    lines = [str(total_nodes)]
    lines.append(" ".join(str(s) for s in switch_ids))
    for link in links:
        bw = link["bandwidth_gbps"]
        lat = link["latency_ms"]
        lines.append(f"{link['src']} {link['dst']} {bw}Gbps {lat}ms {link['error_rate']}")

    dest.write_text("\n".join(lines) + "\n")


def topology_to_analytical_args(topo: dict) -> dict:
    """Return kwargs needed by run_analytical() from topology metadata."""
    return {
        "num_gpus": topo.get("num_gpus"),
        "gpus_per_server": topo.get("gpus_per_server"),
        "nvlink_bandwidth": topo.get("nvlink_bandwidth_gbps"),
        "nic_bandwidth": topo.get("nic_bandwidth_gbps"),
        "nics_per_server": topo.get("nics_per_switch"),
        "gpu_type": topo.get("gpu_type"),
    }
