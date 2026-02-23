"""Result JSON builder and CSV parsers for SimAI simulation outputs."""
from __future__ import annotations

import csv
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from importlib.metadata import version as _pkg_version
    _SIMAI_VERSION = _pkg_version("simai")
except Exception:
    _SIMAI_VERSION = "unknown"


# ---------------------------------------------------------------------------
# CSV parsers
# ---------------------------------------------------------------------------

def parse_analytical_csv(path: Path) -> dict:
    """Parse the analytical backend EndToEnd CSV into structured result data.

    The file is space/tab separated with a header row. Returns a dict with:
      - layers: list of {name, exposed_comm_us, compute_us, ...}
      - summary: {total_time_us, total_exposed_comm_us, total_compute_us}
    """
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return {"layers": [], "summary": None}

    # Find the EndToEnd CSV file
    if path.is_dir():
        candidates = list(path.glob("*EndToEnd*.csv")) + list(path.glob("*.csv"))
        if not candidates:
            return {"layers": [], "summary": None}
        path = candidates[0]

    layers = []
    with open(path, newline="") as f:
        # The file may use spaces/tabs as delimiters
        content = f.read()

    # Normalize: replace tabs with commas, collapse multiple spaces to one comma
    lines = content.splitlines()
    if not lines:
        return {"layers": [], "summary": None}

    # Detect delimiter from first line
    header_line = lines[0]
    if "\t" in header_line:
        delimiter = "\t"
    elif "," in header_line:
        delimiter = ","
    else:
        delimiter = None  # whitespace

    if delimiter:
        reader = csv.DictReader(lines, delimiter=delimiter)
    else:
        # Whitespace-delimited: split manually
        header = lines[0].split()
        rows = [dict(zip(header, ln.split())) for ln in lines[1:] if ln.strip()]
        reader = rows

    total_time = 0.0
    total_comm = 0.0
    total_compute = 0.0

    for row in reader:
        if not row:
            continue
        # Normalize keys: lowercase and strip
        row = {k.strip().lower(): v for k, v in (row.items() if hasattr(row, "items") else row.items())}

        # Try to extract layer name and timing fields
        # Column names vary by binary version — use flexible matching
        name = _pick(row, "layer", "name", "layer_name", "") or ""
        comm = _float_or(row, "exposed_comm", "exposed_comm_us", "comm", "communication", 0.0)
        compute = _float_or(row, "compute", "compute_us", "kernel", "kernel_us", 0.0)
        total = _float_or(row, "total", "total_us", "time", "elapsed", 0.0)

        layers.append({
            "name": name,
            "exposed_comm_us": comm,
            "compute_us": compute,
            "total_us": total,
        })
        total_comm += comm
        total_compute += compute
        total_time += total if total else (comm + compute)

    summary = {
        "total_time_us": total_time,
        "total_exposed_comm_us": total_comm,
        "total_compute_us": total_compute,
    }
    return {"layers": layers, "summary": summary}


def _pick(d: dict, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return default


def _float_or(d: dict, *keys: str, default: float = 0.0) -> float:
    for k in keys:
        if k in d:
            try:
                return float(d[k])
            except (ValueError, TypeError):
                pass
    return default


def parse_ns3_results(result_dir: Path) -> dict:
    """Parse NS-3 result directory: EndToEnd CSV + optional utilization CSVs.

    Missing or empty files are represented as null rather than included.
    """
    result_dir = Path(result_dir)
    layers_data = {"layers": [], "summary": None}
    link_utilization = None
    flow_completion = None

    # EndToEnd CSV
    end_to_end = next(
        (f for f in result_dir.glob("*EndToEnd*.csv") if f.stat().st_size > 0),
        None,
    )
    if end_to_end:
        layers_data = parse_analytical_csv(end_to_end)

    # Flow completion time file
    fct = next(
        (f for f in result_dir.glob("fct*") if f.stat().st_size > 0),
        None,
    )
    if fct:
        flow_completion = _parse_simple_csv(fct)

    # Link utilization / bandwidth monitoring
    bw = next(
        (f for f in result_dir.glob("bw*") if f.stat().st_size > 0),
        None,
    )
    if bw:
        link_utilization = _parse_simple_csv(bw)

    return {
        **layers_data,
        "link_utilization": link_utilization,
        "flow_completion": flow_completion,
    }


def parse_m4_results(result_dir: Path) -> dict:
    """Parse M4 result directory, similar to NS-3 parsing."""
    return parse_ns3_results(result_dir)


def _parse_simple_csv(path: Path) -> list[dict] | None:
    """Parse a simple CSV/whitespace-separated file into a list of dicts."""
    try:
        lines = path.read_text().splitlines()
        if not lines:
            return None
        header_line = lines[0]
        delimiter = "\t" if "\t" in header_line else ("," if "," in header_line else None)
        if delimiter:
            reader = csv.DictReader(lines, delimiter=delimiter)
            return [dict(row) for row in reader]
        else:
            header = lines[0].split()
            return [
                dict(zip(header, ln.split()))
                for ln in lines[1:]
                if ln.strip()
            ]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Workload header parser
# ---------------------------------------------------------------------------

def parse_workload_header(path: Path) -> dict:
    """Extract the comment-header fields from a workload .txt file."""
    result = {}
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            # Lines like: # all_gpus: 128, tp: 8, ...
            for match in re.finditer(r"(\w+):\s*([\w.]+)", line):
                key, val = match.group(1), match.group(2)
                try:
                    result[key] = int(val)
                except ValueError:
                    try:
                        result[key] = float(val)
                    except ValueError:
                        result[key] = val
    return result


# ---------------------------------------------------------------------------
# Result JSON assembly
# ---------------------------------------------------------------------------

def build_result_json(
    *,
    backend: str,
    config: dict,
    topology_meta: dict,
    workload_header: dict,
    results: dict,
    raw_path: str | Path,
) -> dict:
    """Assemble the full result dict."""
    # Strip internal fields from topology meta before storing
    topo_clean = {k: v for k, v in topology_meta.items() if not k.startswith("_")}

    return {
        "simai_version": _SIMAI_VERSION,
        "run_id": uuid.uuid4().hex[:8],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "config": config,
        "topology_metadata": topo_clean,
        "workload_header": workload_header,
        "results": {
            "layers": results.get("layers", []),
            "summary": results.get("summary"),
            "link_utilization": results.get("link_utilization"),
            "flow_completion": results.get("flow_completion"),
        },
        "raw_output_path": str(raw_path),
    }


def write_result_json(data: dict, output: Path) -> None:
    """Write result dict to a JSON file."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Result saved to: {output}")
