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
# Value helpers
# ---------------------------------------------------------------------------

def _parse_value(s: str) -> float:
    """Extract the leading integer/float from a value like '225737 (2.16%)' or '10451678'."""
    s = s.strip()
    m = re.match(r"^([\d.]+)", s)
    if m:
        return float(m.group(1))
    return 0.0


def _pick(d: dict, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return default


def _val_or(d: dict, *keys: str, default: float = 0.0) -> float:
    """Return the parsed numeric value for the first matching key."""
    for k in keys:
        if k in d:
            try:
                return _parse_value(str(d[k]))
            except (ValueError, TypeError):
                pass
    return default


# ---------------------------------------------------------------------------
# EndToEnd CSV column name normalizer
# ---------------------------------------------------------------------------

# Maps normalized column names → canonical field names in result JSON
_ENDTOEND_COL_MAP = {
    # Summary-row columns (analytical + NS3 EndToEnd CSV)
    "expose dp comm":          "expose_dp_comm_us",
    "expose dp_ep comm":       "expose_dp_ep_comm_us",
    "expose tp comm":          "expose_tp_comm_us",
    "expose_ep_comm":          "expose_ep_comm_us",
    "expose ep comm":          "expose_ep_comm_us",
    "expose_pp_comm":          "expose_pp_comm_us",
    "expose pp comm":          "expose_pp_comm_us",
    "bubble time":             "bubble_time_us",
    "total comp":              "total_comp_us",
    "total exposed comm":      "total_exposed_comm_us",
    "total time":              "total_time_us",
    # Per-layer columns
    "file name":               "name",
    "layer":                   "name",
    "layer_name":              "name",
    "name":                    "name",
    "exposed_comm":            "exposed_comm_us",
    "exposed_comm_us":         "exposed_comm_us",
    "comm":                    "exposed_comm_us",
    "compute":                 "compute_us",
    "compute_us":              "compute_us",
    "kernel":                  "compute_us",
    "total":                   "total_us",
    "total_us":                "total_us",
    "elapsed":                 "total_us",
}

# Columns that belong to the summary row (not per-layer)
_SUMMARY_FIELDS = {
    "expose_dp_comm_us", "expose_dp_ep_comm_us", "expose_tp_comm_us",
    "expose_ep_comm_us", "expose_pp_comm_us", "bubble_time_us",
    "total_comp_us", "total_exposed_comm_us", "total_time_us",
}


def _normalize_col(col: str) -> str:
    return col.strip().lower().replace("-", "_").replace("  ", " ")


def _parse_endtoend_csv(path: Path) -> dict:
    """Parse an EndToEnd CSV file produced by any SimAI backend.

    The CSV has a header row followed by data rows. The first row of data is
    typically a summary row with fields like 'Expose DP comm', 'total comp',
    etc., where values may be formatted as '225737 (2.16%)' — the percentage
    is discarded and only the raw integer value is kept.

    Returns:
        {
            "summary": {
                "expose_dp_comm_us": ...,
                "expose_dp_ep_comm_us": ...,
                "expose_tp_comm_us": ...,
                "expose_ep_comm_us": ...,
                "expose_pp_comm_us": ...,
                "bubble_time_us": ...,
                "total_comp_us": ...,
                "total_exposed_comm_us": ...,
                "total_time_us": ...,
            },
            "layers": [
                {"name": ..., "exposed_comm_us": ..., "compute_us": ..., "total_us": ...},
                ...
            ]
        }
    """
    content = path.read_text()
    lines = [ln for ln in content.splitlines() if ln.strip()]
    if not lines:
        return {"summary": None, "layers": []}

    # Detect delimiter
    header_line = lines[0]
    if "," in header_line:
        delimiter = ","
    elif "\t" in header_line:
        delimiter = "\t"
    else:
        delimiter = None

    if delimiter:
        reader = list(csv.DictReader(lines, delimiter=delimiter))
    else:
        header = lines[0].split()
        reader = [dict(zip(header, ln.split())) for ln in lines[1:] if ln.strip()]

    summary: dict[str, Any] = {}
    layers: list[dict] = []

    for row in reader:
        # Normalize all column names
        norm = {_normalize_col(k): v for k, v in row.items() if k is not None}
        canonical = {}
        for col, val in norm.items():
            mapped = _ENDTOEND_COL_MAP.get(col)
            if mapped:
                canonical[mapped] = val

        # Determine whether this row is a summary row or a per-layer row.
        # A row is a summary row if it has any summary-specific fields.
        has_summary_fields = any(k in canonical for k in _SUMMARY_FIELDS)

        if has_summary_fields:
            for field in _SUMMARY_FIELDS:
                if field in canonical:
                    summary[field] = _parse_value(str(canonical[field]))
                else:
                    summary.setdefault(field, 0)
        else:
            name = canonical.get("name", "")
            if name is None:
                name = ""
            layers.append({
                "name": str(name).strip(),
                "exposed_comm_us": _val_or(canonical, "exposed_comm_us"),
                "compute_us": _val_or(canonical, "compute_us"),
                "total_us": _val_or(canonical, "total_us"),
            })

    return {"summary": summary if summary else None, "layers": layers}


# ---------------------------------------------------------------------------
# Public parsers
# ---------------------------------------------------------------------------

def parse_analytical_csv(path: Path) -> dict:
    """Parse the analytical backend EndToEnd CSV.

    Accepts either a file path or a directory (auto-finds the EndToEnd CSV).
    """
    path = Path(path)
    if path.is_dir():
        candidates = sorted(path.glob("*EndToEnd*.csv"))
        if not candidates:
            candidates = sorted(path.glob("*.csv"))
        if not candidates:
            return {"summary": None, "layers": []}
        path = candidates[0]

    if not path.is_file() or path.stat().st_size == 0:
        return {"summary": None, "layers": []}

    return _parse_endtoend_csv(path)


def parse_ns3_results(result_dir: Path) -> dict:
    """Parse NS-3 result directory: EndToEnd CSV + optional utilization CSVs."""
    result_dir = Path(result_dir)

    end_to_end = next(
        (f for f in result_dir.glob("*EndToEnd*.csv") if f.stat().st_size > 0),
        None,
    )
    layers_data = _parse_endtoend_csv(end_to_end) if end_to_end else {"summary": None, "layers": []}

    fct = next(
        (f for f in result_dir.glob("fct*") if f.stat().st_size > 0), None
    )
    bw = next(
        (f for f in result_dir.glob("bw*") if f.stat().st_size > 0), None
    )

    return {
        **layers_data,
        "link_utilization": _parse_simple_csv(bw) if bw else None,
        "flow_completion": _parse_simple_csv(fct) if fct else None,
    }


def parse_m4_results(result_dir: Path) -> dict:
    """Parse M4 result directory — same structure as NS-3."""
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
            "summary": results.get("summary"),
            "layers": results.get("layers", []),
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
