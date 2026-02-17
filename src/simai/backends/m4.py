from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from simai.backends.binary import run_binary

BINARY_NAME = "SimAI_m4"


_M4_MODELS_REL = (
    "astra-sim-alibabacloud", "astra-sim", "network_frontend", "m4", "models"
)
_M4_CACHE_DIR = Path.home() / ".cache" / "simai" / "simai-m4"


def _find_m4_models() -> Path | None:
    """Find the m4 .pt model files directory.

    Search order:
    1. Cache populated by `simai install m4`: ~/.cache/simai/simai-m4/...
    2. Editable install: vendor/simai-m4/...
    3. SIMAI_PATH env var
    """
    # 1. Cache from `simai install m4`
    cached = _M4_CACHE_DIR.joinpath(*_M4_MODELS_REL)
    if cached.is_dir():
        return cached

    # 2. Editable install: __file__ is src/simai/backends/m4.py → project root is 4 levels up
    editable = Path(__file__).resolve().parent.parent.parent.parent / "vendor" / "simai-m4"
    editable = editable.joinpath(*_M4_MODELS_REL)
    if editable.is_dir():
        return editable

    # 3. SIMAI_PATH env var
    env_path = os.environ.get("SIMAI_PATH")
    if env_path:
        candidate = Path(env_path).joinpath(*_M4_MODELS_REL)
        if candidate.is_dir():
            return candidate

    return None


def _find_libtorch_lib_dir() -> str | None:
    """Return the directory containing LibTorch .so files (from the torch package)."""
    try:
        import torch

        lib_dir = Path(torch.__file__).parent / "lib"
        if lib_dir.is_dir():
            return str(lib_dir)
    except ImportError:
        pass
    return None


def _convert_topology(src: Path, dst: Path) -> None:
    """Convert topology file to m4 format.

    Converts bandwidth from raw bps integers to XGbps strings, and
    latency from plain float seconds to Xms strings.
    If the values already carry units (e.g. '7200Gbps', '0.025ms'), they are
    passed through unchanged.

    Format (lines 3+): src_node dst_node bw_bps_or_Gbps latency_sec_or_ms err_rate
    """
    with open(src) as f:
        lines = f.readlines()

    with open(dst, "w") as f:
        for i, line in enumerate(lines):
            if i < 2:
                # Line 0: header, line 1: switch ids — pass through unchanged
                f.write(line)
                continue
            parts = line.strip().split()
            if len(parts) < 5:
                f.write(line)
                continue
            src_node, dst_node = parts[0], parts[1]
            bw_raw, lat_raw, err_rate = parts[2], parts[3], parts[4]

            if bw_raw.lower().endswith("gbps"):
                bw_out = bw_raw
            else:
                bw_out = f"{float(bw_raw) / 1e9:g}Gbps"

            if lat_raw.lower().endswith("ms"):
                lat_out = lat_raw
            else:
                lat_out = f"{float(lat_raw) * 1e3:g}ms"

            f.write(f"{src_node} {dst_node} {bw_out} {lat_out} {err_rate}\n")


def run_m4(
    *,
    workload: Path,
    topology_file: Path,
    threads: int = 1,
    output: Path | None = None,
    verbose: bool = False,
) -> Path:
    """Run the SimAI M4 (flow-level, ML-based) simulator backend.

    Steps:
    1. Convert topology to m4 format (Gbps/ms units) in a temp directory.
    2. Symlink model files to the path the binary hardcodes relative to cwd.
    3. Run the binary from the temp directory.
    4. Move results to the user's chosen output path.

    Returns the output directory path.
    """
    workload = workload.resolve()
    topology_file = topology_file.resolve()

    output_path = Path(output).resolve() if output else Path("results").resolve()

    env: dict[str, str] = {}
    libtorch_dir = _find_libtorch_lib_dir()
    if libtorch_dir:
        existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{libtorch_dir}:{existing_ld}" if existing_ld else libtorch_dir

    with tempfile.TemporaryDirectory(prefix="simai_m4_") as tmpdir:
        tmppath = Path(tmpdir)

        # Convert topology to m4 format
        converted_topo = tmppath / "topology_m4"
        _convert_topology(topology_file, converted_topo)

        # Symlink model files to the hardcoded relative path the binary expects
        models_dir = _find_m4_models()
        if models_dir:
            models_target = (
                tmppath
                / "astra-sim-alibabacloud"
                / "astra-sim"
                / "network_frontend"
                / "m4"
                / "models"
            )
            models_target.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(models_dir, models_target)
        else:
            print(
                "Warning: m4 model files not found. The binary may fail to load models.\n"
                "Install PyTorch and ensure model files are available, or set SIMAI_PATH."
            )

        # Create output dir inside tmpdir
        binary_output_dir = tmppath / "output"
        binary_output_dir.mkdir()

        args: list[str] = [
            "-t", str(threads),
            "-w", str(workload),
            "-n", str(converted_topo),
            "-o", str(binary_output_dir),
        ]

        try:
            run_binary(BINARY_NAME, args, cwd=tmpdir, env=env, verbose=verbose)
        except FileNotFoundError:
            raise FileNotFoundError(
                "SimAI_m4 binary not found.\n"
                "Install it with:\n"
                "  pip install \"simai[m4]\"   # if not already installed\n"
                "  simai install m4\n"
                "Or set SIMAI_BIN_PATH to the directory containing SimAI_m4."
            )

        # Collect result files (prefer binary_output_dir, fall back to tmpdir)
        result_files = [
            f for f in binary_output_dir.iterdir()
        ] if binary_output_dir.exists() else []

        if not result_files:
            # Some versions write results directly to cwd
            result_files = [
                f for f in tmppath.iterdir()
                if f.name not in ("topology_m4", "astra-sim-alibabacloud", "output")
            ]

        if not result_files:
            print("Warning: no result files generated")
            return output_path

        if output_path.suffix and not output_path.is_dir():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(result_files[0]), str(output_path))
            for item in result_files[1:]:
                if item.exists():
                    shutil.move(str(item), str(output_path.parent / item.name))
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            for item in result_files:
                dest = output_path / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(dest))

        print(f"Results saved to: {output_path}")

    return output_path
