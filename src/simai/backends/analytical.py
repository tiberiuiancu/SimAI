from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from simai.backends.binary import find_binary, run_binary

BINARY_NAME = "SimAI_analytical"


@dataclass
class SimulationResult:
    """Structured result from a simulation backend."""
    output_path: Path
    raw_output_path: Path
    parsed: dict


def _find_simai_root() -> Path | None:
    """Find the SimAI repo root for auxiliary data files.

    The binary needs astra-sim-alibabacloud/inputs/ratio/ CSV files.
    """
    # Check SIMAI_PATH env var
    env_path = os.environ.get("SIMAI_PATH")
    if env_path:
        candidate = Path(env_path)
        if (candidate / "astra-sim-alibabacloud").is_dir():
            return candidate

    # Check SIMAI_BIN_PATH parent (e.g. <simai>/bin/ → <simai>/)
    bin_path = os.environ.get("SIMAI_BIN_PATH")
    if bin_path:
        candidate = Path(bin_path).resolve().parent
        if (candidate / "astra-sim-alibabacloud").is_dir():
            return candidate

    # Check relative to the binary location (follow symlinks)
    try:
        binary = find_binary(BINARY_NAME)
        for parent in (binary.parent.parent, binary.resolve().parent.parent):
            if (parent / "astra-sim-alibabacloud").is_dir():
                return parent
    except FileNotFoundError:
        pass

    # Check vendored location (wheel install)
    vendored = Path(__file__).resolve().parent.parent / "_vendor"
    if (vendored / "astra-sim-alibabacloud").is_dir():
        return vendored

    # Check vendor submodule (editable install)
    # __file__ is at src/simai/backends/analytical.py → project root is 4 levels up
    vendor_sub = Path(__file__).resolve().parent.parent.parent.parent / "vendor" / "simai"
    if (vendor_sub / "astra-sim-alibabacloud").is_dir():
        return vendor_sub

    return None


def run_analytical(
    *,
    workload: Path,
    num_gpus: int,
    gpus_per_server: int = 8,
    nvlink_bandwidth: float | None = None,
    nic_bandwidth: float | None = None,
    nics_per_server: int | None = None,
    busbw: Path | None = None,
    gpu_type: str | None = None,
    dp_overlap: float | None = None,
    tp_overlap: float | None = None,
    ep_overlap: float | None = None,
    pp_overlap: float | None = None,
    result_prefix: str | None = None,
    output: Path | None = None,
    verbose: bool = False,
    preserve_raw: bool = True,
) -> SimulationResult:
    """Run the SimAI analytical backend.

    The binary hardcodes output to ./results/ and reads auxiliary data from
    ./astra-sim-alibabacloud/inputs/ratio/. We run it from a temp directory
    with symlinks to the required data.

    When preserve_raw=True (default), the raw binary output directory is kept
    alive after the function returns; its path is included in the result.
    When preserve_raw=False, the tmpdir is cleaned up.

    Returns a SimulationResult with output_path, raw_output_path, and parsed data.
    """
    workload = workload.resolve()

    # Build command-line arguments
    args: list[str] = [
        "-w", str(workload),
        "-g", str(num_gpus),
        "-g_p_s", str(gpus_per_server),
    ]

    if nvlink_bandwidth is not None:
        args += ["-nv", str(nvlink_bandwidth)]
    if nic_bandwidth is not None:
        args += ["-nic", str(nic_bandwidth)]
    if nics_per_server is not None:
        args += ["-n_p_s", str(nics_per_server)]
    if busbw is not None:
        args += ["-busbw", str(busbw.resolve())]
    if gpu_type is not None:
        args += ["-g_type", gpu_type]
    if dp_overlap is not None:
        args += ["-dp_o", str(dp_overlap)]
    if tp_overlap is not None:
        args += ["-tp_o", str(tp_overlap)]
    if ep_overlap is not None:
        args += ["-ep_o", str(ep_overlap)]
    if pp_overlap is not None:
        args += ["-pp_o", str(pp_overlap)]
    # Always pass -r to avoid SIGFPE in the binary's filename parser
    # when the workload filename doesn't match the expected pattern.
    if result_prefix is None:
        result_prefix = workload.stem
    args += ["-r", result_prefix]

    # Determine output path
    output_path = Path(output).resolve() if output else Path("results").resolve()

    # Run from a temp directory; optionally keep it alive for raw file access
    tmpdir_obj = tempfile.mkdtemp(prefix="simai_analytical_")
    tmppath = Path(tmpdir_obj)

    try:
        # The binary writes to ./results/
        (tmppath / "results").mkdir()

        # The binary reads ratio CSVs from ./astra-sim-alibabacloud/inputs/ratio/
        simai_root = _find_simai_root()
        if simai_root:
            astrasim_src = simai_root / "astra-sim-alibabacloud"
            if astrasim_src.is_dir():
                os.symlink(astrasim_src, tmppath / "astra-sim-alibabacloud")

        run_binary(BINARY_NAME, args, cwd=tmpdir_obj, verbose=verbose)

        # Parse results before optionally moving them
        tmp_results = tmppath / "results"
        result_files = list(tmp_results.iterdir()) if tmp_results.is_dir() else []

        # Parse the EndToEnd CSV
        parsed: dict = {"layers": [], "summary": None}
        from simai.output import parse_analytical_csv
        end_to_end = next(
            (f for f in result_files if "EndToEnd" in f.name and f.suffix == ".csv"),
            result_files[0] if result_files else None,
        )
        if end_to_end and end_to_end.is_file():
            parsed = parse_analytical_csv(end_to_end)

        if not preserve_raw:
            # Legacy behavior: move files to output_path
            if not result_files:
                print(f"Warning: no result files found in {tmp_results}")
            elif output_path.suffix and not output_path.is_dir():
                output_path.parent.mkdir(parents=True, exist_ok=True)
                primary = next((f for f in result_files if "EndToEnd" in f.name), result_files[0])
                shutil.move(str(primary), str(output_path))
                for item in result_files:
                    if item.exists():
                        shutil.move(str(item), str(output_path.parent / item.name))
                print(f"Results saved to: {output_path}")
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
            shutil.rmtree(tmpdir_obj, ignore_errors=True)
            raw_path = output_path
        else:
            # Keep tmpdir alive; raw files stay in tmp_results
            print(f"Results saved to: {tmpdir_obj}")
            raw_path = tmppath

    except Exception:
        shutil.rmtree(tmpdir_obj, ignore_errors=True)
        raise

    return SimulationResult(
        output_path=output_path,
        raw_output_path=raw_path,
        parsed=parsed,
    )
