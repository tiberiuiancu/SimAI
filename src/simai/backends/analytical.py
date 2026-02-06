from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from simai.backends.binary import find_binary, run_binary

BINARY_NAME = "SimAI_analytical"


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

    # Check relative to the binary location
    try:
        binary = find_binary(BINARY_NAME)
        # binary might be at <simai>/bin/SimAI_analytical
        simai_root = binary.resolve().parent.parent
        if (simai_root / "astra-sim-alibabacloud").is_dir():
            return simai_root
    except FileNotFoundError:
        pass

    # Check vendored location
    vendored = Path(__file__).resolve().parent.parent / "_vendor"
    if (vendored / "astra-sim-alibabacloud").is_dir():
        return vendored

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
) -> Path:
    """Run the SimAI analytical backend.

    The binary hardcodes output to ./results/ and reads auxiliary data from
    ./astra-sim-alibabacloud/inputs/ratio/. We run it from a temp directory
    with symlinks to the required data, then move results to the user's
    chosen output path.

    Returns the output directory path.
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
    if result_prefix is not None:
        args += ["-r", result_prefix]

    # Determine output directory
    output_dir = Path(output) if output else Path("results")
    output_dir = output_dir.resolve()

    # Run from a temp directory
    with tempfile.TemporaryDirectory(prefix="simai_analytical_") as tmpdir:
        tmppath = Path(tmpdir)

        # The binary writes to ./results/
        (tmppath / "results").mkdir()

        # The binary reads ratio CSVs from ./astra-sim-alibabacloud/inputs/ratio/
        simai_root = _find_simai_root()
        if simai_root:
            astrasim_src = simai_root / "astra-sim-alibabacloud"
            if astrasim_src.is_dir():
                os.symlink(astrasim_src, tmppath / "astra-sim-alibabacloud")

        run_binary(BINARY_NAME, args, cwd=tmpdir)

        # Move results from tmpdir/results/ to output_dir
        tmp_results = tmppath / "results"
        if tmp_results.is_dir() and any(tmp_results.iterdir()):
            output_dir.mkdir(parents=True, exist_ok=True)
            for item in tmp_results.iterdir():
                dest = output_dir / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(dest))
            print(f"Results saved to: {output_dir}")
        else:
            print(f"Warning: no result files found in {tmp_results}")

    return output_dir
