from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from simai.backends.binary import run_binary

BINARY_NAME = "SimAI_simulator"


@dataclass
class SimulationResult:
    """Structured result from a simulation backend."""
    output_path: Path
    raw_output_path: Path
    parsed: dict


def _find_default_config() -> Path:
    """Find the bundled default SimAI.conf."""
    conf_rel = Path("astra-sim-alibabacloud") / "inputs" / "config" / "SimAI.conf"

    # Check in vendored location (wheel install)
    vendored = Path(__file__).resolve().parent.parent / "_vendor" / "SimAI.conf"
    if vendored.is_file():
        return vendored

    # Check vendor submodule (editable install)
    # __file__ is at src/simai/backends/ns3.py → project root is 4 levels up
    vendor_sub = Path(__file__).resolve().parent.parent.parent.parent / "vendor" / "simai" / conf_rel
    if vendor_sub.is_file():
        return vendor_sub

    # Check SIMAI_PATH
    env_path = os.environ.get("SIMAI_PATH")
    if env_path:
        candidate = Path(env_path) / conf_rel
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Cannot find default SimAI.conf. Provide --config or set SIMAI_PATH."
    )


def run_ns3(
    *,
    workload: Path,
    topology: Path,
    config: Path | None = None,
    ns3_conf: dict | None = None,
    threads: int = 8,
    send_latency: int = 0,
    nvls: bool = False,
    pxn: bool = False,
    output: Path | None = None,
    verbose: bool = False,
    preserve_raw: bool = True,
) -> SimulationResult:
    """Run the SimAI NS-3 simulator backend.

    The binary writes output relative to cwd, so we run it from a temp
    directory. Raw files are preserved in tmpdir when preserve_raw=True.

    ns3_conf: optional dict of SimAI.conf key→value pairs. If provided,
      the dict is rendered to SimAI.conf (replacing the bundled config and
      the old regex-patching approach). Path-dependent keys (FLOW_FILE, etc.)
      are set to tmpdir-relative paths automatically.

    Returns a SimulationResult with output_path, raw_output_path, and parsed data.
    """
    workload = workload.resolve()
    topology = topology.resolve()

    if config is None and ns3_conf is None:
        config = _find_default_config()
    if config is not None:
        config = config.resolve()

    # Build command-line arguments
    args: list[str] = [
        "-w", str(workload),
        "-n", str(topology),
        "-t", str(threads),
    ]

    # Build environment variables for the binary
    env: dict[str, str] = {
        # Disable logging to /etc/astra-sim/SimAI.log (requires root to create)
        "AS_LOG_LEVEL": "0",
    }
    env["AS_SEND_LAT"] = str(send_latency)
    if nvls:
        env["AS_NVLS_ENABLE"] = "1"
    if pxn:
        env["AS_PXN_ENABLE"] = "1"

    # Determine output path
    output_path = Path(output).resolve() if output else Path("results").resolve()

    tmpdir_obj = tempfile.mkdtemp(prefix="simai_ns3_")
    tmppath = Path(tmpdir_obj)

    try:
        if ns3_conf is not None:
            # Render the dict directly to SimAI.conf with path substitutions
            patched_config = tmppath / "SimAI.conf"
            conf_lines = []
            for key, value in ns3_conf.items():
                conf_lines.append(f"{key} {value}")
            # Path-dependent entries
            conf_lines += [
                f"FLOW_FILE {tmpdir_obj}/flow1.txt",
                f"TRACE_FILE {tmpdir_obj}/trace1.txt",
                f"FCTFILE {tmpdir_obj}/fct.txt",
                f"PFC_OUTPUT_FILE {tmpdir_obj}/pfc.txt",
                f"QLEN_MON_FILE {tmpdir_obj}/qlen.txt",
                f"BW_MON_FILE {tmpdir_obj}/bw.txt",
                f"QP_MON_FILE {tmpdir_obj}/qp.txt",
            ]
            patched_config.write_text("\n".join(conf_lines) + "\n")
        else:
            # Legacy: patch hardcoded /etc/astra-sim/simulation/ paths
            patched_config = tmppath / "SimAI.conf"
            with open(config) as f:
                conf_text = f.read()
            conf_text = re.sub(
                r"/etc/astra-sim/simulation/",
                tmpdir_obj.rstrip("/") + "/",
                conf_text,
            )
            patched_config.write_text(conf_text)

        # Create dummy input files that the simulator expects to exist
        (tmppath / "flow1.txt").touch()
        (tmppath / "trace1.txt").touch()

        # Add config to args
        args += ["-c", str(patched_config)]

        run_binary(BINARY_NAME, args, cwd=tmpdir_obj, env=env, verbose=verbose)

        result_files = [
            f for f in tmppath.iterdir()
            if f.name != "SimAI.conf" and f.stat().st_size > 0
        ]

        # Parse results
        parsed: dict = {"layers": [], "summary": None}
        from simai.output import parse_ns3_results
        parsed = parse_ns3_results(tmppath)

        if not preserve_raw:
            if not result_files:
                print("Warning: no result files generated")
            elif output_path.suffix and not output_path.is_dir():
                output_path.parent.mkdir(parents=True, exist_ok=True)
                primary = result_files[0]
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
