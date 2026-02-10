from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path

from simai.backends.binary import run_binary

BINARY_NAME = "SimAI_simulator"


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
    threads: int = 8,
    send_latency: int | None = None,
    nvls: bool = False,
    pxn: bool = False,
    output: Path | None = None,
    verbose: bool = False,
) -> Path:
    """Run the SimAI NS-3 simulator backend.

    The binary writes output relative to cwd, so we run it from a temp
    directory and then move results to the user's chosen output path.

    Returns the output directory path.
    """
    workload = workload.resolve()
    topology = topology.resolve()

    if config is None:
        config = _find_default_config()
    config = config.resolve()

    # Build command-line arguments
    args: list[str] = [
        "-w", str(workload),
        "-n", str(topology),
        "-c", str(config),
        "-t", str(threads),
    ]

    # Build environment variables for the binary
    env: dict[str, str] = {
        # Disable logging to /etc/astra-sim/SimAI.log (requires root to create)
        "AS_LOG_LEVEL": "0",
    }
    if send_latency is not None:
        env["AS_SEND_LAT"] = str(send_latency)
    if nvls:
        env["AS_NVLS_ENABLE"] = "1"
    if pxn:
        env["AS_PXN_ENABLE"] = "1"

    # Determine output path
    output_path = Path(output).resolve() if output else Path("results").resolve()

    # Run from a temp directory to capture output
    with tempfile.TemporaryDirectory(prefix="simai_ns3_") as tmpdir:
        # Patch config: replace hardcoded /etc/astra-sim/simulation/ paths
        # with relative paths (relative to cwd=tmpdir where the binary runs).
        # Using relative paths instead of absolute paths avoids potential buffer
        # overflow issues in the C++ binary's path handling code.
        patched_config = Path(tmpdir) / "SimAI.conf"
        with open(config) as f:
            conf_text = f.read()
        conf_text = re.sub(
            r"/etc/astra-sim/simulation/",
            tmpdir.rstrip("/") + "/"
            conf_text,
        )
        with open(patched_config, "w") as f:
            f.write(conf_text)

        # Create dummy input files that the simulator expects to exist
        # These are referenced in the config but may not be used by all workloads
        (Path(tmpdir) / "flow1.txt").touch()
        (Path(tmpdir) / "trace1.txt").touch()

        # Update args to use patched config
        args[args.index("-c") + 1] = str(patched_config)

        run_binary(BINARY_NAME, args, cwd=tmpdir, env=env, verbose=verbose)

        result_files = [f for f in Path(tmpdir).iterdir() if f.name != "SimAI.conf"]

        if not result_files:
            print("Warning: no result files generated")
            return output_path

        # If output looks like a file path (has extension or doesn't exist as dir),
        # and there's a single result, save as that filename.
        if output_path.suffix and not output_path.is_dir():
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

    return output_path
