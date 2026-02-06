from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def find_binary(name: str) -> Path:
    """Locate a SimAI binary (e.g. SimAI_analytical, SimAI_simulator).

    Search order:
    1. Bundled in the package: simai/_binaries/
    2. SIMAI_BIN_PATH environment variable
    3. System PATH (via shutil.which)
    """
    # 1. Bundled
    bundled = Path(__file__).resolve().parent.parent / "_binaries" / name
    if bundled.is_file():
        return bundled

    # 2. SIMAI_BIN_PATH env var
    env_path = os.environ.get("SIMAI_BIN_PATH")
    if env_path:
        candidate = Path(env_path) / name
        if candidate.is_file():
            return candidate

    # 3. System PATH
    found = shutil.which(name)
    if found:
        return Path(found)

    raise FileNotFoundError(
        f"Cannot find '{name}' binary. Either:\n"
        f"  - Install from a wheel that includes pre-built binaries, or\n"
        f"  - Set SIMAI_BIN_PATH to the directory containing {name}, or\n"
        f"  - Ensure {name} is on your system PATH."
    )


def run_binary(
    name: str,
    args: list[str],
    *,
    cwd: Path | str | None = None,
    env: dict[str, str] | None = None,
    verbose: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Find and run a SimAI binary."""
    binary = find_binary(name)
    cmd = [str(binary)] + args

    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    # Ensure shared libraries next to the binary are found (e.g. libns3*.so)
    bin_dir = str(binary.parent)
    ld_path = run_env.get("LD_LIBRARY_PATH", "")
    if bin_dir not in ld_path:
        run_env["LD_LIBRARY_PATH"] = f"{bin_dir}:{ld_path}" if ld_path else bin_dir

    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=run_env,
        stdout=None if verbose else subprocess.DEVNULL,
        stderr=None if verbose else subprocess.PIPE,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip() if result.stderr else ""
        msg = f"'{name}' exited with code {result.returncode}"
        if stderr:
            msg += f":\n{stderr}"
        raise subprocess.CalledProcessError(result.returncode, cmd, output=None, stderr=stderr)
    return result
