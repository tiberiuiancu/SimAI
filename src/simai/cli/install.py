from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True)

_M4_GIT_URL = "https://github.com/liecn/SimAI.git"
_M4_CACHE_DIR = Path.home() / ".cache" / "simai" / "simai-m4"


def _find_m4_src() -> Path | None:
    """Locate the simai-m4 source directory without cloning.

    Search order:
    1. Editable install: vendor/simai-m4/ in the repo root
    2. Previously cloned cache: ~/.cache/simai/simai-m4/
    """
    # 1. Editable install: __file__ is src/simai/cli/install.py → 4 levels up is repo root
    candidate = Path(__file__).resolve().parent.parent.parent.parent / "vendor" / "simai-m4"
    if candidate.is_dir() and (candidate / "scripts" / "build.sh").is_file():
        return candidate

    # 2. Cache from a previous `simai install m4`
    if _M4_CACHE_DIR.is_dir() and (_M4_CACHE_DIR / "scripts" / "build.sh").is_file():
        return _M4_CACHE_DIR

    return None


def _clone_m4_src(git_url: str) -> Path:
    """Clone simai-m4 source to the cache directory."""
    typer.echo(f"Cloning {git_url} into {_M4_CACHE_DIR} ...")
    _M4_CACHE_DIR.parent.mkdir(parents=True, exist_ok=True)
    if _M4_CACHE_DIR.exists():
        shutil.rmtree(_M4_CACHE_DIR)
    subprocess.run(
        ["git", "clone", "--recurse-submodules", "--shallow-submodules", git_url, str(_M4_CACHE_DIR)],
        check=True,
    )
    return _M4_CACHE_DIR


_N_FLOWS_MAX = 500_000  # Upstream hardcodes 50 000; large workloads exceed that


def _patch_n_flows_max(m4_src: Path, n_flows_max: int) -> None:
    """Patch M4.cc to set n_flows_max before compilation."""
    import re
    m4_cc = (
        m4_src
        / "astra-sim-alibabacloud"
        / "astra-sim"
        / "network_frontend"
        / "m4"
        / "M4.cc"
    )
    original = m4_cc.read_text()
    patched = re.sub(
        r"(int32_t\s+M4::n_flows_max\s*=\s*)\d+(\s*;)",
        rf"\g<1>{n_flows_max}\2",
        original,
    )
    if patched == original:
        typer.echo(
            "Warning: could not find 'M4::n_flows_max' in M4.cc — skipping patch.",
            err=True,
        )
        return
    m4_cc.write_text(patched)
    typer.echo(f"Patched M4::n_flows_max → {n_flows_max} in {m4_cc}")


def _build_m4(m4_src: Path, dest_dir: Path, n_flows_max: int = _N_FLOWS_MAX) -> None:
    """Compile SimAI_m4 and place the binary in dest_dir."""
    _patch_n_flows_max(m4_src, n_flows_max)
    # Locate LibTorch directory (parent of share/cmake/Torch) for cmake discovery.
    # The m4 CMakeLists.txt checks LIBTORCH_DIR env var first; setting it explicitly
    # avoids cmake finding the wrong Python interpreter on multi-Python HPC systems.
    libtorch_dir = os.environ.get("LIBTORCH_DIR")
    if not libtorch_dir:
        try:
            import torch
            libtorch_dir = str(Path(torch.__file__).parent)
            typer.echo(f"Found lib torch at: {libtorch_dir}")
        except ImportError:
            typer.echo(
                "Error: torch is not installed.\n"
                "Install PyTorch (CUDA) first, then retry:\n"
                "  pip install torch\n"
                "  simai install m4\n"
                "Or set LIBTORCH_DIR to your LibTorch directory.",
                err=True,
            )
            raise typer.Exit(1)

    gcc = shutil.which("gcc-9") or shutil.which("gcc")
    gxx = shutil.which("g++-9") or shutil.which("g++")
    if not gcc or not gxx:
        typer.echo(
            "Error: gcc/g++ not found. Install GCC and ensure it is on PATH.",
            err=True,
        )
        raise typer.Exit(1)

    cmake_src = m4_src / "astra-sim-alibabacloud" / "build" / "simai_m4"
    build_dir = cmake_src / "build"
    # Clean stale cmake cache to avoid picking up wrong compilers/torch paths
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir()

    typer.echo("Building SimAI_m4 (this may take a few minutes)...")
    torch_cmake_dir = f"{libtorch_dir}/share/cmake/Torch"
    env = {**os.environ, "LIBTORCH_DIR": libtorch_dir}
    subprocess.run(
        [
            "cmake",
            f"-DCMAKE_C_COMPILER={gcc}",
            f"-DCMAKE_CXX_COMPILER={gxx}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -march=native -DNDEBUG",
            "-DCMAKE_CUDA_ARCHITECTURES=80",
            "-DUSE_ANALYTICAL=TRUE",
            # Pin Torch_DIR explicitly so cmake uses this torch regardless of
            # CMAKE_PREFIX_PATH in the environment (e.g. a standalone libtorch build).
            f"-DTorch_DIR={torch_cmake_dir}",
            # Allow unresolved symbols inside shared libs at link time
            # (e.g. newer NCCL symbols in libtorch_cuda.so on HPC systems)
            "-DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined",
            str(cmake_src),
        ],
        cwd=str(build_dir),
        env=env,
        check=True,
    )
    subprocess.run(["make", f"-j{os.cpu_count() or 1}"], cwd=str(build_dir), env=env, check=True)

    built = (
        m4_src
        / "astra-sim-alibabacloud"
        / "build"
        / "simai_m4"
        / "build"
        / "simai_m4"
        / "SimAI_m4"
    )
    if not built.is_file():
        typer.echo(f"Error: build finished but binary not found at {built}", err=True)
        raise typer.Exit(1)

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "SimAI_m4"
    shutil.copy2(str(built), str(dest))
    dest.chmod(dest.stat().st_mode | 0o111)
    typer.echo(f"SimAI_m4 installed to {dest}")


@app.command()
def m4(
    src: Annotated[
        Path | None,
        typer.Option(
            "--src",
            help="Path to simai-m4 source directory. "
                 "Auto-detected for editable installs or after a previous install.",
            exists=True,
            file_okay=False,
        ),
    ] = None,
    git_url: Annotated[
        str,
        typer.Option(
            "--git-url",
            help="Git URL to clone simai-m4 from if source is not found locally.",
        ),
    ] = _M4_GIT_URL,
    force: Annotated[
        bool,
        typer.Option("--force", help="Reinstall even if SimAI_m4 binary already exists."),
    ] = False,
    n_flows_max: Annotated[
        int,
        typer.Option(
            "--n-flows-max",
            help="Maximum concurrent flows (patches M4::n_flows_max at build time). "
                 f"Default: {_N_FLOWS_MAX}.",
        ),
    ] = _N_FLOWS_MAX,
) -> None:
    """Build and install the SimAI_m4 binary from source.

    Requires CUDA-enabled PyTorch and cmake/make/gcc.
    On first run, clones source from GitHub (~/.cache/simai/simai-m4/).
    Subsequent runs reuse the cached clone.
    """
    # Install next to the package so find_binary() picks it up automatically
    bin_dir = Path(__file__).resolve().parent.parent / "_binaries"

    if not force and (bin_dir / "SimAI_m4").is_file():
        typer.echo("SimAI_m4 is already installed. Use --force to reinstall.")
        return

    m4_src = src or _find_m4_src()
    if m4_src is None:
        m4_src = _clone_m4_src(git_url)

    _build_m4(m4_src, bin_dir, n_flows_max=n_flows_max)
