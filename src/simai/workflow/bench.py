"""Workflow for running distributed AICB training benchmarks via torchrun."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from simai.workflow.generator import _find_aicb_root


def _find_torchrun() -> str:
    """Locate the torchrun executable.

    Search order:
    1. Same directory as the current Python interpreter (venv-aware)
    2. System PATH via shutil.which
    """
    venv_torchrun = Path(sys.executable).parent / "torchrun"
    if venv_torchrun.is_file():
        return str(venv_torchrun)

    system_torchrun = shutil.which("torchrun")
    if system_torchrun:
        return system_torchrun

    raise FileNotFoundError(
        "Cannot find 'torchrun'. Install PyTorch:\n"
        "  pip install 'simai[profiling]'\n"
        "Or:\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cu121"
    )


def run_training_benchmark(
    *,
    nnodes: int,
    node_rank: int,
    nproc_per_node: int,
    master_addr: str,
    master_port: int,
    framework: str,
    world_size: int | None,
    tensor_parallel: int,
    pipeline_parallel: int,
    expert_parallel: int,
    global_batch_size: int,
    micro_batch_size: int,
    num_layers: int,
    hidden_size: int,
    sequence_length: int,
    num_heads: int | None,
    vocab_size: int,
    moe: bool,
    num_experts: int,
    top_k: int,
    sequence_parallel: bool,
    flash_attention: bool,
    swiglu: bool,
    distributed_optimizer: bool,
    aiob_enable: bool,
    comp_filepath: str | None,
    epochs: int,
    output_dir: Path,
) -> int:
    """Launch an AICB distributed training benchmark via torchrun.

    Returns the subprocess returncode (0 = success).
    """
    torchrun = _find_torchrun()
    aicb_root = _find_aicb_root()
    aicb_script = aicb_root / "aicb.py"

    ws = world_size if world_size is not None else nnodes * nproc_per_node

    num_attention_heads = num_heads if num_heads is not None else num_layers

    cmd = [
        torchrun,
        f"--nnodes={nnodes}",
        f"--node_rank={node_rank}",
        f"--nproc_per_node={nproc_per_node}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        str(aicb_script),
        "--frame", framework,
        "--world_size", str(ws),
        "--tensor_model_parallel_size", str(tensor_parallel),
        "--pipeline_model_parallel", str(pipeline_parallel),
        "--expert_model_parallel_size", str(expert_parallel),
        "--global_batch", str(global_batch_size),
        "--micro_batch", str(micro_batch_size),
        "--num_layers", str(num_layers),
        "--hidden_size", str(hidden_size),
        "--seq_length", str(sequence_length),
        "--num_attention_heads", str(num_attention_heads),
        "--vocab_size", str(vocab_size),
        "--num_experts", str(num_experts),
        "--moe_router_topk", str(top_k),
        "--epoch_num", str(epochs),
    ]

    # Boolean flags — appended only when True
    if moe:
        cmd.append("--moe_enable")
    if sequence_parallel:
        cmd.append("--enable_sequence_parallel")
    if flash_attention:
        cmd.append("--use_flash_attn")
    if swiglu:
        cmd.append("--swiglu")
    if distributed_optimizer:
        cmd.append("--use-distributed-optimizer")
    if aiob_enable:
        cmd.append("--aiob_enable")
    if comp_filepath is not None:
        cmd.extend(["--comp_filepath", comp_filepath])

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=output_dir)

    # AICB hardcodes output to results/mocked_workload/ relative to cwd.
    # Move results to output_dir root.
    if result.returncode == 0:
        results_subdir = output_dir / "results" / "mocked_workload"
        if results_subdir.is_dir():
            for file in results_subdir.iterdir():
                if file.is_file():
                    shutil.move(str(file), str(output_dir / file.name))
            # Clean up the empty results directory
            shutil.rmtree(output_dir / "results")

    return result.returncode
