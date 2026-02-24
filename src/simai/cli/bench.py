"""CLI commands for running distributed training benchmarks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(no_args_is_help=True)


def _slurm_int(var: str, default: int) -> int:
    """Read an integer from a SLURM environment variable, falling back to default."""
    val = os.environ.get(var)
    if val is not None:
        try:
            return int(val)
        except ValueError:
            pass
    return default


def _slurm_str(var: str, default: str) -> str:
    """Read a string from an environment variable, falling back to default."""
    return os.environ.get(var, default)


@app.command()
def training(
    # --- Distributed / SLURM (auto-detected from env) ---
    nnodes: Annotated[
        int,
        typer.Option("--nnodes", "-N", help="Number of nodes. Auto-detected from SLURM_NNODES."),
    ] = _slurm_int("SLURM_NNODES", 1),
    node_rank: Annotated[
        int,
        typer.Option("--node-rank", help="Rank of this node. Auto-detected from SLURM_NODEID."),
    ] = _slurm_int("SLURM_NODEID", 0),
    nproc_per_node: Annotated[
        int,
        typer.Option(
            "--nproc-per-node", "-g",
            help="GPUs per node. Auto-detected from SLURM_GPUS_PER_NODE / SLURM_NTASKS_PER_NODE.",
        ),
    ] = _slurm_int("SLURM_GPUS_PER_NODE", _slurm_int("SLURM_NTASKS_PER_NODE", 1)),
    master_addr: Annotated[
        str,
        typer.Option("--master-addr", help="Master node address. Auto-detected from MASTER_ADDR."),
    ] = _slurm_str("MASTER_ADDR", "localhost"),
    master_port: Annotated[
        int,
        typer.Option("--master-port", help="Master rendezvous port. Auto-detected from MASTER_PORT."),
    ] = _slurm_int("MASTER_PORT", 29500),
    # --- Framework & parallelism ---
    framework: Annotated[
        str,
        typer.Option(
            "--framework", "-f",
            help="Training framework: Megatron, DeepSpeed, or DeepSeek.",
        ),
    ] = "Megatron",
    world_size: Annotated[
        Optional[int],
        typer.Option(
            "--world-size",
            help="Total GPU count (default: nnodes × nproc-per-node).",
        ),
    ] = None,
    tensor_parallel: Annotated[
        int,
        typer.Option("--tensor-parallel", "--tp", help="Tensor parallelism degree."),
    ] = 1,
    pipeline_parallel: Annotated[
        int,
        typer.Option("--pipeline-parallel", "--pp", help="Pipeline parallelism degree."),
    ] = 1,
    expert_parallel: Annotated[
        int,
        typer.Option("--expert-parallel", "--ep", help="Expert parallelism degree (MoE)."),
    ] = 1,
    # --- Batch sizes ---
    global_batch_size: Annotated[
        int,
        typer.Option("--global-batch-size", help="Global training batch size."),
    ] = 4,
    micro_batch_size: Annotated[
        int,
        typer.Option("--micro-batch-size", help="Micro-batch size per GPU."),
    ] = 1,
    # --- Model architecture ---
    num_layers: Annotated[
        int,
        typer.Option("--num-layers", help="Number of transformer layers."),
    ] = 24,
    hidden_size: Annotated[
        int,
        typer.Option("--hidden-size", help="Transformer hidden dimension."),
    ] = 1024,
    sequence_length: Annotated[
        int,
        typer.Option("--sequence-length", "--seq", help="Maximum sequence length."),
    ] = 2048,
    num_heads: Annotated[
        Optional[int],
        typer.Option("--num-heads", help="Number of attention heads (default: num_layers)."),
    ] = None,
    vocab_size: Annotated[
        int,
        typer.Option("--vocab-size", help="Vocabulary size."),
    ] = 32000,
    # --- MoE ---
    moe: Annotated[
        bool,
        typer.Option("--moe/--no-moe", help="Enable Mixture of Experts."),
    ] = False,
    num_experts: Annotated[
        int,
        typer.Option("--num-experts", help="Number of MoE experts."),
    ] = 1,
    top_k: Annotated[
        int,
        typer.Option("--top-k", help="Number of experts routed per token."),
    ] = 1,
    # --- Optimizations ---
    sequence_parallel: Annotated[
        bool,
        typer.Option("--sequence-parallel/--no-sequence-parallel", "--sp", help="Enable sequence parallelism."),
    ] = False,
    flash_attention: Annotated[
        bool,
        typer.Option("--flash-attention/--no-flash-attention", help="Use FlashAttention."),
    ] = False,
    swiglu: Annotated[
        bool,
        typer.Option("--swiglu/--no-swiglu", help="Use SwiGLU activation."),
    ] = False,
    distributed_optimizer: Annotated[
        bool,
        typer.Option("--distributed-optimizer/--no-distributed-optimizer", help="Use distributed optimizer."),
    ] = False,
    # --- AIOB compute overlap ---
    aiob: Annotated[
        bool,
        typer.Option("--aiob/--no-aiob", help="Enable AIOB compute-communication overlap."),
    ] = False,
    comp_profile: Annotated[
        Optional[Path],
        typer.Option(
            "--comp-profile",
            help="Path to a GPU compute profile (from simai profile gpu). Implies --aiob.",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
    # --- Run config ---
    iters: Annotated[
        int,
        typer.Option("--iters", "--iterations", help="Number of benchmark epochs/iterations."),
    ] = 1,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory for benchmark results."),
    ] = Path("results/bench"),
):
    """Run a distributed AICB training benchmark via torchrun.

    Launches the AICB benchmark using torchrun across the specified nodes and GPUs.
    SLURM environment variables (SLURM_NNODES, SLURM_NODEID, SLURM_GPUS_PER_NODE,
    MASTER_ADDR, MASTER_PORT) are detected automatically when running under srun,
    so no extra flags are needed in SLURM jobs.

    Requirements:
      - PyTorch with CUDA: pip install "simai[profiling]"
      - CUDA-capable GPUs
      - AICB source (vendored, SIMAI_PATH, or sibling directory)

    Single-node smoke test:
      simai bench training --nproc-per-node 1 --world-size 1 \\
          --num-layers 4 --hidden-size 256 --num-heads 4 \\
          --global-batch-size 2 --micro-batch-size 1 --epochs 1 \\
          --output /tmp/bench_smoke/

    Multi-node SLURM (each task reads SLURM vars automatically):
      srun simai bench training --framework Megatron --tensor-parallel 4
    """
    from simai.workflow.bench import run_training_benchmark

    effective_aiob = aiob or (comp_profile is not None)

    try:
        returncode = run_training_benchmark(
            nnodes=nnodes,
            node_rank=node_rank,
            nproc_per_node=nproc_per_node,
            master_addr=master_addr,
            master_port=master_port,
            framework=framework,
            world_size=world_size,
            tensor_parallel=tensor_parallel,
            pipeline_parallel=pipeline_parallel,
            expert_parallel=expert_parallel,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            num_layers=num_layers,
            hidden_size=hidden_size,
            sequence_length=sequence_length,
            num_heads=num_heads,
            vocab_size=vocab_size,
            moe=moe,
            num_experts=num_experts,
            top_k=top_k,
            sequence_parallel=sequence_parallel,
            flash_attention=flash_attention,
            swiglu=swiglu,
            distributed_optimizer=distributed_optimizer,
            aiob_enable=effective_aiob,
            comp_filepath=str(comp_profile) if comp_profile is not None else None,
            epochs=iters,
            output_dir=output,
        )
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    if returncode != 0:
        raise typer.Exit(code=returncode)
