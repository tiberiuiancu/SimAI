"""CLI commands for GPU profiling."""

from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def gpu(
    # --- Parallelism & cluster ---
    framework: Annotated[
        str,
        typer.Option(
            "--framework", "-f",
            help="Training framework: Megatron, DeepSpeed, or DeepSeek.",
        ),
    ] = "Megatron",
    num_gpus: Annotated[
        int,
        typer.Option(
            "--num-gpus", "-g",
            help="Total number of GPUs in the simulated cluster.",
        ),
    ] = 1,
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
    # --- GPU type ---
    gpu_type: Annotated[
        Optional[str],
        typer.Option("--gpu-type", help="GPU type label (e.g., H100, A100) for output naming."),
    ] = None,
    # --- Output ---
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path (default: auto-generated in ./results/profiles/)."),
    ] = None,
):
    """Profile GPU kernel execution times for a model configuration.

    This command profiles GPU kernels and saves the results to a file that
    can be reused for multiple workload generations via the --compute-profile
    flag. This allows you to profile once on a GPU machine, then generate
    multiple workloads without needing GPU access.

    Requirements:
      - PyTorch with CUDA: pip install "simai[profiling]"
      - CUDA-capable GPU

    Example:
      simai profile gpu --framework Megatron --num-gpus 64 --tensor-parallel 4 \\
          --num-layers 32 --hidden-size 4096 --gpu-type H100 -o h100_profile.txt

    Then use the profile when generating workloads:
      simai generate workload --compute-profile h100_profile.txt \\
          --num-gpus 64 --tensor-parallel 4 ...
    """
    from simai.workflow.profiler import profile_gpu_kernels

    try:
        profile_path = profile_gpu_kernels(
            framework=framework,
            world_size=num_gpus,
            tensor_model_parallel_size=tensor_parallel,
            pipeline_model_parallel=pipeline_parallel,
            expert_model_parallel_size=expert_parallel,
            global_batch=global_batch_size,
            micro_batch=micro_batch_size,
            num_layers=num_layers,
            hidden_size=hidden_size,
            seq_length=sequence_length,
            num_attention_heads=num_heads,
            vocab_size=vocab_size,
            moe_enable=moe,
            num_experts=num_experts,
            moe_router_topk=top_k,
            enable_sequence_parallel=sequence_parallel,
            use_flash_attn=flash_attention,
            swiglu=swiglu,
            use_distributed_optimizer=distributed_optimizer,
            gpu_type=gpu_type,
            output=output,
        )

        print(f"GPU profile saved to: {profile_path}")
        print(f"Use with: simai generate workload --compute-profile {profile_path}")

    except ImportError as e:
        typer.echo(f"Error: {e}", err=True)
        typer.echo(
            "\nInstall PyTorch with CUDA support:\n"
            "  pip install 'simai[profiling]'\n"
            "Or:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu121",
            err=True,
        )
        raise typer.Exit(code=1)
    except RuntimeError as e:
        if "CUDA" in str(e) or "GPU" in str(e):
            typer.echo(f"Error: {e}", err=True)
            typer.echo(
                "\nGPU profiling requires a CUDA-capable GPU.\n"
                "Run 'nvidia-smi' to verify GPU accessibility.",
                err=True,
            )
            raise typer.Exit(code=1)
        else:
            raise
