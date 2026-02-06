from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def workload(
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
    # --- Compute profiling ---
    profile_compute: Annotated[
        bool,
        typer.Option("--profile-compute/--no-profile-compute", help="Profile real GPU compute times (requires GPU + torch)."),
    ] = False,
    compute_profile: Annotated[
        Optional[Path],
        typer.Option("--compute-profile", help="Path to pre-recorded compute profile."),
    ] = None,
    # --- Output ---
    gpu_type: Annotated[
        Optional[str],
        typer.Option("--gpu-type", help="GPU type label for output naming."),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path (default: auto-generated in ./results/workload/)."),
    ] = None,
):
    """Generate a training workload description file for SimAI simulation."""
    from simai.workflow.generator import generate_workload

    generate_workload(
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
        aiob_enable=profile_compute,
        comp_filepath=str(compute_profile) if compute_profile else None,
        gpu_type=gpu_type,
        output=output,
    )


def _topology_impl(
    topology_type: Annotated[
        str,
        typer.Option("--type", "-t", help="Topology type: Spectrum-X, AlibabaHPN, or DCN+."),
    ],
    num_gpus: Annotated[
        Optional[int],
        typer.Option("--num-gpus", "-g", help="Total number of GPUs."),
    ] = None,
    gpus_per_server: Annotated[
        Optional[int],
        typer.Option("--gpus-per-server", help="GPUs per server."),
    ] = None,
    gpu_type: Annotated[
        Optional[str],
        typer.Option("--gpu-type", help="GPU type (e.g. H100, A100)."),
    ] = None,
    nic_bandwidth: Annotated[
        Optional[str],
        typer.Option("--nic-bandwidth", help="NIC-to-switch bandwidth (e.g. 400Gbps)."),
    ] = None,
    nvlink_bandwidth: Annotated[
        Optional[str],
        typer.Option("--nvlink-bandwidth", help="NVLink bandwidth (e.g. 2880Gbps)."),
    ] = None,
    nics_per_switch: Annotated[
        Optional[int],
        typer.Option("--nics-per-switch", help="NICs per aggregate switch."),
    ] = None,
    aggregate_switches: Annotated[
        Optional[int],
        typer.Option("--aggregate-switches", help="Number of aggregate switches."),
    ] = None,
    pod_switches: Annotated[
        Optional[int],
        typer.Option("--pod-switches", help="Number of pod switches."),
    ] = None,
    aggregate_bandwidth: Annotated[
        Optional[str],
        typer.Option("--aggregate-bandwidth", help="Aggregate-to-pod bandwidth (e.g. 400Gbps)."),
    ] = None,
    switches_per_pod: Annotated[
        Optional[int],
        typer.Option("--switches-per-pod", help="Aggregate switches per pod switch."),
    ] = None,
    nv_switches_per_server: Annotated[
        Optional[int],
        typer.Option("--nv-switches-per-server", help="NVLink switches per server."),
    ] = None,
    nvlink_latency: Annotated[
        Optional[str],
        typer.Option("--nvlink-latency", help="NVLink latency (e.g. 0.000025ms)."),
    ] = None,
    nic_latency: Annotated[
        Optional[str],
        typer.Option("--nic-latency", help="NIC latency (e.g. 0.0005ms)."),
    ] = None,
    error_rate: Annotated[
        Optional[str],
        typer.Option("--error-rate", help="Link error rate."),
    ] = None,
    dual_tor: Annotated[
        bool,
        typer.Option("--dual-tor/--no-dual-tor", help="Enable dual ToR (DCN+/AlibabaHPN)."),
    ] = False,
    dual_plane: Annotated[
        bool,
        typer.Option("--dual-plane/--no-dual-plane", help="Enable dual plane (AlibabaHPN)."),
    ] = False,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output directory path."),
    ] = None,
):
    """Generate a network topology for SimAI simulation."""
    from simai.topology.generator import generate_topology

    generate_topology(
        topology_type=topology_type,
        num_gpus=num_gpus,
        gpus_per_server=gpus_per_server,
        gpu_type=gpu_type,
        nic_bandwidth=nic_bandwidth,
        nvlink_bandwidth=nvlink_bandwidth,
        nics_per_switch=nics_per_switch,
        aggregate_switches=aggregate_switches,
        pod_switches=pod_switches,
        aggregate_bandwidth=aggregate_bandwidth,
        switches_per_pod=switches_per_pod,
        nv_switches_per_server=nv_switches_per_server,
        nvlink_latency=nvlink_latency,
        nic_latency=nic_latency,
        error_rate=error_rate,
        dual_tor=dual_tor,
        dual_plane=dual_plane,
        output=output,
    )


# Register topology command and its hidden alias
app.command(name="topology")(_topology_impl)
app.command(name="topo", hidden=True)(_topology_impl)
