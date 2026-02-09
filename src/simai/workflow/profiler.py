"""GPU kernel profiling for SimAI workload generation.

This module provides functionality to profile GPU kernel execution times
separately from workload generation, enabling profile reuse across multiple
workload generations without requiring GPU access each time.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from simai.workflow.generator import (
    _aicb_on_path,
    _compute_ffn_hidden_size,
    _find_aicb_root,
    _get_padded_vocab_size,
)


def _patch_optional_cuda_modules():
    """Monkey-patch optional CUDA modules with fallback implementations.

    This allows aicb code to import these modules without errors, falling back
    to PyTorch implementations when the optimized CUDA extensions aren't available.
    """
    import torch

    # Patch apex.contrib.layer_norm if not available
    if 'apex' not in sys.modules:
        try:
            import apex
        except ImportError:
            # Create fake apex module structure
            apex = ModuleType('apex')
            apex.contrib = ModuleType('apex.contrib')
            apex.contrib.layer_norm = ModuleType('apex.contrib.layer_norm')
            apex.contrib.layer_norm.layer_norm = ModuleType('apex.contrib.layer_norm.layer_norm')

            # Create FastLayerNormFN that raises ImportError when used
            class FastLayerNormFN:
                @staticmethod
                def apply(*args, **kwargs):
                    raise ImportError("apex FastLayerNormFN not available")

            apex.contrib.layer_norm.layer_norm.FastLayerNormFN = FastLayerNormFN

            sys.modules['apex'] = apex
            sys.modules['apex.contrib'] = apex.contrib
            sys.modules['apex.contrib.layer_norm'] = apex.contrib.layer_norm
            sys.modules['apex.contrib.layer_norm.layer_norm'] = apex.contrib.layer_norm.layer_norm

    # Patch scaled_upper_triang_masked_softmax_cuda if not available
    if 'scaled_upper_triang_masked_softmax_cuda' not in sys.modules:
        try:
            import scaled_upper_triang_masked_softmax_cuda
        except ImportError:
            # Create fake module that will make is_kernel_available return False
            fake_module = ModuleType('scaled_upper_triang_masked_softmax_cuda')
            sys.modules['scaled_upper_triang_masked_softmax_cuda'] = fake_module

    # Patch deep_gemm if not available
    if 'deep_gemm' not in sys.modules:
        try:
            import deep_gemm
        except ImportError:
            # Create fake deep_gemm module with ceil_div fallback
            deep_gemm = ModuleType('deep_gemm')
            deep_gemm.ceil_div = lambda a, b: (a + b - 1) // b
            sys.modules['deep_gemm'] = deep_gemm


def _create_model_args(
    *,
    framework: str = "Megatron",
    world_size: int = 1,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel: int = 1,
    expert_model_parallel_size: int = 1,
    global_batch: int = 4,
    micro_batch: int = 1,
    num_layers: int = 24,
    hidden_size: int = 1024,
    seq_length: int = 2048,
    num_attention_heads: int | None = None,
    vocab_size: int = 32000,
    moe_enable: bool = False,
    num_experts: int = 1,
    moe_router_topk: int = 1,
    enable_sequence_parallel: bool = False,
    use_flash_attn: bool = False,
    swiglu: bool = False,
    use_distributed_optimizer: bool = False,
    gpu_type: str | None = None,
) -> argparse.Namespace:
    """Build an argparse.Namespace with AICB model configuration.

    This function computes derived parameters and constructs the args
    namespace that AICB's model creation code expects.

    Args:
        framework: Training framework (Megatron, DeepSpeed, or DeepSeek)
        world_size: Total number of GPUs
        tensor_model_parallel_size: Tensor parallelism degree
        pipeline_model_parallel: Pipeline parallelism degree
        expert_model_parallel_size: Expert parallelism degree (for MoE)
        global_batch: Global training batch size
        micro_batch: Micro-batch size per GPU
        num_layers: Number of transformer layers
        hidden_size: Transformer hidden dimension
        seq_length: Maximum sequence length
        num_attention_heads: Number of attention heads (default: num_layers)
        vocab_size: Vocabulary size
        moe_enable: Enable Mixture of Experts
        num_experts: Number of MoE experts
        moe_router_topk: Number of experts routed per token
        enable_sequence_parallel: Enable sequence parallelism
        use_flash_attn: Use FlashAttention
        swiglu: Use SwiGLU activation
        use_distributed_optimizer: Use distributed optimizer
        gpu_type: GPU type label for output naming

    Returns:
        argparse.Namespace with all AICB-required fields populated

    Raises:
        AssertionError: If configuration is invalid (e.g., world_size not
            divisible by tp*pp, or MoE without sequence parallelism)
    """
    # Validate configuration
    assert world_size % (tensor_model_parallel_size * pipeline_model_parallel) == 0, (
        f"world_size ({world_size}) must be divisible by tp*pp "
        f"({tensor_model_parallel_size}*{pipeline_model_parallel})"
    )
    if moe_enable:
        assert enable_sequence_parallel, "MoE requires --sequence-parallel"

    # Compute derived values
    dp_num = world_size // (tensor_model_parallel_size * pipeline_model_parallel)
    num_microbatches = global_batch // (dp_num * micro_batch)

    if num_attention_heads is None:
        num_attention_heads = num_layers

    padded_vocab_size = _get_padded_vocab_size(vocab_size, tensor_model_parallel_size)
    ffn_hidden_size = _compute_ffn_hidden_size(hidden_size, swiglu)

    # Adjust num_layers for pipeline parallelism
    effective_num_layers = num_layers
    if pipeline_model_parallel > 1:
        effective_num_layers = num_layers // pipeline_model_parallel

    # Build the args namespace that AICB expects
    args = argparse.Namespace(
        frame=framework,
        world_size=world_size,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel=pipeline_model_parallel,
        expert_model_parallel_size=expert_model_parallel_size,
        global_batch=global_batch,
        micro_batch=micro_batch,
        num_layers=effective_num_layers,
        hidden_size=hidden_size,
        seq_length=seq_length,
        num_attention_heads=num_attention_heads,
        vocab_size=vocab_size,
        padded_vocab_size=padded_vocab_size,
        ffn_hidden_size=ffn_hidden_size,
        dp_num=dp_num,
        num_microbatches=num_microbatches,
        # MoE
        moe_enable=moe_enable,
        num_experts=num_experts,
        moe_router_topk=moe_router_topk,
        moe_grouped_gemm=False,
        # Optimizations
        enable_sequence_parallel=enable_sequence_parallel,
        use_flash_attn=use_flash_attn,
        swiglu=swiglu,
        gated_linear_unit=swiglu,
        use_distributed_optimizer=use_distributed_optimizer,
        # Misc defaults
        add_bias_linear=False,
        dtype="bfloat16",
        model_name=gpu_type or "default",
        gpu_type=gpu_type or "default",
        max_position_embeddings=4096,
        make_vocab_size_divisible_by=128,
        recompute_activations=False,
        bias_gelu_fusion=False,
        openai_gelu=False,
        onnx_safe=False,
        squared_relu=False,
        overlap_version=False,
        context_parallel_size=1,
        activation_func=None,
        enable_visual=False,
        # DeepSeek-specific defaults
        n_dense_layers=3,
        n_shared_expert=2,
        qk_rope_dim=64,
        qk_nope_dim=128,
        q_lora_rank=1536,
        kv_lora_rank=512,
        v_head_dim=128,
    )

    return args


def _create_model(args: argparse.Namespace):
    """Instantiate a mocked model from AICB.

    Args:
        args: argparse.Namespace with model configuration (from _create_model_args)

    Returns:
        MegatronModel or DeepSeekV3Model instance based on args.frame

    Raises:
        ImportError: If AICB cannot be found or imported
    """
    # Patch optional CUDA modules before importing aicb code
    _patch_optional_cuda_modules()

    with _aicb_on_path():
        from workload_generator.mocked_model.training.MockedMegatron import (
            MegatronModel,
        )
        from workload_generator.mocked_model.training.MockedDeepSeek import (
            DeepSeekV3Model,
        )

        # Build model
        if args.frame == "DeepSeek":
            model = DeepSeekV3Model(args)
        else:
            model = MegatronModel(args)

        return model


def profile_gpu_kernels(
    *,
    framework: str = "Megatron",
    world_size: int = 1,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel: int = 1,
    expert_model_parallel_size: int = 1,
    global_batch: int = 4,
    micro_batch: int = 1,
    num_layers: int = 24,
    hidden_size: int = 1024,
    seq_length: int = 2048,
    num_attention_heads: int | None = None,
    vocab_size: int = 32000,
    moe_enable: bool = False,
    num_experts: int = 1,
    moe_router_topk: int = 1,
    enable_sequence_parallel: bool = False,
    use_flash_attn: bool = False,
    swiglu: bool = False,
    use_distributed_optimizer: bool = False,
    gpu_type: str | None = None,
    output: Path | None = None,
) -> Path:
    """Profile GPU kernel execution times for a model configuration.

    This function profiles GPU kernels using AICB's profiling infrastructure
    and saves the results to a file that can be reused for multiple workload
    generations via the --compute-profile flag.

    Requirements:
        - PyTorch with CUDA support: pip install torch
        - CUDA-capable GPU available
        - AICB source code accessible (see _find_aicb_root)

    Args:
        framework: Training framework (Megatron, DeepSpeed, or DeepSeek)
        world_size: Total number of GPUs
        tensor_model_parallel_size: Tensor parallelism degree
        pipeline_model_parallel: Pipeline parallelism degree
        expert_model_parallel_size: Expert parallelism degree (for MoE)
        global_batch: Global training batch size
        micro_batch: Micro-batch size per GPU
        num_layers: Number of transformer layers
        hidden_size: Transformer hidden dimension
        seq_length: Maximum sequence length
        num_attention_heads: Number of attention heads (default: num_layers)
        vocab_size: Vocabulary size
        moe_enable: Enable Mixture of Experts
        num_experts: Number of MoE experts
        moe_router_topk: Number of experts routed per token
        enable_sequence_parallel: Enable sequence parallelism
        use_flash_attn: Use FlashAttention
        swiglu: Use SwiGLU activation
        use_distributed_optimizer: Use distributed optimizer
        gpu_type: GPU type label (e.g., H100, A100) for output naming
        output: Output file path (default: auto-generated in results/profiles/)

    Returns:
        Path to the generated profile file

    Raises:
        ImportError: If PyTorch is not installed
        RuntimeError: If CUDA is not available or no GPU is found

    Example:
        >>> profile_path = profile_gpu_kernels(
        ...     framework="Megatron",
        ...     world_size=64,
        ...     tensor_model_parallel_size=4,
        ...     num_layers=32,
        ...     hidden_size=4096,
        ...     gpu_type="H100",
        ...     output=Path("h100_profile.txt")
        ... )
        >>> print(f"Profile saved to: {profile_path}")
    """
    # Check PyTorch availability
    if importlib.util.find_spec("torch") is None:
        raise ImportError(
            "PyTorch is required for GPU profiling.\n"
            "Install with: pip install 'simai[profiling]'\n"
            "Or: pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )

    import torch

    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU profiling requires a CUDA-capable GPU.\n"
            "Ensure CUDA drivers are installed and a GPU is available.\n"
            "Run 'nvidia-smi' to verify GPU accessibility."
        )

    # Create model configuration
    args = _create_model_args(
        framework=framework,
        world_size=world_size,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel=pipeline_model_parallel,
        expert_model_parallel_size=expert_model_parallel_size,
        global_batch=global_batch,
        micro_batch=micro_batch,
        num_layers=num_layers,
        hidden_size=hidden_size,
        seq_length=seq_length,
        num_attention_heads=num_attention_heads,
        vocab_size=vocab_size,
        moe_enable=moe_enable,
        num_experts=num_experts,
        moe_router_topk=moe_router_topk,
        enable_sequence_parallel=enable_sequence_parallel,
        use_flash_attn=use_flash_attn,
        swiglu=swiglu,
        use_distributed_optimizer=use_distributed_optimizer,
        gpu_type=gpu_type,
    )

    # Set profiling-specific fields required by AICB (must be set before creating model)
    args.computation_enable = True  # Enable computation profiling
    args.aiob_enable = True  # Enable AIOB profiling
    args.comp_filepath = None  # No pre-existing profile
    args.epoch_num = 1  # Number of iterations for profiling
    args.pp_rank = -1  # Pipeline parallel rank

    # Create model
    model = _create_model(args)

    # Count model parameters
    args.model_param = sum(p.numel() for p in model.parameters())

    # Profile GPU kernels using AICB
    with _aicb_on_path():
        from utils.utils import get_comp_out

        comp_filepath = get_comp_out(args)

    # Determine output path
    if output is not None:
        profile_path = Path(output)
    else:
        # Auto-generate to results/profiles/
        result_dir = Path("results/profiles")
        result_dir.mkdir(parents=True, exist_ok=True)

        # Generate descriptive filename
        model_name = gpu_type or "default"
        filename = (
            f"{model_name}-{framework}-world_size{world_size}"
            f"-tp{tensor_model_parallel_size}-pp{pipeline_model_parallel}"
            f"-ep{expert_model_parallel_size}-gbs{global_batch}"
            f"-mbs{micro_batch}-seq{seq_length}"
            f"-layers{num_layers}-hidden{hidden_size}"
        )
        if moe_enable:
            filename += f"-moe{num_experts}-topk{moe_router_topk}"
        filename += ".txt"

        profile_path = result_dir / filename

    # Copy profile to output location if needed
    if comp_filepath != str(profile_path):
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(comp_filepath, profile_path)

    return profile_path
