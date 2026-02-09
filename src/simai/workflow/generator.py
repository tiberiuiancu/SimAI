from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path


def _find_aicb_root() -> Path:
    """Locate the AICB source tree.

    Search order:
    1. Vendored into the package at build time: simai/_vendor/aicb/
    2. SIMAI_PATH environment variable (points to the SimAI repo root)
    3. Sibling directory: ../simai/aicb (relative to this package)
    """
    # 1. Vendored
    vendored = Path(__file__).resolve().parent.parent / "_vendor" / "aicb"
    if vendored.is_dir():
        return vendored

    # 2. SIMAI_PATH env var
    env_path = os.environ.get("SIMAI_PATH")
    if env_path:
        candidate = Path(env_path) / "aicb"
        if candidate.is_dir():
            return candidate

    # 3. Sibling directory heuristic
    sibling = Path(__file__).resolve().parent.parent.parent.parent.parent / "simai" / "aicb"
    if sibling.is_dir():
        return sibling

    raise FileNotFoundError(
        "Cannot find AICB source code. Either:\n"
        "  - Install from a wheel that includes vendored AICB, or\n"
        "  - Set SIMAI_PATH to the SimAI repository root, or\n"
        "  - Clone SimAI as a sibling directory."
    )


@contextmanager
def _aicb_on_path():
    """Temporarily add the AICB root to sys.path so its internal imports work."""
    aicb_root = str(_find_aicb_root())
    inserted = aicb_root not in sys.path
    if inserted:
        sys.path.insert(0, aicb_root)
    try:
        yield aicb_root
    finally:
        if inserted and aicb_root in sys.path:
            sys.path.remove(aicb_root)


def _get_padded_vocab_size(vocab_size: int, tp: int, divisible_by: int = 128) -> int:
    """Pad vocab size to be divisible by tp * divisible_by."""
    multiple = divisible_by * tp
    after = vocab_size
    while after % multiple != 0:
        after += 1
    return after


def _compute_ffn_hidden_size(hidden_size: int, swiglu: bool) -> int:
    if swiglu:
        return int((4 * hidden_size * 2 / 3) / 64) * 64
    return 4 * hidden_size


def generate_workload(
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
    epoch_num: int = 1,
    aiob_enable: bool = False,
    comp_filepath: str | None = None,
    gpu_type: str | None = None,
    output: Path | None = None,
) -> Path:
    """Generate a SimAI workload file by driving AICB's code directly.

    Returns the path to the generated workload file.
    """
    # Import shared profiling functions
    from simai.workflow.profiler import _create_model_args, _create_model, _patch_optional_cuda_modules

    # Patch optional CUDA modules before any aicb imports
    _patch_optional_cuda_modules()

    # Create model configuration using shared function
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

    # Add workload-specific fields
    args.computation_enable = False
    args.aiob_enable = aiob_enable
    args.comp_filepath = comp_filepath
    args.workload_only = True
    args.epoch_num = epoch_num
    args.pp_rank = -1

    with _aicb_on_path():
        from workload_generator.SimAI_training_workload_generator import SIMAI_workload

        # Build model using shared function
        model = _create_model(args)

        # Handle compute profiling
        compute_cache = None
        if aiob_enable:
            from utils.utils import get_comp_out, extract_averages

            params = model.parameters()
            args.model_param = sum(p.numel() for p in params)
            if comp_filepath is None:
                comp_filepath = get_comp_out(args)
            compute_cache = extract_averages(comp_filepath, args)

        # Generate workload
        # AICB's workload_generate() has bugs where it references bare `model`
        # and `args` as free variables (designed to run as __main__). We inject
        # them into the module's global scope so the code works.
        import workload_generator.SimAI_training_workload_generator as _wg_mod
        _wg_mod.model = model
        _wg_mod.args = args

        work = SIMAI_workload(model, args, compute_cache)
        if aiob_enable:
            work.workload_generate_aiob()
            # Zero out comm_size for NONE comm types
            for item in work.workload:
                if item.forward_comm == "NONE":
                    item.forward_comm_size = 0
                if item.backward_comm == "NONE":
                    item.backward_comm_size = 0
        else:
            work.workload_generate()

        # Determine output path
        if output is not None:
            filepath = Path(output)
            # Strip .txt suffix if provided; dump_file adds it
            if filepath.suffix == ".txt":
                filepath = filepath.with_suffix("")
            filepath.parent.mkdir(parents=True, exist_ok=True)
        else:
            result_dir = Path("results/workload")
            result_dir.mkdir(parents=True, exist_ok=True)
            filename = (
                f"{args.gpu_type}-{args.model_name}-world_size{world_size}"
                f"-tp{tensor_model_parallel_size}-pp{pipeline_model_parallel}"
                f"-ep{expert_model_parallel_size}-gbs{global_batch}"
                f"-mbs{micro_batch}-seq{seq_length}"
                f"-MOE-{moe_enable}-GEMM-False"
                f"-flash_attn-{use_flash_attn}"
            )
            filepath = result_dir / filename

        work.dump_file(str(filepath))
        final_path = Path(str(filepath) + ".txt")
        print(f"Workload saved to: {final_path}")
        return final_path
