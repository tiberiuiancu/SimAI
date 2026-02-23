"""Unified TOML run configuration for SimAI.

Supports loading a run.toml file, merging CLI overrides, and serializing
to a flat dict for experiment tracker logging (MLflow, W&B, etc.).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Config section dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    output: Optional[str] = None
    verbose: bool = False


@dataclass
class WorkloadConfig:
    # Option A – pre-existing file
    file: Optional[str] = None
    # Option B – inline generation params
    framework: str = "Megatron"
    world_size: int = 1
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    expert_parallel: int = 1
    global_batch: int = 4
    micro_batch: int = 1
    num_layers: int = 24
    hidden_size: int = 1024
    seq_length: int = 2048
    num_attention_heads: Optional[int] = None
    vocab_size: int = 32000
    gpu_type: Optional[str] = None
    use_flash_attn: bool = False


@dataclass
class TopologyConfig:
    # Option A – pre-existing file
    file: Optional[str] = None
    # Option B – inline generation params
    type: Optional[str] = None
    num_gpus: Optional[int] = None
    gpus_per_server: Optional[int] = None
    nic_bandwidth_gbps: Optional[float] = None
    nvlink_bandwidth_gbps: Optional[float] = None
    nics_per_switch: Optional[int] = None


@dataclass
class ComputeProfileConfig:
    file: Optional[str] = None


@dataclass
class SimulationConfig:
    # Shared
    threads: int = 8
    # Analytical overlap ratios
    dp_overlap: Optional[float] = None
    tp_overlap: Optional[float] = None
    ep_overlap: Optional[float] = None
    pp_overlap: Optional[float] = None
    # NS-3 specific
    send_latency: Optional[int] = None
    nvls: bool = False
    pxn: bool = False


@dataclass
class NS3Config:
    """NS-3 SimAI.conf parameters — all keys map 1:1 to SimAI.conf lines."""
    ENABLE_QCN: int = 1
    USE_DYNAMIC_PFC_THRESHOLD: int = 1
    PACKET_PAYLOAD_SIZE: int = 9000
    SIMULATOR_STOP_TIME: float = 40000000000000.0
    CC_MODE: int = 1
    ALPHA_RESUME_INTERVAL: int = 1
    RATE_DECREASE_INTERVAL: int = 4
    CLAMP_TARGET_RATE: int = 0
    RP_TIMER: int = 900
    EWMA_GAIN: float = 0.00390625
    FAST_RECOVERY_TIMES: int = 1
    RATE_AI: str = "50Mb/s"
    RATE_HAI: str = "100Mb/s"
    MIN_RATE: str = "100Mb/s"
    DCTCP_RATE_AI: str = "1000Mb/s"
    ERROR_RATE_PER_LINK: float = 0.0
    L2_CHUNK_SIZE: int = 4000
    L2_ACK_INTERVAL: int = 1
    L2_BACK_TO_ZERO: int = 0
    HAS_WIN: int = 1
    GLOBAL_T: int = 0
    VAR_WIN: int = 1
    FAST_REACT: int = 1
    U_TARGET: float = 0.95
    MI_THRESH: int = 0
    INT_MULTI: int = 1
    MULTI_RATE: int = 0
    SAMPLE_FEEDBACK: int = 0
    PINT_LOG_BASE: float = 1.05
    PINT_PROB: float = 1.0
    RATE_BOUND: int = 1
    ACK_HIGH_PRIO: int = 0
    LINK_DOWN: str = "0 0 0"
    ENABLE_TRACE: int = 1
    KMAX_MAP: str = "6 25000000000 400 50000000000 800 100000000000 1600 200000000000 1200 400000000000 3200 1600000000000 2400"
    KMIN_MAP: str = "6 25000000000 100 50000000000 200 100000000000 400 200000000000 300 400000000000 800 1600000000000 600"
    PMAX_MAP: str = "6 25000000000 0.2 50000000000 0.2 100000000000 0.2 200000000000 0.8 400000000000 0.2 1600000000000 0.2"
    BUFFER_SIZE: int = 32
    MON_START: int = 0
    MON_END: int = 20000
    QP_MON_INTERVAL: int = 100
    QLEN_MON_INTERVAL: int = 10000
    BW_MON_INTERVAL: int = 10000


@dataclass
class SimaiConfig:
    run: RunConfig = field(default_factory=RunConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    compute_profile: ComputeProfileConfig = field(default_factory=ComputeProfileConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    ns3: NS3Config = field(default_factory=NS3Config)


# ---------------------------------------------------------------------------
# TOML loading
# ---------------------------------------------------------------------------

def _load_toml(path: Path) -> dict:
    if sys.version_info >= (3, 11):
        import tomllib
        with open(path, "rb") as f:
            return tomllib.load(f)
    else:
        try:
            import tomli
            with open(path, "rb") as f:
                return tomli.load(f)
        except ImportError:
            raise ImportError(
                "Python <3.11 requires 'tomli' to read TOML files.\n"
                "Install it with: pip install tomli"
            )


def _apply_section(dataclass_obj, raw: dict) -> None:
    """Apply raw dict values to a dataclass in-place, ignoring unknown keys."""
    valid = {f.name for f in fields(dataclass_obj)}
    for key, value in raw.items():
        if key in valid:
            setattr(dataclass_obj, key, value)


def load_config(path: Path) -> SimaiConfig:
    """Load a TOML run config file and return a SimaiConfig."""
    raw = _load_toml(Path(path))
    cfg = SimaiConfig()
    if "run" in raw:
        _apply_section(cfg.run, raw["run"])
    if "workload" in raw:
        _apply_section(cfg.workload, raw["workload"])
    if "topology" in raw:
        _apply_section(cfg.topology, raw["topology"])
    if "compute_profile" in raw:
        _apply_section(cfg.compute_profile, raw["compute_profile"])
    if "simulation" in raw:
        _apply_section(cfg.simulation, raw["simulation"])
    if "ns3" in raw:
        _apply_section(cfg.ns3, raw["ns3"])
    return cfg


def merge_cli(config: SimaiConfig, **overrides: Any) -> SimaiConfig:
    """Apply non-None CLI override values onto config sections.

    Each override key is expected to be in the form `section__key` or just `key`.
    Supported shortcuts:
      output, verbose → run
      workload → workload.file
      topology → topology.file
      threads, dp_overlap, tp_overlap, ep_overlap, pp_overlap,
      send_latency, nvls, pxn → simulation
    """
    mapping = {
        "output": ("run", "output"),
        "verbose": ("run", "verbose"),
        "workload": ("workload", "file"),
        "topology": ("topology", "file"),
        "threads": ("simulation", "threads"),
        "dp_overlap": ("simulation", "dp_overlap"),
        "tp_overlap": ("simulation", "tp_overlap"),
        "ep_overlap": ("simulation", "ep_overlap"),
        "pp_overlap": ("simulation", "pp_overlap"),
        "send_latency": ("simulation", "send_latency"),
        "nvls": ("simulation", "nvls"),
        "pxn": ("simulation", "pxn"),
    }
    for key, value in overrides.items():
        if value is None:
            continue
        if key in mapping:
            section_name, attr = mapping[key]
            section = getattr(config, section_name)
            setattr(section, attr, value)
    return config


def to_flat_dict(config: SimaiConfig) -> dict[str, Any]:
    """Flatten the config to 'section.key' format for experiment tracker logging."""
    result: dict[str, Any] = {}
    for section_field in fields(config):
        section = getattr(config, section_field.name)
        for f in fields(section):
            result[f"{section_field.name}.{f.name}"] = getattr(section, f.name)
    return result


def ns3_config_to_dict(ns3_cfg: NS3Config) -> dict[str, Any]:
    """Return the NS3Config as a plain dict (for rendering SimAI.conf)."""
    return asdict(ns3_cfg)


def render_simai_conf(ns3_cfg: NS3Config, tmpdir: str) -> str:
    """Render the NS3Config to SimAI.conf text, with path placeholders filled in.

    FLOW_FILE, TRACE_FILE, and output files are set to tmpdir-relative paths.
    """
    import os
    lines = []
    d = ns3_config_to_dict(ns3_cfg)
    for key, value in d.items():
        lines.append(f"{key} {value}")
    # Add path-dependent entries that the binary hardcodes
    lines += [
        f"FLOW_FILE {os.path.join(tmpdir, 'flow1.txt')}",
        f"TRACE_FILE {os.path.join(tmpdir, 'trace1.txt')}",
        f"FCTFILE {os.path.join(tmpdir, 'fct.txt')}",
        f"PFC_OUTPUT_FILE {os.path.join(tmpdir, 'pfc.txt')}",
        f"QLEN_MON_FILE {os.path.join(tmpdir, 'qlen.txt')}",
        f"BW_MON_FILE {os.path.join(tmpdir, 'bw.txt')}",
        f"QP_MON_FILE {os.path.join(tmpdir, 'qp.txt')}",
    ]
    return "\n".join(lines) + "\n"
