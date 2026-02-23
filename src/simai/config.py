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
class AnalyticalConfig:
    """Parameters specific to the analytical backend."""
    dp_overlap: Optional[float] = None   # DP comm overlap ratio (0.0–1.0)
    tp_overlap: Optional[float] = None   # TP comm overlap ratio
    ep_overlap: Optional[float] = None   # EP comm overlap ratio
    pp_overlap: Optional[float] = None   # PP comm overlap ratio


@dataclass
class M4Config:
    """Parameters specific to the M4 backend."""
    threads: int = 1


@dataclass
class NS3Config:
    """Parameters for the NS-3 backend.

    Runtime flags (threads, send_latency, nvls, pxn) and all SimAI.conf keys.
    All SimAI.conf keys map 1:1 to lines in the generated SimAI.conf file.
    """
    # Runtime flags
    threads: int = 8
    send_latency: Optional[int] = None   # microseconds (AS_SEND_LAT env var)
    nvls: bool = False                   # NVLink Switch (AS_NVLS_ENABLE)
    pxn: bool = False                    # PCIe cross-node (AS_PXN_ENABLE)

    # SimAI.conf parameters
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


# SimAI.conf keys — the subset of NS3Config fields that go into the conf file
_NS3_CONF_KEYS = {
    "ENABLE_QCN", "USE_DYNAMIC_PFC_THRESHOLD", "PACKET_PAYLOAD_SIZE",
    "SIMULATOR_STOP_TIME", "CC_MODE", "ALPHA_RESUME_INTERVAL",
    "RATE_DECREASE_INTERVAL", "CLAMP_TARGET_RATE", "RP_TIMER", "EWMA_GAIN",
    "FAST_RECOVERY_TIMES", "RATE_AI", "RATE_HAI", "MIN_RATE", "DCTCP_RATE_AI",
    "ERROR_RATE_PER_LINK", "L2_CHUNK_SIZE", "L2_ACK_INTERVAL", "L2_BACK_TO_ZERO",
    "HAS_WIN", "GLOBAL_T", "VAR_WIN", "FAST_REACT", "U_TARGET", "MI_THRESH",
    "INT_MULTI", "MULTI_RATE", "SAMPLE_FEEDBACK", "PINT_LOG_BASE", "PINT_PROB",
    "RATE_BOUND", "ACK_HIGH_PRIO", "LINK_DOWN", "ENABLE_TRACE",
    "KMAX_MAP", "KMIN_MAP", "PMAX_MAP", "BUFFER_SIZE",
    "MON_START", "MON_END", "QP_MON_INTERVAL", "QLEN_MON_INTERVAL", "BW_MON_INTERVAL",
}


@dataclass
class SimaiConfig:
    run: RunConfig = field(default_factory=RunConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    compute_profile: ComputeProfileConfig = field(default_factory=ComputeProfileConfig)
    analytical: AnalyticalConfig = field(default_factory=AnalyticalConfig)
    ns3: NS3Config = field(default_factory=NS3Config)
    m4: M4Config = field(default_factory=M4Config)


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
    if "analytical" in raw:
        _apply_section(cfg.analytical, raw["analytical"])
    if "ns3" in raw:
        _apply_section(cfg.ns3, raw["ns3"])
    if "m4" in raw:
        _apply_section(cfg.m4, raw["m4"])
    return cfg


def merge_cli(config: SimaiConfig, **overrides: Any) -> SimaiConfig:
    """Apply non-None CLI override values onto config sections.

    Supported keys and their target section.field:
      output, verbose            → run
      workload                   → workload.file
      topology                   → topology.file
      dp_overlap, tp_overlap,
      ep_overlap, pp_overlap     → analytical
      threads                    → ns3.threads  (or m4.threads — caller decides)
      send_latency, nvls, pxn    → ns3
    """
    mapping = {
        "output": ("run", "output"),
        "verbose": ("run", "verbose"),
        "workload": ("workload", "file"),
        "topology": ("topology", "file"),
        "dp_overlap": ("analytical", "dp_overlap"),
        "tp_overlap": ("analytical", "tp_overlap"),
        "ep_overlap": ("analytical", "ep_overlap"),
        "pp_overlap": ("analytical", "pp_overlap"),
        "send_latency": ("ns3", "send_latency"),
        "nvls": ("ns3", "nvls"),
        "pxn": ("ns3", "pxn"),
    }
    for key, value in overrides.items():
        if value is None:
            continue
        if key in mapping:
            section_name, attr = mapping[key]
            section = getattr(config, section_name)
            setattr(section, attr, value)
        elif key == "threads_ns3":
            config.ns3.threads = value
        elif key == "threads_m4":
            config.m4.threads = value
    return config


def to_flat_dict(config: SimaiConfig) -> dict[str, Any]:
    """Flatten the config to 'section.key' format for experiment tracker logging."""
    result: dict[str, Any] = {}
    for section_field in fields(config):
        section = getattr(config, section_field.name)
        for f in fields(section):
            result[f"{section_field.name}.{f.name}"] = getattr(section, f.name)
    return result


def ns3_config_to_simai_conf_dict(ns3_cfg: NS3Config) -> dict[str, Any]:
    """Return only the SimAI.conf keys from an NS3Config (excludes runtime flags)."""
    return {k: v for k, v in asdict(ns3_cfg).items() if k in _NS3_CONF_KEYS}


def render_simai_conf(ns3_cfg: NS3Config, tmpdir: str) -> str:
    """Render the SimAI.conf-key subset of NS3Config to file text.

    Runtime flags (threads, send_latency, nvls, pxn) are excluded.
    Path-dependent entries (FLOW_FILE, TRACE_FILE, output files) are set
    to tmpdir-relative absolute paths.
    """
    import os
    lines = []
    for key, value in ns3_config_to_simai_conf_dict(ns3_cfg).items():
        lines.append(f"{key} {value}")
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
