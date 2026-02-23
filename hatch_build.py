"""Custom Hatch build hook for SimAI.

Vendors AICB Python code and pre-built binaries into the wheel.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:
        """Vendor AICB code and binaries into the source tree before building."""
        src_root = Path(self.root) / "src" / "simai"

        # --- Vendor AICB Python code ---
        aicb_src = Path(self.root) / "vendor" / "simai" / "aicb"
        aicb_dest = src_root / "_vendor" / "aicb"

        if aicb_src.is_dir():
            if aicb_dest.exists():
                shutil.rmtree(aicb_dest)
            # Copy the required subdirectories
            aicb_dest.mkdir(parents=True, exist_ok=True)
            for subdir in ("workload_generator", "utils", "log_analyzer", "training", "core"):
                src = aicb_src / subdir
                if src.is_dir():
                    shutil.copytree(src, aicb_dest / subdir, dirs_exist_ok=True)
            # Vendor all top-level .py modules (aicb.py, workload_applyer.py, etc.)
            for py_file in aicb_src.glob("*.py"):
                shutil.copy2(py_file, aicb_dest / py_file.name)

        # --- Vendor topology generator ---
        astrasim_src = Path(self.root) / "vendor" / "simai" / "astra-sim-alibabacloud"
        topo_src = astrasim_src / "inputs" / "topo" / "gen_Topo_Template.py"
        if topo_src.is_file():
            topo_dest = src_root / "_vendor" / "topo"
            topo_dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(topo_src, topo_dest / "gen_Topo_Template.py")
            (topo_dest / "__init__.py").touch()

        # --- Vendor auxiliary data files (ratio CSVs + SimAI.conf) ---
        if astrasim_src.is_dir():
            # Ratio CSV files
            ratio_src = astrasim_src / "inputs" / "ratio"
            if ratio_src.is_dir():
                ratio_dest = src_root / "_vendor" / "astra-sim-alibabacloud" / "inputs" / "ratio"
                ratio_dest.mkdir(parents=True, exist_ok=True)
                for csv_file in ratio_src.glob("*.csv"):
                    shutil.copy2(csv_file, ratio_dest / csv_file.name)

            # SimAI.conf
            conf_src = astrasim_src / "inputs" / "config" / "SimAI.conf"
            if conf_src.is_file():
                conf_dest = src_root / "_vendor" / "SimAI.conf"
                conf_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(conf_src, conf_dest)

        # --- Include pre-built binaries ---
        bin_dir = Path(self.root) / "build" / "bin"
        bin_dest = src_root / "_binaries"

        if bin_dir.is_dir():
            if bin_dest.exists():
                shutil.rmtree(bin_dest)
            bin_dest.mkdir(parents=True, exist_ok=True)
            for binary in bin_dir.iterdir():
                if binary.is_file():
                    dest = bin_dest / binary.name
                    shutil.copy2(binary, dest)
                    dest.chmod(dest.stat().st_mode | 0o111)
                    # Strip debug symbols to reduce wheel size
                    import subprocess as _sp
                    _sp.run(["strip", str(dest)], capture_output=True)

        # --- Force-include dynamically created directories ---
        # Hatchling uses git to decide what goes in the wheel, so files
        # created by this hook are excluded by default.
        force_include = build_data.setdefault("force_include", {})
        if (src_root / "_vendor").is_dir():
            force_include[str(src_root / "_vendor")] = "simai/_vendor"
        if (src_root / "_binaries").is_dir():
            force_include[str(src_root / "_binaries")] = "simai/_binaries"

        # --- Set platform tag if specified ---
        platform_tag = os.environ.get("SIMAI_PLATFORM_TAG")
        if platform_tag:
            build_data["tag"] = f"py3-none-{platform_tag}"

    def finalize(self, version: str, build_data: dict, artifact_path: str) -> None:
        """Clean up vendored files after build."""
        src_root = Path(self.root) / "src" / "simai"
        for dirname in ("_vendor", "_binaries"):
            vendored = src_root / dirname
            if vendored.exists():
                shutil.rmtree(vendored)
