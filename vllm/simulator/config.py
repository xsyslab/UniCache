# SPDX-License-Identifier: Apache-2.0
# config.py
import os
from dataclasses import dataclass

from vllm.simulator.opt_utils import (GLOBAL_OFFLINE_OPT_HELPER,
                                      GLOBAL_ONLINE_OPT_HELPER)
from vllm.simulator.wa_helper import GLOBAL_WA_HELPER


def setup_simulator_config(args):
    """
    Initialize global simulator configuration and WA helper from argparse arguments.

    Args:
        args: argparse.Namespace or any object with simulator_* attributes.
    """
    # === Update global simulator configuration ===
    GLOBAL_SIMULATOR_CONFIG.simulator_output_dir = args.simulator_output_dir
    GLOBAL_SIMULATOR_CONFIG.simulator_num_gpu_blocks = args.simulator_num_gpu_blocks
    GLOBAL_SIMULATOR_CONFIG.simulator_num_cpu_blocks = args.simulator_num_cpu_blocks
    GLOBAL_SIMULATOR_CONFIG.simulator_enable_virtual_time = args.simulator_enable_virtual_time
    GLOBAL_SIMULATOR_CONFIG.eviction_policy = args.eviction_policy
    GLOBAL_SIMULATOR_CONFIG.disable_simulation = args.disable_simulation
    GLOBAL_SIMULATOR_CONFIG.enable_lmcache = args.enable_lmcache
    GLOBAL_SIMULATOR_CONFIG.enable_block_hit_track = args.enable_block_hit_track
    # WA
    GLOBAL_SIMULATOR_CONFIG.simulator_wa_fit_interval = args.simulator_wa_fit_interval
    # OPT
    GLOBAL_SIMULATOR_CONFIG.opt_block_use_map_path = args.opt_block_use_map_path
    # OFFSET
    if GLOBAL_SIMULATOR_CONFIG.eviction_policy in ("OFFSET-TAIL",
                                                   "OFFSET-HEAD"):
        GLOBAL_SIMULATOR_CONFIG.tail_first = "TAIL" in GLOBAL_SIMULATOR_CONFIG.eviction_policy
        GLOBAL_SIMULATOR_CONFIG.head_first = "HEAD" in GLOBAL_SIMULATOR_CONFIG.eviction_policy
        assert (GLOBAL_SIMULATOR_CONFIG.tail_first
                ^ GLOBAL_SIMULATOR_CONFIG.head_first
                ), "Exactly one of tail_first/head_first must be True"
        assert ( GLOBAL_SIMULATOR_CONFIG.tail_first or GLOBAL_SIMULATOR_CONFIG.head_first), \
       "Exactly one of --tail_first and --head_first must be set"

    # === Sync settings with WA helper ===
    GLOBAL_WA_HELPER.simulator_enable_virtual_time = args.simulator_enable_virtual_time
    GLOBAL_WA_HELPER.fit_interval = args.simulator_wa_fit_interval

    # === Sync settings with OPT Helper ===
    if GLOBAL_SIMULATOR_CONFIG.eviction_policy == "OPT":
        assert os.getenv(
            "PYTHONHASHSEED"
        ) == "0", "OPT requires PYTHONHASHSEED=0 to make hash deterministic"
        GLOBAL_OFFLINE_OPT_HELPER.set_reuse_track(args.opt_block_use_map_path)
    if GLOBAL_SIMULATOR_CONFIG.eviction_policy == "OPT-Online":
        GLOBAL_ONLINE_OPT_HELPER.block_size = args.block_size
    # HYBRID
    if GLOBAL_SIMULATOR_CONFIG.eviction_policy in ("SimpleHYBRID",
                                                   "SimpleHYBRID-OPT"):
        if "OPT" in GLOBAL_SIMULATOR_CONFIG.eviction_policy:
            GLOBAL_ONLINE_OPT_HELPER.block_size = args.block_size
    # === Sanity checks: device & virtual time consistency ===
    if GLOBAL_SIMULATOR_CONFIG.disable_simulation:
        # Simulation disabled means using CUDA real hardware
        if args.device != "cuda":
            raise ValueError(
                f"[Simulator Config Error] disable_simulation=True requires device='cuda', "
                f"but got device='{args.device}'.")
        if GLOBAL_SIMULATOR_CONFIG.simulator_enable_virtual_time:
            raise ValueError(
                "[Simulator Config Error] disable_simulation=True is incompatible with "
                "simulator_enable_virtual_time=True.")
    else:
        # Simulation enabled means virtual-time scheduling + CPU engine
        if args.device != "cpu":
            raise ValueError(
                f"[Simulator Config Error] disable_simulation=False requires device='cpu', "
                f"but got device='{args.device}'.")

        if not GLOBAL_SIMULATOR_CONFIG.simulator_enable_virtual_time:
            raise ValueError(
                "[Simulator Config Error] Simulation mode requires "
                "simulator_enable_virtual_time=True.")
    if GLOBAL_SIMULATOR_CONFIG.enable_lmcache:
        if args.device != "cuda":
            raise ValueError(
                f"[Simulator Config Error] enable_lmcache=True requires device='cuda', "
                f"but got device='{args.device}'.")

    # === Print full simulator configuration ===
    print("[Simulator Config] Initialized:")
    print(
        f"  Output dir           : {GLOBAL_SIMULATOR_CONFIG.simulator_output_dir}"
    )
    print(
        f"  GPU blocks           : {GLOBAL_SIMULATOR_CONFIG.simulator_num_gpu_blocks}"
    )
    print(
        f"  CPU blocks           : {GLOBAL_SIMULATOR_CONFIG.simulator_num_cpu_blocks}"
    )
    print(
        f"  Virtual time         : {GLOBAL_SIMULATOR_CONFIG.simulator_enable_virtual_time}"
    )
    print(
        f"  WA fit interval      : {GLOBAL_SIMULATOR_CONFIG.simulator_wa_fit_interval}"
    )
    print(
        f"  Eviction policy      : {GLOBAL_SIMULATOR_CONFIG.eviction_policy}")
    print(
        f"  Disable simulation   : {GLOBAL_SIMULATOR_CONFIG.disable_simulation}"
    )


@dataclass
class SimulatorConfig:
    simulator_num_gpu_blocks: int = 100000
    simulator_num_cpu_blocks: int = 100000
    simulator_output_dir: str = "tmp"
    simulator_enable_virtual_time: bool = False
    simulator_wa_fit_interval: int = 300
    eviction_policy: str = "FIFO"
    disable_simulation: bool = False
    enable_block_hit_track: bool = False
    # OPT(offline)
    opt_block_use_map_path: str = None
    # === New: LMCache flags ===
    enable_lmcache: bool = False

    tail_first: bool = False
    head_first: bool = False


GLOBAL_SIMULATOR_CONFIG = SimulatorConfig()
