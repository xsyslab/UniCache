# SPDX-License-Identifier: Apache-2.0
import time
from collections import defaultdict

import numpy as np
from scipy.stats import expon

from vllm.logger import init_logger
from vllm.simulator.timer_utils import GLOBAL_TIMER

logger = init_logger(__name__)


class WAHelper:
    """
    WAHelper:
    Collects reuse-time intervals for each task type,
    and automatically fits exponential distributions
    when get_reuse_probability() is called.
    """

    def __init__(self, fit_interval: float = 5 * 60):
        self.reuse_time = defaultdict(list)  # {task_name: [intervals]}
        self.fit_interval = fit_interval
        self.fitted_params = {}  # Fitted parameters: {task: (λ, mean)}
        self.simulator_enable_virtual_time = True  # Whether to use the global virtual timer
        self.last_fit_time = None  # Timestamp of the last successful fit
        # collect all data points for each task type. for final analysis
        self.all_data = defaultdict(list)  # All data points for each task type

    def _get_time(self) -> float:
        """Return the current time (virtual or real)."""
        return GLOBAL_TIMER.get_current_time(
        ) if self.simulator_enable_virtual_time else time.time()

    def _fit_all(self, clear_after_fit: bool = True):
        """
        Fit exponential distributions for all collected task intervals.

        Args:
            clear_after_fit (bool): Whether to clear the collected data
                                    after fitting to avoid stale samples.
        """
        fitted_count = 0
        for task, intervals in list(self.reuse_time.items()):
            if len(intervals) < 5:
                continue  # Not enough samples for reliable fitting
            data = np.array(intervals)
            # Fit an exponential distribution: loc=0 fixes the origin
            loc, scale = expon.fit(data, floc=0)
            lam = 1 / scale
            self.fitted_params[task] = (lam, scale)
            fitted_count += 1
            if self.last_fit_time is None:
                logger.info(f"[WA Fit] First fit at time {self._get_time()}")

            logger.info(
                f"[WA Fit] Task={task:<15} λ={lam:.5f}, mean={scale:.5f}, samples={len(data)}"
            )
            if clear_after_fit:
                # Clear historical intervals after fitting
                self.reuse_time[task].clear()
                logger.info(f"[WA Clear] Task={task:<15} samples={len(data)}")
        # Update the timestamp only if at least one task was fitted
        if fitted_count > 0:
            self.last_fit_time = self._get_time()

    # === Public interfaces ===
    def collect_hit_interval(self, task_name: str, reuse_time: float):
        #  Record the reuse-time interval for a given task.
        if reuse_time < 0:
            raise ValueError(
                f"[WA] Reuse time must be positive (got {reuse_time}).")
        if reuse_time == 0:
            logger.warning(f"[WA] Reuse time is 0 for task={task_name}.")
        self.reuse_time[task_name].append(reuse_time)
        self.all_data[task_name].append(reuse_time)

    def get_reuse_probability(self, task_name: str, t: float) -> float:
        """
        Query reuse probability P(reuse within t).
        If fitting is outdated or not yet done, perform fitting automatically.
        """
        cur = self._get_time()

        if (self.last_fit_time is None) or (cur - self.last_fit_time
                                            >= self.fit_interval):

            self._fit_all(clear_after_fit=True)

        if task_name not in self.fitted_params:
            return 0.0
        lam, _ = self.fitted_params[task_name]
        cdf = 1 - np.exp(-lam * t)
        reuse_prob = 1 - cdf
        logger.info(
            f"[WA] Reuse probability for {task_name} at {t} is {reuse_prob}")
        return reuse_prob


GLOBAL_WA_HELPER = WAHelper()
