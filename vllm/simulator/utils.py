# SPDX-License-Identifier: Apache-2.0
import os

import matplotlib.pyplot as plt
import numpy as np

from vllm.logger import init_logger

logger = init_logger(__name__)


def plot_hit_rates(hit_rates, output_dir):
    # 1. Hit Rate Scatter Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(hit_rates)), hit_rates, 'b.', markersize=1)
    plt.xlabel('Request ID')
    plt.ylabel('Hit Rate')
    plt.title(f'Hit Rates (First {len(hit_rates)} Prompts)')
    plt.grid(True)
    plt.savefig(f'{output_dir}/hit_rates.png')
    plt.close()

    # 2. Hit Rate CDF
    sorted_rates = np.sort(hit_rates)
    cdf = np.arange(1, len(sorted_rates) + 1) / len(sorted_rates)
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_rates, cdf, color='blue')
    plt.xlabel('Hit Rate')
    plt.ylabel('CDF')
    plt.title('CDF of Hit Rates)')
    plt.grid(True)
    plt.savefig(f'{output_dir}/hit_rates_cdf.png')
    plt.close()


def plot_block_hit_cdf(block_id_2_count: dict[int, int], output_dir: str):
    hit_counts = np.array(list(block_id_2_count.values()))
    sorted_counts = np.sort(hit_counts)
    cdf = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_counts,
             cdf,
             marker='o',
             linestyle='-',
             color='blue',
             linewidth=2)
    plt.xlabel("Hit Count")
    plt.ylabel("CDF")
    plt.title("CDF of Block Hit Counts")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "block_hit_cdf.png")
    plt.savefig(save_path)
    plt.close()


def generate_poisson_requests(lambda_rate, num_sessions, seed=None):
    """
    Generate session arrival times using a Poisson process.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    inter_arrival_times = rng.exponential(scale=1 / lambda_rate,
                                          size=num_sessions)
    arrival_times = np.cumsum(inter_arrival_times)
    return arrival_times


def plot_interval_distribution(task_name, intervals, save_dir="./wa_analysis"):
    data = np.array(intervals)
    if len(data) < 5:
        return  # 样本太少跳过

    plt.figure(figsize=(8, 4))

    # === PDF ===
    plt.subplot(1, 2, 1)
    plt.hist(data, bins=50, density=True, alpha=0.7, color="skyblue")
    plt.xlabel("Interval (s)")
    plt.ylabel("Probability Density")
    plt.title(f"{task_name} - PDF")

    # === CDF ===
    plt.subplot(1, 2, 2)
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, cdf, color="darkblue", linewidth=2)
    plt.xlabel("Interval (s)")
    plt.ylabel("Cumulative Probability")
    plt.title(f"{task_name} - CDF")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{task_name}_distribution.png", dpi=300)

    return
