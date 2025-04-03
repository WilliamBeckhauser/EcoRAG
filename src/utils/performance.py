import psutil
import time
import logging
from typing import Dict, Any
from ..config.settings import MODEL_FLOPS_ESTIMATE


def get_process_metrics() -> Dict[str, float]:
    """
    Returns current process metrics (CPU time and memory usage).
    """
    process = psutil.Process()
    return {
        "cpu_time": process.cpu_percent(interval=0.1),
        "mem_usage": process.memory_info().rss / 1024 / 1024  # Convert to MB
    }


def calculate_throughput(
    total_tokens: int,
    total_latency: float
) -> float:
    """
    Calculates throughput in tokens per second.
    """
    return total_tokens / total_latency if total_latency > 0 else 0


def calculate_energy_consumption(
    cpu_time: float,
    mem_usage: float,
    latency: float
) -> float:
    """
    Estimates energy consumption in joules.
    This is a simplified estimation based on CPU time and memory usage.
    """
    # Constants for energy estimation (these are rough estimates)
    CPU_ENERGY_PER_SECOND = 0.1  # Joules per second
    MEMORY_ENERGY_PER_MB = 0.0001  # Joules per MB per second

    cpu_energy = (cpu_time / 100) * CPU_ENERGY_PER_SECOND * latency
    memory_energy = mem_usage * MEMORY_ENERGY_PER_MB * latency

    return cpu_energy + memory_energy


def calculate_flops_per_token(
    total_tokens: int,
    latency: float
) -> float:
    """
    Calculates FLOPS per token based on model's estimated FLOPS.
    """
    if total_tokens == 0 or latency == 0:
        return 0
    return MODEL_FLOPS_ESTIMATE / (total_tokens / latency)


def measure_performance(
    start_time: float,
    total_tokens: int,
    cpu_time: float,
    mem_usage: float
) -> Dict[str, Any]:
    """
    Measures and calculates all performance metrics.
    """
    latency = time.time() - start_time
    throughput = calculate_throughput(total_tokens, latency)
    energy_consumption = calculate_energy_consumption(
        cpu_time, mem_usage, latency
    )
    flops_per_token = calculate_flops_per_token(total_tokens, latency)

    return {
        "latency": latency,
        "throughput": throughput,
        "energy_consumption": energy_consumption,
        "flops_per_token": flops_per_token,
        "cpu_time": cpu_time,
        "mem_usage": mem_usage
    }


def log_performance_metrics(metrics: Dict[str, Any]) -> None:
    """
    Logs performance metrics in a structured format.
    """
    logging.info("Performance Metrics:")
    logging.info(f"Latency: {metrics['latency']:.2f} seconds")
    logging.info(f"Throughput: {metrics['throughput']:.2f} tokens/second")
    logging.info(
        f"Energy Consumption: {metrics['energy_consumption']:.2f} joules")
    logging.info(f"FLOPS per Token: {metrics['flops_per_token']:.2f}")
    logging.info(f"CPU Time: {metrics['cpu_time']:.2f}%")
    logging.info(f"Memory Usage: {metrics['mem_usage']:.2f} MB")
