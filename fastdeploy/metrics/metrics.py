"""
metrics
"""
import os
import shutil
from typing import Set

from prometheus_client import Gauge, Histogram, multiprocess, CollectorRegistry, generate_latest
from prometheus_client.registry import Collector


def cleanup_prometheus_files(is_main):
    """
       Cleans and recreates the Prometheus multiprocess directory.

       Depending on whether it's the main process or a worker, this function removes the corresponding
       Prometheus multiprocess directory (/tmp/prom_main or /tmp/prom_worker) and recreates it as an empty directory.

       Args:
           is_main (bool): Indicates whether the current process is the main process.

       Returns:
           str: The path to the newly created Prometheus multiprocess directory.
    """
    PROM_DIR = "/tmp/prom_main" if is_main else "/tmp/prom_worker"
    if os.path.exists(PROM_DIR):
        shutil.rmtree(PROM_DIR)
    os.makedirs(PROM_DIR, exist_ok=True)
    return PROM_DIR


class SimpleCollector(Collector):
    """
        A custom Prometheus collector that filters out specific metrics by name.

        This collector wraps an existing registry and yields only those metrics
        whose names are not in the specified exclusion set.
    """
    def __init__(self, base_registry, exclude_names: Set[str]):
        """
            Initializes the SimpleCollector.

            Args:
                base_registry (CollectorRegistry): The source registry from which metrics are collected.
                exclude_names (Set[str]): A set of metric names to exclude from collection.
        """
        self.base_registry = base_registry
        self.exclude_names = exclude_names

    def collect(self):
        """
                Collects and yields metrics not in the exclusion list.

                Yields:
                    Metric: Prometheus Metric objects that are not excluded.
                """
        for metric in self.base_registry.collect():
            if metric.name not in self.exclude_names:
                yield metric


def get_filtered_metrics(exclude_names: Set[str], extra_register_func=None) -> str:
    """
    Get the merged metric text (specified metric name removed)
    :param exclude_names: metric.name set to be excluded
    :param extra_register_func: optional, main process custom metric registration method
    :return: filtered metric text (str)
    """
    base_registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(base_registry)

    filtered_registry = CollectorRegistry()
    filtered_registry.register(SimpleCollector(base_registry, exclude_names))

    if extra_register_func:
        extra_register_func(filtered_registry)

    return generate_latest(filtered_registry).decode("utf-8")


class MetricsManager:
    """Prometheus Metrics Manager handles all metric updates """

    _instance = None

    def __init__(self):
        """Initializes the Prometheus metrics and starts the HTTP server if not already initialized."""

        # Request count gauges
        self.num_requests_running = Gauge(
            'fastdeploy:num_requests_running',
            'Number of requests currently running',
            multiprocess_mode="sum"
        )

        self.num_requests_waiting = Gauge(
            'fastdeploy:num_requests_waiting',
            'Number of requests currently waiting',
            multiprocess_mode="sum"
        )

        # Latency histograms
        self.time_to_first_token = Histogram(
            'fastdeploy:time_to_first_token_seconds',
            'Time to first token in seconds',
            buckets=[0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0]
        )

        self.time_per_output_token = Histogram(
            'fastdeploy:time_per_output_token_seconds',
            'Time per output token in seconds',
            buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
        )



    def register_all(self, registry: CollectorRegistry):
        """Register all metrics to the specified registry"""
        registry.register(self.num_requests_running)
        registry.register(self.num_requests_waiting)
        registry.register(self.time_to_first_token)
        registry.register(self.time_per_output_token)


EXCLUDE_LABELS = {"fastdeploy:num_requests_running",
                  "fastdeploy:num_requests_waiting",
                  "fastdeploy:time_to_first_token_seconds",
                  "fastdeploy:time_per_output_token_seconds"}

main_process_metrics = MetricsManager()
