"""
metrics
"""
import os
import atexit
import shutil
from threading import Lock

from prometheus_client import Histogram

from fastdeploy.metrics.metrics import REQUEST_LATENCY_BUCKETS


class WorkMetricsManager:
    """Prometheus Metrics Manager handles all metric updates """

    _initialized = False

    def __init__(self):
        """Initializes the Prometheus metrics and starts the HTTP server if not already initialized."""

        if self._initialized:
            return

        self.e2e_request_latency = Histogram(
            'fastdeploy:e2e_request_latency_seconds',
            'End-to-end request latency (from request arrival to final response)',
            buckets=REQUEST_LATENCY_BUCKETS
        )

        self._initialized = True


work_process_metrics = WorkMetricsManager()
