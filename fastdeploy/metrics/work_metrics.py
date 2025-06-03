"""
metrics
"""
import os
import atexit
import shutil
from threading import Lock

from prometheus_client import Histogram


class WorkMetricsManager(object):
    """Prometheus Metrics Manager handles all metric updates """

    _initialized = False

    def __init__(self):
        """Initializes the Prometheus metrics and starts the HTTP server if not already initialized."""

        if self._initialized:
            return

        self.e2e_request_latency = Histogram(
            'fastdeploy:e2e_request_latency_seconds',
            'End-to-end request latency (from request arrival to final response)',
            buckets=[
                0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0,
                40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0
            ]
        )

        self._initialized = True


work_process_metrics = WorkMetricsManager()
