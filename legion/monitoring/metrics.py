"""System metrics collection utilities for Legion monitoring"""

import os
import platform
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

import psutil

if TYPE_CHECKING:
    from .events.base import Event

class SystemMetricsCollector:
    """Collects system metrics for event context enrichment"""

    def __init__(self):
        """Initialize the metrics collector"""
        self._process = psutil.Process()
        self._initial_net_io = psutil.net_io_counters()
        self._initial_disk_io = psutil.disk_io_counters()
        self._start_time = time.time()
        self._initial_cpu_times = self._process.cpu_times()

    def get_execution_context(self) -> Dict[str, Any]:
        """Get current execution context

        Returns
        -------
            Dict containing execution context information

        """
        return {
            "thread_id": threading.get_ident(),
            "process_id": os.getpid(),
            "host_name": platform.node(),
            "python_version": sys.version
        }

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics

        Returns
        -------
            Dict containing system metrics

        """
        # Get current measurements
        current_net_io = psutil.net_io_counters()
        current_disk_io = psutil.disk_io_counters()

        # Calculate network IO
        net_bytes_sent = current_net_io.bytes_sent - self._initial_net_io.bytes_sent
        net_bytes_recv = current_net_io.bytes_recv - self._initial_net_io.bytes_recv

        # Calculate disk IO
        disk_bytes_read = current_disk_io.read_bytes - self._initial_disk_io.read_bytes
        disk_bytes_written = current_disk_io.write_bytes - self._initial_disk_io.write_bytes

        return {
            "system_cpu_percent": psutil.cpu_percent(),
            "system_memory_percent": psutil.virtual_memory().percent,
            "system_disk_usage_bytes": disk_bytes_read + disk_bytes_written,
            "system_network_bytes_sent": net_bytes_sent,
            "system_network_bytes_received": net_bytes_recv
        }

    def get_process_metrics(self) -> Dict[str, Any]:
        """Get current process metrics

        Returns
        -------
            Dict containing process metrics

        """
        return {
            "memory_usage_bytes": self._process.memory_info().rss,
            "cpu_usage_percent": self._process.cpu_percent()
        }

class MetricsContext:
    """Context manager for collecting metrics during an operation"""

    def __init__(self, event: Optional["Event"] = None):
        """Initialize metrics context

        Args:
        ----
            event: Optional event to enrich with metrics

        """
        self.event = event
        self.collector = SystemMetricsCollector()
        self._start_time = None

    def __enter__(self):
        """Enter the metrics collection context"""
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the metrics collection context and update event"""
        if not self.event:
            return

        # Calculate duration
        duration_ms = (time.time() - self._start_time) * 1000
        self.event.duration_ms = duration_ms

        # Get metrics
        execution_context = self.collector.get_execution_context()
        system_metrics = self.collector.get_system_metrics()
        process_metrics = self.collector.get_process_metrics()

        # Update event
        for key, value in execution_context.items():
            setattr(self.event, key, value)

        for key, value in system_metrics.items():
            setattr(self.event, key, value)

        for key, value in process_metrics.items():
            setattr(self.event, key, value)
