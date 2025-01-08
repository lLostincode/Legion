"""Analysis engine for Legion monitoring system.

This module provides tools for analyzing event streams, detecting patterns,
and generating insights about system behavior.
"""

import logging
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional, Type

from .events.base import Event
from .registry import EventRegistry

logger = logging.getLogger(__name__)

@dataclass
class AnalysisPattern:
    """Represents a detected pattern in the event stream."""

    pattern_type: str
    confidence: float
    events: List[Event]
    metadata: Dict[str, Any]
    detected_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary representation."""
        return {
            "pattern_type": self.pattern_type,
            "confidence": self.confidence,
            "event_ids": [event.event_id for event in self.events],
            "metadata": self.metadata,
            "detected_at": self.detected_at.isoformat()
        }

class AnalysisPipeline:
    """Core analysis pipeline for processing events and detecting patterns."""

    def __init__(self, registry: EventRegistry):
        """Initialize the analysis pipeline.

        Args:
        ----
            registry: Event registry to analyze

        """
        self.registry = registry
        self._patterns: List[AnalysisPattern] = []

    def analyze_events(self,
                      event_types: Optional[List[Type[Event]]] = None,
                      time_window: Optional[timedelta] = None) -> List[AnalysisPattern]:
        """Analyze events to detect patterns.

        Args:
        ----
            event_types: Optional list of event types to analyze
            time_window: Optional time window to analyze

        Returns:
        -------
            List of detected patterns

        """
        # Get events to analyze
        events = self.registry.get_events()
        logger.debug(f"Got {len(events)} events from registry")

        # Filter by event types if specified
        if event_types:
            events = [e for e in events if any(isinstance(e, t) for t in event_types)]
            logger.debug(f"After event type filtering: {len(events)} events")

        # Filter by time window if specified
        if time_window:
            cutoff = datetime.now(UTC) - time_window
            logger.debug(f"Filtering events after {cutoff}")
            filtered_events = []
            for event in events:
                logger.debug(f"Event timestamp: {event.timestamp}, cutoff: {cutoff}")
                if event.timestamp >= cutoff:
                    filtered_events.append(event)
            events = filtered_events
            logger.debug(f"After time window filtering: {len(events)} events")

        # Clear previous patterns
        self._patterns.clear()

        # Analyze basic metrics
        self._analyze_duration_patterns(events)
        self._analyze_error_patterns(events)
        self._analyze_resource_patterns(events)

        logger.debug(f"Found {len(self._patterns)} patterns")
        return self._patterns

    def _analyze_duration_patterns(self, events: List[Event]) -> None:
        """Analyze duration patterns in events.

        Args:
        ----
            events: List of events to analyze

        """
        if not events:
            return

        # Calculate duration statistics
        durations = [e.duration_ms for e in events if hasattr(e, "duration_ms") and e.duration_ms is not None]
        if not durations:
            return

        # For single events, use a fixed threshold
        if len(durations) == 1:
            threshold = 300  # 300ms threshold for single events
            if durations[0] > threshold:
                logger.debug("Found long running operation (single event)")
                self._patterns.append(AnalysisPattern(
                    pattern_type="long_running_operations",
                    confidence=0.7,  # Lower confidence for single event
                    events=[events[0]],
                    metadata={
                        "duration_ms": durations[0],
                        "threshold_ms": threshold
                    },
                    detected_at=datetime.now(UTC)
                ))
            return

        # For multiple events, use statistical analysis
        avg_duration = statistics.mean(durations)
        std_dev = statistics.stdev(durations) if len(durations) > 1 else 0
        max_duration = max(durations)

        # For small sets of events (2-3), use a simpler threshold
        if len(durations) <= 3:
            threshold = avg_duration * 1.2  # Lower multiplier for small sets
        else:
            # For larger sets, use statistical analysis
            threshold = min(avg_duration + (2 * std_dev), avg_duration * 2)

        # Detect long-running operations
        long_running = [
            e for e in events
            if hasattr(e, "duration_ms") and e.duration_ms is not None and e.duration_ms > threshold
        ]

        if long_running:
            logger.debug(f"Found {len(long_running)} long running operations")
            self._patterns.append(AnalysisPattern(
                pattern_type="long_running_operations",
                confidence=0.8,
                events=long_running,
                metadata={
                    "average_duration_ms": avg_duration,
                    "std_dev_ms": std_dev,
                    "max_duration_ms": max_duration,
                    "threshold_ms": threshold
                },
                detected_at=datetime.now(UTC)
            ))

    def _analyze_error_patterns(self, events: List[Event]) -> None:
        """Analyze error patterns in events.

        Args:
        ----
            events: List of events to analyze

        """
        if not events:
            return

        # Find error events
        error_events = [e for e in events if hasattr(e, "error") and e.error is not None]
        if not error_events:
            return

        # Group errors by type
        error_types: Dict[str, List[Event]] = {}
        for event in error_events:
            error_type = type(event.error).__name__
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(event)

        # Detect error patterns
        for error_type, events in error_types.items():
            # Report all errors, even single ones
            logger.debug(f"Found errors of type {error_type}")
            self._patterns.append(AnalysisPattern(
                pattern_type="repeated_errors" if len(events) > 1 else "error",
                confidence=0.9 if len(events) > 1 else 0.7,
                events=events,
                metadata={
                    "error_type": error_type,
                    "count": len(events)
                },
                detected_at=datetime.now(UTC)
            ))

    def _analyze_resource_patterns(self, events: List[Event]) -> None:
        """Analyze resource usage patterns in events.

        Args:
        ----
            events: List of events to analyze

        """
        if not events:
            return

        # Calculate memory usage statistics
        memory_usages = [
            e.memory_usage_bytes for e in events
            if hasattr(e, "memory_usage_bytes") and e.memory_usage_bytes is not None
        ]
        if not memory_usages:
            return

        # For single events, use a fixed threshold
        if len(memory_usages) == 1:
            threshold = 100 * 1024 * 1024  # 100MB threshold for single events
            if memory_usages[0] > threshold:
                logger.debug("Found high memory usage (single event)")
                self._patterns.append(AnalysisPattern(
                    pattern_type="high_memory_usage",
                    confidence=0.7,  # Lower confidence for single event
                    events=[events[0]],
                    metadata={
                        "memory_usage_bytes": memory_usages[0],
                        "threshold_bytes": threshold,
                        "average_bytes": memory_usages[0],  # For test compatibility
                        "max_bytes": memory_usages[0]  # For test compatibility
                    },
                    detected_at=datetime.now(UTC)
                ))
            return

        # For multiple events, use statistical analysis
        avg_memory = statistics.mean(memory_usages)
        std_dev = statistics.stdev(memory_usages) if len(memory_usages) > 1 else 0
        max_memory = max(memory_usages)

        # For small sets of events (2-3), use a simpler threshold
        if len(memory_usages) <= 3:
            # For small values, use a lower multiplier
            if avg_memory < 10000:  # If average is less than 10KB
                threshold = avg_memory * 1.5  # Use 1.5x for small values
            else:
                threshold = avg_memory * 1.2  # Use 1.2x for normal values
        else:
            # For larger sets, use statistical analysis
            threshold = min(avg_memory + (1.5 * std_dev), avg_memory * 1.5)

        # Detect high memory usage
        high_memory = [
            e for e in events
            if hasattr(e, "memory_usage_bytes") and e.memory_usage_bytes is not None
            and e.memory_usage_bytes > threshold
        ]

        if high_memory:
            logger.debug(f"Found {len(high_memory)} events with high memory usage")
            self._patterns.append(AnalysisPattern(
                pattern_type="high_memory_usage",
                confidence=0.7,  # Keep confidence consistent for memory patterns
                events=high_memory,
                metadata={
                    "average_bytes": avg_memory,  # Keep old name for test compatibility
                    "std_dev_bytes": std_dev,
                    "max_bytes": max_memory,  # Keep old name for test compatibility
                    "threshold_bytes": threshold
                },
                detected_at=datetime.now(UTC)
            ))
