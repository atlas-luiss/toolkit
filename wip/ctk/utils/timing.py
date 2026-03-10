"""
Timing utilities for tracking training and inference performance.
"""

import time
from contextlib import contextmanager
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class TimingStats:
    """Container for timing statistics."""
    total_time: float = 0.0
    count: int = 0
    times: List[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        """Average time."""
        return self.total_time / self.count if self.count > 0 else 0.0

    @property
    def std(self) -> float:
        """Standard deviation of times."""
        return float(np.std(self.times)) if len(self.times) > 0 else 0.0

    @property
    def min(self) -> float:
        """Minimum time."""
        return float(np.min(self.times)) if len(self.times) > 0 else 0.0

    @property
    def max(self) -> float:
        """Maximum time."""
        return float(np.max(self.times)) if len(self.times) > 0 else 0.0

    def add(self, elapsed: float):
        """Add a new timing measurement."""
        self.total_time += elapsed
        self.count += 1
        self.times.append(elapsed)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'total': self.total_time,
            'mean': self.mean,
            'std': self.std,
            'min': self.min,
            'max': self.max,
            'count': self.count
        }


class Timer:
    """Named-operation timer with cumulative stats."""

    def __init__(self):
        self.timings: Dict[str, TimingStats] = {}
        self._start_times: Dict[str, float] = {}

    @contextmanager
    def track(self, name: str):
        """Context manager that records elapsed time under name."""
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            if name not in self.timings:
                self.timings[name] = TimingStats()
            self.timings[name].add(elapsed)

    def start(self, name: str):
        """Start timing for a named operation."""
        self._start_times[name] = time.time()

    def stop(self, name: str) -> float:
        """Stop a named timer and return elapsed seconds."""
        if name not in self._start_times:
            raise ValueError(f"Timer '{name}' was not started")

        elapsed = time.time() - self._start_times[name]
        del self._start_times[name]

        if name not in self.timings:
            self.timings[name] = TimingStats()
        self.timings[name].add(elapsed)

        return elapsed

    def get_stats(self, name: str) -> Optional[TimingStats]:
        """Get timing statistics for a named operation."""
        return self.timings.get(name)

    def get_all_stats(self) -> Dict[str, TimingStats]:
        """Get all timing statistics."""
        return self.timings

    def reset(self, name: Optional[str] = None):
        """Reset one named timer or all timers."""
        if name is not None:
            if name in self.timings:
                del self.timings[name]
            if name in self._start_times:
                del self._start_times[name]
        else:
            self.timings.clear()
            self._start_times.clear()

    def summary(self, names: Optional[List[str]] = None) -> str:
        """Return a formatted timing summary string."""
        if names is None:
            names = sorted(self.timings.keys())

        lines = ["Timing Summary:"]
        lines.append("-" * 60)

        for name in names:
            if name in self.timings:
                stats = self.timings[name]
                lines.append(f"{name}:")
                lines.append(f"  Total: {stats.total_time:.2f}s")
                lines.append(f"  Mean:  {stats.mean:.2f}s ± {stats.std:.2f}s")
                lines.append(f"  Range: [{stats.min:.2f}s, {stats.max:.2f}s]")
                lines.append(f"  Count: {stats.count}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert all timing stats to dictionary."""
        return {name: stats.to_dict() for name, stats in self.timings.items()}


def format_time(seconds: float) -> str:
    """Format seconds as a human-readable string (e.g., "1h 23m 45s")."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"
