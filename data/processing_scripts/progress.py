"""progress.py - phase timing and progress logging for the dataset stages.

The dataset stages run for minutes at a stretch (pairing thousands of Ada
files, hashing every record for the eval guard, writing ~1 GB of JSONL) and
used to print nothing between their first and last line. From the terminal
that is indistinguishable from a hang, so people killed runs that were about
to finish.

Two tools, both cheap and side-effect free:

- :func:`phase` is a context manager that logs "starting" and "done in Ns" for a
  named step. It is the right tool for a step whose length is unknown and whose
  inner loop has no natural progress to report.
- :class:`Progress` counts items in a long loop and logs a rate plus an ETA at a
  bounded cadence: at least every ``min_interval`` seconds, and at most every
  ``every`` items, so a 7000-file loop prints about one line per 15 s instead of
  one per file or none at all.

Both log at INFO on the module logger of the caller, so a stage's output stays
in one stream. Timings are wall clock, which is what a user waiting on the
command cares about.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger("q3as_progress")

# Never log progress more often than this, whatever the item count.
MIN_INTERVAL_S = 15.0


def _fmt_duration(seconds: float) -> str:
    """Human-readable duration: 45s, 12m, 1h04m."""
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


@contextmanager
def phase(label: str, log: logging.Logger | None = None, **fmt: Any) -> Iterator[None]:
    """Log the start and wall-clock duration of a named step.

    ``fmt`` supplies the log fields, e.g. ``phase("dedup", n=len(turns))``.
    """
    log = log or logger
    log.info("[%s] starting%s", label, _fmt_fields(fmt))
    start = time.monotonic()
    try:
        yield
    finally:
        # Logged on the failure path too: a step that dies after 4 minutes is
        # exactly the case where the elapsed time is worth knowing.
        log.info("[%s] done in %s", label, _fmt_duration(time.monotonic() - start))


def _fmt_fields(fields: dict[str, Any]) -> str:
    if not fields:
        return ""
    return " (" + ", ".join(f"{key}={value}" for key, value in fields.items()) + ")"


class Progress:
    """Throttled item counter with a rate and an ETA.

    Usage::

        progress = Progress("Ada files", len(tasks))
        for result in pool.imap(work, tasks):
            handle(result)
            progress.advance()
        progress.close()

    ``advance`` only logs when both the item interval and the time interval
    have passed, so the cost per item is one comparison.
    """

    def __init__(
        self,
        label: str,
        total: int,
        *,
        every: int | None = None,
        min_interval: float = MIN_INTERVAL_S,
        log: logging.Logger | None = None,
    ) -> None:
        self.label = label
        self.total = max(0, int(total))
        self.log = log or logger
        self.min_interval = min_interval
        # Aim for ~20 lines over the whole loop, never more often than `every`.
        self.every = every if every is not None else max(1, self.total // 20 or 1)
        self.done = 0
        self._start = time.monotonic()
        self._last_log = self._start
        self._next_at = self.every
        self._logged_end = False
        self.log.info("[%s] starting (%d items)", self.label, self.total)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def advance(self, n: int = 1) -> None:
        """Count *n* more finished items, logging when a line is due."""
        # Clamped so the count can never read past the total (a 0-item loop
        # must report 0/0, not 1/0).
        self.done = min(self.done + n, self.total)
        if self._logged_end:
            return
        reached_end = self.done >= self.total
        # Two gates: enough items since the last line AND enough wall time.
        # Either alone lets a fast loop burst one line per item.
        if not reached_end and not (self.done >= self._next_at
                                    and time.monotonic() - self._last_log >= self.min_interval):
            return
        self._log()

    def _log(self) -> None:
        now = time.monotonic()
        elapsed = now - self._start
        rate = self.done / elapsed if elapsed > 0 else 0.0
        remaining = max(0, self.total - self.done)
        eta = f", eta {_fmt_duration(remaining / rate)}" if rate > 0 else ""
        self.log.info(
            "[%s] %d/%d (%.0f%%) %.1f items/s, elapsed %s%s",
            self.label, self.done, self.total,
            100.0 * self.done / self.total if self.total else 100.0,
            rate, _fmt_duration(elapsed), eta,
        )
        self._last_log = now
        self._next_at = self.done + self.every
        if self.done >= self.total:
            self._logged_end = True

    def close(self) -> None:
        """Log the final count. Safe to call twice."""
        if self._logged_end:
            return
        self._log()
