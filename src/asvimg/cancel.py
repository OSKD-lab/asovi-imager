"""Cooperative cancellation for long-running pipeline stages.

The preprocessing / warping / extraction loops have no built-in way to be
interrupted.  A :class:`CancellationToken` is a thin wrapper around
``threading.Event`` that those loops check at frame / batch boundaries so a
GUI (or a Ctrl-C handler) can request a graceful stop without killing the
process.
"""

from __future__ import annotations

import threading


class Cancelled(Exception):
    """Raised when a cancellation was requested and a loop chose to abort."""


class CancellationToken:
    """A cooperative, thread-safe cancellation flag.

    Pass an instance into a long loop; the loop periodically calls
    :meth:`is_set` (to stop gracefully and return partial results) or
    :meth:`raise_if_set` (to abort by raising :class:`Cancelled`).  Another
    thread calls :meth:`cancel` to request the stop.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Request cancellation. Safe to call from any thread."""
        self._event.set()

    def reset(self) -> None:
        """Clear a previous cancellation so the token can be reused."""
        self._event.clear()

    def is_set(self) -> bool:
        """Whether cancellation has been requested."""
        return self._event.is_set()

    # Alias kept for readability at call sites that prefer a verb.
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_set(self) -> None:
        """Raise :class:`Cancelled` if cancellation has been requested."""
        if self._event.is_set():
            raise Cancelled()
