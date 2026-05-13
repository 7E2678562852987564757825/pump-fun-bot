"""Strict no-lookahead event bus for the backtest simulation.

The MOST CRITICAL invariant: at any simulation time T, a consumer can ONLY
access events with ts <= T. The bus enforces this by design — events are
stored in a min-heap and only released via pop_until(ts), which advances
the current_ts monotonically.
"""

from __future__ import annotations

import heapq
from typing import Iterator

from hl_liq_cascade.types import Event, EventKind


class EventBus:
    """Min-heap event bus with strict no-lookahead enforcement.

    Time only moves forward: once current_ts reaches T, no event with
    ts < T can ever be injected (ValueError is raised on violation).
    """

    def __init__(self) -> None:
        self._heap: list[tuple[int, int, Event]] = []
        self._seq: int = 0
        self._current_ts: int = 0

    # ------------------------------------------------------------------
    # Producers
    # ------------------------------------------------------------------

    def push(self, event: Event) -> None:
        """Push a single event onto the heap.

        Raises
        ------
        ValueError
            If ``event.ts < self._current_ts`` — retroactive injection is
            forbidden to preserve the no-lookahead guarantee.
        """
        if event.ts < self._current_ts:
            raise ValueError(
                f"EventBus: attempted to push event with ts={event.ts} "
                f"which is before current_ts={self._current_ts}. "
                f"retroactive injection violates no-lookahead invariant. "
                f"Event kind={event.kind}"
            )
        heapq.heappush(self._heap, (event.ts, self._seq, event))
        self._seq += 1

    def push_batch(self, events: list[Event]) -> None:
        """Push multiple events, sorted by ts before insertion.

        Sorting before insertion is not strictly required (the heap handles
        ordering) but ensures that same-ts events pushed together get
        sequenced in chronological order of their occurrence in the batch.
        """
        for event in sorted(events, key=lambda e: e.ts):
            self.push(event)

    # ------------------------------------------------------------------
    # Consumers
    # ------------------------------------------------------------------

    def pop_until(self, ts: int) -> Iterator[Event]:
        """Pop and yield all events with ts <= *ts*, in ts / seq order.

        This is the ONLY way consumers receive events. After each pop the
        internal current_ts is updated, so future pushes of earlier events
        will raise.

        Parameters
        ----------
        ts:
            The upper bound (inclusive) on event timestamps to release.
        """
        while self._heap and self._heap[0][0] <= ts:
            event_ts, _seq, event = heapq.heappop(self._heap)
            # Advance the monotone clock
            if event_ts > self._current_ts:
                self._current_ts = event_ts
            yield event

    # ------------------------------------------------------------------
    # Inspection (read-only, no state change)
    # ------------------------------------------------------------------

    def peek_next_ts(self) -> int | None:
        """Return the timestamp of the next event without consuming it.

        Returns ``None`` if the bus is empty.
        """
        if not self._heap:
            return None
        return self._heap[0][0]

    def is_empty(self) -> bool:
        """Return True if the heap has no events."""
        return len(self._heap) == 0

    def size(self) -> int:
        """Return the number of pending events."""
        return len(self._heap)

    @property
    def current_ts(self) -> int:
        """The highest timestamp seen so far (monotonically non-decreasing)."""
        return self._current_ts


class EventBusValidator:
    """Wraps :class:`EventBus` and records every emitted event for auditing.

    In tests, call ``audit_no_lookahead()`` after a simulation run to verify
    that no signal at time T referenced data with timestamp > T.
    """

    def __init__(self) -> None:
        self.bus: EventBus = EventBus()
        # Full emission log: list of (current_ts_at_emission, event)
        self._emission_log: list[tuple[int, Event]] = []

    # Delegate push / push_batch
    def push(self, event: Event) -> None:
        self.bus.push(event)

    def push_batch(self, events: list[Event]) -> None:
        self.bus.push_batch(events)

    def pop_until(self, ts: int) -> Iterator[Event]:
        """Wrap pop_until and record every emitted event."""
        for event in self.bus.pop_until(ts):
            self._emission_log.append((ts, event))
            yield event

    def peek_next_ts(self) -> int | None:
        return self.bus.peek_next_ts()

    def is_empty(self) -> bool:
        return self.bus.is_empty()

    def size(self) -> int:
        return self.bus.size()

    @property
    def current_ts(self) -> int:
        return self.bus.current_ts

    # ------------------------------------------------------------------
    # Audit helpers
    # ------------------------------------------------------------------

    def emission_log(self) -> list[tuple[int, Event]]:
        """Return the full list of (consumer_ts, event) pairs emitted."""
        return list(self._emission_log)

    def audit_no_lookahead(self) -> list[str]:
        """Check the emission log for any lookahead violations.

        A violation occurs when an emitted event has ``event.ts > consumer_ts``
        (the event is from the future relative to when it was released).

        Returns a list of violation strings (empty means no violations).
        """
        violations: list[str] = []
        for consumer_ts, event in self._emission_log:
            if event.ts > consumer_ts:
                violations.append(
                    f"LOOKAHEAD: event.ts={event.ts} > consumer_ts={consumer_ts} "
                    f"(kind={event.kind})"
                )
        return violations
