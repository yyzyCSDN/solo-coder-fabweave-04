"""Certification-job scheduling for recipes on chambers.

A certification job qualifies a recipe revision on a chamber.  Scheduling must
simultaneously satisfy four constraints:

* **maintenance** -- the job window must not overlap a planned maintenance
  window for the chamber;
* **cooldown** -- consecutive jobs on the same chamber must be separated by at
  least the chamber's cooldown interval;
* **shift** -- the whole job window must lie inside an engineer shift;
* **validity** -- when the certification completes, the remaining validity
  period (``valid_until - end``) must be at least ``min_validity``, so the
  qualification is still usable.

The scheduler is atomic: it tentatively places jobs into a local structure and
only commits when every job fits.  When a batch cannot be scheduled it returns
a denied :class:`~fabweave.core.decision.Decision` listing *all* conflicts
(namespaced by job id) and occupies no resources.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from fabweave.core.decision import Decision, allow, deny
from fabweave.core.timebox import TimeBox
from fabweave.recipe.qualification import Qualification


@dataclass(frozen=True, slots=True)
class CertificationRequest:
    """A recipe/chamber certification job to schedule.

    The qualification expires at ``valid_until``; the scheduler requires the
    job to complete early enough that ``valid_until - end >= min_validity``.
    """

    job_id: str
    tool_id: str
    chamber_id: str
    recipe_revision: int
    duration: int
    release_at: int
    deadline: int
    valid_until: int

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError("certification duration must be positive")
        if self.recipe_revision <= 0:
            raise ValueError("recipe revision must be positive")
        if self.release_at < 0 or self.deadline < 0 or self.valid_until < 0:
            raise ValueError("certification times must be non-negative")


@dataclass(frozen=True, slots=True)
class ChamberProfile:
    """Chamber-specific scheduling context: cooldown and planned maintenance."""

    chamber_id: str
    cooldown: int
    maintenance: tuple[TimeBox, ...] = ()

    def __post_init__(self) -> None:
        if self.cooldown < 0:
            raise ValueError("cooldown must be non-negative")


@dataclass(frozen=True, slots=True)
class _Placement:
    job_id: str
    chamber_id: str
    window: TimeBox


def _gap(a: TimeBox, b: TimeBox) -> int:
    """Gap between two half-open intervals; negative when overlapping."""
    if a.end <= b.start:
        return b.start - a.end
    if b.end <= a.start:
        return a.start - b.end
    return -1


def _within_shift(window: TimeBox, shifts: list[TimeBox]) -> bool:
    return any(shift.start <= window.start and window.end <= shift.end for shift in shifts)


def _next_shift_start(start: int, duration: int, shifts: list[TimeBox]) -> int | None:
    """Earliest start >= ``start`` that fits inside a shift of sufficient length."""
    best: int | None = None
    for shift in shifts:
        if shift.end - shift.start < duration:
            continue
        candidate = max(start, shift.start)
        if candidate + duration <= shift.end and (best is None or candidate < best):
            best = candidate
    return best


def _find_slot(
    request: CertificationRequest,
    chamber: ChamberProfile,
    shifts: list[TimeBox],
    occupied: list[_Placement],
    latest: int,
) -> int | None:
    """Earliest feasible start for ``request``, or None if no slot by ``latest``."""
    duration = request.duration
    start = request.release_at
    while start <= latest:
        window = TimeBox(start, start + duration)
        blocking_maintenance = [item for item in chamber.maintenance if item.overlaps(window)]
        if blocking_maintenance:
            start = max(item.end for item in blocking_maintenance)
            continue
        too_close = [
            item
            for item in occupied
            if item.chamber_id == chamber.chamber_id and _gap(window, item.window) < chamber.cooldown
        ]
        if too_close:
            start = max(item.window.end + chamber.cooldown for item in too_close)
            continue
        if not _within_shift(window, shifts):
            candidate = _next_shift_start(start, duration, shifts)
            if candidate is None or candidate > latest:
                return None
            start = candidate
            continue
        return start
    return None


def _has_feasible_gap(start: int, latest: int, blocked: list[tuple[int, int]]) -> bool:
    """True if some value in ``[start, latest]`` is not inside any blocked interval."""
    if not blocked:
        return True
    merged: list[tuple[int, int]] = []
    for lo, hi in sorted(blocked):
        if merged and lo < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    pos = start
    for lo, hi in merged:
        if pos < lo:
            return True
        pos = max(pos, hi)
        if pos > latest:
            return False
    return pos <= latest


def _shift_coverable(start: int, latest: int, duration: int, shifts: list[TimeBox]) -> bool:
    return any(
        shift.end - shift.start >= duration
        and max(start, shift.start) <= min(latest, shift.end - duration)
        for shift in shifts
    )


def _maintenance_avoidable(
    start: int, latest: int, duration: int, maintenance: tuple[TimeBox, ...]
) -> bool:
    blocked = [(item.start - duration, item.end) for item in maintenance]
    return _has_feasible_gap(start, latest, blocked)


def _cooldown_avoidable(
    start: int,
    latest: int,
    duration: int,
    cooldown: int,
    occupied: list[_Placement],
    chamber_id: str,
) -> bool:
    blocked = [
        (item.window.start - duration - cooldown, item.window.end + cooldown)
        for item in occupied
        if item.chamber_id == chamber_id
    ]
    return _has_feasible_gap(start, latest, blocked)


def _structural_conflicts(
    request: CertificationRequest,
    chamber: ChamberProfile,
    shifts: list[TimeBox],
    occupied: list[_Placement],
    latest: int,
    min_validity: int,
) -> tuple[str, ...]:
    """Every constraint that independently blocks ``request``."""
    reasons: list[str] = []
    duration = request.duration
    start = request.release_at
    if start > request.deadline - duration:
        reasons.append("past_deadline")
    if start > request.valid_until - min_validity - duration:
        reasons.append("validity_period_too_short")
    if start <= latest:
        if not _shift_coverable(start, latest, duration, shifts):
            reasons.append("outside_shift")
        if not _maintenance_avoidable(start, latest, duration, chamber.maintenance):
            reasons.append("maintenance_overlap")
        if not _cooldown_avoidable(start, latest, duration, chamber.cooldown, occupied, chamber.chamber_id):
            reasons.append("cooldown_violation")
    if not reasons:
        reasons.append("no_feasible_slot")
    return tuple(reasons)


def schedule_certifications(
    requests: list[CertificationRequest],
    chambers: Mapping[str, ChamberProfile],
    shifts: list[TimeBox],
    min_validity: int,
    existing: Mapping[str, list[TimeBox]] | None = None,
) -> Decision:
    """Schedule a batch of certification jobs atomically.

    Every job must be placed for the batch to commit.  When one or more jobs
    cannot be placed the returned :class:`Decision` is denied and lists *all*
    conflicts (namespaced ``job_id:reason``); no resources are occupied.
    """
    if min_validity < 0:
        raise ValueError("min_validity must be non-negative")

    occupied: list[_Placement] = []
    if existing:
        for chamber_id, windows in existing.items():
            for window in windows:
                occupied.append(_Placement("__existing__", chamber_id, window))

    ordered = sorted(requests, key=lambda item: (item.release_at, item.job_id))

    scheduled: dict[str, TimeBox] = {}
    qualifications: list[Qualification] = []
    conflicts: dict[str, tuple[str, ...]] = {}

    for request in ordered:
        chamber = chambers.get(request.chamber_id)
        if chamber is None:
            conflicts[request.job_id] = (f"chamber_unknown:{request.chamber_id}",)
            continue

        latest = min(
            request.deadline - request.duration,
            request.valid_until - min_validity - request.duration,
        )
        if request.release_at > latest:
            conflicts[request.job_id] = _structural_conflicts(
                request, chamber, shifts, occupied, latest, min_validity
            )
            continue

        slot = _find_slot(request, chamber, shifts, occupied, latest)
        if slot is None:
            conflicts[request.job_id] = _structural_conflicts(
                request, chamber, shifts, occupied, latest, min_validity
            )
            continue

        window = TimeBox(slot, slot + request.duration)
        scheduled[request.job_id] = window
        occupied.append(_Placement(request.job_id, request.chamber_id, window))
        qualifications.append(
            Qualification(
                tool_id=request.tool_id,
                chamber_id=request.chamber_id,
                recipe_revision=request.recipe_revision,
                qualified_at=window.end,
                expires_at=request.valid_until,
            )
        )

    if conflicts:
        all_reasons = tuple(
            f"{job_id}:{reason}" for job_id in sorted(conflicts) for reason in conflicts[job_id]
        )
        return deny(*all_reasons, conflicts=dict(conflicts), scheduled={})

    return allow(
        scheduled=dict(scheduled),
        qualifications=tuple(qualifications),
        request_count=len(ordered),
    )
