"""Scheduling of recipe-on-chamber qualification runs.

A qualification run certifies one recipe on one chamber.  It must be placed so
that:

* the run never touches planned chamber maintenance (with an optional
  cool-down buffer that has to stay clear before the maintenance starts);
* two runs on the same chamber are separated by the chamber cool-down
  (a request may extend the cool-down that follows itself);
* one named engineer covers the whole run inside a single on-call shift and
  does not exceed their concurrent capacity (engineers may supervise more than
  one chamber at once when their capacity allows);
* the certification is still valid when it is needed: the run finishes no
  later than ``needed_at`` and its expiry (``end + valid_for_minutes``) is no
  earlier than ``needed_at``.

Batches are all-or-nothing: when the whole batch cannot be placed nothing is
committed and every conflict detected is returned, instead of partially
occupying chambers and engineers.

Time is an integer epoch (minutes) and every interval is half-open, matching
:class:`fabweave.core.timebox.TimeBox`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from fabweave.core.timebox import TimeBox


# ---------------------------------------------------------------------------
# Static resources
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MaintenanceSpan:
    """A planned chamber maintenance window.

    ``cool_before_minutes`` is the buffer that must stay clear immediately
    before the window starts (no qualification run may end inside it).
    """

    window: TimeBox
    cool_before_minutes: int = 0

    def __post_init__(self) -> None:
        if self.cool_before_minutes < 0:
            raise ValueError("cool_before_minutes must be non-negative")


@dataclass(frozen=True, slots=True)
class Chamber:
    chamber_id: str
    cooldown_minutes: int = 0
    maintenance: tuple[MaintenanceSpan, ...] = ()

    def __post_init__(self) -> None:
        if not self.chamber_id:
            raise ValueError("chamber_id is required")
        if self.cooldown_minutes < 0:
            raise ValueError("cooldown_minutes must be non-negative")
        windows = [span.window for span in sorted(self.maintenance, key=lambda span: span.window)]
        for previous, current in zip(windows, windows[1:]):
            if previous.end > current.start:
                raise ValueError("maintenance windows must not overlap")
        object.__setattr__(
            self,
            "maintenance",
            tuple(sorted(self.maintenance, key=lambda span: span.window)),
        )


@dataclass(frozen=True, slots=True)
class Engineer:
    engineer_id: str
    skills: frozenset[str] = frozenset()
    shifts: tuple[TimeBox, ...] = ()
    capacity: int = 1

    def __post_init__(self) -> None:
        if not self.engineer_id:
            raise ValueError("engineer_id is required")
        if self.capacity < 1:
            raise ValueError("engineer capacity must be at least 1")
        shifts = tuple(sorted(self.shifts))
        for previous, current in zip(shifts, shifts[1:]):
            if previous.end > current.start:
                raise ValueError("engineer shifts must not overlap")
        object.__setattr__(self, "skills", frozenset(self.skills))
        object.__setattr__(self, "shifts", shifts)


# ---------------------------------------------------------------------------
# Requests and assignments
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CertRequest:
    """A request to certify ``recipe_id`` on ``chamber_id``.

    The resulting certification is consumed at ``needed_at``: the run has to
    finish by then and the certification must not have expired yet, given a
    validity lifetime of ``valid_for_minutes`` after completion.
    """

    request_id: str
    chamber_id: str
    recipe_id: str
    duration_minutes: int
    release_at: int
    needed_at: int
    valid_for_minutes: int
    cooldown_minutes: int | None = None
    required_skill: str | None = None
    eligible_engineers: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id is required")
        if not self.chamber_id or not self.recipe_id:
            raise ValueError("chamber_id and recipe_id are required")
        if self.duration_minutes <= 0:
            raise ValueError("duration_minutes must be positive")
        if self.valid_for_minutes <= 0:
            raise ValueError("valid_for_minutes must be positive")
        if self.cooldown_minutes is not None and self.cooldown_minutes < 0:
            raise ValueError("cooldown_minutes must be non-negative")
        if self.eligible_engineers is not None:
            object.__setattr__(
                self,
                "eligible_engineers",
                tuple(dict.fromkeys(self.eligible_engineers)),
            )


@dataclass(frozen=True, slots=True)
class CertificationAssignment:
    request_id: str
    chamber_id: str
    recipe_id: str
    start: int
    duration_minutes: int
    cooldown_minutes: int
    engineer_id: str
    valid_for_minutes: int

    @property
    def end(self) -> int:
        return self.start + self.duration_minutes

    @property
    def window(self) -> TimeBox:
        return TimeBox(self.start, self.end)

    @property
    def cooldown_until(self) -> int:
        return self.end + self.cooldown_minutes

    @property
    def expires_at(self) -> int:
        return self.end + self.valid_for_minutes


# ---------------------------------------------------------------------------
# Conflicts and results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Conflict:
    code: str
    request_id: str | None = None
    chamber_id: str | None = None
    engineer_id: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))


@dataclass(frozen=True, slots=True)
class ScheduleResult:
    scheduled: bool
    assignments: tuple[CertificationAssignment, ...]
    conflicts: tuple[Conflict, ...]


def _conflict_key(conflict: Conflict) -> tuple[Any, ...]:
    evidence = tuple(sorted(conflict.evidence.items(), key=lambda item: item[0]))
    return (
        conflict.request_id or "",
        conflict.chamber_id or "",
        conflict.engineer_id or "",
        conflict.code,
        evidence,
    )


def _dedupe(conflicts: Sequence[Conflict]) -> tuple[Conflict, ...]:
    by_key: dict[tuple[Any, ...], Conflict] = {}
    for conflict in conflicts:
        by_key.setdefault(_conflict_key(conflict), conflict)
    return tuple(sorted(by_key.values(), key=_conflict_key))


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class CertificationScheduler:
    """Places qualification runs atomically with deterministic results."""

    def __init__(
        self,
        chambers: Sequence[Chamber],
        engineers: Sequence[Engineer],
        committed: Sequence[CertificationAssignment] = (),
        node_limit: int = 100_000,
        offset_seed_cap: int = 256,
    ) -> None:
        self._chambers: dict[str, Chamber] = {item.chamber_id: item for item in chambers}
        if len(self._chambers) != len(chambers):
            raise ValueError("duplicate chamber id")
        self._engineers: dict[str, Engineer] = {item.engineer_id: item for item in engineers}
        if len(self._engineers) != len(engineers):
            raise ValueError("duplicate engineer id")
        self._assignments: list[CertificationAssignment] = list(committed)
        self._batches: dict[str, tuple[tuple[CertRequest, ...], tuple[CertificationAssignment, ...]]] = {}
        self.node_limit = node_limit
        self.offset_seed_cap = offset_seed_cap

    # -- committed state ----------------------------------------------------

    def assignments(self) -> tuple[CertificationAssignment, ...]:
        return tuple(self._assignments)

    def assignment_for(self, request_id: str) -> CertificationAssignment | None:
        for item in self._assignments:
            if item.request_id == request_id:
                return item
        return None

    # -- entry point --------------------------------------------------------

    def schedule_batch(
        self,
        batch_id: str,
        requests: Sequence[CertRequest],
    ) -> ScheduleResult:
        if not batch_id:
            raise ValueError("batch_id is required")
        frozen_requests = tuple(requests)
        if not frozen_requests:
            raise ValueError("empty certification batch")
        ids = [item.request_id for item in frozen_requests]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate request id in batch")
        previous = self._batches.get(batch_id)
        if previous is not None:
            prior_requests, prior_assignments = previous
            if prior_requests != frozen_requests:
                raise ValueError("batch id reused with different requests")
            return ScheduleResult(True, prior_assignments, ())

        committed_ids = {item.request_id for item in self._assignments}
        overlap = sorted(set(ids) & committed_ids)
        if overlap:
            raise ValueError(f"request id already committed: {', '.join(overlap)}")

        search = _Search(self, frozen_requests)
        return search.run(self._commit, batch_id, frozen_requests)

    def _commit(
        self,
        batch_id: str,
        requests: tuple[CertRequest, ...],
        assignments: list[CertificationAssignment],
    ) -> None:
        self._assignments.extend(assignments)
        self._batches[batch_id] = (requests, tuple(assignments))


# ---------------------------------------------------------------------------
# Search internals
# ---------------------------------------------------------------------------


@dataclass
class _Prepared:
    request: CertRequest
    chamber: Chamber | None
    pool: tuple[Engineer, ...]
    start_lo: int
    start_hi: int
    chamber_seeds: tuple[int, ...]
    engineer_seeds: Mapping[str, tuple[int, ...]]
    structural: tuple[Conflict, ...]


class _Search:
    def __init__(self, scheduler: CertificationScheduler, requests: Sequence[CertRequest]) -> None:
        self.scheduler = scheduler
        self.requests = tuple(requests)
        self.fixed = tuple(scheduler._assignments)
        # Global cap on feasibility evaluations across preparation, DFS and
        # pair probing; counting every (start, engineer) check keeps worst-case
        # time proportional to this budget.
        self.eval_budget = scheduler.node_limit
        self.evals = 0
        self.limit_hit = False
        self.dead_end: tuple[int, str, int] | None = None  # (depth, request_id, tried options)
        self.prepared: dict[str, _Prepared] = {
            item.request_id: self._prepare(item) for item in self.requests
        }
        self.order = tuple(
            sorted(
                self.requests,
                key=lambda item: (
                    item.needed_at,
                    item.release_at,
                    -item.duration_minutes,
                    item.request_id,
                ),
            )
        )

    # -- preparation --------------------------------------------------------

    def _prepare(self, request: CertRequest) -> _Prepared:
        static: list[Conflict] = []
        chamber = self.scheduler._chambers.get(request.chamber_id)
        if chamber is None:
            static.append(
                Conflict(
                    "chamber_unknown",
                    request_id=request.request_id,
                    chamber_id=request.chamber_id,
                )
            )

        pool, pool_conflicts = self._resolve_pool(request)
        static.extend(pool_conflicts)

        if request.release_at > request.needed_at:
            static.append(Conflict("release_after_needed", request_id=request.request_id))
        if request.release_at + request.duration_minutes > request.needed_at:
            static.append(
                Conflict(
                    "cannot_complete_by_need",
                    request_id=request.request_id,
                    evidence={
                        "release_at": request.release_at,
                        "needed_at": request.needed_at,
                        "duration_minutes": request.duration_minutes,
                    },
                )
            )

        start_lo = max(
            request.release_at,
            request.needed_at - request.valid_for_minutes - request.duration_minutes,
        )
        start_hi = request.needed_at - request.duration_minutes

        chamber_seeds, engineer_seeds = self._seeds_for(
            request, chamber, pool, start_lo, start_hi
        )

        structural = list(static)
        if chamber is not None and pool and start_lo <= start_hi:
            observed: list[Conflict] = []
            any_feasible = False
            for engineer in pool:
                for seed in set(chamber_seeds).union(engineer_seeds[engineer.engineer_id]):
                    self.evals += 1
                    if self.evals > self.eval_budget:
                        self.limit_hit = True
                        break
                    request_conflicts = self._request_conflicts(
                        request, chamber, seed, ()
                    )
                    engineer_conflicts = self._engineer_conflicts(
                        request, engineer, seed, ()
                    )
                    if not request_conflicts and not engineer_conflicts:
                        any_feasible = True
                        break
                    observed.extend(request_conflicts)
                    observed.extend(engineer_conflicts)
                if any_feasible or self.limit_hit:
                    break
            # Only report placement conflicts when no start/engineer pair works.
            if not any_feasible:
                structural.extend(observed)
        return _Prepared(
            request,
            chamber,
            tuple(sorted(pool, key=lambda item: item.engineer_id)),
            start_lo,
            start_hi,
            chamber_seeds,
            engineer_seeds,
            _dedupe(structural),
        )

    def _resolve_pool(self, request: CertRequest) -> tuple[list[Engineer], list[Conflict]]:
        conflicts: list[Conflict] = []
        engineers = self.scheduler._engineers
        if request.eligible_engineers is None:
            candidates = list(engineers.values())
        else:
            candidates = []
            for engineer_id in request.eligible_engineers:
                engineer = engineers.get(engineer_id)
                if engineer is None:
                    conflicts.append(
                        Conflict(
                            "unknown_engineer",
                            request_id=request.request_id,
                            engineer_id=engineer_id,
                        )
                    )
                else:
                    candidates.append(engineer)
        if request.required_skill is not None:
            skilled = [item for item in candidates if request.required_skill in item.skills]
            if candidates and not skilled:
                conflicts.append(
                    Conflict(
                        "engineer_skill_missing",
                        request_id=request.request_id,
                        evidence={"required_skill": request.required_skill},
                    )
                )
            candidates = skilled
        if not candidates:
            conflicts.append(
                Conflict(
                    "no_eligible_engineer",
                    request_id=request.request_id,
                    evidence={"required_skill": request.required_skill},
                )
            )
        return candidates, conflicts

    def _seeds_for(
        self,
        request: CertRequest,
        chamber: Chamber | None,
        pool: Sequence[Engineer],
        start_lo: int,
        start_hi: int,
    ) -> tuple[tuple[int, ...], dict[str, tuple[int, ...]]]:
        duration = request.duration_minutes
        own_tail = self._effective_cooldown(request, chamber)

        # Cumulative offsets of other requests (each contributes its duration
        # plus an optional cool-down) as subset sums, so packing several
        # not-yet-placed runs around a shift/maintenance edge still surfaces
        # every candidate start an interleaved solution needs.
        #
        # Only offsets that can land a start inside [start_lo, start_hi] are
        # useful; that bounds the distinct sums to the window width, with a hard
        # cap keeping seed generation (outside the node budget) linear.
        max_useful_offset = start_hi - start_lo + duration
        step_sizes: set[int] = set()
        for other in self.requests:
            if other.request_id == request.request_id:
                continue
            other_chamber = self.scheduler._chambers.get(other.chamber_id)
            other_tail = self._effective_cooldown(other, other_chamber)
            step_sizes.add(other.duration_minutes)
            step_sizes.add(other.duration_minutes + other_tail)
        offsets: set[int] = {0}
        for step in sorted(s for s in step_sizes if 0 < s <= max_useful_offset):
            grown = {
                value
                for value in (existing + step for existing in offsets)
                if value <= max_useful_offset
            }
            offsets |= grown
            if len(offsets) > self.scheduler.offset_seed_cap:
                offsets = {0}
                break

        chamber_seeds = {start_lo, start_hi}
        if chamber is not None:
            for span in chamber.maintenance:
                chamber_seeds.add(span.window.end)
                # Latest start that finishes just as maintenance starts, with
                # and without respecting the cool-before buffer.
                chamber_seeds.add(span.window.start - duration)
                chamber_seeds.add(span.window.start - span.cool_before_minutes - duration)
            for fixed in self.fixed:
                if fixed.chamber_id == chamber.chamber_id:
                    chamber_seeds.add(fixed.cooldown_until)
                    chamber_seeds.add(fixed.start - duration - own_tail)

        engineer_seeds: dict[str, tuple[int, ...]] = {}
        for engineer in pool:
            seeds: set[int] = set()
            for shift in engineer.shifts:
                for offset in offsets:
                    seeds.add(shift.start + offset)
                    seeds.add(shift.end - duration - offset)
            for fixed in self.fixed:
                if fixed.engineer_id == engineer.engineer_id:
                    seeds.add(fixed.end)
                    # Slot just before a run already held by this engineer.
                    seeds.add(fixed.start - duration)
            engineer_seeds[engineer.engineer_id] = tuple(sorted(seeds))

        bounded_chamber = tuple(
            sorted(seed for seed in chamber_seeds if start_lo <= seed <= start_hi)
        )
        bounded_engineers = {
            engineer_id: tuple(
                sorted(seed for seed in seeds if start_lo <= seed <= start_hi)
            )
            for engineer_id, seeds in engineer_seeds.items()
        }
        return bounded_chamber, bounded_engineers

    # -- feasibility ---------------------------------------------------------

    @staticmethod
    def _effective_cooldown(request: CertRequest, chamber: Chamber | None) -> int:
        chamber_cooldown = chamber.cooldown_minutes if chamber is not None else 0
        if request.cooldown_minutes is None:
            return chamber_cooldown
        return max(chamber_cooldown, request.cooldown_minutes)

    def _request_conflicts(
        self,
        request: CertRequest,
        chamber: Chamber,
        start: int,
        placed: Sequence[CertificationAssignment],
    ) -> list[Conflict]:
        """Conflicts that are independent of which engineer is chosen."""

        end = start + request.duration_minutes
        conflicts: list[Conflict] = []

        if start < request.release_at:
            conflicts.append(
                Conflict(
                    "starts_before_release",
                    request_id=request.request_id,
                    evidence={"start": start, "release_at": request.release_at},
                )
            )
        stale_bound = request.needed_at - request.valid_for_minutes - request.duration_minutes
        if start < stale_bound:
            conflicts.append(
                Conflict(
                    "certification_stale_before_need",
                    request_id=request.request_id,
                    evidence={
                        "start": start,
                        "expires_at": end + request.valid_for_minutes,
                        "needed_at": request.needed_at,
                    },
                )
            )
        if start > request.needed_at - request.duration_minutes:
            conflicts.append(
                Conflict(
                    "completes_after_need",
                    request_id=request.request_id,
                    evidence={"end": end, "needed_at": request.needed_at},
                )
            )

        job_window = TimeBox(start, end)
        own_tail = self._effective_cooldown(request, chamber)

        for span in chamber.maintenance:
            if job_window.overlaps(span.window):
                conflicts.append(
                    Conflict(
                        "planned_maintenance_overlap",
                        request_id=request.request_id,
                        chamber_id=chamber.chamber_id,
                        evidence={"start": span.window.start, "end": span.window.end},
                    )
                )
            elif span.cool_before_minutes:
                buffer = TimeBox(
                    span.window.start - span.cool_before_minutes, span.window.start
                )
                if job_window.overlaps(buffer):
                    conflicts.append(
                        Conflict(
                            "maintenance_cooldown_overlap",
                            request_id=request.request_id,
                            chamber_id=chamber.chamber_id,
                            evidence={
                                "maintenance_start": span.window.start,
                                "cool_before_minutes": span.cool_before_minutes,
                            },
                        )
                    )

        blockers = [
            item
            for item in (*self.fixed, *placed)
            if item.chamber_id == chamber.chamber_id and item.request_id != request.request_id
        ]
        for blocker in blockers:
            # The gap between the earlier run's end and the later run's start
            # must be at least the earlier run's own cool-down.  Touching at an
            # instant (half-open intervals) is allowed.
            if start >= blocker.end:
                gap = blocker.cooldown_minutes
                earliest = blocker.end + gap
                breach = start < earliest
            elif end <= blocker.start:
                gap = own_tail
                earliest = blocker.start - gap - request.duration_minutes
                latest_end = blocker.start - gap
                breach = end > latest_end
            else:
                # The two run windows overlap, which is never allowed.
                gap = 0
                earliest = blocker.end
                breach = True
            if breach:
                conflicts.append(
                    Conflict(
                        "chamber_cooldown_breach",
                        request_id=request.request_id,
                        chamber_id=chamber.chamber_id,
                        evidence={
                            "blocking_request_id": blocker.request_id,
                            "required_gap_minutes": gap,
                            "earliest_start": earliest,
                        },
                    )
                )
        return conflicts

    def _engineer_conflicts(
        self,
        request: CertRequest,
        engineer: Engineer,
        start: int,
        placed: Sequence[CertificationAssignment],
    ) -> list[Conflict]:
        end = start + request.duration_minutes
        conflicts: list[Conflict] = []
        covered = any(shift.start <= start and end <= shift.end for shift in engineer.shifts)
        if not covered:
            conflicts.append(
                Conflict(
                    "engineer_outside_shift",
                    request_id=request.request_id,
                    engineer_id=engineer.engineer_id,
                    evidence={"start": start, "end": end},
                )
            )
        job_window = TimeBox(start, end)
        busy = sum(
            1
            for item in (*self.fixed, *placed)
            if item.engineer_id == engineer.engineer_id
            and item.request_id != request.request_id
            and item.window.overlaps(job_window)
        )
        if busy >= engineer.capacity:
            conflicts.append(
                Conflict(
                    "engineer_capacity_exceeded",
                    request_id=request.request_id,
                    engineer_id=engineer.engineer_id,
                    evidence={"busy": busy, "capacity": engineer.capacity},
                )
            )
        return conflicts

    def _feasible_starts(
        self,
        prepared: _Prepared,
        engineer: Engineer,
        placed: Sequence[CertificationAssignment],
    ) -> tuple[int, ...]:
        """Every feasible start for this engineer, found by event-point scan.

        Feasibility for a fixed engineer changes only at a finite set of event
        points (maintenance edges, cool-down tails, shift/occupancy edges), so
        evaluating those points is complete without iterating the whole time
        range.
        """

        request = prepared.request
        duration = request.duration_minutes
        own_tail = self._effective_cooldown(request, prepared.chamber)
        chamber = prepared.chamber
        assert chamber is not None
        lo, hi = prepared.start_lo, prepared.start_hi
        if lo > hi:
            return ()

        # Static boundary seeds (maintenance, shifts, cross-request offsets).
        points = set(prepared.chamber_seeds)
        points.update(prepared.engineer_seeds.get(engineer.engineer_id, ()))

        # Dynamic edges from runs already placed during this search.
        for item in placed:
            if item.chamber_id == chamber.chamber_id:
                points.add(item.cooldown_until)
                points.add(item.start - duration - own_tail)
                points.add(item.end - duration)
            if item.engineer_id == engineer.engineer_id:
                points.add(item.end)
                points.add(item.start - duration)

        feasible: list[int] = []
        for start in sorted(point for point in points if lo <= point <= hi):
            self.evals += 1
            if self.evals > self.eval_budget:
                self.limit_hit = True
                break
            request_conflicts = self._request_conflicts(request, chamber, start, placed)
            engineer_conflicts = self._engineer_conflicts(
                request, engineer, start, placed
            )
            if not request_conflicts and not engineer_conflicts:
                feasible.append(start)
        return tuple(dict.fromkeys(feasible))

    def _options(
        self,
        prepared: _Prepared,
        placed: Sequence[CertificationAssignment],
    ) -> tuple[list[tuple[int, Engineer]], int]:
        assert prepared.chamber is not None
        tried = 0
        options: list[tuple[int, Engineer]] = []
        for engineer in prepared.pool:
            starts = self._feasible_starts(prepared, engineer, placed)
            for start in starts:
                tried += 1
                options.append((start, engineer))
        return options, tried

    def _assignment(
        self,
        request: CertRequest,
        start: int,
        engineer: Engineer,
        chamber: Chamber,
    ) -> CertificationAssignment:
        return CertificationAssignment(
            request_id=request.request_id,
            chamber_id=chamber.chamber_id,
            recipe_id=request.recipe_id,
            start=start,
            duration_minutes=request.duration_minutes,
            cooldown_minutes=self._effective_cooldown(request, chamber),
            engineer_id=engineer.engineer_id,
            valid_for_minutes=request.valid_for_minutes,
        )

    # -- depth-first search -------------------------------------------------

    def _search(
        self,
        subset: Sequence[CertRequest],
        placed: list[CertificationAssignment],
        depth: int,
    ) -> list[CertificationAssignment] | None:
        if self.limit_hit:
            return None
        if depth == len(subset):
            return list(placed)
        prepared = self.prepared[subset[depth].request_id]
        options, tried = self._options(prepared, placed)
        for start, engineer in options:
            placed.append(self._assignment(prepared.request, start, engineer, prepared.chamber))
            result = self._search(subset, placed, depth + 1)
            if result is not None:
                return result
            placed.pop()
            if self.limit_hit:
                return None
        if not self.limit_hit:
            candidate = (depth, prepared.request.request_id, tried)
            if self.dead_end is None or depth > self.dead_end[0]:
                self.dead_end = candidate
        return None

    # -- pair contention probing -------------------------------------------

    _DIAGNOSTIC_CODES = frozenset(
        {
            "planned_maintenance_overlap",
            "maintenance_cooldown_overlap",
            "chamber_cooldown_breach",
            "engineer_outside_shift",
            "engineer_capacity_exceeded",
        }
    )

    # Maximum single-feasible placements of one request inspected while
    # diagnosing why a pair cannot be co-scheduled.
    _PAIR_OPTION_CAP = 256

    def _pair_check(
        self, left: CertRequest, right: CertRequest
    ) -> tuple[bool, frozenset[str], bool, bool]:
        """Decide whether two solo-feasible requests can be co-scheduled.

        Bounded and non-recursive: inspect each single-feasible placement of
        ``left`` and ask whether ``right`` then has any start that clears the
        chamber and an engineer.  Returns ``(feasible, codes, chamber_bottleneck,
        engineer_bottleneck)``.
        """

        prepared_left = self.prepared[left.request_id]
        prepared_right = self.prepared[right.request_id]
        codes: set[str] = set()
        # Per-resource: is there a left placement where adding that resource
        # would let the pair succeed?
        more_engineers_would_help = False
        more_chambers_would_help = False

        left_options, _ = self._options(prepared_left, ())
        for index, (start, chosen) in enumerate(left_options):
            if index >= self._PAIR_OPTION_CAP or self.limit_hit:
                # Best-effort diagnosis only; the bounded main search already ran.
                codes.add("pair_probe_capped")
                break
            assignment = self._assignment(left, start, chosen, prepared_left.chamber)

            # For each engineer, the starts that are fully feasible given this
            # left placement (chamber + engineer), plus the starts where only
            # engineer conflicts remain (chamber already free), computed via the
            # budget-bounded scan.
            chamber_seeds = set(prepared_right.chamber_seeds)
            full_feasible = False
            chamber_free_seeds: list[int] = []
            for candidate in prepared_right.pool:
                chamber_seeds.update(
                    self._feasible_starts(prepared_right, candidate, (assignment,))
                )
            for seed in sorted(chamber_seeds):
                if self.limit_hit:
                    codes.add("pair_probe_capped")
                    break
                self.evals += 1
                if self.evals > self.eval_budget:
                    self.limit_hit = True
                    codes.add("pair_probe_capped")
                    break
                request_conflicts = self._request_conflicts(
                    right, prepared_right.chamber, seed, (assignment,)
                )
                codes.update(
                    conflict.code
                    for conflict in request_conflicts
                    if conflict.code in self._DIAGNOSTIC_CODES
                )
                if request_conflicts:
                    continue
                chamber_free_seeds.append(seed)
                for candidate in prepared_right.pool:
                    self.evals += 1
                    engineer_conflicts = self._engineer_conflicts(
                        right, candidate, seed, (assignment,)
                    )
                    codes.update(
                        conflict.code
                        for conflict in engineer_conflicts
                        if conflict.code in self._DIAGNOSTIC_CODES
                    )
                    if not engineer_conflicts:
                        full_feasible = True
                        break
                if self.evals > self.eval_budget:
                    self.limit_hit = True
                    codes.add("pair_probe_capped")
                    break
                if full_feasible:
                    break

            if full_feasible:
                return True, frozenset(), False, False
            if self.limit_hit:
                break

            chamber_feasible = bool(chamber_free_seeds)
            chamber_feasible = bool(chamber_free_seeds)
            # Does an engineer cover any intrinsic-time slot ignoring chamber?
            engineer_available_ignoring_chamber = False
            intrinsic_seeds: set[int] = set()
            for candidate in prepared_right.pool:
                intrinsic_seeds.update(
                    seed
                    for seed in prepared_right.engineer_seeds.get(candidate.engineer_id, ())
                    if prepared_right.start_lo <= seed <= prepared_right.start_hi
                )
            for candidate in prepared_right.pool:
                for seed in intrinsic_seeds:
                    if not self._engineer_conflicts(
                        right, candidate, seed, (assignment,)
                    ):
                        engineer_available_ignoring_chamber = True
                        break
                if engineer_available_ignoring_chamber:
                    break

            if chamber_feasible and not full_feasible:
                # Chamber offers a slot no assigned engineer can cover.
                more_engineers_would_help = True
            if not chamber_feasible and engineer_available_ignoring_chamber:
                # Engineers could cover a slot but the chamber offers none.
                more_chambers_would_help = True

        engineer_bottleneck = more_engineers_would_help
        chamber_bottleneck = more_chambers_would_help
        if not chamber_bottleneck and not engineer_bottleneck:
            chamber_bottleneck = engineer_bottleneck = True
        if engineer_bottleneck:
            codes.add("engineer_capacity_exceeded")
        return False, frozenset(codes), chamber_bottleneck, engineer_bottleneck

    def _pair_conflicts(self, viable: Sequence[CertRequest]) -> list[Conflict]:
        conflicts: list[Conflict] = []
        for index, left in enumerate(viable):
            for right in viable[index + 1 :]:
                if self.limit_hit:
                    return conflicts
                feasible, codes, chamber_blocked, engineer_blocked = self._pair_check(
                    left, right
                )
                if feasible:
                    continue
                if chamber_blocked and engineer_blocked:
                    resource = "chamber+engineer"
                elif chamber_blocked:
                    resource = "chamber"
                else:
                    resource = "engineer"
                same_chamber = left.chamber_id == right.chamber_id
                conflicts.append(
                    Conflict(
                        "pair_contention",
                        request_id=left.request_id,
                        chamber_id=left.chamber_id if same_chamber else None,
                        evidence={
                            "other_request_id": right.request_id,
                            "resource": resource,
                            "observed_codes": tuple(sorted(codes)),
                        },
                    )
                )
        return conflicts

    # -- top level ----------------------------------------------------------

    def run(self, commit, batch_id: str, requests: tuple[CertRequest, ...]) -> ScheduleResult:
        structural: list[Conflict] = []
        viable: list[CertRequest] = []
        for request in self.order:
            prepared = self.prepared[request.request_id]
            if prepared.structural:
                structural.extend(prepared.structural)
            else:
                viable.append(request)

        extra: list[Conflict] = []
        placements: list[CertificationAssignment] | None = None
        if viable:
            placements = self._search(viable, [], 0)
            if placements is None:
                if self.limit_hit:
                    extra.append(
                        Conflict(
                            "search_limit_reached",
                            evidence={"node_limit": self.scheduler.node_limit},
                        )
                    )
                else:
                    extra.extend(self._pair_conflicts(viable))
                    if self.dead_end is not None:
                        depth, request_id, tried = self.dead_end
                        extra.append(
                            Conflict(
                                "search_exhausted",
                                request_id=request_id,
                                evidence={
                                    "depth": depth,
                                    "tried_options": tried,
                                    "evaluations": self.evals,
                                },
                            )
                        )

        conflicts = _dedupe(structural + extra)
        if conflicts or placements is None:
            # All-or-nothing: nothing is committed.
            return ScheduleResult(False, (), conflicts)

        ordered = sorted(placements, key=lambda item: item.request_id)
        commit(batch_id, requests, ordered)
        return ScheduleResult(True, tuple(ordered), ())
