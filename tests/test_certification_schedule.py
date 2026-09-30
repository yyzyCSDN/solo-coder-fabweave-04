import pytest

from fabweave.core.timebox import TimeBox
from fabweave.certification import (
    CertRequest,
    CertificationAssignment,
    CertificationScheduler,
    Chamber,
    Engineer,
    MaintenanceSpan,
)


def chamber(cid="C-A", cooldown=0, maintenance=()):
    return Chamber(cid, cooldown, tuple(maintenance))


def engineer(eid="E-1", shifts=(TimeBox(0, 1000),), skills=frozenset(), capacity=1):
    return Engineer(eid, frozenset(skills), tuple(shifts), capacity)


def request(
    rid="R-1",
    cid="C-A",
    duration=60,
    release=0,
    needed=200,
    valid_for=1000,
    cooldown=None,
    skill=None,
    eligible=None,
    recipe="Q-RCP",
):
    return CertRequest(
        rid, cid, recipe, duration, release, needed, valid_for, cooldown, skill, eligible
    )


def codes(result):
    return {conflict.code for conflict in result.conflicts}


# ---------------------------------------------------------------------------
# Basic placement and the four constraints
# ---------------------------------------------------------------------------


def test_places_earliest_feasible_with_expiry():
    scheduler = CertificationScheduler([chamber()], [engineer()])
    result = scheduler.schedule_batch("B-1", [request(needed=200, valid_for=300)])
    assert result.scheduled
    [item] = result.assignments
    assert (item.start, item.duration_minutes) == (0, 60)
    assert item.expires_at == 360
    assert item.engineer_id == "E-1"


def test_moves_past_planned_maintenance():
    pm = MaintenanceSpan(TimeBox(100, 200))
    scheduler = CertificationScheduler([chamber(maintenance=[pm])], [engineer()])
    result = scheduler.schedule_batch("B-1", [request(release=60, needed=400)])
    assert result.scheduled
    assert result.assignments[0].start == 200
    assert not result.assignments[0].window.overlaps(pm.window)


def test_respects_cool_before_maintenance_buffer():
    pm = MaintenanceSpan(TimeBox(200, 300), cool_before_minutes=30)
    scheduler = CertificationScheduler([chamber(maintenance=[pm])], [engineer()])
    result = scheduler.schedule_batch("B-1", [request(release=120, needed=400)])
    assert result.scheduled
    # 120..180 crosses the 170..200 cool buffer and 200..260 crosses the PM;
    # only a start at 300 or later is feasible inside the need window.
    assert result.assignments[0].start == 300


def test_chamber_cooldown_separates_runs():
    scheduler = CertificationScheduler([chamber(cooldown=30)], [engineer()])
    result = scheduler.schedule_batch(
        "B-1",
        [request("R-1", needed=300), request("R-2", needed=300)],
    )
    assert result.scheduled
    first, second = result.assignments
    assert first.start == 0
    assert second.start == first.end + 30


def test_request_can_extend_chamber_cooldown():
    # Chamber needs 30, the first request asks for 50: the larger value wins.
    scheduler = CertificationScheduler([chamber(cooldown=30)], [engineer()])
    scheduler.schedule_batch(
        "B-1",
        [request("R-1", needed=300, cooldown=50), request("R-2", needed=400)],
    )
    first, second = scheduler.assignments()
    assert first.cooldown_minutes == 50
    assert second.start == first.end + 50


def test_run_must_fit_inside_a_single_shift():
    eng = engineer(shifts=(TimeBox(0, 100), TimeBox(120, 220)))
    scheduler = CertificationScheduler([chamber()], [eng])
    # Released at 70: the first shift only fits a 30-minute run and the second
    # shift cannot finish by 150, so no single shift can host the run.
    result = scheduler.schedule_batch("B-1", [request(release=70, needed=150)])
    assert not result.scheduled
    assert "engineer_outside_shift" in codes(result)


def test_shift_start_becomes_earliest_option():
    eng = engineer(shifts=(TimeBox(100, 300),))
    scheduler = CertificationScheduler([chamber()], [eng])
    result = scheduler.schedule_batch("B-1", [request(needed=250)])
    assert result.scheduled
    assert result.assignments[0].start == 100


def test_validity_window_bounds_latest_completion_and_expiry():
    scheduler = CertificationScheduler([chamber()], [engineer()])
    result = scheduler.schedule_batch("B-1", [request(needed=200, valid_for=60)])
    assert result.scheduled
    item = result.assignments[0]
    assert item.end <= 200
    assert item.expires_at >= 200
    assert item.start == 80  # finishes exactly when needed, still valid then


def test_cannot_complete_by_need_is_structural_conflict():
    scheduler = CertificationScheduler([chamber()], [engineer()])
    result = scheduler.schedule_batch("B-1", [request(release=180, needed=200)])
    assert not result.scheduled
    assert "cannot_complete_by_need" in codes(result)


def test_maintenance_swallowing_validity_window_reported():
    pm = MaintenanceSpan(TimeBox(100, 300))
    scheduler = CertificationScheduler([chamber(maintenance=[pm])], [engineer()])
    # Completion must fall in [240, 300]; the chamber is down the whole time.
    result = scheduler.schedule_batch("B-1", [request(needed=300, valid_for=60)])
    assert not result.scheduled
    assert "planned_maintenance_overlap" in codes(result)


# ---------------------------------------------------------------------------
# Engineers: capacity, skills, whitelist
# ---------------------------------------------------------------------------


def test_engineer_capacity_two_allows_parallel_runs():
    scheduler = CertificationScheduler(
        [chamber("C-1"), chamber("C-2")], [engineer(capacity=2)]
    )
    result = scheduler.schedule_batch(
        "B-1",
        [request("R-1", cid="C-1", needed=100), request("R-2", cid="C-2", needed=100)],
    )
    assert result.scheduled
    assert {item.start for item in result.assignments} == {0}
    assert {item.engineer_id for item in result.assignments} == {"E-1"}


def test_required_skill_missing_lists_skill_and_no_engineer():
    scheduler = CertificationScheduler([chamber()], [engineer(skills=frozenset({"OTHER"}))])
    result = scheduler.schedule_batch("B-1", [request(skill="VACUUM-Q")])
    assert not result.scheduled
    assert {"engineer_skill_missing", "no_eligible_engineer"} <= codes(result)


def test_unknown_engineer_in_whitelist_reported():
    scheduler = CertificationScheduler([chamber()], [engineer()])
    result = scheduler.schedule_batch("B-1", [request(eligible=("E-9",))])
    assert not result.scheduled
    assert "unknown_engineer" in codes(result)
    assert "no_eligible_engineer" in codes(result)


def test_whitelist_restricts_choice():
    scheduler = CertificationScheduler(
        [chamber("C-1"), chamber("C-2")],
        [engineer("E-1"), engineer("E-2")],
    )
    result = scheduler.schedule_batch(
        "B-1", [request(cid="C-1", eligible=("E-2",), needed=100)]
    )
    assert result.scheduled
    assert result.assignments[0].engineer_id == "E-2"


# ---------------------------------------------------------------------------
# Unknown / malformed resources
# ---------------------------------------------------------------------------


def test_unknown_chamber_reported():
    scheduler = CertificationScheduler([chamber()], [engineer()])
    result = scheduler.schedule_batch("B-1", [request(cid="C-NOPE")])
    assert not result.scheduled
    assert "chamber_unknown" in codes(result)
    conflict = next(c for c in result.conflicts if c.code == "chamber_unknown")
    assert conflict.chamber_id == "C-NOPE"


def test_multiple_independent_conflicts_all_listed():
    scheduler = CertificationScheduler([chamber()], [engineer(skills=frozenset())])
    bad = request(
        cid="C-NOPE",
        release=500,
        needed=200,
        skill="VACUUM-Q",
    )
    result = scheduler.schedule_batch("B-1", [bad])
    found = codes(result)
    assert {
        "chamber_unknown",
        "release_after_needed",
        "cannot_complete_by_need",
        "engineer_skill_missing",
        "no_eligible_engineer",
    } <= found


# ---------------------------------------------------------------------------
# All-or-nothing batch semantics
# ---------------------------------------------------------------------------


def test_batch_failure_commits_nothing_and_names_pair_contention():
    # One engineer covering two distinct chambers: two 60-minute certs both due
    # at 100 overlap, but the chambers themselves never contend.
    scheduler = CertificationScheduler(
        [chamber("C-1"), chamber("C-2")], [engineer()]
    )
    result = scheduler.schedule_batch(
        "B-1",
        [request("R-1", cid="C-1", needed=100), request("R-2", cid="C-2", needed=100)],
    )
    assert not result.scheduled
    assert scheduler.assignments() == ()
    pair = [c for c in result.conflicts if c.code == "pair_contention"]
    assert len(pair) == 1
    assert pair[0].evidence["other_request_id"] == "R-2"
    assert pair[0].evidence["resource"] == "engineer"
    assert pair[0].evidence["observed_codes"] == ("engineer_capacity_exceeded",)
    assert "search_exhausted" in codes(result)


def test_chamber_only_pair_contention_named():
    # Two idle engineers but one chamber whose only open slot before the need
    # is a single 60-minute window: contention is purely the chamber.
    pm = MaintenanceSpan(TimeBox(60, 1000))
    scheduler = CertificationScheduler(
        [chamber(maintenance=[pm])], [engineer("E-1"), engineer("E-2")]
    )
    result = scheduler.schedule_batch(
        "B-1",
        [request("R-1", needed=60), request("R-2", needed=60)],
    )
    assert not result.scheduled
    pair = [c for c in result.conflicts if c.code == "pair_contention"]
    assert len(pair) == 1
    assert pair[0].chamber_id == "C-A"
    assert pair[0].evidence["resource"] == "chamber"
    assert "chamber_cooldown_breach" in pair[0].evidence["observed_codes"]
    assert "engineer_capacity_exceeded" not in pair[0].evidence["observed_codes"]


def test_successful_batch_then_blocking_failure_leaves_first_intact():
    scheduler = CertificationScheduler([chamber(cooldown=30)], [engineer()])
    first = scheduler.schedule_batch("B-1", [request("R-1", needed=200)])
    assert first.scheduled
    second = scheduler.schedule_batch(
        "B-2",
        [
            request("R-2", release=0, needed=80),
            request("R-3", release=0, needed=80),
        ],
    )
    assert not second.scheduled
    committed = scheduler.assignments()
    assert [item.request_id for item in committed] == ["R-1"]
    # The failed batch left no phantom reservation: a feasible follow-up works.
    third = scheduler.schedule_batch("B-3", [request("R-4", needed=400)])
    assert third.scheduled
    assert third.assignments[0].start >= committed[0].cooldown_until


def test_three_way_dead_end_reported_without_false_pair_conflicts():
    # Capacity-2 engineer, chamber never blocks: any pair overlaps fine, but
    # three 60-minute certs due at 100 cannot all be placed.
    scheduler = CertificationScheduler(
        [chamber("C-1"), chamber("C-2"), chamber("C-3")], [engineer(capacity=2)]
    )
    result = scheduler.schedule_batch(
        "B-1",
        [request(rid, cid=f"C-{rid[-1]}", needed=100) for rid in ("R-1", "R-2", "R-3")],
    )
    assert not result.scheduled
    assert "pair_contention" not in codes(result)
    assert "search_exhausted" in codes(result)
    assert scheduler.assignments() == ()


# ---------------------------------------------------------------------------
# Committed work, stability and idempotency
# ---------------------------------------------------------------------------


def _committed(cid="C-A", eid="E-1", start=0, duration=60, cooldown=60):
    return CertificationAssignment(
        request_id="FIXED",
        chamber_id=cid,
        recipe_id="Q-RCP",
        start=start,
        duration_minutes=duration,
        cooldown_minutes=cooldown,
        engineer_id=eid,
        valid_for_minutes=1000,
    )


def test_committed_assignment_is_respected_with_cooldown():
    scheduler = CertificationScheduler(
        [chamber(cooldown=30)], [engineer()], committed=[_committed(cooldown=60)]
    )
    result = scheduler.schedule_batch("B-1", [request("R-1", needed=400)])
    assert result.scheduled
    assert result.assignments[0].start == 120  # committed 0..60 plus 60-minute tail


def test_deterministic_regardless_of_input_order():
    requests_a = [request("R-2", needed=300), request("R-1", needed=200)]
    requests_b = [request("R-1", needed=200), request("R-2", needed=300)]

    def run(requests):
        scheduler = CertificationScheduler([chamber(cooldown=30)], [engineer()])
        result = scheduler.schedule_batch("B-1", requests)
        return [(item.request_id, item.start, item.engineer_id) for item in result.assignments]

    assert run(requests_a) == run(requests_b)


def test_batch_idempotent_replay_returns_same_assignments():
    scheduler = CertificationScheduler([chamber(cooldown=30)], [engineer()])
    payload = [request("R-1", needed=300), request("R-2", needed=300)]
    first = scheduler.schedule_batch("B-1", payload)
    second = scheduler.schedule_batch("B-1", payload)
    assert first.assignments == second.assignments
    assert len(scheduler.assignments()) == 2


def test_batch_id_reuse_with_changed_requests_rejected():
    scheduler = CertificationScheduler([chamber()], [engineer()])
    scheduler.schedule_batch("B-1", [request("R-1", needed=300)])
    with pytest.raises(ValueError):
        scheduler.schedule_batch("B-1", [request("R-1", needed=400)])


def test_resubmitting_committed_request_id_rejected():
    scheduler = CertificationScheduler([chamber()], [engineer()])
    scheduler.schedule_batch("B-1", [request("R-1")])
    with pytest.raises(ValueError):
        scheduler.schedule_batch("B-2", [request("R-1", needed=900)])


def test_duplicate_request_id_in_batch_rejected():
    scheduler = CertificationScheduler([chamber()], [engineer()])
    with pytest.raises(ValueError):
        scheduler.schedule_batch("B-1", [request("R-1"), request("R-1")])


def test_overlapping_maintenance_windows_rejected():
    with pytest.raises(ValueError):
        chamber(maintenance=[MaintenanceSpan(TimeBox(0, 100)), MaintenanceSpan(TimeBox(50, 150))])


# ---------------------------------------------------------------------------
# Bounded search and exhaustive conflict reporting
# ---------------------------------------------------------------------------


def test_node_limit_truncation_is_fail_closed_and_commits_nothing():
    # Wide windows with two resources force a large backtracking search.  A tiny
    # budget reports search_limit_reached rather than hanging or returning a
    # partial/incorrect placement.
    chambers = [chamber("C-1"), chamber("C-2")]
    engineers = [engineer("E-1"), engineer("E-2")]
    many = [
        request(f"R-{i}", cid=("C-1" if i % 2 else "C-2"), needed=600)
        for i in range(12)
    ]
    scheduler = CertificationScheduler(chambers, engineers, node_limit=5)
    result = scheduler.schedule_batch("B-1", many)
    assert not result.scheduled
    assert "search_limit_reached" in codes(result)
    assert scheduler.assignments() == ()


def test_successful_batch_lists_every_assignment_with_validity():
    scheduler = CertificationScheduler(
        [chamber("C-1"), chamber("C-2"), chamber("C-3")],
        [engineer("E-1", capacity=3)],
    )
    reqs = [request(f"R-{i}", cid=f"C-{i+1}", needed=100) for i in range(3)]
    result = scheduler.schedule_batch("B-1", reqs)
    assert result.scheduled
    assert len(result.assignments) == 3
    for item in result.assignments:
        assert item.end <= item.start + item.duration_minutes
        assert item.expires_at >= 100


def test_cool_before_buffer_does_not_block_run_fitting_before_it():
    # Buffer [170,200) before maintenance [200,...); a run ending at 160 is fine.
    pm = MaintenanceSpan(TimeBox(200, 260), cool_before_minutes=30)
    scheduler = CertificationScheduler([chamber(maintenance=[pm])], [engineer()])
    result = scheduler.schedule_batch(
        "B-1", [request(release=100, needed=160)]
    )
    assert result.scheduled
    assert result.assignments[0].end == 160
