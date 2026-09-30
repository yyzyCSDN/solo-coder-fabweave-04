# FabWeave Operations Lab

FabWeave is a dependency-light Python reference project for semiconductor-fab
operations.  Its production modules model queue-time protection, dispatch,
reticle reservations, recipe qualification, SPC, metrology sampling,
maintenance, chamber matching, holds, genealogy, chemical aging, cleanroom
excursions, demand control, rework, lot merging, equipment dedication,
contamination, takt, route yield and shipment commitments.  The
`fabweave.certification` package schedules recipe-on-chamber qualification runs
atomically: a batch is placed only if every run clears planned maintenance and
its pre-maintenance cool-down buffer, honors chamber cool-down gaps, is covered
by an on-call engineer within one shift and capacity, and finishes while the
certification is still valid at the need time.  When a batch cannot be placed,
nothing is reserved and every detected conflict (structural conflicts, pairwise
resource contention, exhaustive-search evidence) is returned.

The cross-domain workflow layer composes these services for material
readiness, dispatch-after-hold, qualified reservations and assignments,
metrology/SPC gates, containment, powered scheduling, rework release,
merge-to-commit and performance reporting.  It preserves domain rejection
reasons instead of replacing them with a generic success or failure.

Shared infrastructure supplies half-open time intervals, versioned policy
history, immutable decisions, quantity conservation and an append-only event
log with optimistic concurrency.  Domain services use these primitives to keep
event-time, idempotency and fail-closed behavior consistent.

Run the tests with `python -m pytest -q`.
