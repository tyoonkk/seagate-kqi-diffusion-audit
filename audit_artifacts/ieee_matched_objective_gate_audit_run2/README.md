# Matched-objective gate audit (run2) — hardened confirmation of run1

Generated 2026-08-24 by the hardened build_ieee_matched_objective_gate_audit.py
in response to the 2026-08-23 Codex independent review (Findings 5, 6, 13).

Differences from the run1 builder:
- refuses to run if the output root already exists (no silent overwrite);
- fail-closed legacy reproduction now checks all 225 case rows
  (5 gates x 45 cases: selected_kind and delta at 1e-12) in addition to the
  per-gate summary statistics;
- the manifest binds the builder file, the specification, all three inputs,
  and every output CSV by SHA-256;
- provenance wording corrected: the specification's local file metadata
  predates the audit output, but no independently timestamped preregistration
  exists.

Equivalence: all five output CSVs are byte-identical to
ieee_matched_objective_gate_audit_run1 (verified by diff on 2026-08-24).
run1 remains the artifact cited by the manuscript; run2 is a hardened
confirmation and does not replace it.

Interpretation carries the same limits as run1: within this retrospective
archive the pattern is consistent with gate-family dependence rather than a
candidate-archive or selection-objective effect; this is an association, not
a causal isolation, because the simple gates act as case-level vetoes and the
historical family remains historically test-exposed. No row is prospective
validation.
