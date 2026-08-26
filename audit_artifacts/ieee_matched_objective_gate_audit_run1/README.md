# Matched-objective gate audit (run1) — attribution sensitivity

Generated 2026-08-20 by build_ieee_matched_objective_gate_audit.py under the
pre-frozen spec MATCHED_OBJECTIVE_GATE_AUDIT_SPEC.md
(spec SHA-256 aab73c1cf5e9dd507d8d9ef4ad924e4c00a1c3ccd46b6ab5efa00cea12580df8).
Provenance correction (2026-08-26): the specification's local file metadata
predates the audit output and its SHA-256 is bound in audit_manifest.json, but
no independently timestamped preregistration exists. An earlier version of
this README said "frozen BEFORE any result was viewed"; that wording
overstated the provenance and is withdrawn. The result CSVs are unchanged.

Question: is the primary nested audit's negative result an artifact of its
weaker V9 3-family candidate archive, or of the task-ID-free gate family?

Design: SAME V10 equal-budget 45-case archive for every arm;
{legacy5, nested7_veto} gate families x {conservative, mean_only, harm_first}
outer-LOTO selection objectives. All six arms reported. Legacy5 full-archive
recomputation reproduced ieee_v10_uniform_grid_loto_audit_run1 fail-closed.

Headline (see outer_loto_overall_summary.csv):
- legacy5 x ALL three objectives -> identical: 11/45 selected, task-macro
  +0.008579, 0 harmful rows. The historical positive LOTO profile is
  selection-objective-invariant on this archive.
- nested7_veto x conservative -> reference_only in all 9 folds (0/45).
  The nested family's conservative abstention REPRODUCES on the equal-budget
  archive: every non-reference nested gate has >= 5 negative rows on the
  full archive, so conservative eligibility fails in every development fold.
- nested7_veto x harm_first -> reference_only in all 9 folds (0/45).
- nested7_veto x mean_only -> 32/45 selected, task-macro +0.004942 with
  11 negative rows, 3 negative task means, bootstrap interval spanning zero
  (-0.008831, +0.018602).

Interpretation (corrected 2026-08-26 to match the manuscript's claim
boundary): within this retrospective archive, the generality failure of simple
task-ID-free validation-threshold gates is CONSISTENT WITH gate-family
dependence rather than with a candidate-archive or selection-objective effect;
the positive profile continues to require the historically test-exposed legacy
family under every objective. This is an association, not a causal isolation:
the simple gates act as vetoes rather than candidate selectors here, and the
legacy family's positive profile remains inseparable from its test exposure.
An earlier version of this README said "attributable to the GATE FAMILY";
that causal wording is withdrawn. This is a retrospective
attribution sensitivity on a test-exposed family, not prospective validation,
and it does not replace the primary nested audit or the secondary LOTO audit.

Caveat: nested gates run here as case-level vetoes on the fixed
validation-best hybrid (strictly more conservative than the original
select-best-passing semantics); the candidate-level nested audit is not
reproducible on this archive because 23 of 24 grid candidates have no
archived test outcome.
