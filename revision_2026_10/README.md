# Materials of the October 2026 revision

This folder adds the code, configuration, protocol, and lightweight outputs of the
revision in which the candidate archive was rebuilt. The rest of this repository
describes the original analysis and is unchanged.

## What the revision changed

- The archived diffusion generator had learned little from the data: with the 10
  training epochs of the primary-audit source, its held-out denoising loss was
  about that of predicting zero noise. The candidate archive was rebuilt for all
  11 endpoints and the same five repeats with a corrected mixed-type diffusion
  generator (Gaussian diffusion for continuous features, two-category multinomial
  diffusion for binary features, T = 1000, early stopping on 20% of the
  generator's training rows held out in each class), and the percentile clipping
  after sampling was removed.
- The evaluation changed at the same time. Every candidate was trained with
  LightGBM on the processor and with XGBoost on the graphics card, and its
  validation and test metrics were read from the same fitted model. In the
  original archive, test metrics came from a separate LightGBM refit on the
  graphics card, which is not deterministic.
- The original audit scripts in `analysis_code/` were reused unchanged through a
  wrapper (`build_regen_audits_v1.py`). `validate_regen_audits_v1.py` checks that
  the wrapper reproduces the original numbers from the original inputs, and
  `crosscheck_regen_audits_v1.py` recomputes the rebuilt results independently.
- The protocol, code, generator configuration, classifier settings, and
  interpretation rules were recorded with SHA-256 checksums before the run
  started (`REGEN_PROTOCOL_V1_KO.md`, written in Korean, and `REGEN_FREEZE_V1.json`).
  `REGEN_PROTOCOL_V1_ADDENDUM_1_KO.md` corrects one statement of the protocol.

## Layout

`files/` keeps the paths relative to `experiments/seagate_kqi/` of the original
workspace, with one exception: the files of the original equal-budget runs are
stored under shortened folder names (`outputs/paper_strict/v10_<endpoints>/`) so
that no path becomes too long for Windows. `restore_workspace_layout.py` copies
everything back to the original paths, including the modules and inputs that
already exist elsewhere in this repository, and checks every SHA-256.

| Path under `files/` | Contents |
|---|---|
| `revision_2026_10/` | rebuild entry point `run_regen_archive_v1.py`, corrected generator `augmenters_torch_ddpm_v3.py` with `generator_config_v3_final.json`, audit wrapper and checks, generator and determinism diagnostics, protocol and checksum record |
| `outputs/revision_2026_10/regen_archive_v1_run1/` | run manifest (code and input checksums, package versions) and, for each of the 55 endpoint–repeat cases, `case.json` and `candidates.csv` with the validation and test metrics of every candidate for both classifiers |
| `outputs/revision_2026_10/regen_audits_v1/` | outputs of the rebuilt audits: nested audit, leave-one-task-out analysis, matched comparison, leave-one-endpoint-out sensitivity, classifier agreement, candidate-level differences |
| `outputs/revision_2026_10/regen_results_r1/` | `results_r1.json`, the summary read by the number script |
| `outputs/revision_2026_10/archived_generator_replay_v2/` | refits of the two archived generator configurations on the same rows; `..._v1` stopped after 127 rows, which are identical to the corresponding rows of v2 |
| `outputs/revision_2026_10/` (other folders) | generator diagnostics used to choose the settings (no test data), classifier determinism check, percentile-clipping count, feature-importance tables (endpoints 4 and 8 computed with the archived procedure, and a reproduction check of that procedure on endpoint 9), run logs |
| `outputs/paper_strict/` | files of the original archive runs that the revised numbers read: training logs and results of the equal-budget runs (shortened folder names) and files of the primary-audit source |
| `paper/mdpi_electronics_2026/` | `build_r1_text_numbers.py` and its output `r1_text_numbers.json` (every number of the revised text, with the SHA-256 of all 106 inputs), `build_r1_results_package.py`, Table 1 and supplementary-table builders, figure builders and figure data |
| `analyze_task10_v9_targeted_sweep.py` | module imported by the audit wrapper |

The revision also imports 4 modules that are already in
`analysis_code/` (`run_pipeline.py`, `build_ieee_task_independent_nested_audit.py`, `build_ieee_v10_uniform_grid_loto_audit.py`, `build_ieee_matched_objective_gate_audit.py`); they are byte-identical to the versions used
and are not copied again. `SOURCE_MAP.json` records their hashes, as well as the
2 inputs of the number script that already exist elsewhere in this
repository.

## Not included

The fitted generator weights (110 files) and the generated pools (55
files), 4.1 GB in total, are not stored in this repository because of
their size. `EXCLUDED_HEAVY_FILES.json` lists every such file with its size and
SHA-256, so that a separately deposited copy can be checked.

## Re-running

Environment of the run: Python 3.11.2, PyTorch 2.10.0+cu126 (CUDA 12.6),
scikit-learn 1.8.0, imbalanced-learn 0.14.1, LightGBM
4.6.0, XGBoost 3.2.0, NumPy 2.3.5, pandas 3.0.1, SciPy 1.17.1;
NVIDIA GeForce RTX 3070. The Seagate data are not redistributed (see the top-level README).

First rebuild the workspace layout with
`python revision_2026_10/restore_workspace_layout.py <workspace root>`. Then, from
the workspace root:

    python experiments/seagate_kqi/revision_2026_10/run_regen_archive_v1.py \
        --run-name regen_archive_v1_run1 \
        --generator-config experiments/seagate_kqi/revision_2026_10/generator_config_v3_final.json \
        --tasks <comma-separated endpoints>

The run was split over three processes (logs `regen_v1_proc*`; the log of the
third process is in the Korean Windows code page CP949). Each case was computed
once, because a process skips cases that already exist. The audits, their check,
the summary, and the reported numbers then follow from

    python experiments/seagate_kqi/revision_2026_10/build_regen_audits_v1.py --run-dir <run dir> --out-dir <audit dir>
    python experiments/seagate_kqi/revision_2026_10/crosscheck_regen_audits_v1.py --run-dir <run dir> --audit-dir <audit dir>
    python experiments/seagate_kqi/paper/mdpi_electronics_2026/build_r1_results_package.py --audit-dir <audit dir> --out-dir <results dir>
    python experiments/seagate_kqi/paper/mdpi_electronics_2026/build_r1_text_numbers.py

In a repeat check with identical inputs (`classifier_determinism_v1/result.json`),
LightGBM on the processor and XGBoost on the graphics card gave identical
predictions; LightGBM on the graphics card did not. Generator training on the
graphics card is not guaranteed to be bit-identical across machines, so exact
regeneration of the candidates starts from the retained generator weights and
pools.

## Integrity

`SHA256SUMS.txt` covers every other file in this folder. `SOURCE_MAP.json` maps
each file to its workspace path and, where a hash had been recorded earlier (the
run manifest or `r1_text_numbers.json`), confirms that the file matches it.
