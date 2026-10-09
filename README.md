# Kitaev-ladder Josephson resonance: selected research resources

Research resources for **Resonant suppression of the π-type Josephson component in a Kitaev-ladder junction** (Cheng-Rong Xie, Hiroki Tsuchiura, Yukio Tanaka).

The submission snapshot is `submission-2026-10-09`. This repository supplies selected finite-BdG and recursive-Green-function implementations, figure-generation scripts, numerical scan caches, and the eight figure exports used by the submission manuscript. It is not a complete archive of every original simulation output or every production workflow.

## Install and run

Tested with Python 3.13, NumPy 2.2.1, SciPy 1.16.1, and Matplotlib 3.10.0. Use an isolated environment:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python code/p1b_fingerprint_step1.py
```

The fingerprint check uses finite superconducting leads (40 sites per lead) and compares two representative points: `(L_M, μ_S/t, μ_M/t, t_perp/t)=(4,0,0,0.1)` and `(5,1,1,0.1)`, at `Δ/t=0.6`, `Γ/t=0.1`, and `φ=π/3`. These are finite-BdG checks of π/0 phase selection, not a rerun of the full semi-infinite Matsubara phase diagram. Numerical tolerance and lead-length convergence must be considered for new parameter sets.

## Programs and figure numbering

Historical filenames retain their original numbers. The submission manuscript renumbered some figures:

- `code/general_kitaev_sns.py`: finite S/N/S BdG model, `E_GS=-1/2 sum(E_positive)` and equilibrium current from its phase derivative.
- `code/p1b_sns.py`: a second finite-BdG implementation used by the fingerprint check.
- `code/p1b_rgf_py.py`, `code/mobius_recursive_green.py`, and `code/josephsoncurrentladder.py`: recursive/transfer-matrix implementations and example current calculations. They have explicit example parameters; they are not automatic regenerators of all manuscript panels.
- `code/make_fig5_resonance_diag_v3.py`: manuscript Fig. 4, discrete-level alignment and first-harmonic diagnostics. Reads `data/fig5_J1_cache.npz`; missing parameter/grid combinations are recalculated. Outputs to `generated/`.
- `code/make_fig6_ej_off_vs_res.py`: manuscript Fig. 5, relative Josephson ground-state energies at fixed ladder parameters with `μ_M/t=0.3` versus `0`. Run from `generated/` using `python ../code/make_fig6_ej_off_vs_res.py`.
- `code/make_fig7_clamping_v4.py`: manuscript Fig. 6, analytic cancellation mechanism. Outputs to `generated/`.
- `code/make_fig8_topophase.py`: manuscript Fig. 7, isolated-ladder topological phase boundaries. Outputs to `generated/`.
- Manuscript Figs. 1–3 and 8 are supplied as figure exports. This release does not include a verified, one-command reproduction of their complete original production pipeline. In particular, the schematic and phase-diagram pixels are not raw numerical arrays.

## Numerical files

- `data/fig5_J1_cache.npz`: arrays named `mus_<key>` and `J1_<key>`; `<key>` is computed by `_scan_key` in the associated script from physical parameters and the phase/grid specification. The repository includes the original scan cache unchanged.
- `data/general_kitaev_sns_cases.npz`: example finite-BdG phase grids, energies and currents. These are supporting examples; do not interpret them as raw data for every manuscript figure.
- `figures/`: unchanged submission figure PNGs. Energy and current normalization and physical parameter choices are given in the manuscript captions and code. Clamping describes cancellation in a pair's combined ground-state contribution; individual BdG levels may remain dispersive.
- `checks/`: results of publication-package checks, with their limited scope stated explicitly.
- `provenance.json` and `SHA256SUMS`: source provenance and file integrity checks. Packaging changes affect import/output paths and compatibility/reporting, not Hamiltonians or physical parameters.

## AI assistance and responsibility

The author-developed BCLS/LUS integrated system, using multiple language-model services including the interface label `gpt-5.6-sol`, assisted selected derivations, consistency checks, code/figure modifications, and writing. Model outputs were treated as provisional. The corresponding author independently checked the retained AI-assisted scientific results, code changes, and figures. Underlying versions not exposed by services are not inferred.

## Citation and availability

Cite Cheng-Rong Xie, *Selected code and numerical resources for resonant suppression of the π-type Josephson component in a Kitaev-ladder junction* (2026), GitHub release `submission-2026-10-09`:
https://github.com/Slabestone/kitaev-ladder-josephson-resonance/releases/tag/submission-2026-10-09

Public repository availability is distinct from a DOI archive. No DOI is assigned by this GitHub release. No reuse license has yet been selected by the authors.
