# Kitaev-ladder Josephson resonance: selected research resources

Research resources for **Resonant suppression of the π-type Josephson component in a Kitaev-ladder junction** (Cheng-Rong Xie, Hiroki Tsuchiura, Yukio Tanaka).

The current submission snapshot is `submission-2026-10-10`, corresponding to the seven-figure v6 manuscript. It contains selected finite-BdG and recursive-Green-function implementations, numerical scan caches, the current manuscript figure exports, and independent checks of the analytical coefficients. This is a selected reproducibility archive; it does not contain every original simulation output or a complete production pipeline for every panel.

The earlier eight-figure snapshots remain available in the GitHub release history. Original PNG exports are retained here as historical inputs; the current figure list is below.

## Install and run

Tested with Python 3.13, NumPy 2.2.1, SciPy 1.16.1, and Matplotlib 3.10.0. Use an isolated environment:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-checks.txt
python code/p1b_fingerprint_step1.py
python code/redraw_submission_figures.py
```

`requirements-checks.txt` includes Pillow for figure annotation layout and SymPy for symbolic checks. The redraw command writes to `generated/`; it preserves the schematic and reconstructs the six other exports. For phase diagrams and the off/resonant CPR panels whose original arrays are unavailable, this command retains the supplied image samples and replaces annotations. It is not a fresh numerical reconstruction of those panels. Cached numerical arrays are used for Fig. 4; analytic formulas generate Figs. 5 and 6.

## Current manuscript figures

- Fig. 1: `figures/SIS_junction1.png`, the unchanged junction schematic.
- Fig. 2: `figures/fig2_phases_largefont.pdf`, current-phase diagrams. Original source image: `currentphase_1.png`.
- Fig. 3: `figures/fig3_off_cpr_largefont.pdf`, representative off-resonant CPRs. Original source image: `l4mu0.png`.
- Fig. 4: `figures/fig4_resonance_energy_largefont.pdf`, four panels in one row: normal-level alignment, first harmonic, inter-leg phase dependence, and relative ground-state energy. It combines the earlier manuscript Figs. 4 and 5.
- Fig. 5: `figures/fig5_clamping_largefont.pdf`, the analytical detuning/resonance mechanism, formerly Fig. 6.
- Fig. 6: `figures/fig6_topology_largefont.pdf`, isolated-ladder bulk invariants, formerly Fig. 7. Special decoupled limits and gapless regions require the qualifications in the manuscript.
- Fig. 7: `figures/fig7_res_cpr_largefont.pdf`, representative resonant CPRs, formerly Fig. 8. Original source image: `l5mu1.png`.

The enlarged labels and figure combination do not change the scan arrays. Fig. 4(d) uses `E_GS(theta)-E_GS(0)`. The earlier energy script subtracted the first phase-grid point, `theta=-pi`, despite labeling the reference as zero. The updated script subtracts the actual zero-phase point. This constant shift does not change currents, harmonics, or minimum locations.

## Programs and numerical files

Historical program filenames retain their earlier numbers.

- `code/general_kitaev_sns.py`: finite S/N/S BdG model, `E_GS=-1/2 sum(E_positive)` and equilibrium current from its phase derivative.
- `code/p1b_sns.py`: a second finite-BdG implementation used by the fingerprint check.
- `code/p1b_rgf_py.py`, `code/mobius_recursive_green.py`, and `code/josephsoncurrentladder.py`: recursive/transfer-matrix implementations and example current calculations. Their example parameters do not automatically regenerate all manuscript panels.
- `code/redraw_submission_figures.py`: current six-figure annotation/redraw workflow, with the original Fig. 1 preserved.
- `code/make_fig5_resonance_diag_v3.py`: the earlier standalone resonance diagnostic layout, now represented by Fig. 4(a–c). Reads `data/fig5_J1_cache.npz` and recalculates missing parameter/grid combinations.
- `code/make_fig6_ej_off_vs_res.py`: the earlier standalone energy comparison, now represented by Fig. 4(d); its reference is corrected to zero phase. Run from `generated/` using `python ../code/make_fig6_ej_off_vs_res.py`.
- `code/make_fig7_clamping_v4.py` and `code/make_fig8_topophase.py`: historical standalone layouts for current Figs. 5 and 6.
- `data/fig5_J1_cache.npz`: unchanged original scan cache. Arrays `mus_<key>` and `J1_<key>` use the physical-parameter/grid key computed by `_scan_key` in the associated script.
- `data/general_kitaev_sns_cases.npz`: unchanged supporting finite-BdG examples, not raw data for all figures.
- `data/energy_curve_raw.npz`: the 401-point phase grid and raw finite-BdG energies for the two curves in Fig. 4(d).
- `data/figure4_data.npz`: the numerical arrays used by the combined Fig. 4, including energy differences with the zero-phase reference.

The fingerprint check uses 40 sites per superconducting lead and compares `(L_M, mu_S/t, mu_M/t, t_perp/t)=(4,0,0,0.1)` and `(5,1,1,0.1)`, at `Delta/t=0.6`, `Gamma/t=0.1`, and `phi=pi/3`. These are finite-BdG checks of phase selection, rather than a rerun of the semi-infinite Matsubara phase diagram. Lead-length convergence and tolerances must be assessed for new parameters.

## Independent analytical checks

Run these from the repository root. JSON output goes to `generated/checks/`; the checked results supplied with this release are in `checks/`.

```sh
python code/verify_low_coefficients.py
python code/verify_general_phi.py
python code/verify_detuned_projection.py
python code/verify_projection_bound.py
python code/verify_pi_and_normalization.py
```

- `verify_low_coefficients.py` symbolically checks the eight-Majorana characteristic polynomial and the fourth-order first-harmonic coefficient via positive-energy symmetric polynomials.
- `verify_general_phi.py` evaluates the fourth-order full finite-BdG resolvent coefficient and compares it with isolated-resonance asymptotics, including several normal lengths, selected modes, phase values, and lead lengths.
- `verify_detuned_projection.py` compares the directly projected eight-dimensional resolvent with the detuned closed-form coefficient in 30 representative cases.
- `verify_projection_bound.py` checks representative projection, Hamiltonian, and interface errors for three lead lengths. These finite-system checks support the scaling estimates; they do not replace the manuscript's proof or establish uniformity at phase endpoints.
- `verify_pi_and_normalization.py`, using `verify_B_nambu.py`, checks the contact-reduced determinant identity, the off/on-resonant frequency kernels for normal lengths 1–6, and fourth-order counting against small-interface-coupling BdG extrapolation.

The exact phase-cancellation statement applies at `phi=pi`. General-phase asymptotics use fixed `0<phi<pi`, an isolated normal-level resonance, and the ordered weak-coupling limits stated in the manuscript. No general sign claim is inferred solely from a numerical fit.

Earlier package/figure check files retain their historical scope. Current acceptance is recorded in `checks/current-submission-check.json`, `checks/current-figure-render-check.json`, and the added analytic JSON files.

`provenance.json` records the resource sources and packaging changes; `SHA256SUMS` checks file integrity. The two original numerical archives are unchanged. No Hamiltonian or physical parameter has been changed by this resource update.

## AI assistance and responsibility

The author-developed BCLS/LUS integrated system, using multiple language-model services including the interface label `gpt-5.6-sol`, assisted selected derivations, consistency checks, code/figure modifications, and writing. Model outputs were treated as provisional. The corresponding author independently checked the retained AI-assisted scientific results, code changes, and figures. Underlying versions not exposed by services are not inferred.

## Citation and availability

Cite Cheng-Rong Xie, *Selected code and numerical resources for resonant suppression of the π-type Josephson component in a Kitaev-ladder junction* (2026), GitHub release `submission-2026-10-10`:
https://github.com/Slabestone/kitaev-ladder-josephson-resonance/releases/tag/submission-2026-10-10

Public repository availability is distinct from a DOI archive. No DOI is assigned by this GitHub release. No reuse license has yet been selected by the authors.
