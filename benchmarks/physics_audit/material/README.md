# Physics audit, material lens: how much does the AA5754-O model matter?

**Question.** Is the material model of the springback benchmark (40 x 40 x
1 mm AA5754-O from the precomp library: E 70 GPa, yield 100 MPa, Hollomon
K 420 MPa / n 0.30 fitted by linear + Voce, Hill48 from r 0.75 / 0.70 / 0.80,
isotropic hardening only; friction 0.1) close enough to the real alloy for
the engine to predict the formed and released shape "almost exactly"?
Which material data have to come from coupons?

**Method.** (a) Published AA5754-O data are compared with the library
entry. (b) Three of the benchmark's eight test parts are re-run with the
benchmark's own decks (same commanded shape, tool path, mesh, fixture,
solver), changing one material or friction input at a time to a
literature-supported alternative:

| part | footprint | depth | wall | base deviation RMS (benchmark) |
|---|--:|--:|--:|--:|
| `truncated_cone-s2026-0000` (all variants) | 15.4 mm | 3.09 mm | 38.6 deg | 0.767 mm (0.7665) |
| `pyramid-s2026-0001` | 19.4 mm | 2.58 mm | 41.2 deg | 0.699 mm (0.6991) |
| `elliptic_cone-s2026-0001` | 16.1 mm | 3.26 mm | 42.1 deg | 0.841 mm (0.8407) |

Each variant is compared with its base run: the **released shape** (after
the 3-2-1 release) and the loaded shape (end of forming, tool in place), the
**springback** (released minus loaded surface), and the benchmark's metric
(RMS vertical deviation from the target over the part). The key variants
were repeated with the bending-accurate element (incompatible-mode Hex8, 5
integration points per layer: `_im5`). (c) The runs' own output is used to
check strains, strain paths and rates against what the material law and its
data cover.

Everything here is simulation plus literature; no coupon was tested.
Lengths are in mm. "Verified" sources were read in their own text (URL
given); "inference" marks our reasoning or estimates.

## Bottom line

* **The material card is not what limits the prediction of these parts.**
  One-at-a-time, every input moved across the published AA5754-O spread
  shifts the released shape by **0.003-0.042 mm RMS (at most 0.069 mm at a
  point)**, against deviations from the target of 0.70-0.84 mm RMS. A
  complete "literature" material (von Mises, Coer's Voce curve, Bauschinger
  backstresses from AA5754-O cyclic shear, E degrading with plastic strain)
  moves it by **0.017-0.020 mm RMS** on the standard element (three parts;
  the effects partly cancel) and by **0.045 mm RMS (0.072 max)** on the
  bending-accurate IM5 element, where the Bauschinger effect dominates. The
  0.9-1.1 mm rim sag that dominates the benchmark error is a structural
  outcome of the process: no material variant changed the rim band by more
  than 0.046 mm RMS.
* **Why so little: under displacement control the formed shape does not
  depend on the flow-stress level at all.** Scaling E and every stress
  parameter by 0.9 reproduced the released shape to 1e-13 mm
  (`scaled090`; the tool force fell by exactly 10 %). Shape depends only on
  dimensionless material ratios: sigma/E (springback), the shape of the
  hardening curve, the shape of the yield locus, and the reverse-loading
  (Bauschinger) behaviour. Those are what coupons must measure; the absolute
  strength matters for the forces (robot deflection), not for the FE shape.
* **The two material effects that matter** are the two the library does not
  model: the **Bauschinger effect** (0.013-0.016 mm on the standard
  element; **0.034-0.040 mm and -23 to -31 % springback on IM5**, whose 5
  points per layer resolve the reversed bending the backstresses act on) and
  the **biaxial strength / yield-locus shape** (r-based Hill48 puts
  sigma_b/sigma_0 at 0.95, AA5754-O measures 1.0-1.1: 0.027-0.042 mm). E,
  initial yield, r-values, E(eps_p), friction and a 5 % thinner sheet are
  each <= 0.011 mm; the large-strain hardening extrapolation <= 0.016 mm.
* **The springback proper is small on these parts (0.08-0.18 mm RMS)** and
  the material inputs change it by up to -31 % (Bauschinger, IM5), -20 %
  (biaxial strength), -19 / +16 % (yield -/+15 MPa), +6 to +11 % (E -10 %).
  On larger parts, where springback reaches millimetres (inference: the
  release deflection grows with the span squared at a given residual
  stress), these percentages become the material error budget - that is
  where calibrated coupons pay off.
* **Discretisation outweighs every material input at this mesh**: the IM5
  element instead of the standard Hex8 moves the released shape by
  0.049 / 0.061 mm RMS (cone / pyramid; 0.24 / 0.14 mm max) and the
  springback by +19 / +28 % (element lens). The material sensitivities agree
  between the elements except the Bauschinger effect, which doubles on IM5:
  a model that moves to IM5 should carry kinematic hardening.
* **Achievable accuracy from the material side:** the library card costs
  **0.02-0.05 mm RMS** on parts of this size (literature combination
  0.017-0.045; quadrature sum of the one-at-a-time maxima 0.036-0.051 per
  part and element, `error_budget.json`). With the coupon programme below the
  material share would fall to ~0.01 mm (inference), well under the
  discretisation (0.05-0.24 mm) and process (rim sag ~1 mm) errors.
  "Almost exactly" is decided by mesh, fixture and robot compliance, not by
  the material card - on parts of this size.

## Findings ranked by effect on the released shape

`d released` = RMS (max) over the part of the variant's released surface
minus its base's (IM5 variants against `base_im5`), mm. Springback
(released minus loaded, RMS) of the bases: cone 0.120 (IM5 0.143), pyramid
0.139 (IM5 0.179), elliptic cone 0.082 mm. Evidence: one row per run in
`sensitivity.csv` / `sensitivity.md`; decks in `runs/<part>/<variant>/`.

| # | Input changed (literature basis) | d released, standard Hex8: cone / pyramid / elliptic | on IM5: cone / pyramid | springback change (std; IM5) | verdict |
|---|---|---|---|---|---|
| 0 | *context:* standard Hex8 -> incompatible modes, 5 pts/layer (`base_im5`) | 0.049 (0.242) / 0.061 (0.141) / - | - | +19 / +28 % | discretisation (element lens): larger than any material input |
| 1 | **Bauschinger**: two Armstrong-Frederick backstresses from AA5754-O cyclic shear [4, 5], monotonic curve unchanged (`kin_shutov`) | 0.016 (0.050) / 0.016 (0.045) / 0.013 (0.039) | **0.034 (0.056) / 0.040 (0.069)**, shallower (+0.033 / +0.038 mean) | -12 / -6 / -13 %; **IM5 -31 / -23 %** | largest on the bending-accurate element; grows with through-thickness resolution and with the number of passes |
| 2 | **yield-locus shape**: Hill48 by stress ratios with sigma_b/sigma_0 = 1.10 (measured locus elongation at 10-15 % strain [2]) instead of r-based (0.95) (`hill_sb110`) | 0.036 (0.056) / 0.027 (0.042) / 0.042 (0.064); shallower (+0.025 to +0.040 mean) | 0.032 (0.066) / - | -13 / -6 / -20 %; IM5 -3 % | largest on the standard element; a bound (implies r ~1.5), settle with a bulge test |
| 3 | large-strain hardening: Voce of Coer 2018 (saturates at 292 MPa: -9 % at eps_p 0.4) (`hard_voce_coer`) | 0.014 (0.025) / 0.009 (0.016) / 0.016 (0.030); deeper | 0.008 (0.016) / - | +10 / +4 / +18 %; IM5 +5 % | small; larger on deeper parts |
| 3b | ... power law of Iadicola 2008 (K 474, n 0.317: +11 % at 0.4) (`hard_pow_iadicola`) | 0.012 (0.022) / 0.007 (0.014) / - | - | -7 / -4 % | small |
| 4 | von Mises instead of r-based Hill48 (`vm`) | 0.011 (0.019) / 0.013 (0.018) / 0.017 (0.026); shallower | 0.012 (0.022) / - | -4 / -5 / -8 %; IM5 -2 % | small; literature prefers von Mises over r-based Hill48 for AA5754-O springback [1] |
| 5 | E -10 % (chord-modulus order) (`E63`) | 0.010 (0.017) / 0.005 (0.018) / 0.007 (0.011) | 0.009 (0.013) / - | +11 / +6 / +10 %; IM5 +10 % | small |
| 6 | initial yield -15 / +15 MPa, curve shifted (`ys85`, `ys115`) | cone 0.009 / 0.008, pyramid 0.011 / 0.009 (max 0.032) | - | cone -19 / +16 %, pyramid -15 / +10 % | shape insensitive; springback ~ sigma/E |
| 7 | E(eps_p) per element, -15 % at saturation (`Edeg`) | 0.005 (0.010) / 0.010 (0.023) / - | - | 0 / -2 % | negligible: springback is carried by the lightly strained rim and flange, where E stays ~E0 |
| 8 | weak Bauschinger (one backstress, 20 MPa) (`kin_mild`) | 0.006 (0.016) / - / - | 0.010 (0.019) / - | -8 %; IM5 -9 % | small |
| 9 | measured r-value sets 0.69/0.73/0.87 [2], 0.663/0.860/0.717 [1] (`hill_iadicola`, `hill_coer`) | cone 0.006 (0.014) / 0.005 (0.023) | - | -1 / -5 % | negligible |
| 10 | sheet 5 % thinner (0.95 mm) (`t095`) | cone 0.005 (0.027) | - | -2 % | negligible (data, not model) |
| 11 | friction 0.05 / 0.2 (`mu005`, `mu02`) | cone 0.003 (0.013) / 0.003 (0.010); pyramid 0.2: 0.003 | - | 0 / -2 % | negligible for shape |
| - | E and all stresses x 0.9 (`scaled090`) | cone 0.000 (1e-13) | - | 0 % | shape is flow-stress-level invariant |
| - | literature combination: 4 + 3 + 1 + 7 (`combo_lit`) | 0.020 (0.046) / 0.020 (0.050) / 0.017 (0.052) | **0.045 (0.072)** / - | -9 / -3 / -9 %; IM5 -30 % | the realistic total |

Not a shape effect but a real one: the peak tool force follows the
flow-stress level (cone: 942-1052 N across the variants; x0.9 stresses ->
-10.0 %; yield +15 MPa -> +7 %; biaxial locus +9 %). A serial robot of
~1 N/um Cartesian stiffness (inference, a typical order) turns a 10 %
force error on ~1 kN into ~0.1 mm of tool-position error - more than any
material effect on the FE shape. The absolute flow stress must be right for
the robot-compliance correction (`precomp.robot`, removed in the cleanup; at commit fe29724), not for the FE shape.

## Recommendations

| Priority | Action | Why (evidence) | Cost |
|---|---|---|---|
| 1 | Report the material share of the benchmark error honestly: ~0.02-0.05 mm RMS on these parts; keep the library card for the simulation-only benchmark | every variant <= 0.042 mm, the combination 0.017-0.045 mm (`sensitivity.csv`, `error_budget.json`) | none |
| 2 | For any claim on real parts, characterise the actual coil: 3 tensile tests at 0/45/90 deg with extensometer and width gauge, with 3-4 unload-reload loops (E and chord modulus, Rp0.2, flow curve to uniform elongation, r); 1 hydraulic bulge test (sigma_b/sigma_0 and the flow curve to eps ~0.4-0.5, where tensile tests stop at 0.19-0.22); 1 tension-compression (anti-buckling) or cyclic simple-shear test at 2-3 prestrains (Bauschinger) | the two largest effects (locus, Bauschinger) and the large-strain curve need exactly these; EN 485-2 allows Rp0.2 >= 80 MPa and Rm 190-240 MPa, +-12 % between coils [6] - which moves forces, not shape (scale invariance) | ~2 lab days; order EUR 3-6 k at a test house (inference) |
| 3 | Replace r-based Hill48 for AA5754-O: use von Mises, or Hill48 by `stress_ratios` from the coupon yield stresses and the bulge test (SparLab supports both) | r-based Hill48 (library and both measured r sets) gives sigma_90/sigma_0 1.02-1.07 where AA5754-O measures 0.98, and sigma_b/sigma_0 0.93-0.97 where it measures 1.0-1.1 (`lit_yield_ratios.csv`, [2]); r-based Hill48 under-predicted AA5754-O split-ring springback where von Mises matched [1] | hours (`precomp.materials` + the setup; retrain surrogates afterwards) |
| 4 | Add Chaboche backstresses once reverse-loading data exist - mandatory together with a move to IM5 / more integration points (SparLab takes 4; precomp's `Material` carries one - extend it to a list) | accumulated plastic strain is 2.9-3.2 x the net membrane strain: every pass bends, unbends and shears the wall (`validity_*.json`); on the IM5 element the Bauschinger effect moves the shape 0.034-0.040 mm and cuts springback by 23-31 % | hours for precomp; data needed |
| 5 | Fit hardening to a bulge curve (or Voce / Hockett-Sherby over the measured range) and record the range; 27-46 % of the formed elements exceed eps_p 0.19, beyond any tensile test | published fits differ by -21 / +10 % at eps_p 0.6 (`lit_curves.csv`); effect here 0.007-0.016 mm, growing with depth | hours once data exist |
| 6 | Yld2000-2d in SparLab only if the bulge test shows sigma_b/sigma_0 far from what stress-ratio Hill48 can fit together with the directional stresses | `hill_sb110`: 0.027-0.042 mm, the bound of what the locus shape does here | ~1-2 weeks implementation and verification (inference) |
| 7 | Nothing for E(eps_p), friction, sheet thickness scatter, strain rate or temperature as far as shape is concerned | each <= 0.011 mm; rate and heating bound below | none |

## (a) Literature vs library

| Quantity | Library (`python/precomp/materials.py`) | Published AA5754-O | Source |
|---|---|---|---|
| E, nu | 70 GPa, 0.33 | 68 GPa, 0.33 used for 1 mm AA5754-O [1]; 70.0 GPa, 0.33 from K 68.63 / G 26.31 GPa fitted to AA5754-O shear data [4]; 70.5 GPa, G 26.5 GPa [6] | [1], [4], [6] |
| Rp0.2 RD / TD / DD | 100 MPa | 94.1 / 92.1 / 90.9 MPa (1 mm sheet) [2]; Voce Y0 102.75 MPa [1]; >= 80 MPa (EN 485-2, O/H111, 0.5-1.5 mm) [6] | [2], [1], [6] |
| Rm RD / TD / DD | ~212 MPa (engineering, from the Hollomon law) | 226 / 218 / 216 MPa [2]; 222.2 / 216.5 / 211.0 MPa (RD / TD / DD) [1]; 190-240 MPa [6] | [2], [1], [6] |
| uniform elongation | - | 18.9 / 21.5 / 22.4 % [2] | [2] |
| hardening | Hollomon K 420, n 0.30 from 100 MPa, fitted by linear + Voce over eps_p 0-0.6 (1.6 % RMS) | power law K 474 / 458 / 447 MPa, n 0.317 / 0.326 / 0.323; Voce 289 (1 - 0.686 exp(-12.3 eps)) MPa (RD) [2]; Voce Y0 102.75, Ysat 292.14 MPa, Cy 13.5; Hockett-Sherby 91.74 / 308.63 MPa / 7.98 / 0.831 [1] | [2], [1] |
| flow stress vs library | - | at eps_p 0.05-0.19 the published curves lie 3-12 % *above* the library (0.10: 228-243 vs 218 MPa); beyond the tests they bracket it: Voce and Hockett-Sherby -5 to -21 %, power law +10 to +11 % at eps_p 0.4-0.6 (`lit_curves.csv`) | computed |
| r0 / r45 / r90 | 0.75 / 0.70 / 0.80 | 0.69 / 0.73 / 0.87 [2]; 0.663 / 0.860 / 0.717 [1]; r falls with strain rate [3] | [2], [1], [3] |
| yield stress ratios | r-based Hill48: sigma_45 1.031, sigma_90 1.018, sigma_b 0.950, plane strain 1.111 / 1.132 | measured sigma_45 0.966, sigma_90 0.979 at 0.2 % [2]; balanced biaxial: locus slightly flattened at 1 %, then "elongates in the balanced biaxial direction" with strain (normalised BB > 1 above ~5 %), r_b = 1.13; plane-strain points near von Mises and Hill48; Yld2000-2d fits BB but not PS without "artificially large" r (1.5 / 3.0) [2] | [2] |
| Bauschinger | none | cyclic simple shear at three prestrains, fitted with two backstresses: c1 115.5 MPa, kappa1 0.0885 1/MPa, c2 11500 MPa, kappa2 0.01676 1/MPa, K 31.5 MPa, gamma 1963 MPa, beta 13.33 [4] (data of [5]); "reversed shear results showed a very weak Bauschinger effect" (isotropic hardening used) [1] - i.e. early re-yielding with a short transient, little permanent softening | [4], [1] |
| E degradation | none | no AA5754 value found; EN AW-6082-T6 chord modulus 72 -> 57.4-69 GPa (up to -20 %) [7]; ~-10 % for AA6022-T4 (Cleveland and Ghosh 2002, via search summaries [8]); AA5754 "is known to be sensitive to this phenomenon" [1] | [7], [8], [1] |
| rate, temperature | rate-independent | negative strain-rate sensitivity at room temperature (dynamic strain ageing, serrated PLC flow), positive above ~150 C; yield-point elongation 0.0038-0.0074 [3] | [3] |
| springback benchmark | - | 1 mm AA5754-O cup, split ring: opening 6.00 +- 0.07 mm; von Mises + Voce matched it, r-based Hill48 under-predicted [1] | [1] |

Verdict on the library entry: E, initial yield and r-values are inside the
published spread; the flow stress at eps_p 0.05-0.19 is 3-12 % low (no
consequence for shape: scale invariance); the extrapolation beyond eps_p
0.2 is unconstrained by any tensile test (published fits differ by 30 % at
0.6); r-based Hill48 contradicts the measured directional and biaxial yield
stresses; kinematic hardening and E degradation are absent.

## (b) Sensitivity study

**Setup.** The benchmark decks exactly: `precomp.fea.build_deck` from the
cached run's `precomp_deck.json` and `commanded.npz`
(`/home/user/wt/bench/benchmarks/springback/work/runs/runs/...`), run by
the audit build `sparlab 1.0.0 (b1a81217b6e3-dirty)` with one thread. The
three base runs reproduce the benchmark's cached results to four digits
(0.7665 / 0.6991 / 0.8407 mm, `results.jsonl` vs
`../../springback/stage_n18/headline.csv`), so the audit build computes the
same physics. Variants (`variants.py`) change the precomp `Material`
(`matlens.apply_variant`) or, for keys precomp does not write (a second
backstress, stress-ratio Hill48, `material_regions`, the element
formulation), patch `deck.json` (`matlens.build_variant`).
Kinematic variants keep the library's monotonic uniaxial curve beyond
eps_p 0.01 (the isotropic part is refitted, `voce_fit_with_backstresses`),
so they change the reverse-loading response only; the Shutov-Kreissig
parameters are mapped to Armstrong-Frederick form in uniaxial terms
(C = 1.5 c, gamma = c kappa sqrt(3/2): saturation 13.8 and 73.1 MPa, rates
12.5 and 236; initial elastic limit 29.5 MPa - as in [4], an early
re-yielding model). `Edeg` gives each element E = E0 - (E0 - Ea)(1 -
exp(-xi eps_p)), Ea = 0.85 E0, xi = 20, from its base run's final plastic
strain (8 bins, `material_regions`) - an emulation in which the degraded E
acts from the start, which matters only while an element is still elastic.
The stress-ratio Hill48 of `hill_sb110` (sigma_45 0.966, sigma_90 0.979,
sigma_b 1.10) implies r0 1.55, r90 1.40: it is the bound "artificially
large r" of [2], not a calibrated locus.

The full per-run tables (all columns) are `sensitivity.md` and
`sensitivity.csv`. Three observations beyond the ranking:

1. **Where the shape change goes.** The loaded shape moves about as much
   as the released one (`d loaded` 0.003-0.027 mm): material inputs shift
   the formed shape and its springback by comparable, small amounts, often
   of opposite sign (e.g. `ys85`: loaded 0.021, springback 0.022, released
   0.009 mm).
2. **Direction.** Everything that raises biaxial or plane-strain strength
   relative to uniaxial (von Mises, `hill_sb110`) makes the part come out
   shallower (mean +0.011 to +0.040 mm) and slightly reduces the benchmark
   deviation (-0.008 to -0.032 mm); a saturating (Voce) curve acts the other
   way (deeper by 0.007-0.015 mm). None comes near the 1 mm rim sag.
3. **Element dependence.** On the IM5 element the sensitivities to E
   (0.009 vs 0.010), von Mises (0.012 vs 0.011), Voce (0.008 vs 0.014), the
   locus bound (0.032 vs 0.036) and the weak backstress (0.010 vs 0.006) stay
   at the same small level; the Bauschinger effect grows 2-2.5 times (cone
   0.034 vs 0.016 mm, springback -31 vs -12 %; pyramid 0.040 vs 0.016 mm,
   -23 vs -6 %). Our reading (inference): five points per layer and a
   bending-accurate element resolve the reversed through-thickness bending
   that the backstresses act on, which two Gauss points per standard layer
   smear out. Any production model that adopts IM5 (or more layers) should
   carry kinematic hardening.

## (c) Validity limits of the material model in these runs

From the solver's output (`validity.py` -> `validity_<part>_<variant>_<step>.json`):

| | cone | cone, IM5 | pyramid | elliptic cone |
|---|--:|--:|--:|--:|
| eq. plastic strain in the part: max / 95 % / median | 0.382 / 0.363 / 0.147 | 0.318 / 0.308 / 0.138 | 0.267 / 0.240 / 0.147 | 0.385 / 0.344 / 0.141 |
| rim band (target < 1 mm deep): max / mean | 0.095 / 0.062 | 0.087 / 0.064 | 0.116 / 0.074 | 0.134 / 0.084 |
| share of part elements beyond eps_p 0.19 (beyond tensile data) | 32 % | 32 % | 27 % | 46 % |
| net equivalent membrane strain, max | 0.125 | 0.129 | 0.089 | 0.149 |
| accumulated / net strain (median over strained cells) | 3.0 | 2.9 | 3.2 | 3.0 |
| strain ratio eps2/eps1 where eps1 > 0.05 (median) | 0.59 | 0.62 | 0.41 | 0.55 |
| thickness log strain max (min thickness) | 0.155 (0.857 mm) | - | 0.118 (0.888 mm) | 0.190 (0.827 mm) |
| transverse shear angle (column vs mid-surface normal) max / 95 % [rad] | 0.085 / 0.075 | 0.044 / 0.036 | 0.058 / 0.042 | 0.079 / 0.065 |

* **Strain range.** The benchmark reaches eps_p 0.16-0.59 (its README);
  27-46 % of the formed elements are beyond the uniform elongation of any
  tensile test (0.19-0.22 [2]). There the flow stress is an extrapolation on
  which published fits disagree by up to 30 %; its shape effect here is
  0.007-0.016 mm, growing with depth and wall angle. The rim band, where the
  dominant error sits, stays below eps_p 0.14, inside the tested range.
* **Strain path.** The accumulated plastic strain is ~3 times the net
  equivalent strain of the membrane state: the material under the tool is
  bent, unbent and sheared back and forth at every pass. Isotropic hardening
  on the accumulated strain is the least appropriate law for such paths;
  the Bauschinger variants move the shape by 0.013-0.034 mm and the
  springback by 6-31 %. This is the one material mechanism whose effect grows
  with model fidelity.
* **Strain state.** The walls of these small, shallow parts are stretched
  biaxially (median eps2/eps1 0.4-0.6), not in plane strain as on large SPIF
  walls - the part of the locus where r-based Hill48 is weakest for
  aluminium (sigma_b/sigma_0 0.95 vs measured 1.0-1.1), which is why
  `hill_sb110` is the largest effect.
* **Through-thickness shear** is modest (<= 0.085 rad on the standard
  element, half that on IM5: part of the standard element's shear is
  parasitic). The 3-D continuum carries it (no plane-stress shell
  assumption); DIC on SPIF shows through-thickness shear that FE
  under-predicts ([9], abstract) - a resolution question for the element
  lens, not a gap in the material law.
* **Strain rate** (`rate_temperature.json`, estimate): a wall point gains
  eps_p 0.13-0.19 per pass over a 2-3 mm contact length; at 1-6 m/min feed
  (a common ISF range; RoboForming's speeds are not public) that is
  0.7-10 /s, 3-4 decades above a quasi-static coupon. AA5754's
  rate sensitivity at room temperature is negative [3]; with |m| <= 0.005
  (an assumed bound) the flow stress changes by <= 3-5 %. By the scale
  invariance that moves the shape by << 0.01 mm (a uniform change: 0) and
  the springback by <= 5 %; the forming force moves by the same few per cent.
  Coupons should nevertheless be tested at 1e-2 - 1e-1 /s rather than 1e-3.
* **Temperature.** The plastic work of the most strained element is
  94 MJ/m^3, 35 K if released adiabatically (12-18 K per pass); in the
  0.02-0.18 s of contact heat diffuses 1-3 mm, so the real rise is lower,
  plus friction heating (inference). For AA5754 below ~100 C that is a few
  per cent of flow stress: same conclusion as the rate. Check it at feeds
  well above 6 m/min or with a rotating tool.
* **PLC and Lueders bands.** AA5754-O yields discontinuously and serrates
  [2], [3]; the model averages the serrations in the fitted curve and cannot
  localise bands - a surface-quality question, not a shape one.

## Files

| File | Content |
|---|---|
| `matlens.py` | the benchmark inputs of a part, variant decks, running, measuring (appends to `results.jsonl`) |
| `variants.py` | every variant and where its numbers come from |
| `run_batch.py` | runs `PART:VARIANT` jobs, at most two at a time, one thread each |
| `analyze.py` -> `sensitivity.csv`, `sensitivity.md` | shape, loaded-shape, springback, deviation and force changes per run |
| `validity.py` -> `validity_*.json` | plastic strain, membrane strain state, accumulated vs net strain, transverse shear, thickness |
| `lit_compare.py` -> `lit_curves.csv`, `lit_yield_ratios.csv` | library vs published flow curves; Hill48 stress ratios vs measured |
| `rate_temperature.py` -> `rate_temperature.json` | strain-rate and heating estimate |
| `error_budget.py` -> `error_budget.json` | largest effect per material input and part, quadrature sum, literature combination |
| `runs/<part>/<variant>/` | `deck.json`, `toolpath.csv`, `variant.json`, `run.json`, `sparlab_form.log`, `output/summary.json`, `output/step_3_release_elements.csv`; the heavy outputs (VTK, node tables, `surfaces.npz`) are git-ignored - re-run the deck (runs are deterministic) |
| `log_batch*.txt`, `queue*.sh` | the batches as run, with finish times |

Reproduce (in this directory; the audit build in `../../../build`):

```bash
python3 run_batch.py truncated_cone-s2026-0000:base truncated_cone-s2026-0000:hill_sb110   # any PART:VARIANT
python3 analyze.py && python3 error_budget.py
python3 validity.py truncated_cone-s2026-0000 base release
python3 lit_compare.py && python3 rate_temperature.py
```

Cost: 47 runs, 3.0 h of single-thread solver time (95-300 s per
standard-element run, 325-724 s per IM5 run), at most two at a time on the
shared 4-core machine (load 2-9), 18:12-20:07 UTC.

## Sources

Verified in the source text:

1. J. Coer, H. Laurent, M.C. Oliveira, P.-Y. Manach, L.F. Menezes,
   *Detailed experimental and numerical analysis of a cylindrical cup deep
   drawing: pros and cons of using solid-shell elements*, Int. J. Mater.
   Form. (2018); arXiv:1703.10126 - 1 mm AA5754-O: r-values, UTS, E, Voce and
   Hockett-Sherby fits, weak Bauschinger effect, split-ring opening, Hill48
   vs von Mises. https://arxiv.org/pdf/1703.10126
2. M.A. Iadicola, T. Foecke, S.W. Banovic, *Experimental observations of
   evolving yield loci in biaxially strained AA5754-O*, Int. J. Plasticity
   24 (2008) 2084-2101 - Table 2 (yield, UTS, uniform elongation, r, power
   law and Voce fits), biaxial and plane-strain yield loci, Yld2000-2d.
   https://www.nist.gov/document/ourijpyieldsurf5754pdf
3. A. Pandey, A.S. Khan, E.-Y. Kim, S.-H. Choi, T. Gnaeupel-Herold,
   *Experimental and numerical investigations of yield surface, texture,
   and deformation mechanisms in AA5754 over low to high temperatures and
   strain rates*, Int. J. Plasticity 41 (2013) 165-188 - negative SRS at
   room temperature, PLC, r vs strain rate, yield-point elongation.
   https://www.nist.gov/system/files/documents/lightweighting/1-s2-0-S0749641912001362-main.pdf
4. A.V. Shutov, *Efficient implicit integration for finite-strain
   viscoplasticity with a nested multiplicative split*, arXiv:1510.09163v2
   (2016), section 4.1.1, Table 1 - 5754-O parameters from Bauschinger
   simple-shear tests (data of [5] and Chaparro et al. 2008).
   https://arxiv.org/pdf/1510.09163
6. thyssenkrupp Materials Services, *Material Data Sheet EN AW-5754*
   (06.2017) - EN 485-2 limits for O/H111 sheet; E 70 500 MPa, G 26 500 MPa.
   https://ucpcdn.thyssenkrupp.com/_legacy/UCPthyssenkruppBAMXUK/assets.files/material-data-sheets/aluminium/5754.pdf
7. Nietsch et al., *Comparative study of elastic properties measurement
   techniques during plastic deformation of aluminum, magnesium, and
   titanium alloys: application to springback simulation*, Meccanica 60
   (2025) 55-72 - EN AW-6082-T6 chord modulus 72 -> 57.4-69 GPa; in 3-point
   bending the chord modulus did not improve on the constant modulus.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC11785604/

Cited from abstracts or search summaries only (not verified in full text):

5. H. Laurent, R. Greze, P.Y. Manach, S. Thuillier, *Influence of
   constitutive model in springback prediction using the split-ring test*,
   Int. J. Mech. Sci. 51 (2009) 233-245 (the cyclic shear data behind [4]).
8. Cleveland and Ghosh 2002, Int. J. Plasticity 18 (~10 % modulus loss for
   aluminium), as summarised by search results around [7]:
   https://link.springer.com/article/10.1007/s11012-024-01918-8
9. P. Eyckens et al., *Strain evolution in the single point incremental
   forming process: digital image correlation measurement and finite
   element prediction*, Int. J. Mater. Form. 4 (2011) 55-71 (through-thickness
   shear seen by DIC; bending-dominated in FE).
   https://link.springer.com/article/10.1007/s12289-010-0995-6
10. Machina Labs, *Capabilities* - two robot arms on rails, sheet up to
    6.35 mm, sub-millimetre precision claim; feed speeds not published.
    https://machinalabs.ai/capabilities
