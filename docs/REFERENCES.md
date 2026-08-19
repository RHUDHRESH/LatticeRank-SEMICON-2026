# References and augmentation-choice mapping

These sources justify the mechanisms represented by the generator; they do not
validate the numerical parameter ranges against a proprietary microscope,
process node, or fabrication line. The generator is a synthetic SEM-like
acquisition model.

## Coverage of implemented choices

| Implemented choice | Supporting references |
|---|---|
| 10:1 fields of view, DRAM/FinFET families, independent captures, stronger Search degradation, blur, rotation, and scale variation | [1], [2], [3] |
| Edge brightening, focus/PSF variation, roughness, and sparse image defects | [2], [3], [4] |
| Poisson electron-counting noise and signal-dependent secondary-emission noise | [2], [5], [6] |
| Gaussian detector/read noise, gain, offset, and finite detector response | [2], [5], [6] |
| Smooth charging fields, intensity instability, and occasional streak-like contrast | [2], [7], [8] |
| Translation, small rotation/scale mismatch, and magnification-dependent displacement | [1], [9], [10] |
| Row-dependent drift, scan shear, and jitter | [3], [9], [11] |
| Low-order spatial/radial distortion | [2], [10], [12] |
| Orthogonal DRAM-like word/bit-line arrays and repeated contacts | [1], [13], [14] |
| Parallel FinFET fins crossed by gate structures, with pitch/width variation | [1], [15], [16] |
| Correlation, multiple periodic hypotheses, and candidate ranking | [17], [18], [19] |

The SEM acquisition references support the existence of these effects, not the
specific random distributions in `driftforge/config.py`. Those ranges are
domain-randomization hypotheses and are intentionally described that way in the
documentation.

## Parameter-level traceability

This table is deliberately more specific than the mechanism summary above.
Every implemented parameter family is tied to its actual code range and to two
or three sources. A source can justify the **mechanism and order of magnitude**
without validating the exact endpoints. Where no public source validates an
endpoint for the sponsor's instrument, the final column says so explicitly.

| Implemented parameter | Range in the generator | Physical / challenge rationale | Sources | Status of the numerical range |
|---|---|---|---|---|
| Independent Reference/Search captures | Separate RNG streams and independently sampled acquisition specifications | The two fields are separate acquisitions, so shot noise, read noise, drift and photometric response must not be shared | [1], [5], [9] | Mechanism grounded; seed construction is an engineering choice |
| Edge brightening strength | `0.14–0.42` of normalized edge response | Secondary-electron edge contrast is represented phenomenologically before blur | [2], [3], [4] | Domain-randomization interval, not instrument calibration |
| Edge-response sigma | `1.0–4.5 nm` | Finite interaction/probe response prevents an infinitely sharp edge enhancement | [2], [3], [4] | Domain-randomization interval |
| PSF sigma, x/y | `2–7/8.5 nm`; OOD up to `11/13 nm` | Finite and anisotropic resolution plus defocus/astigmatism | [2], [3], [10] | Broad hypothesis spanning ordinary and stress captures |
| Reference dose | Log-uniform `900–4200` counts at full normalized signal | High-magnification Reference is the cleaner acquisition | [1], [5], [6] | Relative dose model; not electrons-per-probe-current calibration |
| Search dose | Log-uniform `140–700`; hard/OOD lower bound `70` | The specification makes Search noisier/lower dose than Reference | [1], [5], [6] | Relative dose model with explicit stress tail |
| Read-noise sigma | Reference `0.003–0.016`; Search `0.010–0.032`, hard to `0.045` | Adds detector/electronic noise independently of Poisson counting noise | [2], [5], [6] | Normalized-intensity hypothesis |
| Rotation | Reference `±2.2°` (OOD `±3.3°`); Search `±0.35°` (OOD `±0.77°`) | Independent stage/orientation error at the two magnifications | [1], [9], [10] | Challenge-driven domain-randomization interval |
| Scale mismatch | Reference `0.965–1.035` (OOD `0.93–1.07`); Search `0.992–1.008` (OOD `0.975–1.025`) | Magnification calibration and acquisition mismatch | [1], [10], [11] | Challenge-driven interval, not a calibrated tool tolerance |
| Translation | Reference `±1 px`; Search `±3 px` | Independent stage/navigation offset around the nominal scene geometry | [1], [9], [10] | Engineering stress range |
| Scan shear | Reference `±0.25 px`; Search `±2.2 px`, hard `±4.5 px` | Row-time drift produces a non-rigid horizontal displacement | [3], [9], [11] | Phenomenological endpoint |
| Row jitter | Reference `0–0.15 px`; Search `0.05–0.75 px`, hard to `1.7 px` standard deviation | Scan/vibration noise varies row by row | [3], [9], [11] | Phenomenological endpoint |
| Radial distortion coefficient | `±0.006`; OOD `±0.018` | Bounded low-order spatial distortion covers imperfect SEM calibration | [2], [10], [12] | Dimensionless simulation coefficient, not a lens measurement |
| Gain and offset | Gain `0.82–1.20`; offset `−0.08–0.08` | Separate detector response and brightness setup for each capture | [2], [5], [6] | Normalized-intensity hypothesis |
| Gamma | `0.82–1.22`; OOD `0.65–1.45` | Nonlinear display/detector response and contrast setup | [2], [5], [6] | Synthetic photometric stress parameter |
| Vignetting | `0–0.20`; hard `0–0.34` radial strength | Low-frequency collection/illumination nonuniformity | [2], [3], [6] | Synthetic stress range; not instrument-calibrated |
| Charging field | `0–0.10`; hard `0–0.20` normalized amplitude | Smooth spatially varying secondary-electron contrast from charging | [2], [7], [8] | Phenomenological amplitude |
| Charging streak count | `0–2`; hard `0–5`, each `1–4 px` half-width | Rare scan-line/band contrast excursions accompany unstable charging | [3], [7], [8] | Discrete stress hypothesis |
| Hot/impulse pixels | `0–1.5×10⁻⁴`; hard to `7×10⁻⁴` per pixel | Sparse detector/readout outliers should not dominate registration | [2], [5], [6] | Conservative synthetic outlier rate |
| DRAM pitch x/y | Dense `48–72/72–110 nm`; open `75–120/100–170 nm` | Distinct periodic word-/bit-line families and repeated cells | [1], [13], [14] | Geometry-family hypothesis, not a process-node claim |
| DRAM line widths | Dense `15–28/20–38 nm`; open `22–42/28–55 nm` | Width varies independently from pitch in the procedural top view | [2], [13], [14] | Geometry-family hypothesis |
| DRAM contact ellipse | Dense `18–34 nm`; open `25–48 nm` | Repeated contact/via features provide the intersection motif | [1], [13], [14] | Geometry-family hypothesis; checkerboard occupancy is documented below |
| FinFET pitch x/y | Dense `28–50/48–78 nm`; open `50–90/75–130 nm` | Dense parallel fins are crossed by the gate direction | [1], [15], [16] | Geometry-family hypothesis, not a node reconstruction |
| FinFET line widths | Dense `7–18/16–32 nm`; open `12–28/22–45 nm` | Fins are narrower than gate structures and vary across families | [2], [15], [16] | Geometry-family hypothesis |
| Line-position jitter | `0.25–1.8 nm`; zero for exact wallpaper | Local placement variation breaks ideal periodicity | [2], [3], [16] | Synthetic morphology range |
| Width variation | `2–12%`; zero for exact wallpaper | Local width variation represents non-ideal fabrication morphology | [2], [3], [16] | Synthetic morphology range |
| Line-edge roughness magnitude | `0.4–2.0 nm`, ×1.5 in hard/OOD; zero for exact wallpaper | Multi-frequency edge displacement represents roughness without a process simulator | [2], [3], [16] | Domain-randomization hypothesis |
| Roughness wavelength | `90–420 nm` with a second component at `0.43×` | Correlated rather than pixel-independent edge displacement | [2], [3], [16] | Procedural correlation-length hypothesis |
| Defect density and dimensions | Density `0.4–1.4`, ×1.4 hard/OOD; about 10 draws per density unit; radii `18–85 nm` | Missing material, residue, bridges and scratches provide nonperiodic identity evidence | [2], [3], [16] | Synthetic coverage range, not a defect-yield prediction |
| Array-mat period / routing strip | Period `2200–3600 nm`; strip width `220–480 nm`; routing widths `24/28 nm` | Larger-scale array context prevents every normal scene from being infinite wallpaper | [1], [13], [14] | Procedural context model |
| Exact-wallpaper profile | Zero defects, roughness, line jitter and width variation; no routing strips; phase-equivalent target nearest Search centre | Explicitly exercises the challenge's periodic tie rule | [1], [17], [18] | Test construction, not a frequency claim about real wafers |

The checkerboard contact occupancy in `driftforge/scene.py` models a
centred-rectangular repeated sub-lattice and is **not** presented as the only
valid DRAM contact layout. The challenge's simpler “contact at intersections”
description [1] and device examples [13], [14] motivate the contact family;
future sponsor-aligned presets should include full-occupancy and checkerboard
variants as separate configurations.

## Challenge specification

1. i4C, “Drift-Sense: AI-Powered Navigation-Error Recovery for Wafer
   Inspection Tools,” *SEMICON India Hackathon 2026* (2026),
   [official problem and repository requirements](https://i4c.in/hackathon-2026/).

   **Choice used here:** the required 1,000 × 1,000 Reference/Search pair,
   approximately 10× field-of-view difference, DRAM/FinFET styles, independent
   sensor noise, edge brightening, blur, rotation, scale variation, noisier
   Search capture, recorded ground truth, and 30+ pair self-evaluation.

## SEM image formation, edge response, and noise

2. J. I. Goldstein, D. E. Newbury, J. R. Michael, N. W. M. Ritchie,
   J. H. J. Scott, D. C. Joy et al., *Scanning Electron Microscopy and X-Ray
   Microanalysis*, 4th ed., Springer (2018),
   [doi:10.1007/978-1-4939-6676-9](https://link.springer.com/book/10.1007/978-1-4939-6676-9).

   **Choice used here:** a general physical basis for secondary-electron image
   formation, finite probe size, detector response, charging, contrast,
   electron statistics, and instrumental artifacts.

3. P. Cizmar, A. Vladar, B. Ming, and M. T. Postek, “Simulated SEM Images for
   Resolution Measurement,” *Scanning* 30(5) (2008),
   [NIST publication page](https://www.nist.gov/publications/simulated-sem-images-resolution-measurement).

   **Choice used here:** phenomenological edge effect, substrate roughness,
   focus/PSF variation, drift/vibration, noise, and randomized geometry in a
   synthetic SEM test image.

4. G. W. Bailey, “Conditions Required for Detection of Specimen-Specific SE-I
   Secondary Electrons in an Analytical SEM,” *Journal of Microscopy* 154
   (1989),
   [doi:10.1111/j.1365-2818.1989.tb00580.x](https://pubmed.ncbi.nlm.nih.gov/2746638/).

   **Choice used here:** bright edge contrast from localized secondary-electron
   collection. DriftForge uses a simple gradient-based proxy rather than an
   electron-transport calculation.

5. K. S. Sim, J. T. L. Thong, and J. C. H. Phang, “Effect of Shot Noise and
   Secondary Emission Noise in Scanning Electron Microscope Images,”
   *Scanning* 26(1), 36–40 (2004),
   [doi:10.1002/sca.4950260106](https://doi.org/10.1002/sca.4950260106).

   **Choice used here:** Poisson-distributed primary-electron fluctuations,
   secondary-emission noise, and detector-noise context.

6. F. Timischl, M. Date, and S. Nemoto, “A Statistical Model of Signal–Noise in
   Scanning Electron Microscopy,” *Scanning* 34(3), 137–144 (2012),
   [doi:10.1002/sca.20282](https://doi.org/10.1002/sca.20282).

   **Choice used here:** separate signal-dependent counting statistics and
   detector/read contributions. The code approximates them with Poisson and
   Gaussian terms.

## Charging, drift, and geometric distortion

7. M. T. Postek and A. E. Vladar, “Does Your SEM Really Tell the Truth?—How
   Would You Know? Part 4: Charging and its Mitigation,” *Proceedings of SPIE*
   9636, 963605 (2015),
   [doi:10.1117/12.2195344](https://www.nist.gov/publications/does-your-sem-really-tell-truth-how-would-you-know-part-4-charging-and-its-mitigation).

   **Choice used here:** spatially and temporally varying image intensity from
   specimen charging. The generator uses smooth fields and rare streaks, not a
   charge-transport solver.

8. H.-B. Zhang, R.-J. Feng, and K. Ura, “Utilizing the Charging Effect in
   Scanning Electron Microscopy,” *Science Progress* 87(4), 249–268 (2004),
   [doi:10.3184/003685004783238490](https://doi.org/10.3184/003685004783238490).

   **Choice used here:** charging contrast and secondary-electron
   redistribution as motivation for low-frequency intensity variation.

9. P. Cizmar, A. Vladar, and M. T. Postek, “Real-Time Image Composition with
   Correction of Drift Distortion,” *Microscopy and Microanalysis* 17 (2011),
   [NIST publication page](https://www.nist.gov/publications/real-time-image-composition-correction-drift-distortion).

   **Choice used here:** scan-time drift as blur and non-rigid displacement,
   motivating independent row-dependent shear/jitter in each acquisition.

10. A. C. Malti, S. Dembélé, N. Piat, C. Arnoult, and N. Marturi, “Toward Fast
    Calibration of Global Drift in Scanning Electron Microscopes with Respect
    to Time and Magnification,” *International Journal of Optomechatronics*
    6(1), 1–16 (2012),
    [doi:10.1080/15599612.2012.663462](https://doi.org/10.1080/15599612.2012.663462).

    **Choice used here:** magnification- and time-dependent image displacement,
    motivating independent translation and small scale mismatch.

11. W. F. Maune, “Photogrammetric Self-Calibration of Scanning Electron
    Microscopes,” *Photogrammetric Engineering and Remote Sensing* 42,
    1161–1172 (1976).

    **Choice used here:** SEM calibration must account for affine and
    non-linear geometric effects rather than treating pixel coordinates as an
    exact Euclidean camera.

12. Q. Zhang, H. Xie, W. Shi, and B. Fan, “A Novel Sampling Moiré Method and
    Its Application for Distortion Calibration in Scanning Electron
    Microscope,” *Optics and Lasers in Engineering* 128, 105990 (2020),
    [doi:10.1016/j.optlaseng.2019.105990](https://doi.org/10.1016/j.optlaseng.2019.105990).

    **Choice used here:** spatial distortion varies with scale and image
    location, motivating a bounded low-order radial distortion term.

## Semiconductor geometry priors

13. T. Takahashi et al., “A Multigigabit DRAM Technology With 6F² Open-Bitline
    Cell, Distributed Overdriven Sensing, and Stacked-Flash Fuse,” *IEEE
    Journal of Solid-State Circuits* 36(11), 1721–1727 (2001),
    [doi:10.1109/4.962294](https://doi.org/10.1109/4.962294).

    **Choice used here:** compact periodic DRAM arrays with distinct bit-line
    and word-line pitches, contacts, and repeated cell structure.

14. J.-S. Kim, Y.-S. Choi, H.-J. Yoo, and K.-S. Seo, “A Low-Noise Folded
    Bit-Line Sensing Architecture for Multigigabit DRAM with Ultrahigh-Density
    6F² Cell,” *IEEE Journal of Solid-State Circuits* 33(7), 1096–1102 (1998),
    [doi:10.1109/4.701271](https://doi.org/10.1109/4.701271).

    **Choice used here:** repeated 6F² cell geometry and coupled word/bit-line
    organization. The rendered top view is illustrative and not a mask-level
    reconstruction.

15. X. Huang et al., “Sub 50-nm FinFET: PMOS,” *IEEE International Electron
    Devices Meeting*, 67–70 (1999),
    [doi:10.1109/IEDM.1999.823848](https://doi.org/10.1109/IEDM.1999.823848).

    **Choice used here:** thin parallel fins with a self-aligned gate crossing
    the fin direction.

16. H. Trombini et al., “Unraveling Structural and Compositional Information in
    3D FinFET Electronic Devices,” *Scientific Reports* 9, 11629 (2019),
    [doi:10.1038/s41598-019-48117-0](https://doi.org/10.1038/s41598-019-48117-0).

    **Choice used here:** non-ideal local morphology, represented only by
    procedural line-edge roughness, width variation, and sparse defects.

## Registration and ranking

17. J. P. Lewis, “Fast Normalized Cross-Correlation,” expanded author version
    of “Fast Template Matching,” *Vision Interface*, 120–123 (1995),
    [author-hosted paper](https://www.scribblethink.org/Work/nvisionInterface/nip.pdf).

    **Choice used here:** efficient zero-mean normalized cross-correlation with
    local window statistics.

18. P. Doubek, J. Matas, M. Perdoch, and O. Chum, “Image Matching and
    Retrieval by Repetitive Patterns,” *ICPR* 2010, 3195–3198,
    [doi:10.1109/ICPR.2010.782](https://doi.org/10.1109/ICPR.2010.782).

    **Choice used here:** retain multiple periodic hypotheses and treat lattice
    translation as genuine ambiguity.

19. F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” *Journal
    of Machine Learning Research* 12, 2825–2830 (2011),
    [JMLR article](https://jmlr.org/papers/v12/pedregosa11a.html).

    **Choice used here:** the packaged `HistGradientBoostingClassifier`
    candidate ranker.

20. P. Virtanen et al., “SciPy 1.0: Fundamental Algorithms for Scientific
    Computing in Python,” *Nature Methods* 17, 261–272 (2020),
    [doi:10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).

    **Choice used here:** Gaussian/Sobel filters, interpolation, affine and
    coordinate warps, morphology, and FFT convolution.

21. E. B. Wilson, “Probable Inference, the Law of Succession, and Statistical
    Inference,” *Journal of the American Statistical Association* 22(158),
    209–212 (1927),
    [doi:10.1080/01621459.1927.10502953](https://doi.org/10.1080/01621459.1927.10502953).

    **Choice used here:** Wilson score intervals for binomial candidate-recall
    reporting.

## Deliberate exclusions

No citation is presented as proof that the generator is physically exact.
Methods and structures not used by the final system are omitted rather than
cited as if they were implemented.
