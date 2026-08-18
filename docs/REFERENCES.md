# References and model-choice mapping

Links and bibliographic details were checked against publisher, institutional,
or author-hosted public pages on 18 August 2026. These sources motivate
mechanisms and algorithms; they do not validate DriftForge's numerical
parameter ranges against a proprietary fabrication line.

## SEM-like acquisition

1. P. Cizmar, A. Vladar, B. Ming, and M. T. Postek, “Simulated SEM Images for
   Resolution Measurement,” *Scanning* (2008).
   [NIST publication page](https://www.nist.gov/publications/simulated-sem-images-resolution-measurement).

   **Choice used here:** independent phenomenological edge enhancement,
   roughness, focus/PSF variation, drift-like distortion, vibration/jitter, and
   noise. The source explicitly describes deterministic simulated images with
   edge effect, substrate roughness, focus, drift/vibration, and noise.

2. P. Cizmar, A. Vladar, and M. T. Postek, “Real-Time Image Composition with
   Correction of Drift Distortion,” *Microscopy and Microanalysis* 17 (2011),
   [doi:10.1017/S1431927610094250](https://doi.org/10.1017/S1431927610094250).

   **Choice used here:** row-dependent Search displacement (`scan_shear_px`
   plus independently sampled row jitter) and cross-correlation as a relevant
   registration primitive. DriftForge does not claim to reproduce this
   paper's correction system or exact temporal drift behaviour.

## Semiconductor geometry priors

3. T. Takahashi et al., “A Multigigabit DRAM Technology With 6F² Open-Bitline
   Cell, Distributed Overdriven Sensing, and Stacked-Flash Fuse,” *IEEE Journal
   of Solid-State Circuits* 36(11), 1721–1727 (2001),
   [doi:10.1109/4.962294](https://doi.org/10.1109/4.962294).

   **Choice used here:** an orthogonal, periodic DRAM-like family with distinct
   bit-line and word-line pitches and repeated contacts. DriftForge varies
   these pitches broadly and does not assert a literal 6F² mask reconstruction.

4. C. Hu, “FinFET 3D Transistor & the Concept Behind It,” UC Berkeley seminar
   (2011), [public course slides](https://microlab.berkeley.edu/text/seminars/slides/2011-8_FinFET_and_the_Concept_Behind_It.pdf).

   **Choice used here:** parallel thin fins crossed by gate structures, with
   fin width and pitch as separate parameters.

5. H. Trombini et al., “Unraveling Structural and Compositional Information in
   3D FinFET Electronic Devices,” *Scientific Reports* 9, 11629 (2019),
   [doi:10.1038/s41598-019-48117-0](https://doi.org/10.1038/s41598-019-48117-0).

   **Choice used here:** non-ideal local morphology, represented only as
   procedural line-edge roughness, width variation, and sparse defects. The
   2D generator is not electron tomography and does not model materials.

## Registration and repetitive structure

6. J. P. Lewis, “Fast Normalized Cross-Correlation,” expanded author version
   of “Fast Template Matching,” *Vision Interface*, 120–123 (1995),
   [author-hosted PDF](https://www.scribblethink.org/Work/nvisionInterface/nip.pdf).

   **Choice used here:** zero-mean normalized cross-correlation response maps
   computed efficiently with FFT convolution and local window statistics.

7. P. Doubek, J. Matas, M. Perdoch, and O. Chum, “Image Matching and Retrieval
   by Repetitive Patterns,” *ICPR* 2010, 3195–3198,
   [doi:10.1109/ICPR.2010.782](https://doi.org/10.1109/ICPR.2010.782).

   **Choice used here:** treat lattice translation as a real ambiguity rather
   than collapsing every repeat into one answer. LatticeRank keeps multiple
   local maxima, builds spatially resolved descriptors, and applies the centre
   rule only to an evidential equivalence set. It does not implement the
   paper's retrieval descriptor.

## Software and statistical reporting

8. F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” *Journal
   of Machine Learning Research* 12, 2825–2830 (2011),
   [JMLR](https://jmlr.org/papers/v12/pedregosa11a.html).

   **Choice used here:** the packaged
   `HistGradientBoostingClassifier` candidate ranker. The exact estimator
   contract is documented by
   [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html).

9. P. Virtanen et al., “SciPy 1.0: Fundamental Algorithms for Scientific
   Computing in Python,” *Nature Methods* 17, 261–272 (2020),
   [doi:10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).

   **Choice used here:** Gaussian/Sobel filters, interpolation, affine and
   coordinate warps, morphology, and FFT convolution from `scipy.ndimage` and
   `scipy.signal`.

10. E. B. Wilson, “Probable Inference, the Law of Succession, and Statistical
    Inference,” *Journal of the American Statistical Association* 22(158),
    209–212 (1927),
    [doi:10.1080/01621459.1927.10502953](https://doi.org/10.1080/01621459.1927.10502953).

    **Choice used here:** 95% Wilson score intervals for binomial candidate
    recall in the curated diagnostic artifact.

## Deliberate exclusions

No source is cited as proof that the generator is physically exact. Published
FinFET/DRAM dimensions are used only to establish plausible geometry families;
the code's ranges remain domain-randomization hypotheses. Registration methods
not present in the final pipeline are omitted rather than cited as if they were
implemented.
