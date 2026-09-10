# Third-party notices

`LICENSE` (MIT, OSKD-lab) covers the code written for this project. Some of that
code is *derived from* work by others, which keeps its own terms; this file records
what, from where, and under what licence.

This repository is Python-only. The MATLAB originals below are **not distributed
here** — only the Python implementations derived from them. They live in the lab's
internal repository.

---

## Guizar-Sicairos DFT registration — ported to Python

**Derived work in this repository** — `src/asvimg/registration/{numpy,fast,torch}.py`,
which implement the same algorithm. `numpy.py` is a close port of the original
MATLAB, keeping its variable names and branch structure.

**Original** — `dftregistration.m` by Manuel Guizar-Sicairos, **not included here**.
Its header reads:

> Manuel Guizar - Dec 13, 2007
>
> Portions of this code were taken from code written by Ann M. Kowalczyk
> and James R. Fienup.
> J.R. Fienup and A.M. Kowalczyk, "Phase retrieval for a complex-valued
> object by using a low-resolution image," J. Opt. Soc. Am. A 7, 450-458 (1990).

**Source** — MATLAB Central File Exchange submission 18401, *Efficient subpixel
image registration by cross-correlation*:
<https://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation>

**License** — BSD 3-Clause, added to the submission in version 1.1.0.0
(16 June 2016). It is not in the `.m` file header: it ships as `license.txt` in
the File Exchange download, and is reproduced verbatim:

```text
Copyright (c) 2016, Manuel Guizar Sicairos, James R. Fienup, University of Rochester
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the distribution
    * Neither the name of the University of Rochester nor the names
      of its contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

**Citation for the algorithm** — Manuel Guizar-Sicairos, Samuel T. Thurman, and
James R. Fienup, "Efficient subpixel image registration algorithms," *Optics
Letters* 33(2):156–158 (2008). <https://doi.org/10.1364/OL.33.000156>

---

## Whitney rolling-percentile baseline — reimplemented

**Derived work in this repository** — `src/asvimg/wfci.py`
(`_rolling_percentile_baseline`, reached through the `baseline_percentile*`
configuration fields). It follows the method of the original but is an independent
implementation built on SciPy's `percentile_filter` / `uniform_filter1d` /
`interp1d`, not a transcription.

**Original** — `baselinePercentileFilter.m`, **not included here**. Its header
reads:

> by David Whitney (david.whitney@mpfi.org), Max Planck Florida Institute, 2016.

**License** — none stated in the original, and no accompanying licence was found.

---

## `src/asvimg/data/wfciAnnotationData.mat`

**Origin** — the lab's own top-view cortical parcellation, made in 2024. It is not
vendored third-party source, and the MIT `LICENSE` covers it; it is listed here
because it is *derived from* the Allen Mouse Brain Common Coordinate Framework and
now ships inside the wheel.

**Underlying data** — Allen Mouse Brain Common Coordinate Framework (CCFv3):
Wang Q, et al. *The Allen Mouse Brain Common Coordinate Framework: A 3D Reference
Atlas.* Cell 181(4):936–953 (2020). <https://doi.org/10.1016/j.cell.2020.04.007> ·
<https://atlas.brain-map.org/>. Allen Institute data are subject to the Allen
Institute Terms of Use.

**Known limitation** — its region *names* are wrong (scrambled ID → acronym
mapping; the `_R` / `_L` suffixes are fictional because the ID map is bilateral).
`ACCFv3.get_mask()` and `ACCFv3.region_names` raise rather than answer from them.
Geometry, boundaries, midline landmarks and point ROIs are correct.
