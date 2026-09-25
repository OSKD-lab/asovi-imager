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

## The cortical atlas — fetched and built, never redistributed

**Nothing here carries Allen Institute data.** The top-view atlas the annotation
stage registers to is derived from the Allen Mouse Brain Common Coordinate
Framework, so `asovi-atlas --download` builds it on the machine that uses it,
into `~/.asovi/atlas/`. An earlier version of this package shipped a prebuilt
copy inside the wheel; that was withdrawn, because the MIT `LICENSE` this code
carries is not the licence that data is under, and a wheel saying MIT over the
whole of its contents would have told users something untrue.

**What the build downloads**

| | |
| --- | --- |
| CCF volumes | figshare [25365829](https://doi.org/10.6084/m9.figshare.25365829), *Modified Allen CCF 2017 for cortex-lab/allenCCF* (Nick Steinmetz) — **CC BY 4.0**, ~4.8 GB |
| Structure tree | `structure_tree_safe_2017.csv` from [cortex-lab/allenCCF](https://github.com/cortex-lab/allenCCF) — it is not in the figshare article, and it cannot be replaced by a fresh Allen API query: the volume is stored *by index* into this file's row order |

**Underlying data** — Allen Mouse Brain Common Coordinate Framework (CCFv3):
Wang Q, et al. *The Allen Mouse Brain Common Coordinate Framework: A 3D Reference
Atlas.* Cell 181(4):936–953 (2020). <https://doi.org/10.1016/j.cell.2020.04.007> ·
<https://atlas.brain-map.org/>. Allen Institute Content is subject to the
[Allen Institute Terms of Use](https://alleninstitute.org/terms-of-use/), which
permit research and other non-commercial use, and which travel with derived
work. **Cite the paper above if you publish results that used the atlas.**
