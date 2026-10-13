# Notice: scope of the license, data sources and dependencies

Copyright (c) 2026 Tianyi Luo, Yangyang Liu, Xidian University

## Scope of the MIT License

The MIT License in [`LICENSE`](LICENSE) covers the source code, analysis
scripts, configuration files, tests and documentation of the NDIF
repository written by the copyright holders above.

It does **not** cover:

* **Third-party datasets** (listed below). They remain under their original
terms. This repository does not include the raw recordings of these datasets.
Any derived file in this repository that was computed from one of them stays
subject to the terms of the dataset it came from.
* **The manuscript and its journal-submission figures.** They are not
distributed with this repository, and the MIT License does not grant any
rights to them.

## Datasets used in the accompanying study

The study re-analyzes previously published datasets. Obtain them from the
original sources under their own terms.

|Dataset|Source|Terms|
|-|-|-|
|Digit-span EEG, OpenNeuro ds003838|https://doi.org/10.18112/openneuro.ds003838.v1.0.2|CC0|
|COG-BCI database|https://doi.org/10.5281/zenodo.6874128|See the license on the Zenodo record|
|Human Connectome Project (working-memory fMRI and MEG)|https://db.humanconnectome.org|WU-Minn HCP Consortium Open Access Data Use Terms. Registration and acceptance are required. HCP data may be redistributed only under these same terms.|
|Sleep-EDF Expanded|https://doi.org/10.13026/C2X676|Open Data Commons Attribution License v1.0|
|EEG Motor Movement/Imagery Dataset (eegmmidb)|https://doi.org/10.13026/C28G6P|Open Data Commons Attribution License v1.0|

The `nile`, `sunspots`, `co2` and `macrodata` series used as non-neural
examples are loaded from the `statsmodels` package at run time and are not
redistributed here.

### Required acknowledgments

**Human Connectome Project.** Data were provided in part by the Human
Connectome Project, WU-Minn Consortium (Principal Investigators: David Van
Essen and Kamil Ugurbil; 1U54MH091657) funded by the 16 NIH Institutes and
Centers that support the NIH Blueprint for Neuroscience Research; and by the
McDonnell Center for Systems Neuroscience at Washington University.

**PhysioNet datasets (Sleep-EDF Expanded, eegmmidb).** Please cite the
original publications and the PhysioNet platform paper:

* Kemp, B., Zwinderman, A. H., Tuk, B., Kamphuisen, H. A. C. \& Oberyé, J. J. L.
Analysis of a sleep-dependent neuronal feedback loop: the slow-wave
microcontinuity of the EEG. *IEEE Trans. Biomed. Eng.* 47, 1185–1194 (2000).
* Schalk, G., McFarland, D. J., Hinterberger, T., Birbaumer, N. \& Wolpaw, J. R.
BCI2000: a general-purpose brain-computer interface (BCI) system.
*IEEE Trans. Biomed. Eng.* 51, 1034–1043 (2004).
* Pollard, T. et al. PhysioNet as a global platform for biomedical research.
*Nat. Health* 1, 792–795 (2026).

## Third-party code

`analyses/landscape_readouts/elapy_functions.py` contains functions copied from
elapy (https://github.com/okumakito/elapy), a Python implementation of the
Energy Landscape Analysis Toolbox (T. Ezaki et al., *Philos. Trans. A Math.
Phys. Eng. Sci.* 375, 20160287 (2017)). These functions remain under the Apache
License, Version 2.0; the license text is in
`analyses/landscape_readouts/LICENSE_elapy.txt`. The MIT License of this
repository does not apply to them.

## Software dependencies

The `ndif` package depends on NumPy and SciPy. The analysis scripts also
use statsmodels, pandas, scikit-learn, Matplotlib, MNE-Python, h5py,
xlrd, networkx and PyTorch (`requirements.txt`, `requirements-analyses.txt`),
and the tests use pytest. These packages are installed separately, are not
bundled in this repository, and are distributed under their own licenses.

