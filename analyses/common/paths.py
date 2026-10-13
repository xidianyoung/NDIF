"""Locations of the source data, the intermediate files and the outputs of the analysis scripts.

Every location is read from an environment variable and otherwise defaults to a path relative to the repository
root.  The source datasets are obtained from their providers (top-level README, *Data*) and are not part of the
repository; the intermediate files are rebuilt by the scripts that write them.

==========================  ======================================  ===============================================
variable                    default (relative to the repository)    content
==========================  ======================================  ===============================================
``NDIF_DS003838_RAW``       ``data/ds003838``                       OpenNeuro ds003838 (BIDS)
``NDIF_DS003838_DERIV``     ``data/ds003838_preproc``               preprocessed ds003838 recordings
``NDIF_EEGMMIDB_RAW``       ``data/eegmmidb``                       PhysioNet EEG Motor Movement/Imagery
``NDIF_HCP_FMRI_PARCELS``   ``data/hcp_wm_parcels``                 parcel time series of the HCP working-memory fMRI
``NDIF_HCP_FMRI_LABELS``    ``data/hcp_wm_labels``                  block onsets of the HCP working-memory runs
``NDIF_HCP_MEG_RAW``        ``data/hcp_meg``                        HCP MEG working memory (``tmegpreproc``)
``NDIF_COGBCI_RAW``         ``COG-BCI``                             COG-BCI release (Zenodo)
``NDIF_SLEEPEDF_RAW``       ``data/sleep-edf``                      Sleep-EDF Expanded (cassette and telemetry)
``NDIF_WORK``               ``analyses/_work``                      intermediate caches written by the scripts
``NDIF_OUT``                (unset)                                 if set, outputs are written below this
                                                                    directory instead of their stored locations
==========================  ======================================  ===============================================

An intermediate cache ``name`` is ``NDIF_WORK/<name>`` unless ``NDIF_<NAME>_CACHE`` names another directory
(:func:`cache`).  The per-trial HCP MEG state cache and the COG-BCI band-power cache keep the locations that the
drivers in ``analyses/primary_configuration`` and ``analyses/workload_certificates`` read (``NDIF_MEG_CACHE`` and
``NDIF_COGBCI_CACHE``).
"""

from __future__ import annotations

import os

#: The repository root.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Default locations of the source data, relative to the repository root.
DATA_DEFAULTS = {
    "NDIF_DS003838_RAW": os.path.join("data", "ds003838"),
    "NDIF_DS003838_DERIV": os.path.join("data", "ds003838_preproc"),
    "NDIF_EEGMMIDB_RAW": os.path.join("data", "eegmmidb"),
    "NDIF_HCP_FMRI_PARCELS": os.path.join("data", "hcp_wm_parcels"),
    "NDIF_HCP_FMRI_LABELS": os.path.join("data", "hcp_wm_labels"),
    "NDIF_HCP_MEG_RAW": os.path.join("data", "hcp_meg"),
    "NDIF_COGBCI_RAW": "COG-BCI",
    "NDIF_SLEEPEDF_RAW": os.path.join("data", "sleep-edf"),
    "NDIF_WORK": os.path.join("analyses", "_work"),
}

#: Caches whose default location is fixed by the drivers that read them.
CACHE_DEFAULTS = {
    "meg": os.path.join("analyses", "primary_configuration", "meg_feature_cache"),
    "cogbci": os.path.join("analyses", "workload_certificates", "features", "cog_feature_cache"),
}


def repo(*parts: str) -> str:
    """A path inside the repository."""
    return os.path.join(REPO, *parts)


def path(name: str) -> str:
    """The location named by the environment variable ``name``, else its default (:data:`DATA_DEFAULTS`)."""
    value = os.environ.get(name)
    if value:
        return os.path.abspath(value)
    if name not in DATA_DEFAULTS:
        raise KeyError(f"unknown location {name!r}; known: {sorted(DATA_DEFAULTS)}")
    return repo(DATA_DEFAULTS[name])


def work(*parts: str) -> str:
    """A path in the directory of intermediate files (``NDIF_WORK``)."""
    return os.path.join(path("NDIF_WORK"), *parts)


def cache(name: str) -> str:
    """The directory of the intermediate cache ``name``.

    ``NDIF_<NAME>_CACHE`` if set; otherwise the fixed default of :data:`CACHE_DEFAULTS`, or ``NDIF_WORK/<name>``.
    """
    value = os.environ.get(f"NDIF_{name.upper()}_CACHE")
    if value:
        return os.path.abspath(value)
    if name in CACHE_DEFAULTS:
        return repo(CACHE_DEFAULTS[name])
    return work(name)


def out(*parts: str) -> str:
    """The location of an output, given relative to the repository root.

    Outputs are written to their stored locations in the repository, which are the files that the README tables
    and the Source Data refer to.  With ``NDIF_OUT`` set, the same relative path is used below that directory, so
    that a run can be compared with the stored files without replacing them.
    """
    base = os.environ.get("NDIF_OUT")
    target = os.path.join(os.path.abspath(base), *parts) if base else repo(*parts)
    os.makedirs(os.path.dirname(target) if os.path.splitext(target)[1] else target, exist_ok=True)
    return target
