"""
uncertainty_core.py
===================

Core code for the paper
    "A Framework for Characterising Dataset-Level Uncertainty in Experimental
     Condition-Monitoring Datasets: A Rolling-Element Bearing Case Study"

The module is organised into small, independently testable classes:

    Config              all experiment and processing settings in one place
    RawDataLoader       reads one recording from the raw Excel files
    SESProcessor        Hilbert squared envelope spectrum and band-limited peak picking
    FeatureExtractor    full-signal and 1 s segmented (A1, A2) features for every recording
    FeatureStore        saves / loads the feature tables (CSV) and returns point sets
    Geometry            centroid, hull dispersion, covariance dispersion and bias
    MetricTable         D_hull, D_Sigma, B and rho_Sigma for every assembly state
    Inference           recording-level bootstrap, jackknife and exact permutation test
    ScreeningSweep      sensitivity of D_hull to the outlier-screening threshold
    VarianceComponents  ISO 5725 style nested variance components (s_r, s_R, ICC)
    TaskCheck           separation Delta between health states and range delta within each
    SampleSizeCheck     full-signal vs segmented dispersion (hull vs covariance, subsampling)
    SyntheticCheck      behaviour of rho_Sigma and the permutation test on synthetic data
    Plotter             all data figures of the paper and the supplementary material

Notation (paper):
    feature vector  x_k = (A1_k, A2_k), the SES peak amplitudes at 1x and 2x BPFO
    assembly state  one intervention case (C1 to C4) within one test session (T1 to T4)
    centroid        mu = mean of all points in a set
    dispersion      D_hull = sqrt(Area(convex hull));  D_Sigma = det(Sigma_hat)^(1/4)
    bias            B(Cj) = || mu(Cj) - mu(C1) ||_2  (centroid shift)
    effect size     rho_Sigma(Cj) = B(Cj) / D_Sigma,pool,  Sigma_pool = (Sigma_C1 + Sigma_Cj) / 2
    task check      Delta = smallest distance between healthy and damaged centroids,
                    delta = largest distance between two centroids of the same health state
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.fft import fft as _fft
from scipy.signal import hilbert as _hilbert
from scipy.spatial import ConvexHull
from scipy.spatial import QhullError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HEALTH_STATES: Tuple[str, ...] = ("healthy", "damaged")
TESTS: Tuple[str, ...] = ("T1", "T2", "T3", "T4")
CASES: Tuple[str, ...] = ("C1", "C2", "C3", "C4")
LEVELS: Tuple[str, ...] = ("full", "segmented")


@dataclass
class Config:
    """All settings for the case study"""

    # Paths
    data_root: Path = Path(".")
    output_dir: Path = Path("outputs")

    # Acquisition
    fs: int = 48_000                 # sampling frequency [Hz]
    duration_s: float = 10.0         # analysed length of every recording [s]
    channel_col: int = 2             # Excel column of the accelerometer used ("X - Axis", hardware channel 2)
    channel_label_contains: str = "X"  # sanity check on the column header text
    data_start_row: int = 49         # first numeric row when the sheet is read with header=None

    # Signal processing
    segment_s: float = 1.0           # segment length [s]
    nfft: int = 2 ** 22              # FFT length
    bpfo_hz: float = 276.27          # BPFO at the nominal 2000 rpm [Hz]
    harmonics: Tuple[int, ...] = (1, 2)
    half_band_hz: float = 5.0        # peak search band: theoretical harmonic +/- half_band_hz

    # Experimental design and folder names
    n_recordings: int = 5
    health_folder: Dict[str, str] = field(default_factory=lambda: {"healthy": "Healthy", "damaged": "Damage"})
    case_folder: Dict[str, str] = field(default_factory=lambda: {
        "C1": "Baseline", "C2": "Visual", "C3": "No Outerrace", "C4": "Outerrace"})

    # Analysis settings
    outlier_pct: float = 85.0        # hull screening threshold used for segmented sets (dissertation)
    sweep_pcts: Tuple[float, ...] = (70.0, 80.0, 85.0, 90.0, 95.0, 100.0)
    n_boot: int = 2000
    seed: int = 20261003
    representative_test: str = "T2"

    # ----- derived ---------------------------------------------------------
    @property
    def n_samples(self) -> int:
        return int(round(self.fs * self.duration_s))

    @property
    def segment_len(self) -> int:
        return int(round(self.fs * self.segment_s))

    @property
    def n_segments(self) -> int:
        return self.n_samples // self.segment_len

    @property
    def targets_hz(self) -> List[float]:
        return [h * self.bpfo_hz for h in self.harmonics]

    def rng(self, offset: int = 0) -> np.random.Generator:
        """Independent, reproducible random stream for each analysis."""
        return np.random.default_rng(self.seed + offset)

    def subdir(self, name: str) -> Path:
        p = Path(self.output_dir) / name
        p.mkdir(parents=True, exist_ok=True)
        return p


# ---------------------------------------------------------------------------
# Raw data
# ---------------------------------------------------------------------------

class RawDataLoader:
    """Reads the analysed channel of one recording from the raw Excel files.

    Folder layout:  <data_root>/<Healthy|Damage>/Test <k>/<case folder>/Test <r>.xlsx
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        try:
            import python_calamine  # noqa: F401
            self.engine = "calamine"     # about 3x faster than openpyxl
        except ImportError:
            self.engine = "openpyxl"

    def path(self, health: str, test: str, case: str, recording: int) -> Path:
        folder = (Path(self.cfg.data_root) / self.cfg.health_folder[health]
                  / f"Test {test[1:]}" / self.cfg.case_folder[case])
        stem = f"Test {recording}"
        for ext in (".xlsx", ".XLSX", ".Xlsx"):
            p = folder / f"{stem}{ext}"
            if p.exists():
                return p
        # case-insensitive fallback
        if folder.exists():
            for p in folder.iterdir():
                if p.stem == stem and p.suffix.lower() == ".xlsx":
                    return p
        raise FileNotFoundError(f"No Excel file for {health} {test} {case} R{recording} in {folder}")

    def design(self) -> List[Tuple[str, str, str, int]]:
        """All (health, test, case, recording) combinations of the design."""
        return [(h, t, c, r) for h in HEALTH_STATES for t in TESTS for c in CASES
                for r in range(1, self.cfg.n_recordings + 1)]

    def check_files(self) -> pd.DataFrame:
        rows = []
        for h, t, c, r in self.design():
            try:
                p = self.path(h, t, c, r)
                rows.append(dict(health=h, test=t, case=c, recording=r, path=str(p),
                                 exists=True, size_bytes=p.stat().st_size))
            except FileNotFoundError:
                rows.append(dict(health=h, test=t, case=c, recording=r, path="", exists=False, size_bytes=np.nan))
        return pd.DataFrame(rows)

    def load(self, path: Path) -> np.ndarray:
        """Return the first `duration_s` seconds of the analysed channel as float64."""
        df = pd.read_excel(path, header=None, usecols=[self.cfg.channel_col], engine=self.engine)
        label = str(df.iloc[1, 0])
        if self.cfg.channel_label_contains not in label:
            raise ValueError(f"{path}: column {self.cfg.channel_col} header is '{label}', "
                             f"expected it to contain '{self.cfg.channel_label_contains}'")
        x = pd.to_numeric(df.iloc[self.cfg.data_start_row:, 0], errors="coerce").to_numpy(dtype=float)
        x = x[: self.cfg.n_samples]
        if x.size < self.cfg.n_samples:
            raise ValueError(f"{path}: only {x.size} samples, {self.cfg.n_samples} required")
        if not np.all(np.isfinite(x)):
            raise ValueError(f"{path}: non-numeric values inside the analysed {self.cfg.duration_s} s")
        return x


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

class SESProcessor:
    """Hilbert-transform squared envelope spectrum (SES) and band-limited peak picking.

    SES(f) = (2 / N) |FFT_nfft( |x + j H{x}|^2 - mean )|  (one-sided, N = number of samples)
    The selected peak for harmonic h is the maximum of the SES inside
    [h*BPFO - half_band, h*BPFO + half_band].
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.freq = np.arange(cfg.nfft // 2) * (cfg.fs / cfg.nfft)
        self._bands = [np.where((self.freq >= f - cfg.half_band_hz) & (self.freq <= f + cfg.half_band_hz))[0]
                       for f in cfg.targets_hz]

    def ses(self, x: np.ndarray) -> np.ndarray:
        env2 = np.abs(_hilbert(x)) ** 2
        env2 = env2 - env2.mean()
        spec = np.abs(_fft(env2, self.cfg.nfft)) / len(x)
        return spec[: self.cfg.nfft // 2] * 2.0

    def pick_peaks(self, spec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (amplitudes, frequencies) of the selected peaks, one per harmonic."""
        amps, freqs = [], []
        for idx in self._bands:
            k = idx[np.argmax(spec[idx])]
            amps.append(spec[k])
            freqs.append(self.freq[k])
        return np.array(amps), np.array(freqs)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """Full-signal and segmented (A1, A2) features for every recording."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.proc = SESProcessor(cfg)
        self.loader = RawDataLoader(cfg)

    def recording_features(self, x: np.ndarray, keep_spectra: bool = False):
        """Features of one recording.

        Returns (full_row, segment_rows, spectra) where spectra = (ses_full, ses_seg1)
        when keep_spectra is True (used for the peak verification plots).
        """
        spec_full = self.proc.ses(x)
        a_full, f_full = self.proc.pick_peaks(spec_full)
        full_row = dict(A1=a_full[0], A2=a_full[1], f1=f_full[0], f2=f_full[1])
        seg_rows, spec_seg1 = [], None
        L = self.cfg.segment_len
        for s in range(self.cfg.n_segments):
            spec = self.proc.ses(x[s * L:(s + 1) * L])
            a, f = self.proc.pick_peaks(spec)
            seg_rows.append(dict(segment=s + 1, A1=a[0], A2=a[1], f1=f[0], f2=f[1]))
            if s == 0 and keep_spectra:
                spec_seg1 = spec
        spectra = (spec_full, spec_seg1) if keep_spectra else None
        return full_row, seg_rows, spectra

    def process_file(self, key: Tuple[str, str, str, int], peak_plot_dir: Optional[Path] = None):
        h, t, c, r = key
        x = self.loader.load(self.loader.path(h, t, c, r))
        full_row, seg_rows, spectra = self.recording_features(x, keep_spectra=peak_plot_dir is not None)
        ids = dict(health=h, test=t, case=c, recording=r)
        full = [{**ids, **full_row}]
        seg = [{**ids, **row} for row in seg_rows]
        if peak_plot_dir is not None:
            pl = Plotter(self.cfg)
            tag = f"{h}_{t}_{c}_R{r}"
            pl.peak_verification(self.proc.freq, spectra[0], full_row,
                                 save_path=Path(peak_plot_dir) / f"peaks_{tag}_full.png")
            s1 = seg_rows[0]
            pl.peak_verification(self.proc.freq, spectra[1], s1,
                                 save_path=Path(peak_plot_dir) / f"peaks_{tag}_segment01.png")
        return full, seg

    def extract_all(self, keys: Optional[Sequence[Tuple[str, str, str, int]]] = None,
                    n_jobs: int = 1, peak_plot_dir: Optional[Path] = None,
                    verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract features for all recordings (or the given subset).

        n_jobs > 1 uses separate processes (works in Jupyter on Windows because the
        worker function lives in this importable module).
        """
        keys = list(keys) if keys is not None else self.loader.design()
        if peak_plot_dir is not None:
            Path(peak_plot_dir).mkdir(parents=True, exist_ok=True)
        full_rows, seg_rows = [], []
        t0 = time.time()
        if n_jobs <= 1:
            for i, key in enumerate(keys, 1):
                f, s = self.process_file(key, peak_plot_dir)
                full_rows += f
                seg_rows += s
                if verbose and (i % 10 == 0 or i == len(keys)):
                    print(f"[{i:3d}/{len(keys)}] done  ({time.time() - t0:6.0f} s)", flush=True)
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                futs = {ex.submit(_process_file_worker, self.cfg, key, peak_plot_dir): key for key in keys}
                for i, fut in enumerate(as_completed(futs), 1):
                    f, s = fut.result()
                    full_rows += f
                    seg_rows += s
                    if verbose and (i % 10 == 0 or i == len(keys)):
                        print(f"[{i:3d}/{len(keys)}] done  ({time.time() - t0:6.0f} s)", flush=True)
        full = pd.DataFrame(full_rows).sort_values(["health", "test", "case", "recording"]).reset_index(drop=True)
        seg = (pd.DataFrame(seg_rows).sort_values(["health", "test", "case", "recording", "segment"])
               .reset_index(drop=True))
        return full, seg


def _process_file_worker(cfg: Config, key, peak_plot_dir):
    import matplotlib
    matplotlib.use("Agg")
    return FeatureExtractor(cfg).process_file(key, peak_plot_dir)


# ---------------------------------------------------------------------------
# Feature storage and access
# ---------------------------------------------------------------------------

class FeatureStore:
    """Feature tables on disk (CSV) and convenient access to point sets."""

    FULL_FILE = "features_full.csv"
    SEG_FILE = "features_segmented.csv"

    def __init__(self, full: pd.DataFrame, segmented: pd.DataFrame):
        self.full = full
        self.segmented = segmented

    # ----- I/O ---------------------------------------------------------------
    def save(self, folder: Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.full.to_csv(folder / self.FULL_FILE, index=False, float_format="%.10e")
        self.segmented.to_csv(folder / self.SEG_FILE, index=False, float_format="%.10e")

    @classmethod
    def load(cls, folder: Path) -> "FeatureStore":
        folder = Path(folder)
        return cls(pd.read_csv(folder / cls.FULL_FILE), pd.read_csv(folder / cls.SEG_FILE))

    # ----- access ------------------------------------------------------------
    def table(self, level: str) -> pd.DataFrame:
        return self.full if level == "full" else self.segmented

    def points(self, health: str, test: str, case: str, level: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (points (n, 2), recording label per point)."""
        df = self.table(level)
        sub = df[(df.health == health) & (df.test == test) & (df.case == case)]
        sort_cols = ["recording", "segment"] if level == "segmented" else ["recording"]
        sub = sub.sort_values(sort_cols)
        return sub[["A1", "A2"]].to_numpy(dtype=float), sub["recording"].to_numpy()

    def cells(self) -> List[Tuple[str, str, str]]:
        """All (health state, test session, case) combinations, i.e. all assembly states."""
        return [(h, t, c) for h in HEALTH_STATES for t in TESTS for c in CASES]


# ---------------------------------------------------------------------------
# Geometry of point sets
# ---------------------------------------------------------------------------

class Geometry:
    """Point-set indicators in the 2-D feature space."""

    @staticmethod
    def n_distinct(P: np.ndarray) -> int:
        return len(np.unique(np.round(P, 15), axis=0))

    @staticmethod
    def centroid(P: np.ndarray) -> np.ndarray:
        return P.mean(axis=0)

    @staticmethod
    def screen_mask(P: np.ndarray, pct: float) -> np.ndarray:
        """Dissertation screening rule: keep points whose distance from the mean is
        at most pct % of the largest distance. pct = 100 keeps every point."""
        if pct >= 100.0 or len(P) < 3:
            return np.ones(len(P), dtype=bool)
        d = np.linalg.norm(P - P.mean(axis=0), axis=1)
        return d <= (pct / 100.0) * d.max()

    @staticmethod
    def hull(P: np.ndarray) -> Optional[ConvexHull]:
        """Convex hull, or None for degenerate sets (fewer than 3 distinct or collinear points)."""
        if Geometry.n_distinct(P) < 3:
            return None
        try:
            return ConvexHull(P)
        except QhullError:
            return None

    @staticmethod
    def hull_dispersion(P: np.ndarray, pct: float = 100.0) -> float:
        """D_hull = sqrt(Area(H)) of the (optionally screened) set; NaN when the hull is degenerate."""
        Q = P[Geometry.screen_mask(P, pct)]
        H = Geometry.hull(Q)
        return float(np.sqrt(H.volume)) if H is not None else np.nan

    @staticmethod
    def cov(P: np.ndarray) -> np.ndarray:
        return np.cov(P, rowvar=False, ddof=1)

    @staticmethod
    def cov_dispersion(P: np.ndarray) -> float:
        """D_Sigma = det(Sigma_hat)^(1/4); same units as the features."""
        if Geometry.n_distinct(P) < 3:
            return np.nan
        det = float(np.linalg.det(Geometry.cov(P)))
        return det ** 0.25 if det > 0 else np.nan

    @staticmethod
    def pooled_cov_dispersion(P1: np.ndarray, P2: np.ndarray) -> float:
        """det((Sigma_1 + Sigma_2) / 2)^(1/4)."""
        if Geometry.n_distinct(P1) < 3 or Geometry.n_distinct(P2) < 3:
            return np.nan
        det = float(np.linalg.det(0.5 * (Geometry.cov(P1) + Geometry.cov(P2))))
        return det ** 0.25 if det > 0 else np.nan

    @staticmethod
    def shift(P_case: np.ndarray, P_base: np.ndarray) -> float:
        """Bias B = || mu(case) - mu(baseline) ||_2 (centroid shift)."""
        return float(np.linalg.norm(P_case.mean(axis=0) - P_base.mean(axis=0)))


# ---------------------------------------------------------------------------
# Metric table
# ---------------------------------------------------------------------------

class MetricTable:
    """D_hull, D_Sigma, B and rho_Sigma for every assembly state (health state,
    test session, case) and signal length (full or segmented).

    Screening (cfg.outlier_pct) is applied to the hull of segmented sets only,
    as in the dissertation; B and D_Sigma always use all points.
    """

    def __init__(self, cfg: Config, store: FeatureStore):
        self.cfg = cfg
        self.store = store

    def hull_pct(self, level: str) -> float:
        return self.cfg.outlier_pct if level == "segmented" else 100.0

    def cell(self, health: str, test: str, case: str, level: str, pct: Optional[float] = None) -> Dict:
        pct = self.hull_pct(level) if pct is None else pct
        P, _ = self.store.points(health, test, case, level)
        Pb, _ = self.store.points(health, test, "C1", level)
        mu = Geometry.centroid(P)
        row = dict(health=health, test=test, case=case, level=level, n=len(P),
                   mu_A1=mu[0], mu_A2=mu[1],
                   D_hull=Geometry.hull_dispersion(P, pct), D_sigma=Geometry.cov_dispersion(P))
        if case == "C1":
            row.update(B=np.nan, D_sigma_pool=np.nan, rho_sigma=np.nan)
        else:
            B = Geometry.shift(P, Pb)
            Dp = Geometry.pooled_cov_dispersion(P, Pb)
            row.update(B=B, D_sigma_pool=Dp, rho_sigma=B / Dp)
        return row

    def build(self) -> pd.DataFrame:
        rows = [self.cell(h, t, c, lv) for lv in LEVELS for (h, t, c) in self.store.cells()]
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Inference: bootstrap, jackknife, permutation
# ---------------------------------------------------------------------------

class Inference:
    """Uncertainty on the indicators, with the recording as the unit of repetition.

    * Bootstrap: whole recordings are resampled with replacement (for segmented sets
      a recording carries all of its segments). Used for B, D_Sigma and rho_Sigma.
    * Jackknife: leave-one-recording-out, used for the convex-hull dispersion D_hull
      (a percentile bootstrap is not valid for a hull area because a resample's hull
      always lies inside the original hull).
    * Permutation test of "no intervention effect": the recordings of C1 and Cj are
      relabelled in all C(10, 5) = 252 ways; statistic B.
    """

    def __init__(self, cfg: Config, store: FeatureStore):
        self.cfg = cfg
        self.store = store

    @staticmethod
    def _groups(P: np.ndarray, g: np.ndarray) -> List[np.ndarray]:
        return [P[g == k] for k in np.unique(g)]

    # ----- bootstrap ---------------------------------------------------------
    def bootstrap_cell(self, health: str, test: str, case: str, level: str,
                       rng: np.random.Generator, n_boot: Optional[int] = None) -> Dict:
        n_boot = n_boot or self.cfg.n_boot
        P, g = self.store.points(health, test, case, level)
        Pb, gb = self.store.points(health, test, "C1", level)
        Gc, Gb = self._groups(P, g), self._groups(Pb, gb)
        out = {k: [] for k in ("B", "D_sigma", "rho_sigma")}
        for _ in range(n_boot):
            Pc_s = np.vstack([Gc[i] for i in rng.integers(0, len(Gc), len(Gc))])
            Pb_s = np.vstack([Gb[i] for i in rng.integers(0, len(Gb), len(Gb))])
            out["D_sigma"].append(Geometry.cov_dispersion(Pc_s))
            if case != "C1":
                B = Geometry.shift(Pc_s, Pb_s)
                out["B"].append(B)
                out["rho_sigma"].append(B / Geometry.pooled_cov_dispersion(Pc_s, Pb_s))
        row = dict(health=health, test=test, case=case, level=level)
        for k, v in out.items():
            v = np.asarray(v, dtype=float)
            ok = v[np.isfinite(v)]
            row[f"{k}_lo"] = np.quantile(ok, 0.025) if ok.size else np.nan
            row[f"{k}_hi"] = np.quantile(ok, 0.975) if ok.size else np.nan
            row[f"{k}_valid_frac"] = ok.size / n_boot if v.size else np.nan
        return row

    # ----- jackknife ---------------------------------------------------------
    def jackknife_hull(self, health: str, test: str, case: str, level: str, pct: float) -> Dict:
        P, g = self.store.points(health, test, case, level)
        labels = np.unique(g)
        D0 = Geometry.hull_dispersion(P, pct)
        Dj = np.array([Geometry.hull_dispersion(P[g != k], pct) for k in labels])
        n = len(labels)
        Dbar = np.nanmean(Dj)
        se = math.sqrt((n - 1) / n * np.nansum((Dj - Dbar) ** 2))
        return dict(health=health, test=test, case=case, level=level, D_hull=D0,
                    D_hull_jk_bias=(n - 1) * (Dbar - D0), D_hull_jk_se=se,
                    D_hull_jk_min=np.nanmin(Dj), D_hull_jk_max=np.nanmax(Dj))

    # ----- permutation -------------------------------------------------------
    @staticmethod
    def permutation_pvalue(P: np.ndarray, g: np.ndarray, Pb: np.ndarray, gb: np.ndarray) -> Tuple[float, int]:
        """Exact permutation p-value of B over all relabellings of whole recordings."""
        groups = Inference._groups(P, g) + Inference._groups(Pb, gb)
        sums = np.array([G.sum(axis=0) for G in groups])
        counts = np.array([len(G) for G in groups])
        n_c = len(Inference._groups(P, g))
        idx_all = range(len(groups))
        B_obs = Geometry.shift(P, Pb)
        stats = []
        for comb in itertools.combinations(idx_all, n_c):
            mask = np.zeros(len(groups), dtype=bool)
            mask[list(comb)] = True
            m1 = sums[mask].sum(axis=0) / counts[mask].sum()
            m2 = sums[~mask].sum(axis=0) / counts[~mask].sum()
            stats.append(np.linalg.norm(m1 - m2))
        stats = np.asarray(stats)
        return float(np.mean(stats >= B_obs - 1e-15)), len(stats)

    def permutation_cell(self, health: str, test: str, case: str, level: str) -> Dict:
        P, g = self.store.points(health, test, case, level)
        Pb, gb = self.store.points(health, test, "C1", level)
        p, n_perm = self.permutation_pvalue(P, g, Pb, gb)
        return dict(health=health, test=test, case=case, level=level, perm_p=p, n_perm=n_perm)

    # ----- all assembly states ----------------------------------------------
    def run(self) -> pd.DataFrame:
        rows = []
        for i, (lv, (h, t, c)) in enumerate(itertools.product(LEVELS, self.store.cells())):
            pct = self.cfg.outlier_pct if lv == "segmented" else 100.0
            row = self.bootstrap_cell(h, t, c, lv, self.cfg.rng(1000 + i))
            row.update({k: v for k, v in self.jackknife_hull(h, t, c, lv, pct).items()
                        if k.startswith("D_hull")})
            if c != "C1":
                row.update({k: v for k, v in self.permutation_cell(h, t, c, lv).items()
                            if k in ("perm_p", "n_perm")})
            rows.append(row)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Outlier-screening sensitivity
# ---------------------------------------------------------------------------

class ScreeningSweep:
    """Hull dispersion D_hull of every segmented assembly state for each screening threshold.

    Only D_hull depends on the screening rule; B, D_Sigma, rho_Sigma, the permutation
    test and the task check use all points.
    """

    def __init__(self, cfg: Config, store: FeatureStore):
        self.cfg = cfg
        self.mt = MetricTable(cfg, store)

    def run(self) -> pd.DataFrame:
        rows = []
        for pct in self.cfg.sweep_pcts:
            for h, t, c in self.mt.store.cells():
                r = self.mt.cell(h, t, c, "segmented", pct=pct)
                rows.append(dict(health=h, test=t, case=c, pct=pct, D_hull=r["D_hull"]))
        return pd.DataFrame(rows)

    def summary(self, sweep: pd.DataFrame) -> pd.DataFrame:
        """Per threshold: change of D_hull relative to its value at the reference threshold (85 %)."""
        ref = sweep[sweep.pct == self.cfg.outlier_pct].set_index(["health", "test", "case"])
        out = []
        for pct, sub in sweep.groupby("pct"):
            s = sub.set_index(["health", "test", "case"])
            rel = (s.D_hull / ref.D_hull - 1.0) * 100.0
            out.append(dict(pct=pct,
                            D_hull_change_median_pct=float(np.nanmedian(rel)),
                            D_hull_change_max_abs_pct=float(np.nanmax(np.abs(rel))),
                            n_assembly_states=int(rel.notna().sum())))
        return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# ISO 5725 style variance components
# ---------------------------------------------------------------------------

class VarianceComponents:
    """Balanced nested random-effects model, method of moments (ANOVA estimators).

    Full-signal model (one value per recording):
        y = mu + test + assembly(test) + recording(assembly)
    Segmented model (one value per 1 s segment):
        y = mu + test + assembly(test) + recording(assembly) + segment(recording)

    'assembly' is one intervention case within one test session (a distinct assembly
    state). Repeatability s_r is the recording-to-recording standard deviation within
    an assembly state; reproducibility s_R adds the assembly and test-session components.
    Negative component estimates are set to zero (reported in the table).
    """

    def __init__(self, cfg: Config, store: FeatureStore):
        self.cfg = cfg
        self.store = store

    @staticmethod
    def nested_anova(y: np.ndarray) -> Tuple[List[float], List[float]]:
        """y has shape (a, b, c[, d]) for a balanced nested design.

        Returns (mean squares from top to bottom level, variance components top to bottom).
        """
        shape = y.shape
        k = y.ndim
        means = [y]
        for ax in range(k - 1, 0, -1):
            means.insert(0, means[0].mean(axis=ax))
        # means[i] has shape shape[:i+1]; grand mean
        grand = y.mean()
        ms = []
        for lvl in range(k):
            m = means[lvl]
            parent = grand if lvl == 0 else means[lvl - 1][(...,) + (None,)]
            reps = int(np.prod(shape[lvl + 1:]))
            ss = reps * np.sum((m - parent) ** 2)
            df = int(np.prod(shape[:lvl])) * (shape[lvl] - 1)
            ms.append(ss / df)
        comps = [0.0] * k
        comps[k - 1] = ms[k - 1]
        for lvl in range(k - 2, -1, -1):
            reps = int(np.prod(shape[lvl + 1:]))
            comps[lvl] = (ms[lvl] - ms[lvl + 1]) / reps
        return ms, comps

    def _array(self, health: str, feature: str, level: str) -> np.ndarray:
        df = self.store.table(level)
        sub = df[df.health == health]
        keys = ["test", "case", "recording"] + (["segment"] if level == "segmented" else [])
        sub = sub.sort_values(keys)
        dims = [len(TESTS), len(CASES), self.cfg.n_recordings] + ([self.cfg.n_segments] if level == "segmented" else [])
        return sub[feature].to_numpy(dtype=float).reshape(dims)

    def run(self) -> pd.DataFrame:
        rows = []
        for level in LEVELS:
            names = ["test", "assembly", "recording"] + (["segment"] if level == "segmented" else [])
            for h in HEALTH_STATES:
                for feat in ("A1", "A2"):
                    y = self._array(h, feat, level)
                    ms, comps = self.nested_anova(y)
                    neg = [n for n, c in zip(names, comps) if c < 0]
                    comps = [max(c, 0.0) for c in comps]
                    var = dict(zip(names, comps))
                    total = sum(comps)
                    s_r2 = var["recording"] + var.get("segment", 0.0)
                    s_R2 = s_r2 + var["assembly"] + var["test"]
                    rows.append(dict(level=level, health=h, feature=feat, mean=float(y.mean()),
                                     **{f"var_{n}": var[n] for n in names},
                                     s_r=math.sqrt(s_r2), s_R=math.sqrt(s_R2),
                                     ratio_sR_sr=math.sqrt(s_R2 / s_r2) if s_r2 > 0 else np.nan,
                                     ICC_assembly=var["assembly"] / total if total > 0 else np.nan,
                                     ICC_test=var["test"] / total if total > 0 else np.nan,
                                     negative_set_to_zero=",".join(neg)))
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Task check: separation between health states vs range within each health state
# ---------------------------------------------------------------------------

class TaskCheck:
    """Separation Delta between the health states and range delta within each health state.

    Every assembly state (test session, case) is represented by its centroid. Any healthy
    assembly state could be met with any damaged assembly state in practice, so all
    16 x 16 healthy-damaged pairs are compared:

        Delta          smallest distance between a healthy and a damaged centroid
        delta_healthy  largest distance between two healthy centroids (fault detection)
        delta_damaged  largest distance between two damaged centroids (tasks within the
                       damaged state, such as severity estimation or trending)

    The ratio Delta / delta is judged from its bootstrap interval (whole recordings of
    every assembly state are resampled): negligible if the interval lies above one,
    ambiguous if it contains one, influential if it lies below one.
    """

    def __init__(self, cfg: Config, store: FeatureStore):
        self.cfg = cfg
        self.store = store

    def centroids(self, level: str, rng: Optional[np.random.Generator] = None) -> Dict[str, np.ndarray]:
        out = {}
        for h in HEALTH_STATES:
            cs = []
            for t in TESTS:
                for c in CASES:
                    P, g = self.store.points(h, t, c, level)
                    if rng is not None:  # resample whole recordings
                        G = Inference._groups(P, g)
                        P = np.vstack([G[i] for i in rng.integers(0, len(G), len(G))])
                    cs.append(P.mean(axis=0))
            out[h] = np.array(cs)
        return out

    @staticmethod
    def _stats(cent: Dict[str, np.ndarray]) -> Dict[str, float]:
        H, D = cent["healthy"], cent["damaged"]
        between = np.linalg.norm(H[:, None, :] - D[None, :, :], axis=2)
        def within(X):
            d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
            return d[np.triu_indices(len(X), 1)]
        wh, wd = within(H), within(D)
        return dict(Delta=between.min(), between_max=between.max(), between_median=np.median(between),
                    delta_healthy=wh.max(), delta_damaged=wd.max(),
                    Delta_over_delta_healthy=between.min() / wh.max(),
                    Delta_over_delta_damaged=between.min() / wd.max())

    def run(self, n_boot: Optional[int] = None) -> pd.DataFrame:
        n_boot = n_boot or self.cfg.n_boot
        rows = []
        for j, level in enumerate(LEVELS):
            est = self._stats(self.centroids(level))
            rng = self.cfg.rng(5000 + j)
            boots = pd.DataFrame([self._stats(self.centroids(level, rng)) for _ in range(n_boot)])
            for k, v in est.items():
                rows.append(dict(level=level, quantity=k, estimate=v,
                                 boot_lo=boots[k].quantile(0.025), boot_hi=boots[k].quantile(0.975)))
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Full-signal vs segmented dispersion (sample-size effect)
# ---------------------------------------------------------------------------

class SampleSizeCheck:
    """How much of the full vs segmented dispersion gap is sample size and how much is real.

    * ratio_hull  = D_hull(segmented) / D_hull(full)  : sample size + genuine difference
    * ratio_sigma = D_Sigma(segmented) / D_Sigma(full): genuine difference only
      (D_Sigma is approximately unbiased in n)
    * subsampling: D_hull of 5 points drawn at random from the segmented set (no
      screening, like the 5 full-signal points), compared with D_hull(full).
    """

    def __init__(self, cfg: Config, store: FeatureStore):
        self.cfg = cfg
        self.store = store

    def run(self, n_sub: int = 2000) -> Tuple[pd.DataFrame, Dict[Tuple[str, str, str], np.ndarray]]:
        rows, draws = [], {}
        for i, (h, t, c) in enumerate(self.store.cells()):
            Pf, _ = self.store.points(h, t, c, "full")
            Ps, _ = self.store.points(h, t, c, "segmented")
            rng = self.cfg.rng(8000 + i)
            sub = np.array([Geometry.hull_dispersion(Ps[rng.choice(len(Ps), len(Pf), replace=False)])
                            for _ in range(n_sub)])
            draws[(h, t, c)] = sub
            Dh_f, Dh_s = Geometry.hull_dispersion(Pf), Geometry.hull_dispersion(Ps, self.cfg.outlier_pct)
            Ds_f, Ds_s = Geometry.cov_dispersion(Pf), Geometry.cov_dispersion(Ps)
            rows.append(dict(health=h, test=t, case=c,
                             D_hull_full=Dh_f, D_hull_seg=Dh_s, D_sigma_full=Ds_f, D_sigma_seg=Ds_s,
                             ratio_hull=Dh_s / Dh_f, ratio_sigma=Ds_s / Ds_f,
                             sub5_median=np.nanmedian(sub), sub5_lo=np.nanquantile(sub, 0.025),
                             sub5_hi=np.nanquantile(sub, 0.975),
                             full_D_percentile_in_sub5=float(np.nanmean(sub <= Dh_f) * 100)))
        return pd.DataFrame(rows), draws


# ---------------------------------------------------------------------------
# Synthetic check of the effect size and the permutation test
# ---------------------------------------------------------------------------

class SyntheticCheck:
    """Two isotropic Gaussian sets (sigma = 1), one shifted by a known amount m along A1.

    For each m/sigma and set size n, the fraction of trials with rho_Sigma > 1 and, for
    n = 5, the fraction with a permutation p-value below 0.05 are recorded. m = 0 gives
    the false-positive rate when there is no intervention effect; m > 0 gives the
    detection rate. The permutation test is checked for n = 5 only because, in the case
    study, it always relabels the five recordings of each case.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self, shifts: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0),
            sizes: Sequence[int] = (5, 50), n_rep: int = 1000) -> pd.DataFrame:
        rows = []
        for n in sizes:
            rng = self.cfg.rng(9000 + n)
            for m in shifts:
                rs, pp = [], []
                for _ in range(n_rep):
                    base = rng.normal(0.0, 1.0, (n, 2))
                    case = rng.normal(0.0, 1.0, (n, 2)) + np.array([m, 0.0])
                    B = Geometry.shift(case, base)
                    rs.append(B / Geometry.pooled_cov_dispersion(case, base))
                    if n <= 5:
                        g = np.arange(n)
                        pp.append(Inference.permutation_pvalue(case, g, base, g + n)[0])
                rs = np.asarray(rs)
                rows.append(dict(n_per_case=n, m_over_sigma=m,
                                 P_rho_sigma_gt_1=float(np.nanmean(rs > 1)),
                                 P_perm_p_lt_005=float(np.mean(np.asarray(pp) < 0.05)) if pp else np.nan))
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def set_plot_style() -> None:
    import matplotlib
    matplotlib.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": "cmr10",
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "axes.formatter.use_mathtext": True,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


class Plotter:
    """All figures. Every figure is saved as an individual PNG (and PDF) so that
    panels can be assembled as subfigures in LaTeX; captions are written in LaTeX."""

    REC_COLOURS = ["blue", "black", "green", "red", "purple"]
    REC_MARKERS = ["o", "s", "^", "*", "P"]
    CASE_COLOURS = ["blue", "black", "green", "red"]
    CASE_MARKERS = ["o", "s", "^", "*"]

    def __init__(self, cfg: Config):
        self.cfg = cfg
        set_plot_style()

    # ----- helpers -----------------------------------------------------------
    @staticmethod
    def _save(fig, path: Optional[Path], pdf: bool = True) -> None:
        import matplotlib.pyplot as plt
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path.with_suffix(".png"))
            if pdf:
                fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)

    @staticmethod
    def _sci(ax, axis: str = "both", power: int = -3) -> None:
        import matplotlib.ticker as mt
        f = mt.ScalarFormatter(useMathText=True)
        f.set_powerlimits((power, power))
        if axis in ("x", "both"):
            ax.xaxis.set_major_formatter(f)
        if axis in ("y", "both"):
            f2 = mt.ScalarFormatter(useMathText=True)
            f2.set_powerlimits((power, power))
            ax.yaxis.set_major_formatter(f2)

    # ----- peak verification ---------------------------------------------------
    def peak_verification(self, freq: np.ndarray, spec: np.ndarray, peaks: Dict,
                          save_path: Optional[Path] = None, fmax: float = 1000.0,
                          zoom_half_width_hz: float = 15.0):
        """SES 0-fmax Hz with the selected peaks as red dots (top), and zoomed views
        around each harmonic with the +/- half-band search window shaded (bottom)."""
        import matplotlib.pyplot as plt
        amps = [peaks["A1"], peaks["A2"]]
        fpk = [peaks["f1"], peaks["f2"]]
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.25)
        ax = fig.add_subplot(gs[0, :])
        m = freq <= fmax
        ax.plot(freq[m], spec[m], color="black", lw=0.6)
        ymax = max(spec[m & (freq >= 2.0)].max(), max(amps)) * 1.15
        for h, (a, f) in enumerate(zip(amps, fpk), start=1):
            ax.scatter(f, a, color="red", s=60, zorder=3)
            ax.text(f + 8, a + 0.04 * ymax, f"{h} x BPFO", color="red", fontsize=14, ha="left", va="bottom",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))
        ax.set_xlim(0, fmax)
        ax.set_ylim(0, ymax)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Amplitude")
        self._sci(ax, "y")
        for h, (a, f, ft) in enumerate(zip(amps, fpk, self.cfg.targets_hz)):
            axz = fig.add_subplot(gs[1, h])
            mz = (freq >= ft - zoom_half_width_hz) & (freq <= ft + zoom_half_width_hz)
            axz.axvspan(ft - self.cfg.half_band_hz, ft + self.cfg.half_band_hz, color="0.85", zorder=0)
            axz.plot(freq[mz], spec[mz], color="black", lw=0.8)
            axz.scatter(f, a, color="red", s=60, zorder=3)
            axz.set_xlim(ft - zoom_half_width_hz, ft + zoom_half_width_hz)
            axz.set_ylim(0, max(spec[mz].max(), a) * 1.15)
            axz.set_xlabel("Frequency [Hz]")
            axz.set_ylabel("Amplitude")
            axz.text(0.02, 0.95, f"{h + 1} x BPFO", transform=axz.transAxes, color="red", fontsize=14, va="top")
            self._sci(axz, "y")
        self._save(fig, save_path, pdf=False)
        return fig

    # ----- feature-space plots -------------------------------------------------
    @staticmethod
    def _hull_patch(ax, P: np.ndarray, colour: str, pct: float):
        Q = P[Geometry.screen_mask(P, pct)]
        H = Geometry.hull(Q)
        if H is None:
            return np.nan
        v = np.append(H.vertices, H.vertices[0])
        ax.plot(Q[v, 0], Q[v, 1], color=colour, ls="--", lw=1.5, alpha=0.8)
        ax.fill(Q[v, 0], Q[v, 1], color=colour, alpha=0.10)
        return float(np.sqrt(H.volume))

    @staticmethod
    def _fmt_e(v: float) -> str:
        if not np.isfinite(v):
            return "n/a"
        m, e = f"{v:.1e}".split("e")
        return rf"{m}\times 10^{{{int(e)}}}"

    def within_case(self, store: FeatureStore, health: str, test: str, out_dir: Path,
                    pct: Optional[float] = None) -> Dict[str, Path]:
        """One panel per case: 1 s segments of each recording with its convex hull and
        centroid (mean of the recording's segments). Axis limits are shared by the four panels."""
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        pct = self.cfg.outlier_pct if pct is None else pct
        allP = np.vstack([store.points(health, test, c, "segmented")[0] for c in CASES])
        pad = 0.05 * (allP.max(axis=0) - allP.min(axis=0))
        lo, hi = np.maximum(allP.min(axis=0) - pad, 0), allP.max(axis=0) + pad
        paths = {}
        for c in CASES:
            P, g = store.points(health, test, c, "segmented")
            fig, ax = plt.subplots(figsize=(7, 6))
            handles = []
            for i, r in enumerate(np.unique(g)):
                Q = P[g == r]
                ax.scatter(Q[:, 0], Q[:, 1], color=self.REC_COLOURS[i], marker=self.REC_MARKERS[i], s=40, alpha=0.45)
                D = self._hull_patch(ax, Q, self.REC_COLOURS[i], pct)
                mu = Q.mean(axis=0)
                ax.scatter(*mu, marker="D", s=90, color=self.REC_COLOURS[i], edgecolor="black", lw=1.2, zorder=4)
                handles.append(Line2D([], [], color=self.REC_COLOURS[i], marker=self.REC_MARKERS[i], ls="None",
                                      markersize=9, label=rf"{c}-R{r}: $\sqrt{{\mathrm{{Area}}}}={self._fmt_e(D)}$"))
            handles.append(Line2D([], [], color="black", marker="D", ls="None", markerfacecolor="white",
                                  markersize=9, label="Centroid"))
            ax.legend(handles=handles, loc="best", fontsize=11)
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
            ax.set_xlabel(r"1$\times$BPFO amplitude"); ax.set_ylabel(r"2$\times$BPFO amplitude")
            ax.grid(True, ls="--", alpha=0.4)
            self._sci(ax)
            p = Path(out_dir) / f"fig_within_case_{health}_{test}_{c}"
            self._save(fig, p)
            paths[c] = p
        return paths

    def across_case(self, store: FeatureStore, health: str, test: str, out_dir: Path) -> Dict[str, Path]:
        """Cases C1-C4 of one test with case hulls and centroids; one figure per level."""
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        paths = {}
        for level in LEVELS:
            pct = self.cfg.outlier_pct if level == "segmented" else 100.0
            fig, ax = plt.subplots(figsize=(7, 6))
            handles = []
            for i, c in enumerate(CASES):
                P, _ = store.points(health, test, c, level)
                ax.scatter(P[:, 0], P[:, 1], color=self.CASE_COLOURS[i], marker=self.CASE_MARKERS[i],
                           s=40 if level == "segmented" else 90, alpha=0.45 if level == "segmented" else 0.8)
                D = self._hull_patch(ax, P, self.CASE_COLOURS[i], pct)
                ax.scatter(*P.mean(axis=0), marker="D", s=110, color=self.CASE_COLOURS[i], edgecolor="black",
                           lw=1.2, zorder=4)
                handles.append(Line2D([], [], color=self.CASE_COLOURS[i], marker=self.CASE_MARKERS[i], ls="None",
                                      markersize=9, label=rf"{c}: $\sqrt{{\mathrm{{Area}}}}={self._fmt_e(D)}$"))
            handles.append(Line2D([], [], color="black", marker="D", ls="None", markerfacecolor="white",
                                  markersize=9, label="Centroid"))
            ax.legend(handles=handles, loc="best", fontsize=11)
            ax.set_xlabel(r"1$\times$BPFO amplitude"); ax.set_ylabel(r"2$\times$BPFO amplitude")
            ax.grid(True, ls="--", alpha=0.4)
            self._sci(ax)
            p = Path(out_dir) / f"fig_across_case_{health}_{test}_{level}"
            self._save(fig, p)
            paths[level] = p
        return paths

    def cluster_map(self, store: FeatureStore, level: str, save_path: Path):
        """Centroid of every assembly state of both health states (the centroids used in the task check)."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        for h, col, mk in (("healthy", "tab:blue", "o"), ("damaged", "tab:red", "s")):
            C = np.array([store.points(h, t, c, level)[0].mean(axis=0) for t in TESTS for c in CASES])
            ax.scatter(C[:, 0], C[:, 1], color=col, marker=mk, s=60, label=f"{h.capitalize()} assembly states",
                       edgecolor="black", lw=0.6)
        ax.set_xlabel(r"1$\times$BPFO amplitude"); ax.set_ylabel(r"2$\times$BPFO amplitude")
        ax.legend()
        ax.grid(True, ls="--", alpha=0.4)
        self._sci(ax)
        self._save(fig, save_path)

    # ----- heatmaps (one PNG per panel) ----------------------------------------
    def heatmap_panel(self, data: np.ndarray, col_labels: Sequence[str], unit_text: str,
                      cmap_name: str, log_scale: bool, save_path: Path,
                      vmin: Optional[float] = None, vmax: Optional[float] = None, fmt: str = "{:.2f}"):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.0, 4.2))
        cmap = plt.get_cmap(cmap_name)
        if log_scale:
            plot_data = np.log10(data)
            lmin, lmax = float(np.nanmin(plot_data)), float(np.nanmax(plot_data))
            norm = plt.Normalize(vmin=lmin, vmax=lmax)
        else:
            plot_data = data
            norm = plt.Normalize(vmin=np.nanmin(data) if vmin is None else vmin,
                                 vmax=np.nanmax(data) if vmax is None else vmax)
        im = ax.imshow(plot_data, cmap=cmap, norm=norm, aspect="auto")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if not np.isfinite(v):
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=12)
                    continue
                rgba = cmap(norm(plot_data[i, j]))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        color="white" if lum < 0.55 else "black", fontsize=13)
        ax.set_xticks(range(data.shape[1])); ax.set_xticklabels(col_labels)
        ax.set_yticks(range(data.shape[0])); ax.set_yticklabels(TESTS)
        ax.set_xlabel("Intervention case"); ax.set_ylabel("Test session")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=11)
        if log_scale:
            ticks = np.linspace(lmin, lmax, 5)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{10 ** t:.2f}" for t in ticks])
        cbar.set_label(unit_text, fontsize=16)
        fig.tight_layout()
        self._save(fig, save_path, pdf=False)

    def heatmaps(self, metrics: pd.DataFrame, out_dir: Path) -> List[Path]:
        """D_hull, D_Sigma, B and rho_Sigma; D_hull, D_Sigma and B share a colour scale within
        each health state, rho_Sigma uses an independent log scale per panel."""
        specs = [
            ("D_hull", CASES, r"$D_{\mathrm{hull}}~(\times 10^{-4})$", "viridis", False, 1e4),
            ("D_sigma", CASES, r"$D_\Sigma~(\times 10^{-4})$", "viridis", False, 1e4),
            ("B", CASES[1:], r"$B~(\times 10^{-4})$", "plasma", False, 1e4),
            ("rho_sigma", CASES[1:], r"$\rho_\Sigma$", "RdBu_r", True, 1.0),
        ]
        out = []
        for metric, cols, unit, cmap, logs, scale in specs:
            for h in HEALTH_STATES:
                mats = {}
                for lv in LEVELS:
                    sub = metrics[(metrics.health == h) & (metrics.level == lv)]
                    mats[lv] = (sub.pivot(index="test", columns="case", values=metric)
                                .reindex(index=list(TESTS), columns=list(cols)).to_numpy() * scale)
                both = np.concatenate([m.ravel() for m in mats.values()])
                vmin, vmax = (None, None) if logs else (np.nanmin(both), np.nanmax(both))
                for lv, mat in mats.items():
                    p = Path(out_dir) / f"fig_heatmap_{metric}_{h}_{lv}.png"
                    self.heatmap_panel(mat, cols, unit, cmap, logs, p, vmin=vmin, vmax=vmax)
                    out.append(p)
        return out

    # ----- robustness figures ---------------------------------------------------
    def interval_plot(self, metrics: pd.DataFrame, intervals: pd.DataFrame, metric: str,
                      health: str, level: str, save_path: Path, ref_line: Optional[float] = None,
                      ylabel: str = ""):
        """Point estimates with 95 % bootstrap intervals (or +/- 2 jackknife SE for D_hull)."""
        import matplotlib.pyplot as plt
        m = metrics[(metrics.health == health) & (metrics.level == level)]
        iv = intervals[(intervals.health == health) & (intervals.level == level)]
        df = m.merge(iv, on=["health", "test", "case", "level"], suffixes=("", "_iv"))
        if metric != "D_hull" and metric != "D_sigma":
            df = df[df.case != "C1"]
        df = df.sort_values(["test", "case"]).reset_index(drop=True)
        x = np.arange(len(df))
        y = df[metric].to_numpy()
        if metric == "D_hull":
            lo, hi = y - 2 * df["D_hull_jk_se"].to_numpy(), y + 2 * df["D_hull_jk_se"].to_numpy()
        else:
            lo, hi = df[f"{metric}_lo"].to_numpy(), df[f"{metric}_hi"].to_numpy()
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.errorbar(x, y, yerr=[np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)], fmt="o", capsize=4,
                    color="black", ecolor="0.4")
        if ref_line is not None:
            ax.axhline(ref_line, color="red", ls="--", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t}-{c}" for t, c in zip(df.test, df.case)], rotation=60, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.grid(True, ls="--", alpha=0.4)
        fig.tight_layout()
        self._save(fig, save_path)

    def rho_interval_grid(self, metrics: pd.DataFrame, intervals: pd.DataFrame, save_path: Path,
                          alpha: float = 0.05):
        """rho_Sigma of every intervention with its 95 % recording-level bootstrap interval,
        for the four combinations of health state and signal length in one 2 x 2 grid.
        Filled markers: permutation p-value below alpha; open markers: the others."""
        import matplotlib.pyplot as plt
        df = metrics.merge(intervals, on=["health", "test", "case", "level"], suffixes=("", "_iv"))
        df = df[df.case != "C1"]
        fig, axes = plt.subplots(2, 2, figsize=(10, 6.6))
        titles = {("healthy", "segmented"): "(a) Healthy bearing, segmented signal",
                  ("healthy", "full"): "(b) Healthy bearing, full signal",
                  ("damaged", "segmented"): "(c) Damaged bearing, segmented signal",
                  ("damaged", "full"): "(d) Damaged bearing, full signal"}
        for ax, ((h, lv), title) in zip(axes.ravel(), titles.items()):
            d = df[(df.health == h) & (df.level == lv)].sort_values(["test", "case"]).reset_index(drop=True)
            x = np.arange(len(d))
            y = d.rho_sigma.to_numpy()
            lo, hi = d.rho_sigma_lo.to_numpy(), d.rho_sigma_hi.to_numpy()
            sig = (d.perm_p < alpha).to_numpy()
            ax.errorbar(x, y, yerr=[np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)], fmt="none",
                        ecolor="0.45", capsize=3, lw=1)
            ax.scatter(x[sig], y[sig], color="black", s=36, zorder=3, label=rf"$p<{alpha}$")
            ax.scatter(x[~sig], y[~sig], facecolor="white", edgecolor="black", s=36, zorder=3,
                       label=rf"$p\geq{alpha}$")
            ax.axhline(1.0, color="red", ls="--", lw=1)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{t}-{c}" for t, c in zip(d.test, d.case)], rotation=60, fontsize=10)
            ax.set_ylabel(r"$\rho_\Sigma$", fontsize=14)
            ax.set_title(title, fontsize=12)
            ax.tick_params(axis="y", labelsize=11)
            ax.grid(True, ls="--", alpha=0.4)
        axes[0, 0].legend(fontsize=10, loc="upper right")
        fig.tight_layout()
        self._save(fig, save_path)

    def screening_sweep(self, sweep: pd.DataFrame, health: str, save_path: Path):
        """Change of the segmented D_hull of every assembly state relative to its value at
        the reference threshold (cfg.outlier_pct), for each screening threshold."""
        import matplotlib.pyplot as plt
        ref_pct = self.cfg.outlier_pct
        sub = sweep[sweep.health == health]
        ref = sub[sub.pct == ref_pct].set_index(["test", "case"]).D_hull
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for (t, c), g in sub.groupby(["test", "case"]):
            g = g.sort_values("pct")
            change = (g.D_hull.to_numpy() / ref.loc[(t, c)] - 1.0) * 100.0
            ax.plot(g.pct, change, marker="o", lw=1, alpha=0.7)
        ax.axvline(ref_pct, color="red", ls="--", lw=1)
        ax.axhline(0.0, color="0.5", ls=":", lw=1)
        ax.set_xlabel("Screening threshold [%]  (100 = no screening)")
        ax.set_ylabel(r"Change in $D_{\mathrm{hull}}$ [%]")
        ax.grid(True, ls="--", alpha=0.4)
        fig.tight_layout()
        self._save(fig, save_path)

    def dispersion_comparison(self, metrics: pd.DataFrame, save_path: Path):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        for (h, lv), g in metrics.groupby(["health", "level"]):
            ax.scatter(g.D_hull, g.D_sigma, label=f"{h}, {lv}", s=40,
                       marker="o" if lv == "segmented" else "s", alpha=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$D_{\mathrm{hull}}=\sqrt{\mathrm{Area}(H)}$")
        ax.set_ylabel(r"$D_\Sigma=(\det\hat\Sigma)^{1/4}$")
        ax.legend(fontsize=11)
        ax.grid(True, which="both", ls="--", alpha=0.3)
        fig.tight_layout()
        self._save(fig, save_path)

    def subsample_hist(self, draws: np.ndarray, D_full: float, D_seg: float, save_path: Path):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(draws[np.isfinite(draws)], bins=40, color="0.7", edgecolor="black")
        ax.axvline(D_full, color="red", lw=2, label="Full signal (5 recordings)")
        ax.axvline(D_seg, color="blue", lw=2, ls="--", label="All segments")
        ax.set_xlabel(r"$D_{\mathrm{hull}}$ of 5 segments drawn at random")
        ax.set_ylabel("Count")
        ax.legend(fontsize=11)
        self._sci(ax, "x")
        fig.tight_layout()
        self._save(fig, save_path)

    def synthetic_check(self, synth: pd.DataFrame, save_path: Path):
        """Fraction of synthetic trials with rho_Sigma > 1 (n = 5 and n = 50) and with a
        permutation p-value below 0.05 (n = 5), against the true shift m / sigma."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for n, g in synth.groupby("n_per_case"):
            ls = "--" if n == 5 else "-"
            ax.plot(g.m_over_sigma, g.P_rho_sigma_gt_1, ls=ls, marker="o",
                    label=rf"$\rho_\Sigma>1$, $n={n}$")
            if g.P_perm_p_lt_005.notna().any():
                ax.plot(g.m_over_sigma, g.P_perm_p_lt_005, ls=":", marker="s",
                        label=rf"permutation $p<0.05$, $n={n}$")
        ax.axhline(0.05, color="0.5", ls=":", lw=1)
        ax.set_xlabel(r"True shift in units of the spread, $m/\sigma$")
        ax.set_ylabel("Fraction of trials")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=11)
        ax.grid(True, ls="--", alpha=0.4)
        fig.tight_layout()
        self._save(fig, save_path)


# ---------------------------------------------------------------------------
# Self-tests (run from the notebooks; each checks one class in isolation)
# ---------------------------------------------------------------------------

def run_self_tests(verbose: bool = True) -> None:
    rng = np.random.default_rng(0)
    cfg = Config(nfft=2 ** 16)

    # SESProcessor: amplitude-modulated carrier, peak at BPFO found inside the band
    t = np.arange(cfg.n_samples) / cfg.fs
    x = (1 + 0.5 * np.cos(2 * np.pi * cfg.bpfo_hz * t)) * np.sin(2 * np.pi * 3000 * t)
    sp = SESProcessor(cfg)
    a, f = sp.pick_peaks(sp.ses(x))
    assert abs(f[0] - cfg.bpfo_hz) <= cfg.half_band_hz and a[0] > a[1], "SES peak picking"

    # Geometry: degenerate sets give NaN, square of side 1 has D = 1
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    assert abs(Geometry.hull_dispersion(sq) - 1.0) < 1e-12, "hull area"
    assert np.isnan(Geometry.hull_dispersion(np.array([[0, 0], [0, 0], [1, 1]], float))), "degenerate hull"
    assert np.isnan(Geometry.hull_dispersion(np.array([[0, 0], [1, 1], [2, 2]], float))), "collinear hull"
    assert Geometry.screen_mask(sq, 100).all(), "no screening at 100 %"

    # Permutation: identical sets give p = 1; widely separated sets give the minimum p
    P = rng.normal(size=(5, 2))
    g = np.arange(5)
    p_same, n_perm = Inference.permutation_pvalue(P, g, P.copy(), g + 5)
    assert n_perm == 252 and p_same == 1.0, "permutation, identical sets"
    p_far, _ = Inference.permutation_pvalue(P + 100, g, P, g + 5)
    assert abs(p_far - 2 / 252) < 1e-12, "permutation, separated sets"

    # Variance components recover known values in a large balanced design
    a_, b_, c_ = 400, 4, 5
    y = (rng.normal(0, 2.0, (a_, 1, 1)) + rng.normal(0, 1.0, (a_, b_, 1)) + rng.normal(0, 0.5, (a_, b_, c_)))
    _, comps = VarianceComponents.nested_anova(y)
    assert abs(comps[0] - 4.0) < 1.5 and abs(comps[1] - 1.0) < 0.3 and abs(comps[2] - 0.25) < 0.05, comps
    if verbose:
        print("All self-tests passed.")
