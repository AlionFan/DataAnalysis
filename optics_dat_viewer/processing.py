from __future__ import annotations

from dataclasses import dataclass

import numpy as np
try:
    from scipy.signal import find_peaks, peak_widths, savgol_filter
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
try:
    from scipy.optimize import curve_fit
except Exception:
    curve_fit = None


@dataclass
class PeakResult:
    peak_indices: np.ndarray
    peak_x: np.ndarray
    peak_y: np.ndarray
    fwhm: np.ndarray


@dataclass
class GaussianFitResult:
    amplitude: float
    center: float
    sigma: float
    offset: float
    fitted_y: np.ndarray
    success: bool


@dataclass
class QualityReport:
    nan_ratio: float
    zero_ratio: float
    saturated_ratio: float
    outlier_ratio: float
    std_dev: float
    mean_val: float


def smooth_signal(y: np.ndarray, window_length: int, polyorder: int = 2) -> np.ndarray:
    valid = np.where(np.isfinite(y), y, np.nanmedian(y))
    if window_length < 3 or window_length % 2 == 0:
        return valid
    if window_length >= len(valid):
        return valid
    if SCIPY_AVAILABLE:
        return savgol_filter(valid, window_length=window_length, polyorder=polyorder)
    kernel = np.ones(window_length) / window_length
    return np.convolve(valid, kernel, mode="same")


def normalize_minmax(y: np.ndarray) -> np.ndarray:
    y_min = np.nanmin(y)
    y_max = np.nanmax(y)
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max == y_min:
        return y.copy()
    return (y - y_min) / (y_max - y_min)


def crop_roi_2d(
    matrix: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_mask = (x_coords >= x_min) & (x_coords <= x_max)
    y_mask = (y_coords >= y_min) & (y_coords <= y_max)
    cropped = matrix[np.ix_(y_mask, x_mask)]
    return cropped, x_coords[x_mask], y_coords[y_mask]


def detect_peaks_with_fwhm(
    x: np.ndarray,
    y: np.ndarray,
    prominence: float,
    distance: int,
) -> PeakResult:
    signal = np.where(np.isfinite(y), y, np.nanmedian(y))
    if SCIPY_AVAILABLE:
        peaks, _ = find_peaks(signal, prominence=prominence, distance=max(1, distance))
    else:
        peaks = _find_peaks_fallback(signal, prominence=prominence, distance=max(1, distance))
    if len(peaks) == 0:
        return PeakResult(
            peak_indices=np.array([], dtype=int),
            peak_x=np.array([]),
            peak_y=np.array([]),
            fwhm=np.array([]),
        )

    if SCIPY_AVAILABLE:
        widths, _, left_ips, right_ips = peak_widths(signal, peaks, rel_height=0.5)
        fwhm = np.interp(right_ips, np.arange(len(x)), x) - np.interp(
            left_ips, np.arange(len(x)), x
        )
    else:
        widths = np.array([_estimate_fwhm_fallback(x, signal, p) for p in peaks])
        fwhm = widths
    return PeakResult(
        peak_indices=peaks,
        peak_x=x[peaks],
        peak_y=signal[peaks],
        fwhm=fwhm if len(fwhm) == len(widths) else widths,
    )


def _find_peaks_fallback(signal: np.ndarray, prominence: float, distance: int) -> np.ndarray:
    candidate = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] >= signal[i + 1]:
            local_base = min(signal[i - 1], signal[i + 1])
            if signal[i] - local_base >= prominence:
                candidate.append(i)
    if not candidate:
        return np.array([], dtype=int)

    filtered = [candidate[0]]
    for idx in candidate[1:]:
        if idx - filtered[-1] >= distance:
            filtered.append(idx)
        elif signal[idx] > signal[filtered[-1]]:
            filtered[-1] = idx
    return np.array(filtered, dtype=int)


def _estimate_fwhm_fallback(x: np.ndarray, y: np.ndarray, peak_idx: int) -> float:
    half = y[peak_idx] * 0.5
    left = peak_idx
    right = peak_idx
    while left > 0 and y[left] > half:
        left -= 1
    while right < len(y) - 1 and y[right] > half:
        right += 1
    return float(x[right] - x[left]) if right > left else 0.0


def integrate_range(x: np.ndarray, y: np.ndarray, x_min: float, x_max: float) -> float:
    mask = (x >= x_min) & (x <= x_max)
    if mask.sum() < 2:
        return float("nan")
    ys = np.where(np.isfinite(y[mask]), y[mask], 0.0)
    return float(np.trapezoid(ys, x[mask]))


def extract_profile(matrix: np.ndarray, axis: str, index: int) -> np.ndarray:
    if axis == "x":
        return matrix[index, :]
    return matrix[:, index]


def gaussian_model(x: np.ndarray, amplitude: float, center: float, sigma: float, offset: float) -> np.ndarray:
    sigma = max(abs(sigma), 1e-9)
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) + offset


def fit_gaussian(x: np.ndarray, y: np.ndarray) -> GaussianFitResult:
    signal = np.where(np.isfinite(y), y, np.nanmedian(y))
    if len(x) < 4:
        return GaussianFitResult(0.0, float(x[0]) if len(x) else 0.0, 0.0, 0.0, signal, False)

    amp0 = float(np.nanmax(signal) - np.nanmin(signal))
    center0 = float(x[int(np.nanargmax(signal))])
    sigma0 = float((x[-1] - x[0]) / 10.0 if x[-1] != x[0] else 1.0)
    offset0 = float(np.nanmin(signal))

    if curve_fit is None:
        fitted = gaussian_model(x, amp0, center0, sigma0, offset0)
        return GaussianFitResult(amp0, center0, sigma0, offset0, fitted, False)

    try:
        popt, _ = curve_fit(
            gaussian_model,
            x,
            signal,
            p0=[amp0, center0, sigma0, offset0],
            maxfev=10000,
        )
        fitted = gaussian_model(x, *popt)
        return GaussianFitResult(float(popt[0]), float(popt[1]), float(abs(popt[2])), float(popt[3]), fitted, True)
    except Exception:
        fitted = gaussian_model(x, amp0, center0, sigma0, offset0)
        return GaussianFitResult(amp0, center0, sigma0, offset0, fitted, False)


def compute_quality_report(matrix: np.ndarray, saturation_quantile: float = 0.999) -> QualityReport:
    arr = matrix[np.isfinite(matrix)]
    if arr.size == 0:
        return QualityReport(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    nan_ratio = 1.0 - (arr.size / matrix.size)
    zero_ratio = float(np.mean(arr == 0))
    sat_threshold = float(np.quantile(arr, saturation_quantile))
    saturated_ratio = float(np.mean(arr >= sat_threshold))
    mean_val = float(np.mean(arr))
    std_dev = float(np.std(arr))
    if std_dev == 0:
        outlier_ratio = 0.0
    else:
        z = np.abs((arr - mean_val) / std_dev)
        outlier_ratio = float(np.mean(z > 3))
    return QualityReport(
        nan_ratio=float(nan_ratio),
        zero_ratio=zero_ratio,
        saturated_ratio=saturated_ratio,
        outlier_ratio=outlier_ratio,
        std_dev=std_dev,
        mean_val=mean_val,
    )


def radial_bin_stats(
    matrix: np.ndarray,
    center_row: float,
    center_col: float,
    offset_row: float = 0.0,
    offset_col: float = 0.0,
    bins: int = 10,
) -> np.ndarray:
    rows, cols = matrix.shape
    rr, cc = np.indices((rows, cols))
    rr = rr - offset_row
    cc = cc - offset_col
    radial = np.sqrt((rr - center_row) ** 2 + (cc - center_col) ** 2)
    valid = np.isfinite(matrix)
    r = radial[valid]
    v = matrix[valid]
    if r.size == 0:
        return np.empty((0, 5))
    edges = np.linspace(float(r.min()), float(r.max()), bins + 1)
    out = []
    for i in range(bins):
        mask = (r >= edges[i]) & (r < edges[i + 1] if i < bins - 1 else r <= edges[i + 1])
        if not np.any(mask):
            out.append([edges[i], edges[i + 1], np.nan, np.nan, 0])
            continue
        vals = v[mask]
        out.append([edges[i], edges[i + 1], float(np.mean(vals)), float(np.sum(vals)), int(vals.size)])
    return np.array(out, dtype=float)
