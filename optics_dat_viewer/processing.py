from __future__ import annotations

from dataclasses import dataclass

import numpy as np
try:
    from scipy.signal import find_peaks, peak_widths, savgol_filter
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


@dataclass
class PeakResult:
    peak_indices: np.ndarray
    peak_x: np.ndarray
    peak_y: np.ndarray
    fwhm: np.ndarray


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
