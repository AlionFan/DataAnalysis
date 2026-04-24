from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from .dat_parser import DatGrid, load_dat_from_path, load_dat_from_upload
    from .processing import (
        PeakResult,
        crop_roi_2d,
        detect_peaks_with_fwhm,
        extract_profile,
        integrate_range,
        normalize_minmax,
        smooth_signal,
    )
except ImportError:
    from dat_parser import DatGrid, load_dat_from_path, load_dat_from_upload
    from processing import (
        PeakResult,
        crop_roi_2d,
        detect_peaks_with_fwhm,
        extract_profile,
        integrate_range,
        normalize_minmax,
        smooth_signal,
    )


# ── 数据变换辅助 ──────────────────────────────

def preprocess_profile(y: np.ndarray, do_smooth: bool, do_norm: bool, window: int) -> np.ndarray:
    processed = y.copy()
    if do_smooth:
        processed = smooth_signal(processed, window_length=window)
    if do_norm:
        processed = normalize_minmax(processed)
    return processed


def to_angle_axis(length: int, start_deg: float = 0.0, end_deg: float = 4.0) -> np.ndarray:
    if length <= 1:
        return np.array([start_deg], dtype=float)
    return np.linspace(start_deg, end_deg, length)


def remap_grid_to_angle(grid: DatGrid) -> DatGrid:
    col_delta = 4.0 / max(1, grid.cols - 1)
    row_delta = 4.0 / max(1, grid.rows - 1)
    return DatGrid(
        rows=grid.rows,
        cols=grid.cols,
        hole_value=grid.hole_value,
        row_delta=row_delta,
        col_delta=col_delta,
        row_origin=0.0,
        col_origin=0.0,
        data=grid.data,
    )


def radial_to_angle_axis(radial_raw: np.ndarray, end_deg: float = 4.0) -> np.ndarray:
    """将径向分箱结果的 r_min/r_max 从像素单位映射到 0~end_deg 角度。"""
    result = radial_raw.copy()
    r_max_val = float(result[:, 1].max()) if result.size > 0 else 1.0
    if r_max_val == 0:
        r_max_val = 1.0
    scale = end_deg / r_max_val
    result[:, 0] *= scale
    result[:, 1] *= scale
    return result


def align_curve_by_peak(x: np.ndarray, y: np.ndarray, target_peak_deg: float = 2.0) -> np.ndarray:
    if len(x) == 0:
        return x
    signal = np.where(np.isfinite(y), y, np.nanmedian(y))
    peak_idx = int(np.nanargmax(signal))
    peak_x = float(x[peak_idx])
    return x - peak_x + target_peak_deg


# ── 侧边栏：文件输入 ─────────────────────────

def choose_input_files() -> list[DatGrid]:
    st.sidebar.header("文件输入")
    uploaded = st.sidebar.file_uploader("上传 .dat 文件（可多选）", type=["dat"], accept_multiple_files=True)
    default_path = "/Users/fan/Downloads/guangyuan1000wan.dat"
    local_path = st.sidebar.text_input("或输入本地文件路径", value=default_path)

    grids: list[DatGrid] = []
    for uf in uploaded or []:
        grids.append(load_dat_from_upload(uf))

    if local_path.strip():
        path = Path(local_path.strip())
        if path.exists():
            grids.append(load_dat_from_path(path))
        else:
            st.sidebar.warning("本地路径不存在，将仅使用上传文件。")

    return grids


# ── 表格编辑 ──────────────────────────────────

def show_editable_table(grid: DatGrid) -> np.ndarray:
    st.subheader("表格信息与数据编辑")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("行数", f"{grid.rows}")
    c2.metric("列数", f"{grid.cols}")
    c3.metric("最小值", f"{np.nanmin(grid.data):.6g}" if np.isfinite(np.nanmin(grid.data)) else "NaN")
    c4.metric("最大值", f"{np.nanmax(grid.data):.6g}" if np.isfinite(np.nanmax(grid.data)) else "NaN")

    table_df = pd.DataFrame(grid.data)
    table_df.index = np.arange(grid.rows)
    table_df.columns = np.arange(grid.cols)
    table_df.index.name = "row_index"

    st.caption(
        "表格显示行号/列号索引，可直接修改单元格；修改会立即用于后续可视化、分析和导出。"
    )
    edited_df = st.data_editor(
        table_df,
        use_container_width=True,
        num_rows="fixed",
        key="roi_data_editor",
    )
    return edited_df.to_numpy(dtype=float)


# ── 剖面分析 ──────────────────────────────────

def show_profile_analysis(x: np.ndarray, y: np.ndarray, axis_label: str) -> tuple[go.Figure, PeakResult, float]:
    st.subheader(f"{axis_label} 剖面分析")
    col1, col2, col3 = st.columns(3)
    with col1:
        do_smooth = st.checkbox("平滑", value=True, key=f"smooth_{axis_label}")
    with col2:
        do_norm = st.checkbox("归一化", value=False, key=f"norm_{axis_label}")
    with col3:
        window = st.slider("平滑窗口(奇数)", 3, 41, 9, 2, key=f"window_{axis_label}")

    processed = preprocess_profile(y, do_smooth, do_norm, window)
    prom = st.slider("峰值显著性", 0.0, float(np.nanmax(processed) if np.nanmax(processed) > 0 else 1.0), 0.05, key=f"prom_{axis_label}")
    dist = st.slider("峰间最小间距", 1, max(2, len(processed) // 5), 5, key=f"dist_{axis_label}")
    peak_res = detect_peaks_with_fwhm(x, processed, prominence=prom, distance=dist)

    integral_min, integral_max = st.select_slider(
        "积分区间",
        options=list(x),
        value=(float(x[0]), float(x[-1])),
        key=f"int_{axis_label}",
    )
    area = integrate_range(x, processed, integral_min, integral_max)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=processed, mode="lines", name=f"{axis_label} profile"))
    if len(peak_res.peak_x) > 0:
        fig.add_trace(
            go.Scatter(
                x=peak_res.peak_x,
                y=peak_res.peak_y,
                mode="markers",
                marker={"size": 9, "color": "red"},
                name="Peaks",
            )
        )
    fig.update_layout(xaxis_title=axis_label, yaxis_title="Intensity", height=420)
    return fig, peak_res, area


# ── 多文件剖面对比 ────────────────────────────

def compare_profiles(
    grids: list[DatGrid],
    axis: str,
    index: int,
    use_angle_axis: bool = True,
    do_align: bool = False,
    target_peak_deg: float = 2.0,
) -> go.Figure:
    fig = go.Figure()
    for idx, grid in enumerate(grids):
        if axis == "x":
            x = grid.x_coords
            y = extract_profile(grid.data, "x", min(index, grid.rows - 1))
            label = f"File {idx + 1} - row {min(index, grid.rows - 1)}"
        else:
            x = grid.y_coords
            y = extract_profile(grid.data, "y", min(index, grid.cols - 1))
            label = f"File {idx + 1} - col {min(index, grid.cols - 1)}"
        y = normalize_minmax(np.where(np.isfinite(y), y, 0.0))
        if use_angle_axis:
            x = to_angle_axis(len(y), 0.0, 4.0)
        if do_align:
            x = align_curve_by_peak(x, y, target_peak_deg=target_peak_deg)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=label))
    x_label = "Angle (deg)" if use_angle_axis else ("X" if axis == "x" else "Y")
    fig.update_layout(height=420, title="多文件剖面对比（归一化）", xaxis_title=x_label, yaxis_title="Intensity")
    return fig


# ── 多 ROI 区域统计 ───────────────────────────

def multi_roi_stats(grid: DatGrid) -> pd.DataFrame:
    st.subheader("多 ROI 区域统计")
    roi_count = st.number_input("ROI 数量", min_value=1, max_value=6, value=2, step=1)
    records: list[dict] = []
    for i in range(int(roi_count)):
        st.markdown(f"**ROI {i + 1}**")
        c1, c2 = st.columns(2)
        with c1:
            x_range = st.slider(
                f"ROI{i+1} X范围",
                float(grid.x_coords[0]),
                float(grid.x_coords[-1]),
                (float(grid.x_coords[0]), float(grid.x_coords[-1])),
                key=f"roi_x_{i}",
            )
        with c2:
            y_range = st.slider(
                f"ROI{i+1} Y范围",
                float(grid.y_coords[0]),
                float(grid.y_coords[-1]),
                (float(grid.y_coords[0]), float(grid.y_coords[-1])),
                key=f"roi_y_{i}",
            )
        sub, _, _ = crop_roi_2d(
            grid.data, grid.x_coords, grid.y_coords, x_range[0], x_range[1], y_range[0], y_range[1]
        )
        vals = sub[np.isfinite(sub)]
        if vals.size == 0:
            records.append({"roi": i + 1, "mean": np.nan, "std": np.nan, "sum": np.nan, "max": np.nan})
        else:
            records.append(
                {
                    "roi": i + 1,
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "sum": float(np.sum(vals)),
                    "max": float(np.max(vals)),
                }
            )
    return pd.DataFrame(records)
