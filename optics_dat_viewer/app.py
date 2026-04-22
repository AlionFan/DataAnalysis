from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dat_parser import DatGrid, dat_to_csv_bytes, load_dat_from_path, load_dat_from_upload
from processing import (
    PeakResult,
    crop_roi_2d,
    detect_peaks_with_fwhm,
    extract_profile,
    integrate_range,
    normalize_minmax,
    smooth_signal,
)

st.set_page_config(page_title="Optical DAT Viewer", layout="wide")
st.title("光学 .dat 可视化与分析工具")


def plot_heatmap(grid: DatGrid, title: str) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=grid.data,
            x=grid.x_coords,
            y=grid.y_coords,
            colorscale="Viridis",
            colorbar={"title": "Intensity"},
        )
    )
    fig.update_layout(title=title, xaxis_title="X", yaxis_title="Y")
    return fig


def plot_surface(grid: DatGrid, title: str) -> go.Figure:
    fig = go.Figure(
        data=go.Surface(z=grid.data, x=grid.x_coords, y=grid.y_coords, colorscale="Viridis")
    )
    fig.update_layout(title=title, scene={"xaxis_title": "X", "yaxis_title": "Y", "zaxis_title": "I"})
    return fig


def export_plot_png(fig: go.Figure) -> bytes:
    buffer = BytesIO()
    fig.write_image(buffer, format="png")
    return buffer.getvalue()


def preprocess_profile(y: np.ndarray, do_smooth: bool, do_norm: bool, window: int) -> np.ndarray:
    processed = y.copy()
    if do_smooth:
        processed = smooth_signal(processed, window_length=window)
    if do_norm:
        processed = normalize_minmax(processed)
    return processed


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


def compare_profiles(grids: list[DatGrid], axis: str, index: int) -> go.Figure:
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
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=label))
    fig.update_layout(height=420, title="多文件剖面对比（归一化）")
    return fig


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
    with st.expander("查看索引与物理坐标对应关系"):
        mapping_df = pd.DataFrame(
            {
                "row_index": np.arange(grid.rows),
                "y_coord": grid.y_coords,
            }
        )
        st.dataframe(mapping_df, use_container_width=True, hide_index=True)
        col_mapping_df = pd.DataFrame(
            {
                "col_index": np.arange(grid.cols),
                "x_coord": grid.x_coords,
            }
        )
        st.dataframe(col_mapping_df, use_container_width=True, hide_index=True)
    edited_df = st.data_editor(
        table_df,
        use_container_width=True,
        num_rows="fixed",
        key="roi_data_editor",
    )
    return edited_df.to_numpy(dtype=float)


def main() -> None:
    try:
        grids = choose_input_files()
    except Exception as exc:
        st.error(f"文件解析失败：{exc}")
        return

    if not grids:
        st.info("请上传或指定至少一个 .dat 文件。")
        return

    grid = grids[0]
    st.sidebar.header("ROI 选择")
    x_min, x_max = st.sidebar.slider("X 范围", float(grid.x_coords[0]), float(grid.x_coords[-1]), (float(grid.x_coords[0]), float(grid.x_coords[-1])))
    y_min, y_max = st.sidebar.slider("Y 范围", float(grid.y_coords[0]), float(grid.y_coords[-1]), (float(grid.y_coords[0]), float(grid.y_coords[-1])))

    cropped_data, cropped_x, cropped_y = crop_roi_2d(
        grid.data, grid.x_coords, grid.y_coords, x_min, x_max, y_min, y_max
    )
    cropped_grid = DatGrid(
        rows=len(cropped_y),
        cols=len(cropped_x),
        hole_value=grid.hole_value,
        row_delta=grid.row_delta,
        col_delta=grid.col_delta,
        row_origin=float(cropped_y[0]) if len(cropped_y) else grid.row_origin,
        col_origin=float(cropped_x[0]) if len(cropped_x) else grid.col_origin,
        data=cropped_data,
    )
    edited_data = show_editable_table(cropped_grid)
    cropped_grid = DatGrid(
        rows=cropped_grid.rows,
        cols=cropped_grid.cols,
        hole_value=cropped_grid.hole_value,
        row_delta=cropped_grid.row_delta,
        col_delta=cropped_grid.col_delta,
        row_origin=cropped_grid.row_origin,
        col_origin=cropped_grid.col_origin,
        data=edited_data,
    )

    mode = st.radio("选择可视化模式", ["2D 热力图", "3D 表面图", "剖面图分析", "多文件对比"], horizontal=True)
    fig = None
    peak_export = pd.DataFrame()

    if mode == "2D 热力图":
        fig = plot_heatmap(cropped_grid, "ROI 热力图")
        st.plotly_chart(fig, use_container_width=True)
    elif mode == "3D 表面图":
        fig = plot_surface(cropped_grid, "ROI 3D 表面")
        st.plotly_chart(fig, use_container_width=True)
    elif mode == "剖面图分析":
        axis = st.selectbox("剖面方向", ["x", "y"])
        if axis == "x":
            row_idx = st.slider("选择行索引", 0, max(0, cropped_grid.rows - 1), min(10, max(0, cropped_grid.rows - 1)))
            x = cropped_grid.x_coords
            y = extract_profile(cropped_grid.data, axis="x", index=row_idx)
        else:
            col_idx = st.slider("选择列索引", 0, max(0, cropped_grid.cols - 1), min(10, max(0, cropped_grid.cols - 1)))
            x = cropped_grid.y_coords
            y = extract_profile(cropped_grid.data, axis="y", index=col_idx)

        fig, peaks, area = show_profile_analysis(x, y, axis_label=axis.upper())
        st.plotly_chart(fig, use_container_width=True)
        st.metric("积分面积", f"{area:.6g}" if np.isfinite(area) else "NaN")
        peak_export = pd.DataFrame({"peak_x": peaks.peak_x, "peak_y": peaks.peak_y, "fwhm": peaks.fwhm})
        st.dataframe(peak_export, use_container_width=True)
    else:
        axis = st.selectbox("对比方向", ["x", "y"])
        max_idx = min(g.rows if axis == "x" else g.cols for g in grids) - 1
        idx = st.slider("剖面索引", 0, max(0, max_idx), min(10, max(0, max_idx)))
        fig = compare_profiles(grids, axis=axis, index=idx)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("导出")
    st.download_button(
        "导出当前 ROI 矩阵 CSV",
        data=dat_to_csv_bytes(cropped_grid.data),
        file_name="roi_data.csv",
        mime="text/csv",
    )
    if not peak_export.empty:
        st.download_button(
            "导出峰值分析 CSV",
            data=peak_export.to_csv(index=False).encode("utf-8"),
            file_name="peaks_summary.csv",
            mime="text/csv",
        )
        st.download_button(
            "导出峰值分析 JSON",
            data=peak_export.to_json(orient="records", force_ascii=False).encode("utf-8"),
            file_name="peaks_summary.json",
            mime="application/json",
        )
    if fig is not None:
        try:
            st.download_button(
                "导出当前图像 PNG",
                data=export_plot_png(fig),
                file_name="figure.png",
                mime="image/png",
            )
        except Exception:
            st.caption("导出 PNG 需要安装 kaleido：`pip install kaleido`。")


if __name__ == "__main__":
    main()
