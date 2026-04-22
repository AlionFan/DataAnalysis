from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from .dat_parser import DatGrid, dat_to_csv_bytes, load_dat_from_path, load_dat_from_upload
    from .processing import (
        PeakResult,
        compute_quality_report,
        crop_roi_2d,
        detect_peaks_with_fwhm,
        extract_profile,
        fit_gaussian,
        integrate_range,
        normalize_minmax,
        radial_bin_stats,
        smooth_signal,
    )
except ImportError:
    from dat_parser import DatGrid, dat_to_csv_bytes, load_dat_from_path, load_dat_from_upload
    from processing import (
        PeakResult,
        compute_quality_report,
        crop_roi_2d,
        detect_peaks_with_fwhm,
        extract_profile,
        fit_gaussian,
        integrate_range,
        normalize_minmax,
        radial_bin_stats,
        smooth_signal,
    )

st.set_page_config(page_title="Optical DAT Viewer", layout="wide")
st.title("光学 .dat 可视化与分析工具")
st.caption("作者：胡一凡 | 邮箱：h1317483655@gmail.com | 有问题请通过邮箱反馈。")
WORKFLOW_DIR = Path("optics_dat_viewer/workflows")
WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)


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


def apply_visual_transform(data: np.ndarray, log_scale: bool) -> np.ndarray:
    transformed = data.copy()
    if log_scale:
        transformed = np.log1p(np.clip(transformed, a_min=0, a_max=None))
    return transformed


def plot_contour_overlay(grid: DatGrid, levels: int) -> go.Figure:
    fig = go.Figure(
        data=go.Contour(
            z=grid.data,
            x=grid.x_coords,
            y=grid.y_coords,
            ncontours=levels,
            colorscale="Viridis",
        )
    )
    fig.update_layout(title="等高线图", xaxis_title="X", yaxis_title="Y", height=500)
    return fig


def save_workflow(workflow_name: str, payload: dict) -> None:
    path = WORKFLOW_DIR / f"{workflow_name}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_workflow(workflow_name: str) -> dict:
    path = WORKFLOW_DIR / f"{workflow_name}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_workflow_names() -> list[str]:
    return sorted([p.stem for p in WORKFLOW_DIR.glob("*.json")])


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


def batch_process(folder_path: str) -> pd.DataFrame:
    folder = Path(folder_path)
    rows = []
    if not folder.exists():
        return pd.DataFrame()
    for file in sorted(folder.glob("*.dat")):
        try:
            g = load_dat_from_path(file)
            vals = g.data[np.isfinite(g.data)]
            rows.append(
                {
                    "file": file.name,
                    "rows": g.rows,
                    "cols": g.cols,
                    "mean": float(np.mean(vals)) if vals.size else np.nan,
                    "max": float(np.max(vals)) if vals.size else np.nan,
                    "sum": float(np.sum(vals)) if vals.size else np.nan,
                }
            )
        except Exception as exc:
            rows.append({"file": file.name, "error": str(exc)})
    return pd.DataFrame(rows)


def generate_html_report(quality_df: pd.DataFrame, roi_df: pd.DataFrame, radial_df: pd.DataFrame) -> str:
    return f"""
<html><head><meta charset='utf-8'><title>Optics Report</title></head><body>
<h1>光学数据分析报告</h1>
<h2>质量诊断</h2>{quality_df.to_html(index=False)}
<h2>多ROI统计</h2>{roi_df.to_html(index=False)}
<h2>径向分箱统计</h2>{radial_df.to_html(index=False)}
</body></html>
"""


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

    mode = st.radio(
        "选择可视化模式",
        ["2D 热力图", "3D 表面图", "剖面图分析", "多文件对比", "时序对比"],
        horizontal=True,
    )
    fig = None
    peak_export = pd.DataFrame()

    if mode == "2D 热力图":
        c1, c2 = st.columns(2)
        with c1:
            log_scale = st.checkbox("对数色阶(log1p)", value=False)
        with c2:
            show_contour = st.checkbox("叠加等高线", value=False)
        vis_grid = DatGrid(
            rows=cropped_grid.rows,
            cols=cropped_grid.cols,
            hole_value=cropped_grid.hole_value,
            row_delta=cropped_grid.row_delta,
            col_delta=cropped_grid.col_delta,
            row_origin=cropped_grid.row_origin,
            col_origin=cropped_grid.col_origin,
            data=apply_visual_transform(cropped_grid.data, log_scale),
        )
        fig = plot_heatmap(vis_grid, "ROI 热力图")
        st.plotly_chart(fig, use_container_width=True)
        if show_contour:
            contour_levels = st.slider("等高线层数", 5, 50, 20)
            st.plotly_chart(plot_contour_overlay(vis_grid, contour_levels), use_container_width=True)
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
        do_fit = st.checkbox("执行高斯拟合", value=True)
        if do_fit:
            fit_res = fit_gaussian(x, np.where(np.isfinite(y), y, np.nanmedian(y)))
            fig.add_trace(go.Scatter(x=x, y=fit_res.fitted_y, mode="lines", name="Gaussian Fit"))
            st.write(
                {
                    "fit_success": fit_res.success,
                    "amplitude": fit_res.amplitude,
                    "center": fit_res.center,
                    "sigma": fit_res.sigma,
                    "offset": fit_res.offset,
                }
            )
        st.plotly_chart(fig, use_container_width=True)
        st.metric("积分面积", f"{area:.6g}" if np.isfinite(area) else "NaN")
        peak_export = pd.DataFrame({"peak_x": peaks.peak_x, "peak_y": peaks.peak_y, "fwhm": peaks.fwhm})
        st.dataframe(peak_export, use_container_width=True)
    elif mode == "多文件对比":
        axis = st.selectbox("对比方向", ["x", "y"])
        max_idx = min(g.rows if axis == "x" else g.cols for g in grids) - 1
        idx = st.slider("剖面索引", 0, max(0, max_idx), min(10, max(0, max_idx)))
        fig = compare_profiles(grids, axis=axis, index=idx)
        st.plotly_chart(fig, use_container_width=True)
    else:
        if len(grids) < 2:
            st.info("时序对比至少需要两个文件（上传多文件即可）。")
        else:
            frame = st.slider("帧序号", 0, len(grids) - 1, 0)
            frame_grid = grids[frame]
            st.plotly_chart(plot_heatmap(frame_grid, f"Frame {frame + 1}"), use_container_width=True)
            if frame > 0:
                prev = grids[frame - 1]
                diff = frame_grid.data - prev.data
                diff_grid = DatGrid(
                    rows=frame_grid.rows,
                    cols=frame_grid.cols,
                    hole_value=frame_grid.hole_value,
                    row_delta=frame_grid.row_delta,
                    col_delta=frame_grid.col_delta,
                    row_origin=frame_grid.row_origin,
                    col_origin=frame_grid.col_origin,
                    data=diff,
                )
                st.plotly_chart(plot_heatmap(diff_grid, "与前一帧差分图"), use_container_width=True)

    st.subheader("数据质量诊断")
    qr = compute_quality_report(cropped_grid.data)
    quality_df = pd.DataFrame(
        [
            {
                "nan_ratio": qr.nan_ratio,
                "zero_ratio": qr.zero_ratio,
                "saturated_ratio": qr.saturated_ratio,
                "outlier_ratio": qr.outlier_ratio,
                "std_dev": qr.std_dev,
                "mean": qr.mean_val,
            }
        ]
    )
    st.dataframe(quality_df, use_container_width=True, hide_index=True)

    roi_df = multi_roi_stats(cropped_grid)
    st.dataframe(roi_df, use_container_width=True, hide_index=True)

    st.subheader("径向分箱统计（支持中心平移）")
    c1, c2, c3 = st.columns(3)
    with c1:
        center_row = st.number_input("center_row", value=50.5)
    with c2:
        center_col = st.number_input("center_col", value=50.5)
    with c3:
        bins = st.slider("分箱数量", 4, 40, 10)
    use_offset = st.checkbox("先做坐标平移(减去offset)", value=True)
    if use_offset:
        o1, o2 = st.columns(2)
        with o1:
            offset_row = st.number_input("offset_row", value=1.0)
        with o2:
            offset_col = st.number_input("offset_col", value=1.0)
    else:
        offset_row = 0.0
        offset_col = 0.0
    radial_raw = radial_bin_stats(
        cropped_grid.data,
        center_row=center_row,
        center_col=center_col,
        offset_row=offset_row,
        offset_col=offset_col,
        bins=bins,
    )
    radial_df = pd.DataFrame(radial_raw, columns=["r_min", "r_max", "mean", "sum", "count"])
    st.dataframe(radial_df, use_container_width=True, hide_index=True)

    st.subheader("流程模板（参数保存/加载）")
    workflow_name = st.text_input("模板名", value="default_workflow")
    payload = {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "center_row": center_row,
        "center_col": center_col,
        "offset_row": offset_row,
        "offset_col": offset_col,
        "bins": bins,
    }
    wf_col1, wf_col2 = st.columns(2)
    with wf_col1:
        if st.button("保存当前模板"):
            save_workflow(workflow_name, payload)
            st.success("模板已保存")
    with wf_col2:
        names = get_workflow_names()
        selected = st.selectbox("加载模板", [""] + names)
        if selected:
            st.json(load_workflow(selected))

    st.subheader("批量处理")
    batch_path = st.text_input("批处理目录（扫描所有 .dat）", value="")
    batch_df = pd.DataFrame()
    if batch_path.strip():
        batch_df = batch_process(batch_path.strip())
        if batch_df.empty:
            st.warning("目录不存在或无可处理文件。")
        else:
            st.dataframe(batch_df, use_container_width=True, hide_index=True)
            st.download_button(
                "导出批处理汇总CSV",
                data=batch_df.to_csv(index=False).encode("utf-8"),
                file_name="batch_summary.csv",
                mime="text/csv",
            )

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
    html_report = generate_html_report(quality_df, roi_df, radial_df)
    st.download_button(
        "导出分析报告 HTML",
        data=html_report.encode("utf-8"),
        file_name="analysis_report.html",
        mime="text/html",
    )


if __name__ == "__main__":
    main()
