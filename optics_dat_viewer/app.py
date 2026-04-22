from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from .dat_parser import DatGrid, dat_to_csv_bytes
    from .processing import (
        compute_quality_report,
        crop_roi_2d,
        extract_profile,
        fft2d_filter,
        fft_analysis_1d,
        fit_gaussian,
        fit_multi_gaussian,
        radial_bin_stats,
        repair_outlier_point,
    )
    from .plots import apply_visual_transform, export_plot_png, plot_contour_overlay, plot_heatmap, plot_surface
    from .ui import (
        align_curve_by_peak,
        choose_input_files,
        compare_profiles,
        multi_roi_stats,
        remap_grid_to_angle,
        show_editable_table,
        show_profile_analysis,
        to_angle_axis,
    )
    from .llm import call_siliconflow_llm
    from .reports import (
        batch_process,
        generate_html_report,
        generate_markdown_report,
        get_workflow_names,
        load_workflow,
        save_workflow,
    )
except ImportError:
    from dat_parser import DatGrid, dat_to_csv_bytes
    from processing import (
        compute_quality_report,
        crop_roi_2d,
        extract_profile,
        fft2d_filter,
        fft_analysis_1d,
        fit_gaussian,
        fit_multi_gaussian,
        radial_bin_stats,
        repair_outlier_point,
    )
    from plots import apply_visual_transform, export_plot_png, plot_contour_overlay, plot_heatmap, plot_surface
    from ui import (
        align_curve_by_peak,
        choose_input_files,
        compare_profiles,
        multi_roi_stats,
        remap_grid_to_angle,
        show_editable_table,
        show_profile_analysis,
        to_angle_axis,
    )
    from llm import call_siliconflow_llm
    from reports import (
        batch_process,
        generate_html_report,
        generate_markdown_report,
        get_workflow_names,
        load_workflow,
        save_workflow,
    )

st.set_page_config(page_title="Optical DAT Viewer", layout="wide")
st.title("光学 .dat 可视化与分析工具")
st.caption("作者：胡一凡 | 邮箱：h1317483655@gmail.com | 有问题请通过邮箱反馈。")


# ══════════════════════════════════════════════
# 综合分析：各可视化模式
# ══════════════════════════════════════════════

def _render_heatmap_mode(cropped_grid: DatGrid) -> go.Figure:
    c1, c2 = st.columns(2)
    with c1:
        log_scale = st.checkbox("对数色阶(log1p)", value=False)
    with c2:
        show_contour = st.checkbox("叠加等高线", value=False)

    st.markdown("**标注工具**")
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        ann_x = st.number_input("标注X", value=float(cropped_grid.x_coords[0]))
    with a2:
        ann_y = st.number_input("标注Y", value=float(cropped_grid.y_coords[0]))
    with a3:
        ann_text = st.text_input("标注文本", value="")
    with a4:
        if st.button("添加标注"):
            st.session_state.setdefault("annotations", [])
            st.session_state["annotations"].append({"x": ann_x, "y": ann_y, "text": ann_text})

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
    anns = st.session_state.get("annotations", [])
    if anns:
        fig.add_trace(
            go.Scatter(
                x=[a["x"] for a in anns],
                y=[a["y"] for a in anns],
                mode="markers+text",
                text=[a["text"] for a in anns],
                textposition="top center",
                marker={"size": 8, "color": "white"},
                name="Annotations",
            )
        )
    st.plotly_chart(fig, use_container_width=True)
    if show_contour:
        contour_levels = st.slider("等高线层数", 5, 50, 20)
        st.plotly_chart(plot_contour_overlay(vis_grid, contour_levels), use_container_width=True)
    return fig


def _render_surface_mode(cropped_grid: DatGrid) -> go.Figure:
    fig = plot_surface(cropped_grid, "ROI 3D 表面")
    st.plotly_chart(fig, use_container_width=True)
    return fig


def _render_profile_mode(cropped_grid: DatGrid) -> tuple[go.Figure, pd.DataFrame]:
    axis = st.selectbox("剖面方向", ["x", "y"])
    do_align = st.checkbox("按峰位对齐", value=False)
    target_peak_deg = st.slider("对齐目标角度(°)", 0.0, 4.0, 2.0, 0.05) if do_align else 2.0
    if axis == "x":
        row_idx = st.slider("选择行索引", 0, max(0, cropped_grid.rows - 1), min(10, max(0, cropped_grid.rows - 1)))
        x = cropped_grid.x_coords
        y = extract_profile(cropped_grid.data, axis="x", index=row_idx)
    else:
        col_idx = st.slider("选择列索引", 0, max(0, cropped_grid.cols - 1), min(10, max(0, cropped_grid.cols - 1)))
        x = cropped_grid.y_coords
        y = extract_profile(cropped_grid.data, axis="y", index=col_idx)
    x = to_angle_axis(len(y), 0.0, 4.0)
    if do_align:
        x = align_curve_by_peak(x, y, target_peak_deg=target_peak_deg)

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
        multi_fit = st.checkbox("多峰拟合增强")
        if multi_fit:
            n_peaks = st.slider("多峰数量", 2, 4, 2)
            params, fitted_multi, ok = fit_multi_gaussian(x, y, n_peaks=n_peaks)
            fig.add_trace(go.Scatter(x=x, y=fitted_multi, mode="lines", name="Multi-Gaussian"))
            st.write({"multi_fit_success": ok, "params": params.tolist() if len(params) else []})
    st.plotly_chart(fig, use_container_width=True)
    st.metric("积分面积", f"{area:.6g}" if np.isfinite(area) else "NaN")
    peak_export = pd.DataFrame({"peak_x": peaks.peak_x, "peak_y": peaks.peak_y, "fwhm": peaks.fwhm})
    st.dataframe(peak_export, use_container_width=True)

    # 傅立叶变换（FFT）分析
    st.subheader("傅立叶变换（FFT）分析")
    fft_enable = st.checkbox("启用 FFT 分析", value=True)
    if fft_enable:
        cutoff_ratio = st.slider("低通保留比例", 0.01, 1.0, 0.2, 0.01)
        fft_res = fft_analysis_1d(x, np.where(np.isfinite(y), y, np.nanmedian(y)), cutoff_ratio=cutoff_ratio)

        fft_fig = go.Figure()
        fft_fig.add_trace(
            go.Scatter(x=fft_res.frequencies, y=fft_res.magnitude, mode="lines", name="FFT Magnitude")
        )
        fft_fig.update_layout(title="频谱幅度图", xaxis_title="Frequency", yaxis_title="Magnitude", height=360)
        st.plotly_chart(fft_fig, use_container_width=True)

        recon_fig = go.Figure()
        recon_fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Original"))
        recon_fig.add_trace(
            go.Scatter(x=x, y=fft_res.reconstructed, mode="lines", name="Low-pass Reconstructed")
        )
        recon_fig.update_layout(title="原始信号 vs 低通重建", xaxis_title=axis.upper(), yaxis_title="Intensity", height=360)
        st.plotly_chart(recon_fig, use_container_width=True)

    return fig, peak_export


def _render_compare_mode(grids: list[DatGrid]) -> go.Figure:
    axis = st.selectbox("对比方向", ["x", "y"])
    align_cmp = st.checkbox("对比曲线按峰位对齐", value=True)
    align_target_cmp = st.slider("对比对齐目标角度(°)", 0.0, 4.0, 2.0, 0.05) if align_cmp else 2.0
    max_idx = min(g.rows if axis == "x" else g.cols for g in grids) - 1
    idx = st.slider("剖面索引", 0, max(0, max_idx), min(10, max(0, max_idx)))
    fig = compare_profiles(grids, axis=axis, index=idx, use_angle_axis=True, do_align=align_cmp, target_peak_deg=align_target_cmp)
    st.plotly_chart(fig, use_container_width=True)
    return fig


def _render_timeseries_mode(grids: list[DatGrid]) -> go.Figure | None:
    if len(grids) < 2:
        st.info("时序对比至少需要两个文件（上传多文件即可）。")
        return None
    frame = st.slider("帧序号", 0, len(grids) - 1, 0)
    frame_grid = grids[frame]
    fig = plot_heatmap(frame_grid, f"Frame {frame + 1}")
    st.plotly_chart(fig, use_container_width=True)
    if frame > 0:
        prev = grids[frame - 1]
        diff = frame_grid.data - prev.data
        diff_grid = DatGrid(
            rows=frame_grid.rows, cols=frame_grid.cols, hole_value=frame_grid.hole_value,
            row_delta=frame_grid.row_delta, col_delta=frame_grid.col_delta,
            row_origin=frame_grid.row_origin, col_origin=frame_grid.col_origin, data=diff,
        )
        st.plotly_chart(plot_heatmap(diff_grid, "与前一帧差分图"), use_container_width=True)
    return fig


def _render_overview_page(cropped_grid: DatGrid, grids: list[DatGrid]) -> tuple[go.Figure | None, pd.DataFrame]:
    """综合分析：选择可视化模式并渲染"""
    mode = st.radio(
        "选择可视化模式",
        ["2D 热力图", "3D 表面图", "剖面图分析", "多文件对比", "时序对比"],
        horizontal=True,
    )
    if mode == "2D 热力图":
        return _render_heatmap_mode(cropped_grid), pd.DataFrame()
    elif mode == "3D 表面图":
        return _render_surface_mode(cropped_grid), pd.DataFrame()
    elif mode == "剖面图分析":
        return _render_profile_mode(cropped_grid)
    elif mode == "多文件对比":
        return _render_compare_mode(grids), pd.DataFrame()
    else:
        return _render_timeseries_mode(grids), pd.DataFrame()


# ══════════════════════════════════════════════
# 频域分析
# ══════════════════════════════════════════════

def _render_fft_page(cropped_grid: DatGrid) -> None:
    st.subheader("2D FFT 频域分析")
    fm = st.selectbox("滤波类型", ["lowpass", "highpass", "bandpass"])
    c1, c2 = st.columns(2)
    with c1:
        low_ratio = st.slider("低频阈值比例", 0.01, 1.0, 0.1, 0.01)
    with c2:
        high_ratio = st.slider("高频阈值比例", 0.01, 1.0, 0.4, 0.01)
    fft2 = fft2d_filter(cropped_grid.data, filter_mode=fm, low_ratio=low_ratio, high_ratio=high_ratio)

    mag_grid = DatGrid(
        rows=cropped_grid.rows, cols=cropped_grid.cols, hole_value=cropped_grid.hole_value,
        row_delta=cropped_grid.row_delta, col_delta=cropped_grid.col_delta,
        row_origin=cropped_grid.row_origin, col_origin=cropped_grid.col_origin,
        data=fft2.magnitude,
    )
    recon_grid = DatGrid(
        rows=cropped_grid.rows, cols=cropped_grid.cols, hole_value=cropped_grid.hole_value,
        row_delta=cropped_grid.row_delta, col_delta=cropped_grid.col_delta,
        row_origin=cropped_grid.row_origin, col_origin=cropped_grid.col_origin,
        data=fft2.filtered_reconstructed,
    )
    st.plotly_chart(plot_heatmap(mag_grid, "2D FFT 频谱幅度"), use_container_width=True)
    st.plotly_chart(plot_heatmap(recon_grid, "滤波后重建图"), use_container_width=True)


# ══════════════════════════════════════════════
# AI 大模型分析
# ══════════════════════════════════════════════

def _render_ai_page(quality_df, roi_df, radial_df, cropped_grid, peak_export, batch_df):
    st.subheader("SiliconFlow 大模型分析")
    st.caption("填写你自己的 SiliconFlow API Key 后可调用模型分析。")
    st.markdown("注册链接：[SiliconFlow 注册与API创建](https://cloud.siliconflow.cn/i/tOIjnzot)")
    api_key = st.text_input("SiliconFlow API Key", type="password", value=st.session_state.get("sf_api_key", ""))
    if api_key:
        st.session_state["sf_api_key"] = api_key
    model = st.selectbox(
        "选择模型",
        [
            "Pro/MiniMaxAI/MiniMax-M2.5",
            "Pro/moonshotai/Kimi-K2.6",
            "Pro/zai-org/GLM-5.1",
            "Pro/deepseek-ai/DeepSeek-V3.2",
        ],
    )
    system_prompt = st.text_area(
        "System Prompt",
        value=(
            "你是光学数据分析助手。请基于用户提供的统计指标，输出：1)关键现象，"
            "2)可能原因，3)下一步实验建议，4)参数调优建议。输出结构化中文。"
        ),
    )
    auto_context = {
        "quality": quality_df.to_dict(orient="records"),
        "roi": roi_df.to_dict(orient="records"),
        "radial": radial_df.head(20).to_dict(orient="records"),
    }
    llm_upload = st.file_uploader("上传补充分析文件（txt/csv/json）", type=["txt", "csv", "json"])
    uploaded_text = ""
    if llm_upload is not None:
        uploaded_text = llm_upload.getvalue().decode("utf-8", errors="ignore")[:5000]
    user_prompt = st.text_area(
        "User Prompt",
        value=(
            "请分析这组光学数据并给出建议："
            f"{json.dumps(auto_context, ensure_ascii=False)}\n"
            f"补充文件内容：{uploaded_text}"
        ),
        height=180,
    )
    if st.button("上传并进行大模型分析"):
        if not api_key.strip():
            st.error("请先输入 API Key")
        else:
            with st.spinner("模型分析中..."):
                parts = []
                parts.append("=== 质量诊断 ===")
                parts.append(quality_df.to_csv(index=False))
                parts.append("\n=== ROI 统计 ===")
                parts.append(roi_df.to_csv(index=False))
                parts.append("\n=== 径向统计（前20行） ===")
                parts.append(radial_df.head(20).to_csv(index=False))
                parts.append("\n=== ROI 数据矩阵 ===")
                parts.append(pd.DataFrame(cropped_grid.data).to_csv(index=False))
                if not peak_export.empty:
                    parts.append("\n=== 峰值分析 ===")
                    parts.append(peak_export.to_csv(index=False))
                if not batch_df.empty:
                    parts.append("\n=== 批处理汇总 ===")
                    parts.append(batch_df.head(20).to_csv(index=False))
                data_text = "\n".join(parts)
                result = call_siliconflow_llm(api_key.strip(), model, system_prompt, user_prompt, data_text=data_text)
            st.markdown("#### 模型分析结果")
            st.markdown(result)


# ══════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════

def main() -> None:
    # ── 1. 数据加载 ──
    try:
        grids = choose_input_files()
    except Exception as exc:
        st.error(f"文件解析失败：{exc}")
        return

    if not grids:
        st.info("请上传或指定至少一个 .dat 文件。")
        return
    grids = [remap_grid_to_angle(g) for g in grids]
    grid = grids[0]

    # ── 2. 侧边栏：导航 + ROI ──
    st.sidebar.header("导航目录")
    nav = st.sidebar.radio(
        "跳转到功能区",
        ["综合分析", "频域分析(2D FFT)", "AI大模型分析"],
        index=0,
    )
    st.sidebar.header("ROI 选择")
    x_min, x_max = st.sidebar.slider(
        "X 范围", float(grid.x_coords[0]), float(grid.x_coords[-1]),
        (float(grid.x_coords[0]), float(grid.x_coords[-1])),
    )
    y_min, y_max = st.sidebar.slider(
        "Y 范围", float(grid.y_coords[0]), float(grid.y_coords[-1]),
        (float(grid.y_coords[0]), float(grid.y_coords[-1])),
    )
    cropped_data, cropped_x, cropped_y = crop_roi_2d(
        grid.data, grid.x_coords, grid.y_coords, x_min, x_max, y_min, y_max
    )
    cropped_grid = DatGrid(
        rows=len(cropped_y), cols=len(cropped_x),
        hole_value=grid.hole_value,
        row_delta=grid.row_delta, col_delta=grid.col_delta,
        row_origin=float(cropped_y[0]) if len(cropped_y) else grid.row_origin,
        col_origin=float(cropped_x[0]) if len(cropped_x) else grid.col_origin,
        data=cropped_data,
    )

    # ── 3. 表格编辑 ──
    edited_data = show_editable_table(cropped_grid)
    cropped_grid = DatGrid(
        rows=cropped_grid.rows, cols=cropped_grid.cols,
        hole_value=cropped_grid.hole_value,
        row_delta=cropped_grid.row_delta, col_delta=cropped_grid.col_delta,
        row_origin=cropped_grid.row_origin, col_origin=cropped_grid.col_origin,
        data=edited_data,
    )

    # ── 4. 可视化（按导航分支） ──
    fig = None
    peak_export = pd.DataFrame()
    if nav == "综合分析":
        fig, peak_export = _render_overview_page(cropped_grid, grids)
    elif nav == "频域分析(2D FFT)":
        _render_fft_page(cropped_grid)

    # ── 5. 异常修复 ──
    st.subheader("异常点交互修复")
    r1, r2, r3 = st.columns(3)
    with r1:
        bad_row = st.number_input("异常点行号", min_value=0, max_value=max(0, cropped_grid.rows - 1), value=0)
    with r2:
        bad_col = st.number_input("异常点列号", min_value=0, max_value=max(0, cropped_grid.cols - 1), value=0)
    with r3:
        repair_method = st.selectbox("修复方式", ["median_3x3", "mean_3x3"])
    if st.button("执行异常点修复"):
        cropped_grid = DatGrid(
            rows=cropped_grid.rows, cols=cropped_grid.cols,
            hole_value=cropped_grid.hole_value,
            row_delta=cropped_grid.row_delta, col_delta=cropped_grid.col_delta,
            row_origin=cropped_grid.row_origin, col_origin=cropped_grid.col_origin,
            data=repair_outlier_point(cropped_grid.data, int(bad_row), int(bad_col), repair_method),
        )
        st.success("已修复指定异常点（当前会话内生效）")

    # ── 6. 数据质量与统计 ──
    st.subheader("数据质量诊断")
    qr = compute_quality_report(cropped_grid.data)
    quality_df = pd.DataFrame(
        [{
            "nan_ratio": qr.nan_ratio,
            "zero_ratio": qr.zero_ratio,
            "saturated_ratio": qr.saturated_ratio,
            "outlier_ratio": qr.outlier_ratio,
            "std_dev": qr.std_dev,
            "mean": qr.mean_val,
        }]
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
        center_row=center_row, center_col=center_col,
        offset_row=offset_row, offset_col=offset_col, bins=bins,
    )
    radial_df = pd.DataFrame(radial_raw, columns=["r_min", "r_max", "mean", "sum", "count"])
    st.dataframe(radial_df, use_container_width=True, hide_index=True)

    # ── 7. 流程模板 ──
    st.subheader("流程模板（参数保存/加载）")
    workflow_name = st.text_input("模板名", value="default_workflow")
    payload = {
        "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max,
        "center_row": center_row, "center_col": center_col,
        "offset_row": offset_row, "offset_col": offset_col, "bins": bins,
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

    # ── 8. 批量处理 ──
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
            st.subheader("实验批次对比看板")
            if {"mean", "max"}.issubset(set(batch_df.columns)):
                dash_fig = go.Figure()
                dash_fig.add_trace(go.Bar(x=batch_df["file"], y=batch_df["mean"], name="mean"))
                dash_fig.add_trace(go.Scatter(x=batch_df["file"], y=batch_df["max"], mode="lines+markers", name="max"))
                dash_fig.update_layout(height=360, xaxis_title="file", yaxis_title="value")
                st.plotly_chart(dash_fig, use_container_width=True)
                mean_series = pd.to_numeric(batch_df["mean"], errors="coerce")
                if mean_series.notna().sum() > 2:
                    z = (mean_series - mean_series.mean()) / (mean_series.std() + 1e-9)
                    anomaly = batch_df[np.abs(z) > 2].copy()
                    st.write("异常批次（|z|>2）")
                    st.dataframe(anomaly, use_container_width=True, hide_index=True)

    # ── 9. 导出 ──
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
    md_report = generate_markdown_report(quality_df, roi_df, radial_df, batch_df)
    st.download_button(
        "导出升级报告 Markdown",
        data=md_report.encode("utf-8"),
        file_name="analysis_report.md",
        mime="text/markdown",
    )

    # ── 10. AI 大模型分析 ──
    if nav == "AI大模型分析":
        _render_ai_page(quality_df, roi_df, radial_df, cropped_grid, peak_export, batch_df)


if __name__ == "__main__":
    main()
