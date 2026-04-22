from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .dat_parser import load_dat_from_path
except ImportError:
    from dat_parser import load_dat_from_path

WORKFLOW_DIR = Path("optics_dat_viewer/workflows")
WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)


# ── 报告生成 ──────────────────────────────────

def generate_html_report(quality_df: pd.DataFrame, roi_df: pd.DataFrame, radial_df: pd.DataFrame) -> str:
    return f"""
<html><head><meta charset='utf-8'><title>Optics Report</title></head><body>
<h1>光学数据分析报告</h1>
<h2>质量诊断</h2>{quality_df.to_html(index=False)}
<h2>多ROI统计</h2>{roi_df.to_html(index=False)}
<h2>径向分箱统计</h2>{radial_df.to_html(index=False)}
</body></html>
"""


def generate_markdown_report(
    quality_df: pd.DataFrame,
    roi_df: pd.DataFrame,
    radial_df: pd.DataFrame,
    batch_df: pd.DataFrame,
) -> str:
    q = quality_df.iloc[0].to_dict() if not quality_df.empty else {}
    roi_line = (
        f"```csv\n{roi_df.to_csv(index=False)}\n```"
        if not roi_df.empty
        else "无 ROI 数据"
    )
    radial_line = (
        f"```csv\n{radial_df.head(10).to_csv(index=False)}\n```"
        if not radial_df.empty
        else "无径向数据"
    )
    batch_line = (
        f"```csv\n{batch_df.head(20).to_csv(index=False)}\n```"
        if not batch_df.empty
        else "未执行批处理"
    )
    return f"""# 光学数据自动分析报告

## 1. 质量诊断摘要
- nan_ratio: {q.get('nan_ratio', 'N/A')}
- zero_ratio: {q.get('zero_ratio', 'N/A')}
- saturated_ratio: {q.get('saturated_ratio', 'N/A')}
- outlier_ratio: {q.get('outlier_ratio', 'N/A')}
- mean: {q.get('mean', 'N/A')}
- std_dev: {q.get('std_dev', 'N/A')}

## 2. ROI 统计
{roi_line}

## 3. 径向分箱统计（前10条）
{radial_line}

## 4. 批次看板（前20条）
{batch_line}

## 5. 自动结论
- 若 outlier_ratio > 0.01，建议执行异常点修复与频域滤波。
- 若 saturated_ratio 偏高，建议检查曝光或增益设置。
- 若批次中 mean/max 波动较大，建议做时序稳定性评估与设备校准。
"""


# ── 流程模板 ──────────────────────────────────

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


# ── 批量处理 ──────────────────────────────────

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
