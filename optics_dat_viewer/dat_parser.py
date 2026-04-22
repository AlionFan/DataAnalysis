from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import TextIO

import numpy as np


@dataclass
class DatGrid:
    rows: int
    cols: int
    hole_value: float
    row_delta: float
    col_delta: float
    row_origin: float
    col_origin: float
    data: np.ndarray

    @property
    def x_coords(self) -> np.ndarray:
        return self.col_origin + np.arange(self.cols) * self.col_delta

    @property
    def y_coords(self) -> np.ndarray:
        return self.row_origin + np.arange(self.rows) * self.row_delta


def parse_dat_file(file_obj: TextIO) -> DatGrid:
    lines = [line.strip() for line in file_obj if line.strip()]
    clean_lines = [line for line in lines if not line.startswith("!")]

    if len(clean_lines) < 5:
        raise ValueError("DAT 文件内容不完整，至少需要头部与数据矩阵。")

    grid_tokens = clean_lines[0].split()
    if len(grid_tokens) < 3 or grid_tokens[0].upper() != "GRID":
        raise ValueError("DAT 文件格式错误，未找到 GRID 行。")

    rows = int(grid_tokens[1])
    cols = int(grid_tokens[2])
    hole_value = float(clean_lines[1].split()[0])

    delta_tokens = clean_lines[2].split()
    origin_tokens = clean_lines[3].split()
    if len(delta_tokens) < 2 or len(origin_tokens) < 2:
        raise ValueError("DAT 文件头部 row/column 元数据不足。")

    row_delta = float(delta_tokens[0])
    col_delta = float(delta_tokens[-1])
    row_origin = float(origin_tokens[0])
    col_origin = float(origin_tokens[-1])

    data_lines = clean_lines[4:]
    data_text = "\n".join(data_lines)
    matrix = np.loadtxt(StringIO(data_text), dtype=float)
    matrix = np.atleast_2d(matrix)

    if matrix.shape != (rows, cols):
        raise ValueError(
            f"矩阵维度不匹配：头部为 {(rows, cols)}，实际为 {matrix.shape}。"
        )

    matrix = np.where(matrix == hole_value, np.nan, matrix)
    return DatGrid(
        rows=rows,
        cols=cols,
        hole_value=hole_value,
        row_delta=row_delta,
        col_delta=col_delta,
        row_origin=row_origin,
        col_origin=col_origin,
        data=matrix,
    )


def load_dat_from_path(path: str | Path) -> DatGrid:
    with open(path, "r", encoding="utf-8") as f:
        return parse_dat_file(f)


def load_dat_from_upload(uploaded_file) -> DatGrid:
    raw = uploaded_file.getvalue()
    content = raw.decode("utf-8", errors="replace")
    return parse_dat_file(StringIO(content))


def dat_to_csv_bytes(data: np.ndarray) -> bytes:
    output = BytesIO()
    np.savetxt(output, data, delimiter=",", fmt="%.8g")
    return output.getvalue()
