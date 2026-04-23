from __future__ import annotations

from io import BytesIO

import numpy as np
import plotly.graph_objects as go

try:
    from .dat_parser import DatGrid
except ImportError:
    from dat_parser import DatGrid


def plot_heatmap(grid: DatGrid, title: str) -> go.Figure:
    x = np.linspace(-10, 10, grid.cols)
    y = np.linspace(-10, 10, grid.rows)
    fig = go.Figure(
        data=go.Heatmap(
            z=grid.data,
            x=x,
            y=y,
            colorscale="Viridis",
            colorbar={"title": "Intensity"},
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="X",
        yaxis_title="Y",
        xaxis={"constrain": "range"},
        yaxis={"scaleanchor": "x", "scaleratio": 1, "constrain": "range"},
    )
    return fig


def plot_surface(grid: DatGrid, title: str) -> go.Figure:
    fig = go.Figure(
        data=go.Surface(z=grid.data, x=grid.x_coords, y=grid.y_coords, colorscale="Viridis")
    )
    fig.update_layout(title=title, scene={"xaxis_title": "X", "yaxis_title": "Y", "zaxis_title": "I"})
    return fig


def plot_contour_overlay(grid: DatGrid, levels: int) -> go.Figure:
    x = np.linspace(-10, 10, grid.cols)
    y = np.linspace(-10, 10, grid.rows)
    fig = go.Figure(
        data=go.Contour(
            z=grid.data,
            x=x,
            y=y,
            ncontours=levels,
            colorscale="Viridis",
        )
    )
    fig.update_layout(title="等高线图", xaxis_title="X", yaxis_title="Y", height=500)
    return fig


def export_plot_png(fig: go.Figure) -> bytes:
    buffer = BytesIO()
    fig.write_image(buffer, format="png")
    return buffer.getvalue()


def apply_visual_transform(data: np.ndarray, log_scale: bool) -> np.ndarray:
    transformed = data.copy()
    if log_scale:
        transformed = np.log1p(np.clip(transformed, a_min=0, a_max=None))
    return transformed
