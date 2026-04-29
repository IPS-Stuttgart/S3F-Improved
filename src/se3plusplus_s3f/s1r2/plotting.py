"""Plotting helpers shared by S1 x R2 benchmark outputs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt


def write_metric_line_plots(
    output_dir: Path,
    rows: list[dict[str, float | int | str]],
    plot_specs: Sequence[tuple[str, str, str]],
    variants: Sequence[str],
    variant_labels: dict[str, str],
) -> list[Path]:
    """Write one grid-size line plot per metric specification."""

    paths = []
    for metric, ylabel, filename in plot_specs:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        for variant in variants:
            variant_rows = sorted(
                [row for row in rows if row["variant"] == variant],
                key=lambda row: int(row["grid_size"]),
            )
            xs = [int(row["grid_size"]) for row in variant_rows]
            ys = [float(row[metric]) for row in variant_rows]
            ax.plot(xs, ys, marker="o", linewidth=1.8, label=variant_labels[variant])

        ax.set_xlabel("Number of circular cells")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        path = save_figure(fig, output_dir, filename)
        paths.append(path)
    return paths


def save_figure(fig, output_dir: Path, filename: str) -> Path:
    """Save a Matplotlib figure and close it."""

    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def format_plot_list(plot_paths: list[Path]) -> str:
    """Format plot filenames for notes."""

    if not plot_paths:
        return "- plots were disabled for this run"
    return "\n".join(f"- `{plot_path.name}`" for plot_path in plot_paths)
