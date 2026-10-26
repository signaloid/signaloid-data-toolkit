#   Copyright (c) 2026, Signaloid.
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to
#   deal in the Software without restriction, including without limitation the
#   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
#   sell copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#   DEALINGS IN THE SOFTWARE.


import math
from typing import Any
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from signaloid.distributional_information_plotting.plot_histogram_dirac_deltas import (
    PlotData,
)


def printv(verbose: bool, formatString: str, *args) -> None:
    """
    A wrapper for `print()` that allows for controlled verbosity.

    Args:
        verbose: Flag that controls verbosity.
        formatString: Formatted string to print.
        args: Tuple holding values to be used in the formatted string.
    """
    if verbose:
        print(formatString % args)


def plot(
    plot_data: PlotData,
    path: str = "./plot.png",
    plot_expected_value_line: bool = True,
    no_special_y: bool = False,
    save: bool = False,
    verbose: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    x_label: str | None = None,
    x_tick_label_rotation: float | None = None,
    font_size: int = 20,
    matplotlib_rc_params_override: dict[str, str] | None = None,
) -> bool:
    """
    Args:
        plot_data: `PlotData` to plot.
        path: Path to save the output if saving is enabled.
        plot_expected_value_line: Flag toggling whether the plot should have a vertical
            line at the expected value of the input distribution.
        no_special_y: Flag toggling the plotting of special values, e.g., `NaN`, `INF`, and `-INF`.
        save: Flag toggling if the plot should be saved to a file or just shown.
        verbose: Flag controlling printing verbosity.
        xlim: Input x-axis limits for the plot.
        ylim: Input y-axis limits for the plot.
        x_label: x-axis label.
        x_tick_label_rotation: Rotation of x-axis tick labels.
        font_size: Font size to use for the plot labels.
        matplotlib_rc_params_override: Dictionary to specify custom plotting parameters.
    Returns:
        `True` if successful, `False` else.
    """

    matplotlib_rcParams_update_defaults: dict[str, int | str | bool] = {
        "font.size": font_size,
        "figure.facecolor": "FFFFFF30",
        "axes.facecolor": "FFFFFF30",
        "xtick.top": True,
        "ytick.right": True,
    }

    if save:
        matplotlib.use("Agg")

    if matplotlib_rc_params_override is not None:
        matplotlib_rcParams_update_defaults.update(matplotlib_rc_params_override)

    # matplotlib >= 3.11 types rcParams keys as literals.
    matplotlib.rcParams.update(
        matplotlib_rcParams_update_defaults  # type: ignore[arg-type]
    )

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2 if plot_data.dist.has_special_values else 1,
        sharey=False,
        figsize=(10 + (3 if plot_data.dist.has_special_values else 0), 6),
        gridspec_kw=(
            {"width_ratios": [4.2, 1]} if plot_data.dist.has_special_values else None
        ),
    )

    if not plot_data.dist.has_special_values:
        # Force axes to be a list (Thanks pyplot for the wonderful semantics)
        axes = [axes]

    fig.sca(axes[0])
    printv(verbose, "Plotting...")

    # If there is only one finite Dirac delta, then plot just an
    # arrow representing a Dirac delta.
    if len(plot_data.positions) == 1:
        plt.annotate(
            text="",
            xy=(plot_data.positions[0], plot_data.masses[0]),
            xytext=(plot_data.positions[0], 0),
            arrowprops={"arrowstyle": "->", "facecolor": "black", "lw": 3},
        )
    elif len(plot_data.positions) > 1:
        # Plot the binning: filled step area + step outline + bin edges.
        # When there are no finite Dirac deltas at all (all mass on NaN or
        # Inf), leave the main axis empty and only draw the special-values
        # panel below.
        ax = plt.gca()

        # Plot the binning: filled step area + step outline + bin edges.
        step_patch = ax.stairs(
            plot_data.masses,
            plot_data.positions,
            fill=True,
            facecolor="#33A333" + "40",
            hatch="\\",
        )
        step_patch.set_edgecolor("#33A333")
        step_patch.set_linewidth(0)

        ax.stairs(
            plot_data.masses,
            plot_data.positions,
            fill=False,
            color="#33A333",
            linewidth=1.0,
        )

        # Vertical lines at internal bin edges via LineCollection (faster than plt.vlines).
        internal_edges = plot_data.positions[1:-1]
        edge_heights = np.maximum(plot_data.masses[:-1], plot_data.masses[1:])
        n_edges = len(internal_edges)
        segments = np.empty((n_edges, 2, 2))
        segments[:, 0, 0] = internal_edges
        segments[:, 0, 1] = 0
        segments[:, 1, 0] = internal_edges
        segments[:, 1, 1] = edge_heights
        lc = LineCollection(segments.tolist(), colors="#33A333", linewidths=0.5)
        ax.add_collection(lc)

    # Default kwargs for plt.annotate
    annotation_default_args: dict[str, Any] = {
        "xycoords": ("data", "axes fraction"),
        "textcoords": "offset points",
        "xytext": (3, 0),
        "fontsize": 16,
        "rotation": 90,
        "verticalalignment": "top",
        "parse_math": True,
    }

    # Plot mean value (if finite) and standard deviation (if available)
    if (
        plot_expected_value_line is True
        and plot_data.dist.mean is not None
        and math.isfinite(plot_data.dist.mean)
    ):
        axes[0].axvline(plot_data.dist.mean, lw=2, color="#29782d")
        plt.annotate(
            "$E(X)$",
            (plot_data.dist.mean, 0.9),
            color="#206024",
            **annotation_default_args,
        )

    if plot_data.dist.has_special_values:
        fig.sca(axes[1])
        plt.bar(
            x=["NaN", "-Inf", "Inf"],
            height=[
                plot_data.dist.nan_dirac_delta.mass,
                plot_data.dist.neg_inf_dirac_delta.mass,
                plot_data.dist.pos_inf_dirac_delta.mass,
            ],
            width=0.55,
            edgecolor="#33A333",
            facecolor="#757575" + "40",
            hatch="\\",
        )

    printv(verbose, "Adjusting plot...")
    for i, ax in enumerate(axes):
        if i == 0:
            if len(plot_data.positions):
                if xlim is None:
                    # This prevents the plots failing if mean value is
                    # incorrect and way off the range.
                    if (
                        plot_data.dist.mean is None
                        or math.isnan(plot_data.dist.mean)
                        or math.isinf(plot_data.dist.mean)
                    ):  # type: ignore[arg-type]
                        min_x = plot_data.min_range
                        max_x = plot_data.max_range
                    else:
                        min_x = min(plot_data.dist.mean, plot_data.min_range)
                        max_x = max(plot_data.dist.mean, plot_data.max_range)
                    range_spacing = 0.05 * (max_x - min_x)
                    xlim = (min_x - range_spacing, max_x + range_spacing)
                ax.set_xlim(*xlim)

                if ylim is None:
                    ylim = (0, 1.1 * plot_data.max_value)
                ax.set_ylim(*ylim)

            ax.set_xlabel(x_label if x_label else "Distribution Support")
            ax.set_ylabel("Probability Density")
        else:
            ax.set_ylim(0, 1)
            if not no_special_y:
                ax.set_ylabel("Probability Amplitude")
            else:
                ax.set_yticklabels([])

        ax.grid(visible=True, which="major", color="#999999aa", linestyle="--")

        plt.subplots_adjust(
            left=0.16, bottom=0.14, right=1.0, top=0.9, hspace=0.0, wspace=0.05
        )

        ax.spines["right"].set_alpha(0.5)
        ax.spines["left"].set_alpha(0.5)
        ax.spines["top"].set_alpha(1)
        ax.spines["top"].set_linewidth(3)
        ax.spines["bottom"].set_linewidth(3)
        ax.tick_params(axis="x", which="major", width=2, length=10, direction="in")
        ax.tick_params(axis="y", which="major", width=1, length=4, direction="in")

        if x_tick_label_rotation is not None:
            plt.setp(
                ax.get_xticklabels(),
                rotation=x_tick_label_rotation,
                horizontalalignment="right",
            )

        ax.minorticks_on()
        ax.tick_params(axis="x", which="minor", width=1, length=5, direction="in")
        ax.tick_params(axis="y", which="minor", left=False, right=False)

    fig.tight_layout()

    if save:
        printv(verbose, f"Saving figure as `{path}`...")
        fig.savefig(path, format="png", bbox_inches="tight", pad_inches=0.2)
        printv(verbose, "Closing figure...")
        plt.close(fig=fig)
    else:
        plt.show()

    return True
