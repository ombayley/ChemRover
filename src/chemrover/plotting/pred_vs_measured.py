#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: O. Bayley
Description: Publication-style parity ("predicted vs. actual") plots for
             regression models, with inline error metrics.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
_RC = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#3A3A3A",
    "axes.linewidth": 0.9,
    # pin every text colour so the plot is immune to a dark ambient theme
    # (otherwise titles/legend inherit a light global text.color -> invisible)
    "text.color": "#1A1A1A",
    "axes.labelcolor": "#1A1A1A",
    "axes.titlecolor": "#1A1A1A",
    "legend.labelcolor": "#1A1A1A",
    "axes.labelsize": 12,
    "axes.titlesize": 13.5,
    "axes.titleweight": 700,
    "axes.titlepad": 12,
    "xtick.color": "#3A3A3A",
    "ytick.color": "#3A3A3A",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "legend.frameon": False,
    "legend.fontsize": 9.5,
    "savefig.bbox": "tight",
}

_ACCENT = "#2F5D75"  # ink blue, used when not colouring by error
_BAND = "#7FA8BC"  # ±RMSE band
_GUIDE = "#9AA0A6"  # 1:1 line, zero line


def _link_widths(fig, ax_top, ax_bottom) -> None:
    """
    Keep the residual strip exactly as wide as the main panel.

    The main axes uses ``aspect='equal'``, so matplotlib shrinks its box at
    draw time and the two panels drift out of alignment. Re-syncing on every
    draw keeps them locked together through resizes and tight_bbox saves.
    """

    def _sync(_event=None):
        top, bot = ax_top.get_position(), ax_bottom.get_position()
        if abs(top.x0 - bot.x0) < 1e-6 and abs(top.width - bot.width) < 1e-6:
            return  # already aligned; avoids redraw ping-pong
        ax_bottom.set_position([top.x0, bot.y0, top.width, bot.height])

    fig.canvas.mpl_connect("draw_event", _sync)
    fig.canvas.draw()  # align once up front so savefig is correct


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute the standard regression report."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def plot_parity(
        y_true: Sequence[float],
        y_pred: Sequence[float],
        *,
        title: Optional[str] = None,
        quantity: str = r"$\lambda_{\mathrm{max}}$",
        unit: str = "nm",
        color_by_error: bool = True,
        cmap: str = "cividis",
        show_residuals: bool = True,
        annotate_worst: int = 0,
        labels: Optional[Sequence[str]] = None,
        save_path: Optional[str] = None,
        dpi: int = 300,
        grid: bool = False,
        show: bool = False,
        target_fig: Optional[mpl.figure.FigureBase] = None,
):
    """
    Draw a parity plot of predictions against experimental values.

    Parameters
    ----------
    y_true, y_pred : array-like
        Measured and predicted values (pandas Series are fine).
    title : str, optional
        Figure title. Defaults to a generic one.
    quantity, unit : str
        Used to build axis labels, e.g. "$\\lambda_{max}$" and "nm".
    color_by_error : bool
        Colour each point by its absolute residual and add a colourbar.
        Set False for a single-colour scatter.
    cmap : str
        Perceptually-uniform colormap name used when ``color_by_error``.
    show_residuals : bool
        Add a residual strip beneath the main panel, sharing the x-axis.
    annotate_worst : int
        Label the N worst-predicted points (needs ``labels`` for names,
        otherwise falls back to the index).
    labels : sequence of str, optional
        Per-point identifiers (e.g. SMILES or compound names) for annotation.
    save_path : str, optional
        If given, write the figure here at ``dpi`` (standalone only).
    show : bool
        Call ``plt.show()`` at the end (standalone only).
    target_fig : Figure or SubFigure, optional
        Draw the whole parity composition (main panel + residual strip +
        colorbar) into this figure instead of creating a new one. Pass a
        ``SubFigure`` (e.g. from ``fig.subfigures(2, 2)``) to tile several
        parity plots into one grid. When given, ``save_path``/``show`` are
        ignored so the caller owns the parent figure.

    Returns
    -------
    (fig, ax_m, ax_r, stats) : the figure, main axis, residuals axis, and the metrics dict (keys: "rmse", "mae", "r2", "n")
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")

    stats = _metrics(y_true, y_pred)
    resid = y_pred - y_true
    abs_err = np.abs(resid)

    # symmetric, padded limits so the 1:1 line is a true diagonal
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = 0.06 * (hi - lo) if hi > lo else 1.0
    lim = (lo - pad, hi + pad)

    with mpl.rc_context(_RC):
        if show_residuals:
            fig = target_fig if target_fig is not None else plt.figure(figsize=(6.4, 7.2))
            gs = fig.add_gridspec(
                2, 1, height_ratios=[3.1, 1.0], hspace=0.06,
                left=0.14, right=0.86, top=0.89, bottom=0.09,
            )
            ax = fig.add_subplot(gs[0])
            ax_r = fig.add_subplot(gs[1], sharex=ax)
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            fig = target_fig if target_fig is not None else plt.figure(figsize=(6.4, 6.0))
            gs = fig.add_gridspec(
                1, 1, left=0.14, right=0.86, top=0.88, bottom=0.11
            )
            ax = fig.add_subplot(gs[0])
            ax_r = None

        # ---------------- main panel ---------------- #
        # ±1 RMSE envelope around the parity line
        edge = np.array(lim)
        ax.fill_between(
            edge, edge - stats["rmse"], edge + stats["rmse"],
            color=_BAND, alpha=0.16, lw=0, zorder=0,
        )
        ax.plot(edge, edge, ls="--", lw=1.2, color=_GUIDE, zorder=1)

        if grid:
            ax_r.grid(grid, color="#E3E6E8", lw=0.7, zorder=0)
        else:
            ax_r.grid(grid)
        ax.set_axisbelow(True)

        if color_by_error:
            sc = ax.scatter(
                y_true, y_pred, c=abs_err, cmap=cmap, s=52,
                edgecolor="white", linewidth=0.6, alpha=0.95, zorder=3,
            )
            cax = ax.inset_axes([1.03, 0.0, 0.035, 1.0])
            cb = fig.colorbar(sc, cax=cax)
            cb.set_label(f"|residual| ({unit})", fontsize=10, labelpad=8)
            cb.outline.set_linewidth(0.6)
            cb.outline.set_edgecolor("#B9BDC1")
            cb.ax.tick_params(labelsize=9, length=2.5)
        else:
            ax.scatter(
                y_true, y_pred, s=52, facecolor=_ACCENT, alpha=0.75,
                edgecolor="white", linewidth=0.6, zorder=3,
            )

        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_ylabel(f"Predicted {quantity}  /  {unit}")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(title or f"Predicted vs. experimental {quantity}", loc="left")

        # ---------------- metric card ---------------- #
        card = "\n".join([
            f"$R^2$ = {stats['r2']:.3f}",
            f"RMSE = {stats['rmse']:.1f} {unit}",
            f"MAE  = {stats['mae']:.1f} {unit}",
            f"$n$ = {stats['n']}",
        ])
        ax.text(
            0.035, 0.965, card, transform=ax.transAxes,
            ha="left", va="top", fontsize=10.5, linespacing=1.55,
            color="#1A1A1A",
            bbox=dict(boxstyle="round,pad=0.55", facecolor="white",
                      edgecolor="#D5D9DC", linewidth=0.9, alpha=0.94),
            zorder=5,
        )

        handles = [
            Line2D([], [], ls="--", lw=1.2, color=_GUIDE, label="ideal (1:1)"),
            mpl.patches.Patch(facecolor=_BAND, alpha=0.3, lw=0,
                              label=f"±1 RMSE ({stats['rmse']:.1f} {unit})"),
        ]
        ax.legend(handles=handles, loc="lower right", handlelength=1.8,
                  borderpad=0.4, labelspacing=0.5)

        # ---------------- worst offenders ---------------- #
        if annotate_worst > 0:
            worst = np.argsort(abs_err)[::-1][:annotate_worst]
            for i in worst:
                tag = str(labels[i]) if labels is not None else f"#{i}"
                ax.annotate(
                    tag, (y_true[i], y_pred[i]),
                    textcoords="offset points", xytext=(7, 5),
                    fontsize=8.5, color="#5A6066", zorder=4,
                )

        # ---------------- residual strip ---------------- #
        if ax_r is not None:
            ax_r.axhspan(-stats["rmse"], stats["rmse"], color=_BAND,
                         alpha=0.16, lw=0, zorder=0)
            ax_r.axhline(0.0, ls="--", lw=1.2, color=_GUIDE, zorder=1)
            if grid:
                ax_r.grid(grid, color="#E3E6E8", lw=0.7, zorder=0)
            else:
                ax_r.grid(grid)
            ax_r.set_axisbelow(True)

            if color_by_error:
                ax_r.scatter(y_true, resid, c=abs_err, cmap=cmap, s=34,
                             edgecolor="white", linewidth=0.5, alpha=0.95,
                             zorder=3)
            else:
                ax_r.scatter(y_true, resid, s=34, facecolor=_ACCENT, alpha=0.75,
                             edgecolor="white", linewidth=0.5, zorder=3)

            rmax = 1.25 * float(np.abs(resid).max()) if len(resid) else 1.0
            ax_r.set_ylim(-rmax, rmax)
            ax_r.set_xlabel(f"Experimental {quantity}  /  {unit}")
            ax_r.set_ylabel(f"Residual / {unit}", fontsize=11)
            ax_r.spines[["top", "right"]].set_visible(False)
        else:
            ax.set_xlabel(f"Experimental {quantity}  /  {unit}")

        if ax_r is not None:
            _link_widths(fig, ax, ax_r)

        # save/show only when we own the figure; embedded plots leave the
        # parent figure to the caller
        if target_fig is None:
            if save_path:
                fig.savefig(save_path, dpi=dpi, facecolor="white")
            if show:
                plt.show()

    return fig, ax, ax_r, stats


if __name__ == "__main__":
    # smoke test with synthetic azobenzene-like data
    rng = np.random.default_rng(42)
    y = rng.normal(360, 38, 42).clip(300, 460)
    yh = y + rng.normal(0, 11, y.size) + 0.05 * (y - y.mean())
    plot_parity(y, yh, title="XGBoost — azobenzene $\\lambda_{max}$ (test set)",
                annotate_worst=3, save_path="demo_parity.png")
