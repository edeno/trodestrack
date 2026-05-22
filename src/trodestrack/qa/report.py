"""Quality assurance report generation for filter diagnostics.

This module generates comprehensive PDF reports containing:
- Summary statistics (RMSE, NEES, NIS)
- Time series plots (position/velocity error, residuals)
- Consistency checks (NEES/NIS histograms)
- 2D trajectory visualization
- Filter configuration parameters
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from numpy.typing import NDArray

from trodestrack.qa.metrics import (
    compute_heading_error,
    compute_heading_rmse,
    compute_nees_stats,
    compute_nis_stats,
    compute_position_rmse,
    compute_velocity_rmse,
)
from trodestrack.qa.plots import (
    plot_heading_error,
    plot_nees_histogram,
    plot_nis_histogram,
    plot_position_error,
    plot_velocity_error,
)
from trodestrack.viz.styles import COLORS, apply_tufte_style

# Project acceptance targets for filter quality (rat tracking, ground-truth-aligned)
TARGET_POSITION_RMSE_M = 0.02  # 2 cm
TARGET_VELOCITY_RMSE_MS = 0.10  # 10 cm/s
TARGET_HEADING_MAE_DEG = 7.0  # 7 degrees


def generate_qa_report(
    pdf_path: Path | str,
    t: NDArray[np.float64],
    positions_true: NDArray[np.float64],
    positions_est: NDArray[np.float64],
    velocities_true: NDArray[np.float64],
    velocities_est: NDArray[np.float64],
    headings_true: NDArray[np.float64],
    headings_est: NDArray[np.float64],
    nees: NDArray[np.float64],
    state_dim: int,
    nis: NDArray[np.float64] | None = None,
    measurement_dim: int | None = None,
    config: dict | None = None,
    title: str = "Filter QA Report",
) -> None:
    """Generate comprehensive PDF quality assurance report.

    Creates a multi-page PDF with summary statistics, time series plots,
    consistency checks, and configuration details.

    Parameters
    ----------
    pdf_path : Path or str
        Output PDF file path.
    t : NDArray[np.float64]
        Time vector (N,) in seconds.
    positions_true : NDArray[np.float64]
        Ground truth positions (N, 2) in meters.
    positions_est : NDArray[np.float64]
        Estimated positions (N, 2) in meters.
    velocities_true : NDArray[np.float64]
        Ground truth velocities (N, 2) in m/s.
    velocities_est : NDArray[np.float64]
        Estimated velocities (N, 2) in m/s.
    headings_true : NDArray[np.float64]
        Ground truth headings (N,) in radians.
    headings_est : NDArray[np.float64]
        Estimated headings (N,) in radians.
    nees : NDArray[np.float64]
        NEES values (N,).
    state_dim : int
        State dimensionality for NEES χ² bounds.
    nis : NDArray[np.float64] | None, optional
        NIS values (N,). If provided, ``measurement_dim`` is required.
    measurement_dim : int | None, optional
        Measurement dimensionality for NIS χ² bounds.
    config : dict | None, optional
        Filter configuration to embed on the summary page.
    title : str, default "Filter QA Report"
        Report title.

    Raises
    ------
    ValueError
        If array shapes are inconsistent or required parameters are missing.
    FileNotFoundError
        If ``pdf_path`` directory doesn't exist.
    OSError
        If the PDF cannot be created (permissions, disk space, etc.).

    Example:
        >>> import numpy as np
        >>> from pathlib import Path
        >>> t = np.linspace(0, 10, 100)
        >>> pos_true = np.column_stack([t * 0.1, np.zeros(100)])
        >>> pos_est = pos_true + np.random.randn(100, 2) * 0.01
        >>> vel_true = np.column_stack([np.ones(100) * 0.1, np.zeros(100)])
        >>> vel_est = vel_true + np.random.randn(100, 2) * 0.01
        >>> heading_true = np.zeros(100)
        >>> heading_est = np.random.randn(100) * 0.1
        >>> nees = np.random.chisquare(df=8, size=100)
        >>> generate_qa_report(
        ...     pdf_path=Path("report.pdf"),
        ...     t=t,
        ...     positions_true=pos_true,
        ...     positions_est=pos_est,
        ...     velocities_true=vel_true,
        ...     velocities_est=vel_est,
        ...     headings_true=heading_true,
        ...     headings_est=heading_est,
        ...     nees=nees,
        ...     state_dim=8,
        ... )

    Notes:
        Report sections:
        1. Title page with summary statistics
        2. Position error time series
        3. Velocity error time series
        4. Heading error time series
        5. 2D trajectory plot (true vs estimated)
        6. NEES histogram with chi-squared bounds
        7. NIS histogram (if provided)
    """
    # Convert to Path
    pdf_path = Path(pdf_path)

    # Validate the time axis itself before using its length to bind the
    # other shape checks. The PDF embeds time-domain plots that index
    # ``t`` directly, so a non-1D / non-finite / non-monotonic ``t``
    # silently produced a finished-looking but wrong report.
    t_arr = np.asarray(t)
    if t_arr.ndim != 1:
        raise ValueError(f"t must be 1D (N,) in seconds; got shape {t_arr.shape}.")
    if t_arr.size < 2:
        raise ValueError(
            f"t must have at least two samples for a meaningful report; "
            f"got shape {t_arr.shape}."
        )
    if not np.all(np.isfinite(t_arr)):
        n_bad = int(np.sum(~np.isfinite(t_arr)))
        raise ValueError(
            f"t contains {n_bad} non-finite value(s) (NaN/inf); the report's "
            "time-axis plots require finite seconds."
        )
    if not np.all(np.diff(t_arr) > 0):
        first_bad = int(np.argmax(np.diff(t_arr) <= 0))
        raise ValueError(
            f"t must be strictly increasing; first non-increasing step at "
            f"index {first_bad + 1} (t[{first_bad}]={t_arr[first_bad]!r}, "
            f"t[{first_bad + 1}]={t_arr[first_bad + 1]!r})."
        )

    # Validation: Check array shapes
    N = t_arr.shape[0]
    if positions_true.shape != (N, 2):
        raise ValueError(
            f"Shape mismatch: positions_true {positions_true.shape} vs t {t.shape}"
        )
    if positions_est.shape != (N, 2):
        raise ValueError(
            f"Shape mismatch: positions_est {positions_est.shape} vs t {t.shape}"
        )
    if velocities_true.shape != (N, 2):
        raise ValueError(
            f"Shape mismatch: velocities_true {velocities_true.shape} vs t {t.shape}"
        )
    if velocities_est.shape != (N, 2):
        raise ValueError(
            f"Shape mismatch: velocities_est {velocities_est.shape} vs t {t.shape}"
        )
    if headings_true.shape != (N,):
        raise ValueError(
            f"Shape mismatch: headings_true {headings_true.shape} vs t {t.shape}"
        )
    if headings_est.shape != (N,):
        raise ValueError(
            f"Shape mismatch: headings_est {headings_est.shape} vs t {t.shape}"
        )
    if nees.shape[0] != N:
        raise ValueError(f"Shape mismatch: nees {nees.shape} vs t {t.shape}")

    if nis is not None and measurement_dim is None:
        raise ValueError("measurement_dim required when nis is provided")

    # NIS shape: documented as (N,), aligned to the same time base as the
    # other arrays. Without this guard a mismatched NIS (e.g. NIS over a
    # different sample set) silently summarised next to the trajectory
    # plots, attaching consistency stats to the wrong frames.
    if nis is not None and nis.shape != (N,):
        raise ValueError(
            f"Shape mismatch: nis {nis.shape} vs t {t.shape}; nis must be "
            f"({N},) so its time alignment matches positions/headings."
        )

    # Validation: Check PDF path is writable
    if not pdf_path.parent.exists():
        raise FileNotFoundError(f"Directory does not exist: {pdf_path.parent}")

    # Apply visualization style
    apply_tufte_style()

    # Compute summary statistics
    pos_rmse = compute_position_rmse(positions_true, positions_est)
    vel_rmse = compute_velocity_rmse(velocities_true, velocities_est)
    heading_mae = compute_heading_error(headings_true, headings_est)
    heading_rmse = compute_heading_rmse(headings_true, headings_est)
    nees_stats = compute_nees_stats(nees, state_dim=state_dim, confidence=0.95)

    if nis is not None and measurement_dim is not None:
        nis_stats = compute_nis_stats(
            nis, measurement_dim=measurement_dim, confidence=0.95
        )
    else:
        nis_stats = None

    # Create PDF
    try:
        with PdfPages(pdf_path) as pdf:
            # Page 1: Title and summary statistics
            fig_summary = _create_summary_page(
                title=title,
                pos_rmse=pos_rmse,
                vel_rmse=vel_rmse,
                heading_mae=heading_mae,
                heading_rmse=heading_rmse,
                nees_stats=nees_stats,
                nis_stats=nis_stats,
                config=config,
            )
            pdf.savefig(fig_summary, bbox_inches="tight")
            plt.close(fig_summary)

            # Page 2: Position error time series
            fig_pos, _ = plot_position_error(
                t, positions_true, positions_est, target_threshold_m=0.02
            )
            pdf.savefig(fig_pos, bbox_inches="tight")
            plt.close(fig_pos)

            # Page 3: Velocity error time series
            fig_vel, _ = plot_velocity_error(t, velocities_true, velocities_est)
            pdf.savefig(fig_vel, bbox_inches="tight")
            plt.close(fig_vel)

            # Page 4: Heading error time series
            fig_heading, _ = plot_heading_error(
                t,
                headings_true,
                headings_est,
                target_threshold_deg=TARGET_HEADING_MAE_DEG,
            )
            pdf.savefig(fig_heading, bbox_inches="tight")
            plt.close(fig_heading)

            # Page 5: 2D trajectory plot
            fig_traj = _create_trajectory_plot(positions_true, positions_est)
            pdf.savefig(fig_traj, bbox_inches="tight")
            plt.close(fig_traj)

            # Page 6: NEES histogram
            fig_nees, _ = plot_nees_histogram(
                nees, state_dim=state_dim, confidence=0.95
            )
            pdf.savefig(fig_nees, bbox_inches="tight")
            plt.close(fig_nees)

            # Page 7 (optional): NIS histogram
            if nis is not None and measurement_dim is not None:
                fig_nis, _ = plot_nis_histogram(
                    nis, measurement_dim=measurement_dim, confidence=0.95
                )
                pdf.savefig(fig_nis, bbox_inches="tight")
                plt.close(fig_nis)

            # Set PDF metadata
            d = pdf.infodict()
            d["Title"] = title
            d["Author"] = "trodestrack QA"
            d["Subject"] = "Filter Quality Assurance Report"
            d["Keywords"] = "EKF UKF tracking NEES NIS RMSE"

    except OSError as e:
        raise OSError(f"Failed to create PDF at {pdf_path}: {e}") from e


def _create_summary_page(
    title: str,
    pos_rmse: float,
    vel_rmse: float,
    heading_mae: float,
    heading_rmse: float,
    nees_stats: dict,
    nis_stats: dict | None,
    config: dict | None,
) -> Figure:
    """Create summary page with metrics and configuration.

    Parameters
    ----------
    title : str
        Report title.
    pos_rmse : float
        Position RMSE (m).
    vel_rmse : float
        Velocity RMSE (m/s).
    heading_mae : float
        Heading MAE (rad).
    heading_rmse : float
        Heading RMSE (rad).
    nees_stats : dict
        NEES statistics dictionary.
    nis_stats : dict | None
        Optional NIS statistics dictionary.
    config : dict | None
        Optional configuration dictionary.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    fig = plt.figure(figsize=(8.5, 11))  # US Letter size
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    # Compute PASS/FAIL verdict against project acceptance targets so the
    # report leads with the bottom line. Heading is compared in degrees
    # because ``TARGET_HEADING_MAE_DEG`` is degrees but ``heading_mae`` is
    # radians (matching the convention used elsewhere in this function).
    heading_mae_deg = float(np.rad2deg(heading_mae))
    position_pass = pos_rmse <= TARGET_POSITION_RMSE_M
    velocity_pass = vel_rmse <= TARGET_VELOCITY_RMSE_MS
    heading_pass = heading_mae_deg <= TARGET_HEADING_MAE_DEG
    all_pass = position_pass and velocity_pass and heading_pass
    if all_pass:
        verdict = "RESULT: PASS"
        verdict_color = COLORS["verdict_pass"]
    else:
        failed = [
            name
            for name, ok in [
                ("position", position_pass),
                ("velocity", velocity_pass),
                ("heading", heading_pass),
            ]
            if not ok
        ]
        verdict = f"RESULT: FAIL ({', '.join(failed)})"
        verdict_color = COLORS["verdict_fail"]
    fig.text(
        0.5,
        0.93,
        verdict,
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
        color="white",
        bbox={
            "facecolor": verdict_color,
            "edgecolor": verdict_color,
            "boxstyle": "round,pad=0.4",
        },
        transform=fig.transFigure,
    )

    # Create text content
    text_lines = []

    # Section 1: Accuracy Metrics
    text_lines.append("=" * 60)
    text_lines.append("ACCURACY METRICS")
    text_lines.append("=" * 60)
    text_lines.append("")
    text_lines.append(
        f"Position RMSE:    {pos_rmse * 100:.2f} cm    "
        f"(target: ≤{TARGET_POSITION_RMSE_M * 100:.1f} cm)"
    )
    text_lines.append(
        f"Velocity RMSE:    {vel_rmse * 100:.2f} cm/s  "
        f"(target: ≤{TARGET_VELOCITY_RMSE_MS * 100:.1f} cm/s)"
    )
    text_lines.append(
        f"Heading MAE:      {np.rad2deg(heading_mae):.2f}°     "
        f"(target: ≤{TARGET_HEADING_MAE_DEG:.1f}°)"
    )
    text_lines.append(f"Heading RMSE:     {np.rad2deg(heading_rmse):.2f}°")
    text_lines.append("")

    # Section 2: NEES Consistency
    text_lines.append("=" * 60)
    text_lines.append("NEES CONSISTENCY (State Estimation)")
    text_lines.append("=" * 60)
    text_lines.append("")
    text_lines.append(f"Mean NEES:        {nees_stats['mean']:.2f}")
    text_lines.append(f"Std NEES:         {nees_stats['std']:.2f}")
    text_lines.append(
        f"95% CI bounds:    [{nees_stats['chi2_lower']:.2f}, {nees_stats['chi2_upper']:.2f}]"
    )
    text_lines.append(
        f"Within bounds:    {nees_stats['pct_in_bounds']:.1f}%  (expect ~95%)"
    )
    text_lines.append("")

    # Section 3 (optional): NIS Consistency
    if nis_stats is not None:
        text_lines.append("=" * 60)
        text_lines.append("NIS CONSISTENCY (Measurement Validation)")
        text_lines.append("=" * 60)
        text_lines.append("")
        text_lines.append(f"Mean NIS:         {nis_stats['mean']:.2f}")
        text_lines.append(f"Std NIS:          {nis_stats['std']:.2f}")
        text_lines.append(
            f"95% CI bounds:    [{nis_stats['chi2_lower']:.2f}, {nis_stats['chi2_upper']:.2f}]"
        )
        text_lines.append(
            f"Within bounds:    {nis_stats['pct_in_bounds']:.1f}%  (expect ~95%)"
        )
        text_lines.append("")

    # Section 4 (optional): Configuration
    if config is not None:
        text_lines.append("=" * 60)
        text_lines.append("FILTER CONFIGURATION")
        text_lines.append("=" * 60)
        text_lines.append("")
        for key, value in config.items():
            # Format value nicely
            if isinstance(value, float):
                if abs(value) < 1e-3 or abs(value) > 1e3:
                    value_str = f"{value:.2e}"
                else:
                    value_str = f"{value:.6f}".rstrip("0").rstrip(".")
            else:
                value_str = str(value)

            # Truncate long keys
            key_str = key[:40]
            text_lines.append(f"{key_str:45s} {value_str}")
        text_lines.append("")

    # Render text on figure. Start below the verdict banner so it doesn't
    # collide with the title or the PASS/FAIL block.
    text_content = "\n".join(text_lines)
    fig.text(
        0.1,
        0.88,
        text_content,
        verticalalignment="top",
        fontfamily="monospace",
        fontsize=9,
        transform=fig.transFigure,
    )

    return fig


def _create_trajectory_plot(
    positions_true: NDArray[np.float64],
    positions_est: NDArray[np.float64],
) -> Figure:
    """Create 2D trajectory comparison plot.

    Parameters
    ----------
    positions_true : NDArray[np.float64]
        Ground truth positions (N, 2) in meters.
    positions_est : NDArray[np.float64]
        Estimated positions (N, 2) in meters.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    # Plot trajectories
    ax.plot(
        positions_true[:, 0],
        positions_true[:, 1],
        color=COLORS["gray"],
        linewidth=1.5,
        alpha=0.7,
        label="Ground Truth",
    )
    ax.plot(
        positions_est[:, 0],
        positions_est[:, 1],
        color=COLORS["blue"],
        linewidth=1.0,
        alpha=0.8,
        label="Estimate",
    )

    # Mark start and end. Shapes (circle vs square) carry the start/end
    # distinction; the Okabe-Ito blue/orange pair is colorblind-safe (the
    # previous green/red pair was unreadable for deuteranopes).
    ax.scatter(
        positions_true[0, 0],
        positions_true[0, 1],
        marker="o",
        s=80,
        c=COLORS["okabe_ito_blue"],
        edgecolor="black",
        linewidth=1.5,
        label="Start",
        zorder=10,
    )
    ax.scatter(
        positions_true[-1, 0],
        positions_true[-1, 1],
        marker="s",
        s=80,
        c=COLORS["okabe_ito_orange"],
        edgecolor="black",
        linewidth=1.5,
        label="End",
        zorder=10,
    )

    # Labels and formatting
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("2D Trajectory")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    return fig
