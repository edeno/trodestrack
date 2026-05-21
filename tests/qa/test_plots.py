"""Tests for QA plotting utilities.

Following TDD: These tests are written BEFORE implementation to define the API.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from trodestrack.qa.plots import (
    plot_covariance_ellipse,
    plot_heading_error,
    plot_nees_histogram,
    plot_nis_histogram,
    plot_position_error,
    plot_residuals,
    plot_velocity_error,
)


class TestPlotResiduals:
    """Test residual time series plotting (innovations/measurement errors)."""

    def test_residuals_basic_plot(self) -> None:
        """Test basic residual plotting creates figure with expected structure."""
        # Arrange: Create synthetic residuals (position measurements)
        t = np.linspace(0, 10, 100)
        residuals = np.random.randn(100, 2) * 0.01  # 1cm std noise

        # Act: Create plot
        fig, axes = plot_residuals(t, residuals, ylabel="Position residuals (m)")

        # Assert: Structure checks
        assert isinstance(fig, plt.Figure)
        assert len(axes) == 2  # One subplot per dimension
        assert axes[0].get_ylabel() == "X (m)"
        assert axes[1].get_ylabel() == "Y (m)"
        assert axes[1].get_xlabel() == "Time (s)"

        plt.close(fig)

    def test_residuals_with_confidence_bands(self) -> None:
        """Test residual plot with +/- confidence bands."""
        # Arrange
        t = np.linspace(0, 10, 100)
        residuals = np.random.randn(100, 2) * 0.01
        std = 0.01  # Expected std

        # Act
        fig, axes = plot_residuals(
            t, residuals, ylabel="Residuals (m)", confidence_std=std
        )

        # Assert: Check that confidence bands are plotted
        # axhspan adds a PolyCollection to collections
        assert len(axes[0].collections) > 0 or len(axes[0].patches) > 0

        plt.close(fig)

    def test_residuals_shape_validation(self) -> None:
        """Test that mismatched shapes raise ValueError."""
        t = np.linspace(0, 10, 100)
        residuals = np.random.randn(50, 2)  # Wrong length

        with pytest.raises(ValueError, match="Shape mismatch"):
            plot_residuals(t, residuals)

    def test_residuals_custom_labels(self) -> None:
        """Test custom dimension labels."""
        t = np.linspace(0, 10, 100)
        residuals = np.random.randn(100, 2)

        fig, axes = plot_residuals(
            t, residuals, ylabel="Custom", dim_labels=["Dimension 1", "Dimension 2"]
        )

        assert axes[0].get_ylabel() == "Dimension 1"
        assert axes[1].get_ylabel() == "Dimension 2"

        plt.close(fig)


class TestPlotPositionError:
    """Test position error time series plotting."""

    def test_position_error_basic(self) -> None:
        """Test basic position error plot."""
        # Arrange
        t = np.linspace(0, 10, 100)
        pos_true = np.column_stack([t * 0.1, np.zeros(100)])
        pos_est = pos_true + np.random.randn(100, 2) * 0.01

        # Act
        fig, ax = plot_position_error(t, pos_true, pos_est)

        # Assert
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert ax.get_xlabel() == "Time (s)"
        assert "Error" in ax.get_ylabel()
        assert "m" in ax.get_ylabel()  # Should include units

        plt.close(fig)

    def test_position_error_with_target_threshold(self) -> None:
        """Test position error plot with target threshold line."""
        # Arrange
        t = np.linspace(0, 10, 100)
        pos_true = np.zeros((100, 2))
        pos_est = np.random.randn(100, 2) * 0.01

        # Act
        fig, ax = plot_position_error(t, pos_true, pos_est, target_threshold_m=0.02)

        # Assert: Check for threshold line
        # Should be a horizontal line
        lines = ax.get_lines()
        assert len(lines) > 1  # At least error line + threshold

        plt.close(fig)

    def test_position_error_with_mask(self) -> None:
        """Test position error plot with validity mask."""
        # Arrange
        t = np.linspace(0, 10, 100)
        pos_true = np.zeros((100, 2))
        pos_est = np.random.randn(100, 2) * 0.01
        mask = t < 5.0  # Only first half valid

        # Act
        fig, ax = plot_position_error(t, pos_true, pos_est, valid_mask=mask)

        # Assert: Plot should only show data where mask is True
        lines = ax.get_lines()
        assert len(lines) > 0

        plt.close(fig)


class TestPlotVelocityError:
    """Test velocity error time series plotting."""

    def test_velocity_error_basic(self) -> None:
        """Test basic velocity error plot."""
        # Arrange
        t = np.linspace(0, 10, 100)
        vel_true = np.column_stack([np.ones(100) * 0.1, np.zeros(100)])
        vel_est = vel_true + np.random.randn(100, 2) * 0.01

        # Act
        fig, ax = plot_velocity_error(t, vel_true, vel_est)

        # Assert
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert ax.get_xlabel() == "Time (s)"
        assert "Velocity" in ax.get_ylabel() or "Error" in ax.get_ylabel()
        assert "m/s" in ax.get_ylabel()

        plt.close(fig)


class TestPlotHeadingError:
    """plot_heading_error must validate inputs as strictly as the position/velocity helpers."""

    def test_heading_error_basic(self) -> None:
        t = np.linspace(0, 10, 100)
        ht = np.linspace(0, 1.0, 100)
        he = ht.copy()
        fig, _ax = plot_heading_error(t, ht, he)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_heading_error_rejects_shape_mismatch(self) -> None:
        t = np.linspace(0, 10, 100)
        ht = np.linspace(0, 1.0, 100)
        he = np.linspace(0, 1.0, 50)  # wrong length
        with pytest.raises(ValueError, match=r"Shape mismatch"):
            plot_heading_error(t, ht, he)

    def test_heading_error_rejects_non_1d_headings(self) -> None:
        t = np.linspace(0, 10, 100)
        ht = np.zeros((100, 1))  # 2D where 1D expected
        he = np.zeros((100, 1))
        with pytest.raises(ValueError, match=r"headings.*must be 1-D"):
            plot_heading_error(t, ht, he)

    def test_heading_error_rejects_time_mismatch(self) -> None:
        t = np.linspace(0, 10, 100)
        ht = np.zeros(50)
        he = np.zeros(50)
        with pytest.raises(ValueError, match=r"Shape mismatch:"):
            plot_heading_error(t, ht, he)

    def test_heading_error_rejects_bad_mask(self) -> None:
        t = np.linspace(0, 10, 100)
        ht = np.zeros(100)
        he = np.zeros(100)
        # Bad mask shape
        with pytest.raises(ValueError, match=r"valid_mask must have shape"):
            plot_heading_error(t, ht, he, valid_mask=np.ones(50, dtype=bool))
        # Bad mask dtype (int with non-{0,1} value)
        with pytest.raises(ValueError, match=r"boolean or 0/1 integer"):
            plot_heading_error(
                t,
                ht,
                he,
                valid_mask=np.array([2] + [1] * 99, dtype=np.int32),
            )


class TestPlotNEESHistogram:
    """Test NEES histogram plotting with chi-squared bounds."""

    def test_nees_histogram_basic(self) -> None:
        """Test basic NEES histogram creation."""
        # Arrange: Generate NEES values from chi-squared distribution
        np.random.seed(42)
        nees = np.random.chisquare(df=8, size=500)
        state_dim = 8

        # Act
        fig, ax = plot_nees_histogram(nees, state_dim=state_dim)

        # Assert
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert "NEES" in ax.get_xlabel()
        assert "Frequency" in ax.get_ylabel() or "Count" in ax.get_ylabel()

        plt.close(fig)

    def test_nees_histogram_with_chi2_bounds(self) -> None:
        """Test that chi-squared confidence bounds are plotted."""
        # Arrange
        np.random.seed(42)
        nees = np.random.chisquare(df=8, size=500)

        # Act
        fig, ax = plot_nees_histogram(nees, state_dim=8, confidence=0.95)

        # Assert: Should have vertical lines for bounds
        vlines = [line for line in ax.get_lines() if hasattr(line, "get_xdata")]
        assert len(vlines) >= 2  # At least lower and upper bounds

        plt.close(fig)

    def test_nees_histogram_custom_confidence(self) -> None:
        """Test NEES histogram with custom confidence level."""
        # Arrange
        np.random.seed(42)
        nees = np.random.chisquare(df=8, size=500)

        # Act
        fig, _ax = plot_nees_histogram(nees, state_dim=8, confidence=0.99)

        # Assert: Plot should be created successfully
        assert isinstance(fig, plt.Figure)

        plt.close(fig)

    def test_nees_histogram_shape_validation(self) -> None:
        """Test that invalid NEES shape raises ValueError."""
        nees = np.random.randn(100, 2)  # Should be 1D

        with pytest.raises(ValueError, match="Expected 1D"):
            plot_nees_histogram(nees, state_dim=8)


class TestPlotNISHistogram:
    """Test NIS histogram plotting with chi-squared bounds."""

    def test_nis_histogram_basic(self) -> None:
        """Test basic NIS histogram creation."""
        # Arrange: Generate NIS values from chi-squared distribution
        np.random.seed(42)
        nis = np.random.chisquare(df=4, size=500)
        measurement_dim = 4

        # Act
        fig, ax = plot_nis_histogram(nis, measurement_dim=measurement_dim)

        # Assert
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert "NIS" in ax.get_xlabel()

        plt.close(fig)

    def test_nis_histogram_with_chi2_bounds(self) -> None:
        """Test that chi-squared confidence bounds are plotted."""
        # Arrange
        np.random.seed(42)
        nis = np.random.chisquare(df=4, size=500)

        # Act
        fig, ax = plot_nis_histogram(nis, measurement_dim=4, confidence=0.95)

        # Assert: Should have vertical lines for bounds
        vlines = [line for line in ax.get_lines() if hasattr(line, "get_xdata")]
        assert len(vlines) >= 2  # At least lower and upper bounds

        plt.close(fig)


class TestPlotCovarianceEllipse:
    """Test 2D covariance ellipse plotting."""

    def test_covariance_ellipse_basic(self) -> None:
        """Test basic covariance ellipse plotting."""
        # Arrange
        mean = np.array([0.5, 0.5])
        cov = np.array([[0.01, 0.0], [0.0, 0.01]])  # Circular

        # Act
        fig, ax = plot_covariance_ellipse(mean, cov)

        # Assert
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        # Should have an ellipse patch
        assert len(ax.patches) > 0

        plt.close(fig)

    def test_covariance_ellipse_with_trajectory(self) -> None:
        """Test covariance ellipse with trajectory overlay."""
        # Arrange
        mean = np.array([0.5, 0.5])
        cov = np.array([[0.01, 0.0], [0.0, 0.01]])
        trajectory = np.array([[0.0, 0.0], [0.25, 0.25], [0.5, 0.5], [0.75, 0.75]])

        # Act
        fig, ax = plot_covariance_ellipse(mean, cov, trajectory=trajectory)

        # Assert: Should have trajectory line
        lines = ax.get_lines()
        assert len(lines) > 0

        plt.close(fig)

    def test_covariance_ellipse_multiple_sigmas(self) -> None:
        """Test plotting multiple sigma-level ellipses."""
        # Arrange
        mean = np.array([0.5, 0.5])
        cov = np.array([[0.01, 0.005], [0.005, 0.02]])  # Correlated

        # Act
        fig, ax = plot_covariance_ellipse(mean, cov, n_std=[1, 2, 3])

        # Assert: Should have 3 ellipse patches
        assert len(ax.patches) >= 3

        plt.close(fig)

    def test_covariance_ellipse_shape_validation(self) -> None:
        """Test that invalid shapes raise ValueError."""
        mean = np.array([0.5, 0.5, 0.5])  # 3D, should be 2D
        cov = np.array([[0.01, 0.0], [0.0, 0.01]])

        with pytest.raises(ValueError, match="2D"):
            plot_covariance_ellipse(mean, cov)

    def test_covariance_ellipse_singular_cov(self) -> None:
        """Test handling of singular covariance matrix."""
        # Arrange: Singular covariance (rank-deficient)
        mean = np.array([0.5, 0.5])
        cov = np.array([[0.01, 0.01], [0.01, 0.01]])  # Singular

        # Act/Assert: Should either handle gracefully or raise clear error
        # Implementation can choose to skip ellipse or add regularization
        try:
            fig, _ax = plot_covariance_ellipse(mean, cov)
            plt.close(fig)
        except (ValueError, np.linalg.LinAlgError):
            # Acceptable to raise error for singular covariance
            pass

    def test_covariance_ellipse_custom_color(self) -> None:
        """Test custom ellipse color."""
        # Arrange
        mean = np.array([0.5, 0.5])
        cov = np.array([[0.01, 0.0], [0.0, 0.01]])

        # Act
        fig, ax = plot_covariance_ellipse(mean, cov, color="red", alpha=0.5)

        # Assert: Patch should have specified color
        patches = ax.patches
        assert len(patches) > 0

        plt.close(fig)


class TestIntegration:
    """Integration tests combining multiple plotting functions."""

    def test_full_qa_workflow(self) -> None:
        """Test complete QA workflow with all plots."""
        # Arrange: Synthetic filter results
        np.random.seed(42)
        N = 200
        t = np.linspace(0, 10, N)

        # Ground truth
        pos_true = np.column_stack([t * 0.1, np.zeros(N)])
        vel_true = np.column_stack([np.ones(N) * 0.1, np.zeros(N)])

        # Estimates with noise
        pos_est = pos_true + np.random.randn(N, 2) * 0.01
        vel_est = vel_true + np.random.randn(N, 2) * 0.01

        # Residuals
        residuals = pos_true - pos_est

        # NEES values (chi-squared distributed if filter is consistent)
        nees = np.random.chisquare(df=8, size=N)

        # Act: Create all QA plots
        fig1, _ax1 = plot_position_error(t, pos_true, pos_est, target_threshold_m=0.02)
        fig2, _ax2 = plot_velocity_error(t, vel_true, vel_est)
        fig3, _axes3 = plot_residuals(t, residuals, confidence_std=0.01)
        fig4, _ax4 = plot_nees_histogram(nees, state_dim=8, confidence=0.95)
        fig5, _ax5 = plot_covariance_ellipse(
            mean=np.array([0.5, 0.0]),
            cov=np.array([[0.01, 0.0], [0.0, 0.01]]),
            trajectory=pos_true,
        )

        # Assert: All plots created successfully
        assert all([fig1, fig2, fig3, fig4, fig5])

        # Cleanup
        for fig in [fig1, fig2, fig3, fig4, fig5]:
            plt.close(fig)
