"""
Tests for LED wall reflection artifacts in rat_imu.py

LED reflections near walls simulate realistic vision artifacts where
LED detections appear at mirrored positions when the rat is near arena boundaries.
"""

import numpy as np
import pytest

from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu


class TestLEDWallReflectionConfiguration:
    """Test LED wall reflection config parameters."""

    def test_default_no_reflections(self):
        """Default config should have no wall reflections enabled."""
        cfg = RatIMUSimConfig()
        assert cfg.led_wall_reflection_prob == 0.0
        assert cfg.led_wall_reflection_distance == 0.2

    def test_enable_reflections(self):
        """Can enable wall reflections with non-zero probability."""
        cfg = RatIMUSimConfig(led_wall_reflection_prob=0.3)
        assert cfg.led_wall_reflection_prob == 0.3

    def test_configure_reflection_distance(self):
        """Can configure distance threshold for reflections."""
        cfg = RatIMUSimConfig(
            led_wall_reflection_prob=0.5,
            led_wall_reflection_distance=0.15,
        )
        assert cfg.led_wall_reflection_distance == 0.15


class TestLEDReflectionGeometry:
    """Test geometric properties of reflected LED positions."""

    def test_reflection_creates_mirrored_position(self):
        """Reflected LEDs should appear at mirrored positions across walls."""
        cfg = RatIMUSimConfig(
            duration_s=2.0,
            fs_imu=100.0,
            fs_cam=30.0,
            arena_w=1.0,
            arena_h=1.0,
            led_wall_reflection_prob=1.0,  # Always reflect
            led_wall_reflection_distance=0.3,  # Trigger within 30cm of wall
            cam_dropout_prob=0.0,  # No dropouts
            cam_sigma_m=0.0,  # No noise for exact geometry check
            use_second_led=True,
            # Start near left wall (x=0)
            m0=np.array([0.15, 0.5, 0.0, 0.0, 0.0]),  # x=15cm from left wall
        )
        sim = simulate_rat_imu(cfg, seed=42)

        # Find frames where rat is near left wall (x < 0.3m)
        # Note: LED1 is rear (-2cm), LED2 is front (+2cm) from center
        Z1 = sim["Z_cam_led1"]
        truth_x = np.interp(sim["t_cam_exp"], sim["t_imu"], sim["X_truth"][:, 0])

        # Frames near left wall
        near_wall = truth_x < 0.3
        reflected_frames = sim["led_reflection_applied"]  # Should exist

        # At least some frames should have reflections
        assert np.sum(reflected_frames) > 0, "Expected reflections to occur near wall"

        # Reflected LEDs should be mirrored across wall (x → -x)
        # Non-reflected: LED_x ≈ truth_x ± offset
        # Reflected: LED_x ≈ -(truth_x ± offset)
        reflected_idx = np.where(reflected_frames & near_wall)[0]
        if len(reflected_idx) > 0:
            # Check that reflected x-coordinates are negative (mirrored)
            reflected_x_led1 = Z1[reflected_idx, 0]
            assert np.all(reflected_x_led1 < 0), "Reflected LED1 should have negative x"

    def test_reflection_probability_controls_frequency(self):
        """Higher reflection probability should cause more reflections."""
        cfg_low = RatIMUSimConfig(
            duration_s=10.0,
            fs_cam=30.0,
            led_wall_reflection_prob=0.1,
            led_wall_reflection_distance=0.3,
            use_second_led=True,
            m0=np.array([0.15, 0.5, 0.0, 0.0, 0.0]),  # Near wall
        )
        cfg_high = RatIMUSimConfig(
            duration_s=10.0,
            fs_cam=30.0,
            led_wall_reflection_prob=0.9,
            led_wall_reflection_distance=0.3,
            use_second_led=True,
            m0=np.array([0.15, 0.5, 0.0, 0.0, 0.0]),  # Near wall
        )

        sim_low = simulate_rat_imu(cfg_low, seed=123)
        sim_high = simulate_rat_imu(cfg_high, seed=123)

        n_reflect_low = np.sum(sim_low["led_reflection_applied"])
        n_reflect_high = np.sum(sim_high["led_reflection_applied"])

        assert n_reflect_high > n_reflect_low, "Higher prob should cause more reflections"

    def test_reflection_distance_threshold(self):
        """Reflections only occur within configured distance from walls."""
        cfg = RatIMUSimConfig(
            duration_s=5.0,
            fs_cam=30.0,
            led_wall_reflection_prob=1.0,  # Always reflect if near wall
            led_wall_reflection_distance=0.2,  # 20cm threshold
            use_second_led=True,
            m0=np.array([0.5, 0.5, 0.0, 0.0, 0.0]),  # Start in center (far from walls)
        )
        sim = simulate_rat_imu(cfg, seed=42)

        reflected = sim["led_reflection_applied"]

        # For all reflected frames, at least one LED should be near a wall
        reflected_idx = np.where(reflected)[0]
        for idx in reflected_idx:
            # Reflections are based on rat center position, not individual LED positions
            # Check that rat is near wall
            truth_x = np.interp(sim["t_cam_exp"][idx], sim["t_imu"], sim["X_truth"][:, 0])
            truth_y = np.interp(sim["t_cam_exp"][idx], sim["t_imu"], sim["X_truth"][:, 1])
            dist_rat = min(truth_x, cfg.arena_w - truth_x, truth_y, cfg.arena_h - truth_y)
            assert (
                dist_rat <= cfg.led_wall_reflection_distance
            ), f"Reflection occurred when rat was {dist_rat:.2f}m from wall (threshold={cfg.led_wall_reflection_distance})"


class TestReflectionAppliedMask:
    """Test the led_reflection_applied output mask."""

    def test_reflection_mask_exists(self):
        """Simulation should return led_reflection_applied mask."""
        cfg = RatIMUSimConfig(
            duration_s=2.0,
            led_wall_reflection_prob=0.5,
        )
        sim = simulate_rat_imu(cfg, seed=42)

        assert "led_reflection_applied" in sim, "Missing led_reflection_applied mask"
        assert sim["led_reflection_applied"].shape == (sim["Z_cam_led1"].shape[0],)

    def test_mask_is_boolean(self):
        """led_reflection_applied should be boolean array."""
        cfg = RatIMUSimConfig(duration_s=2.0, led_wall_reflection_prob=0.5)
        sim = simulate_rat_imu(cfg, seed=42)

        assert sim["led_reflection_applied"].dtype == bool

    def test_zero_probability_gives_no_reflections(self):
        """Zero reflection probability should result in all-False mask."""
        cfg = RatIMUSimConfig(
            duration_s=2.0,
            led_wall_reflection_prob=0.0,
        )
        sim = simulate_rat_imu(cfg, seed=42)

        assert np.sum(sim["led_reflection_applied"]) == 0


class TestReflectionInteractionWithDropouts:
    """Test interaction between reflections and camera dropouts."""

    def test_reflections_respect_dropout_mask(self):
        """Reflections should only apply to visible LEDs."""
        cfg = RatIMUSimConfig(
            duration_s=5.0,
            fs_cam=30.0,
            led_wall_reflection_prob=0.8,
            led_wall_reflection_distance=0.3,
            cam_dropout_prob=0.5,
            use_second_led=True,
            m0=np.array([0.15, 0.5, 0.0, 0.0, 0.0]),  # Near wall
        )
        sim = simulate_rat_imu(cfg, seed=42)

        # Reflections should only apply when LEDs are visible
        reflected = sim["led_reflection_applied"]
        mask_led1 = sim["mask_led1"]
        mask_led2 = sim["mask_led2"]

        # Reflection can only happen if at least one LED is visible
        # (Otherwise the position would be NaN anyway)
        for idx in np.where(reflected)[0]:
            assert (
                mask_led1[idx] or mask_led2[idx]
            ), f"Frame {idx}: reflection applied but both LEDs dropped"

    def test_nan_positions_unchanged_by_reflections(self):
        """NaN LED positions (dropouts) should remain NaN after reflections."""
        cfg = RatIMUSimConfig(
            duration_s=3.0,
            led_wall_reflection_prob=1.0,
            cam_dropout_prob=0.5,
            use_second_led=True,
        )
        sim = simulate_rat_imu(cfg, seed=42)

        Z1 = sim["Z_cam_led1"]
        Z2 = sim["Z_cam_led2"]
        mask_led1 = sim["mask_led1"]
        mask_led2 = sim["mask_led2"]

        # Dropped LEDs should still be NaN
        assert np.all(np.isnan(Z1[~mask_led1]))
        assert np.all(np.isnan(Z2[~mask_led2]))


class TestReflectionDeterminism:
    """Test deterministic behavior with seeds."""

    def test_same_seed_gives_same_reflections(self):
        """Same seed should produce identical reflection patterns."""
        cfg = RatIMUSimConfig(
            duration_s=3.0,
            led_wall_reflection_prob=0.5,
            use_second_led=True,
        )

        sim1 = simulate_rat_imu(cfg, seed=999)
        sim2 = simulate_rat_imu(cfg, seed=999)

        np.testing.assert_array_equal(
            sim1["led_reflection_applied"], sim2["led_reflection_applied"]
        )
        np.testing.assert_allclose(sim1["Z_cam_led1"], sim2["Z_cam_led1"])
        np.testing.assert_allclose(sim1["Z_cam_led2"], sim2["Z_cam_led2"])

    def test_different_seed_gives_different_reflections(self):
        """Different seeds should produce different reflection patterns."""
        cfg = RatIMUSimConfig(
            duration_s=5.0,
            led_wall_reflection_prob=0.5,
            led_wall_reflection_distance=0.5,  # Large threshold to ensure near-wall frames
            use_second_led=True,
            m0=np.array([0.2, 0.5, 0.0, 0.0, 0.0]),  # Start near wall
        )

        sim1 = simulate_rat_imu(cfg, seed=111)
        sim2 = simulate_rat_imu(cfg, seed=222)

        # At least one simulation should have reflections
        # (If both have zero, test is not meaningful)
        total_reflections = np.sum(sim1["led_reflection_applied"]) + np.sum(
            sim2["led_reflection_applied"]
        )
        if total_reflections == 0:
            # Skip if no reflections occurred (motion didn't stay near walls)
            pytest.skip("No reflections occurred in either simulation")

        # Reflection masks should differ (with high probability given different seeds)
        assert not np.array_equal(
            sim1["led_reflection_applied"], sim2["led_reflection_applied"]
        ), "Different seeds should give different reflections"


class TestReflectionEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_single_led_reflections(self):
        """Reflections should work with single LED (no LED2)."""
        cfg = RatIMUSimConfig(
            duration_s=3.0,
            led_wall_reflection_prob=0.5,
            use_second_led=False,  # Single LED
            m0=np.array([0.15, 0.5, 0.0, 0.0, 0.0]),
        )
        sim = simulate_rat_imu(cfg, seed=42)

        # Should still have reflection mask
        assert "led_reflection_applied" in sim
        # LED2 should be all NaN
        assert np.all(np.isnan(sim["Z_cam_led2"]))

    def test_corner_position_reflections(self):
        """Test reflections when rat is in corner (near multiple walls)."""
        cfg = RatIMUSimConfig(
            duration_s=2.0,
            fs_cam=30.0,
            led_wall_reflection_prob=1.0,
            led_wall_reflection_distance=0.3,
            use_second_led=True,
            # Start in corner
            m0=np.array([0.1, 0.1, 0.0, 0.0, 0.0]),
        )
        sim = simulate_rat_imu(cfg, seed=42)

        # Should have reflections when in corner
        assert np.sum(sim["led_reflection_applied"]) > 0

    def test_reflection_with_zero_distance_threshold(self):
        """Zero distance threshold should prevent all reflections."""
        cfg = RatIMUSimConfig(
            duration_s=2.0,
            led_wall_reflection_prob=1.0,
            led_wall_reflection_distance=0.0,  # Never near enough
        )
        sim = simulate_rat_imu(cfg, seed=42)

        # Should have no reflections
        assert np.sum(sim["led_reflection_applied"]) == 0
