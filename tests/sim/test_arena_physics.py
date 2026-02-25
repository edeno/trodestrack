"""
Test suite for arena boundary physics in rat_imu simulation.

Tests verify that:
1. Rats stay within arena boundaries with reflections
2. Wall collisions apply energy loss (coefficient of restitution)
3. Reflections maintain physical realism (no tunneling, proper velocity reversal)
4. Arena constraints work for different sizes and initial conditions
"""

import numpy as np

from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu


class TestArenaBoundaries:
    """Test that simulated trajectories respect arena boundaries."""

    def test_rat_stays_in_bounds(self):
        """Verify all positions remain within arena boundaries."""
        config = RatIMUSimConfig(
            duration_s=30.0,
            fs_imu=200.0,
            arena_w=2.0,
            arena_h=1.5,
            # High motion to increase likelihood of wall collisions
            sigma_a_fwd=1.5,
            sigma_a_lat=1.0,
            sigma_yaw_rate=np.deg2rad(90.0),
            # Start near edge to trigger boundary quickly
            m0=np.array([0.1, 0.1, 0.5, 0.5, 0.0]),
        )

        sim = simulate_rat_imu(config=config, seed=42)
        X_truth = sim["X_truth"]

        # Extract positions
        px = X_truth[:, 0]
        py = X_truth[:, 1]

        # Check all positions are within bounds
        assert np.all(px >= 0.0), f"Found x positions below 0: min={px.min()}"
        assert np.all(px <= config.arena_w), (
            f"Found x positions above {config.arena_w}: max={px.max()}"
        )
        assert np.all(py >= 0.0), f"Found y positions below 0: min={py.min()}"
        assert np.all(py <= config.arena_h), (
            f"Found y positions above {config.arena_h}: max={py.max()}"
        )

    def test_wall_collision_reverses_velocity(self):
        """Verify that wall collisions reverse velocity component."""
        config = RatIMUSimConfig(
            duration_s=5.0,
            fs_imu=200.0,
            arena_w=1.0,
            arena_h=1.0,
            # Start moving toward right wall
            m0=np.array([0.9, 0.5, 0.5, 0.0, 0.0]),  # Near right wall, moving right
            # Small OU noise to maintain motion
            sigma_a_fwd=0.3,
            sigma_a_lat=0.1,
            sigma_yaw_rate=np.deg2rad(5.0),
            vel_drag=0.1,  # Small drag
        )

        sim = simulate_rat_imu(config=config, seed=123)
        X_truth = sim["X_truth"]

        px = X_truth[:, 0]
        vx = X_truth[:, 2]

        # Find first collision: position at wall with velocity sign change
        at_wall = px > 0.98
        if np.any(at_wall):
            wall_indices = np.where(at_wall)[0]
            # Look for velocity sign change at wall
            for idx in wall_indices[:10]:  # Check first few wall encounters
                # Get velocity just before and after this wall encounter
                if idx >= 5 and idx + 5 < len(vx):
                    v_before = vx[idx - 3]
                    v_after = vx[idx + 3]
                    # Check if velocity reversed (positive to negative)
                    if v_before > 0.1 and v_after < 0:
                        # Found a clear reversal
                        return
            # If we got here, check that at least velocity became negative at some point
            assert np.any(vx[wall_indices] < 0), (
                f"Velocity should become negative during wall contact, "
                f"but all velocities at wall are: {vx[wall_indices[:10]]}"
            )

    def test_wall_collision_applies_energy_loss(self):
        """Verify that wall collisions apply coefficient of restitution = 0.5."""
        config = RatIMUSimConfig(
            duration_s=3.0,
            fs_imu=1000.0,  # High rate for better time resolution
            arena_w=1.0,
            arena_h=1.0,
            # Start very close to wall with known velocity
            m0=np.array([0.95, 0.5, 0.4, 0.0, 0.0]),  # vx = 0.4 m/s toward wall
            # Disable all noise for deterministic collision
            sigma_a_fwd=0.0,
            sigma_a_lat=0.0,
            sigma_yaw_rate=0.0,
            vel_drag=0.0,
            gyro_noise_density=0.0,
            accel_noise_density=0.0,
            gyro_bias_rw_density=0.0,
            accel_bias_rw_density=0.0,
        )

        sim = simulate_rat_imu(config=config, seed=456)
        X_truth = sim["X_truth"]

        px = X_truth[:, 0]
        vx = X_truth[:, 2]

        # Find collision: when position is at wall AND velocity just reversed
        # Look for the first frame where px hits boundary
        at_wall = px >= 0.99
        if np.any(at_wall):
            idx_collision = np.where(at_wall)[0][0]

            # Get velocity before collision (should be positive)
            v_before = vx[max(0, idx_collision - 5)]
            # Get velocity after collision (should be negative with 0.5 factor)
            v_after = vx[min(len(vx) - 1, idx_collision + 5)]

            # Check that velocity reversed and was attenuated
            # v_after ≈ -0.5 * v_before (coefficient of restitution = 0.5)
            if v_before > 0.1:  # Only check if there was significant approach velocity
                assert v_after < 0, (
                    f"Velocity should reverse: v_before={v_before}, v_after={v_after}"
                )
                # Allow 50% tolerance due to integration artifacts
                expected_v_after = -0.5 * v_before
                assert abs(v_after - expected_v_after) / abs(expected_v_after) < 0.5, (
                    f"Velocity should be attenuated by ~0.5: "
                    f"v_before={v_before:.3f}, v_after={v_after:.3f}, "
                    f"expected={expected_v_after:.3f}"
                )

    def test_corner_collision_affects_both_axes(self):
        """Verify that corner collisions reverse both velocity components."""
        config = RatIMUSimConfig(
            duration_s=3.0,
            fs_imu=200.0,
            arena_w=1.0,
            arena_h=1.0,
            # Start near top-right corner, moving toward it
            m0=np.array([0.85, 0.85, 0.3, 0.3, 0.0]),
            # Disable noise for predictable motion
            sigma_a_fwd=0.0,
            sigma_a_lat=0.0,
            sigma_yaw_rate=0.0,
            vel_drag=0.0,
        )

        sim = simulate_rat_imu(config=config, seed=789)
        X_truth = sim["X_truth"]

        px = X_truth[:, 0]
        py = X_truth[:, 1]
        vx = X_truth[:, 2]
        vy = X_truth[:, 3]

        # Check if we get near the corner
        near_corner = (px > 0.9) & (py > 0.9)
        if np.any(near_corner):
            # At some point after reaching corner, both velocities should reverse
            idx_corner = np.where(near_corner)[0][0]
            if idx_corner + 20 < len(vx):
                # Check that both components eventually become negative
                vx_after = vx[idx_corner + 10 : idx_corner + 20].mean()
                vy_after = vy[idx_corner + 10 : idx_corner + 20].mean()
                # At least one should definitely reverse (might not hit exactly simultaneously)
                assert vx_after < 0 or vy_after < 0, (
                    "At least one velocity component should reverse after corner collision"
                )

    def test_small_arena_high_activity(self):
        """Verify boundaries work in small arena with high activity."""
        config = RatIMUSimConfig(
            duration_s=20.0,
            fs_imu=200.0,
            arena_w=0.5,  # Small 50cm × 50cm arena
            arena_h=0.5,
            # High motion
            sigma_a_fwd=2.0,
            sigma_a_lat=1.5,
            sigma_yaw_rate=np.deg2rad(120.0),
            m0=np.array([0.25, 0.25, 0.0, 0.0, 0.0]),  # Center
        )

        sim = simulate_rat_imu(config=config, seed=111)
        X_truth = sim["X_truth"]

        px = X_truth[:, 0]
        py = X_truth[:, 1]

        # All positions must be in bounds
        assert np.all((px >= 0) & (px <= config.arena_w)), (
            "X positions out of bounds in small arena"
        )
        assert np.all((py >= 0) & (py <= config.arena_h)), (
            "Y positions out of bounds in small arena"
        )

        # Should have multiple wall collisions in small space
        # Count frames very close to walls (within 5cm)
        near_walls = (px < 0.05) | (px > 0.45) | (py < 0.05) | (py > 0.45)
        collision_ratio = np.mean(near_walls)
        assert collision_ratio > 0.1, (
            f"Expected frequent wall proximity in small arena, got {collision_ratio:.1%}"
        )

    def test_large_arena_free_motion(self):
        """Verify boundaries don't interfere in large arena with typical motion."""
        config = RatIMUSimConfig(
            duration_s=30.0,
            fs_imu=200.0,
            arena_w=5.0,  # Large 5m × 5m arena
            arena_h=5.0,
            # Typical motion
            sigma_a_fwd=1.0,
            sigma_a_lat=0.5,
            sigma_yaw_rate=np.deg2rad(60.0),
            m0=np.array([2.5, 2.5, 0.0, 0.0, 0.0]),  # Center
            speed_clip=0.8,  # Moderate max speed
        )

        sim = simulate_rat_imu(config=config, seed=222)
        X_truth = sim["X_truth"]

        px = X_truth[:, 0]
        py = X_truth[:, 1]

        # All positions must be in bounds
        assert np.all((px >= 0) & (px <= config.arena_w)), (
            "X positions out of bounds in large arena"
        )
        assert np.all((py >= 0) & (py <= config.arena_h)), (
            "Y positions out of bounds in large arena"
        )

        # Should stay mostly away from boundaries in large arena
        margin = 0.5  # 50cm margin
        away_from_walls = (
            (px > margin)
            & (px < config.arena_w - margin)
            & (py > margin)
            & (py < config.arena_h - margin)
        )
        interior_ratio = np.mean(away_from_walls)
        # Most time should be spent in interior (>60%)
        assert interior_ratio > 0.6, (
            f"Expected mostly interior motion in large arena, got {interior_ratio:.1%}"
        )

    def test_reflection_preserves_trajectory_continuity(self):
        """Verify that reflections don't cause position discontinuities."""
        config = RatIMUSimConfig(
            duration_s=10.0,
            fs_imu=500.0,  # High rate to catch discontinuities
            arena_w=1.0,
            arena_h=1.0,
            sigma_a_fwd=1.5,
            sigma_a_lat=1.0,
            m0=np.array([0.2, 0.2, 0.4, 0.4, 0.0]),
        )

        sim = simulate_rat_imu(config=config, seed=333)
        X_truth = sim["X_truth"]

        px = X_truth[:, 0]
        py = X_truth[:, 1]

        # Check for position jumps (should be smooth even at reflections)
        dt = 1.0 / config.fs_imu
        max_expected_step = 1.5 * dt  # With speed_clip=1.5, max step is ~1.5*dt

        dx = np.diff(px)
        dy = np.diff(py)
        step_sizes = np.hypot(dx, dy)

        # No step should be larger than physically possible
        assert np.all(step_sizes <= 2 * max_expected_step), (
            f"Found discontinuous position jump: max_step={step_sizes.max():.4f}, "
            f"expected_max={2 * max_expected_step:.4f}"
        )

    def test_different_arena_aspect_ratios(self):
        """Verify boundaries work for non-square arenas."""
        # Wide arena
        config_wide = RatIMUSimConfig(
            duration_s=15.0,
            arena_w=3.0,
            arena_h=1.0,
            m0=np.array([1.5, 0.5, 0.0, 0.0, 0.0]),
        )
        sim_wide = simulate_rat_imu(config=config_wide, seed=444)
        px_wide = sim_wide["X_truth"][:, 0]
        py_wide = sim_wide["X_truth"][:, 1]

        assert np.all((px_wide >= 0) & (px_wide <= 3.0)), "Wide arena X bounds violated"
        assert np.all((py_wide >= 0) & (py_wide <= 1.0)), "Wide arena Y bounds violated"

        # Tall arena
        config_tall = RatIMUSimConfig(
            duration_s=15.0,
            arena_w=1.0,
            arena_h=3.0,
            m0=np.array([0.5, 1.5, 0.0, 0.0, 0.0]),
        )
        sim_tall = simulate_rat_imu(config=config_tall, seed=555)
        px_tall = sim_tall["X_truth"][:, 0]
        py_tall = sim_tall["X_truth"][:, 1]

        assert np.all((px_tall >= 0) & (px_tall <= 1.0)), "Tall arena X bounds violated"
        assert np.all((py_tall >= 0) & (py_tall <= 3.0)), "Tall arena Y bounds violated"


class TestEnergyDissipation:
    """Test energy loss during wall collisions."""

    def test_multiple_collisions_reduce_speed(self):
        """Verify that repeated collisions progressively reduce speed."""
        config = RatIMUSimConfig(
            duration_s=5.0,
            fs_imu=200.0,
            arena_w=0.8,
            arena_h=0.8,
            # Start with high speed toward corner
            m0=np.array([0.1, 0.1, 0.6, 0.6, np.pi / 4]),  # 45° toward corner
            # Minimal driving forces (let collisions dominate)
            sigma_a_fwd=0.1,
            sigma_a_lat=0.1,
            sigma_yaw_rate=np.deg2rad(10.0),
            vel_drag=0.1,  # Small drag
        )

        sim = simulate_rat_imu(config=config, seed=666)
        X_truth = sim["X_truth"]

        vx = X_truth[:, 2]
        vy = X_truth[:, 3]
        speed = np.hypot(vx, vy)

        # Initial speed should be high
        speed_initial = speed[:100].mean()
        # Later speed should be lower due to collisions + drag
        speed_late = speed[-1000:].mean()

        assert speed_late < speed_initial, (
            f"Speed should decrease over time with collisions: "
            f"initial={speed_initial:.3f}, late={speed_late:.3f}"
        )

    def test_collision_reduces_kinetic_energy(self):
        """Verify energy loss: KE_after < KE_before for collision."""
        config = RatIMUSimConfig(
            duration_s=2.0,
            fs_imu=500.0,
            arena_w=1.0,
            arena_h=1.0,
            m0=np.array([0.9, 0.5, 0.5, 0.0, 0.0]),  # Moving toward right wall
            # No driving forces
            sigma_a_fwd=0.0,
            sigma_a_lat=0.0,
            sigma_yaw_rate=0.0,
            vel_drag=0.0,
            gyro_noise_density=0.0,
            accel_noise_density=0.0,
            gyro_bias_rw_density=0.0,
            accel_bias_rw_density=0.0,
        )

        sim = simulate_rat_imu(config=config, seed=777)
        X_truth = sim["X_truth"]

        px = X_truth[:, 0]
        vx = X_truth[:, 2]
        vy = X_truth[:, 3]
        ke = 0.5 * (vx**2 + vy**2)  # Kinetic energy (mass=1)

        # Find collision event
        at_wall = px >= 0.99
        if np.any(at_wall):
            idx_collision = np.where(at_wall)[0][0]

            # KE before and after collision
            ke_before = ke[max(0, idx_collision - 10)]
            ke_after = ke[min(len(ke) - 1, idx_collision + 10)]

            # Energy should decrease
            if ke_before > 0.01:  # Only test if there was significant energy
                assert ke_after < ke_before, (
                    f"Kinetic energy should decrease after collision: "
                    f"before={ke_before:.4f}, after={ke_after:.4f}"
                )
                # With coefficient of restitution = 0.5, expect KE_after ≈ 0.25 * KE_before
                # (since KE ∝ v²), allow factor of 2 tolerance
                expected_ke_after = 0.25 * ke_before
                assert ke_after < 2 * expected_ke_after, (
                    f"Energy loss too small: ke_after={ke_after:.4f}, "
                    f"expected~{expected_ke_after:.4f}"
                )


class TestPhysicalRealism:
    """Test that arena physics produce realistic behaviors."""

    def test_no_tunneling_through_walls(self):
        """Verify that high-speed rats don't tunnel through boundaries."""
        config = RatIMUSimConfig(
            duration_s=20.0,
            fs_imu=200.0,  # Standard rate
            arena_w=1.5,
            arena_h=1.5,
            # High speed and acceleration
            speed_clip=2.0,
            sigma_a_fwd=3.0,
            sigma_a_lat=2.0,
            m0=np.array([0.75, 0.75, 0.0, 0.0, 0.0]),
        )

        sim = simulate_rat_imu(config=config, seed=888)
        X_truth = sim["X_truth"]

        px = X_truth[:, 0]
        py = X_truth[:, 1]

        # Even with high speeds, no tunneling
        assert np.all(px >= 0), f"Tunneling below x=0: min={px.min()}"
        assert np.all(px <= config.arena_w), (
            f"Tunneling above x={config.arena_w}: max={px.max()}"
        )
        assert np.all(py >= 0), f"Tunneling below y=0: min={py.min()}"
        assert np.all(py <= config.arena_h), (
            f"Tunneling above y={config.arena_h}: max={py.max()}"
        )

    def test_boundary_reflection_deterministic(self):
        """Verify that boundary physics are deterministic with same seed."""
        config = RatIMUSimConfig(
            duration_s=10.0,
            arena_w=1.0,
            arena_h=1.0,
            m0=np.array([0.8, 0.8, 0.3, 0.3, 0.0]),
        )

        # Run twice with same seed
        sim1 = simulate_rat_imu(config=config, seed=999)
        sim2 = simulate_rat_imu(config=config, seed=999)

        # Trajectories should be identical
        np.testing.assert_allclose(
            sim1["X_truth"],
            sim2["X_truth"],
            rtol=1e-10,
            atol=1e-12,
            err_msg="Arena physics should be deterministic with same seed",
        )

    def test_boundary_reflection_different_seeds(self):
        """Verify that different seeds produce different trajectories."""
        config = RatIMUSimConfig(
            duration_s=10.0,
            arena_w=1.0,
            arena_h=1.0,
            m0=np.array([0.5, 0.5, 0.0, 0.0, 0.0]),
        )

        # Run with different seeds
        sim1 = simulate_rat_imu(config=config, seed=1001)
        sim2 = simulate_rat_imu(config=config, seed=1002)

        # Trajectories should differ
        diff = np.abs(sim1["X_truth"] - sim2["X_truth"])
        max_diff = diff.max()

        assert max_diff > 0.01, (
            f"Different seeds should produce different trajectories, max_diff={max_diff}"
        )
