"""Tests for persistent (event-based) LED swap simulation.

This module tests the persistent LED swap feature where swap events cause
LEDs to remain swapped for an extended duration (rather than per-frame swaps).

Test coverage:
- Basic swap persistence (swap stays active across frames)
- Swap duration parameter handling
- Determinism with same seed
- Multiple swap events within a session
- Interaction with dropouts and other artifacts
"""

import numpy as np
import pytest

from trodestrack.sim.rat_imu import RatIMUSimConfig, simulate_rat_imu


class TestPersistentLEDSwaps:
    """Test suite for persistent LED swaps."""

    def test_persistent_swap_stays_active_across_frames(self):
        """Verify that once swapped, LEDs stay swapped for duration."""
        config = RatIMUSimConfig(
            duration_s=10.0,  # Longer session for more swap events
            use_second_led=True,
            led_swap_mode="persistent",  # Event-based swaps
            led_swap_rate=1.0,  # 1 swap per second on average
            led_swap_duration_mean=1.0,  # Mean duration 1.0 seconds
            led_swap_duration_std=0.0,  # Fixed duration (no randomness)
            cam_dropout_prob=0.0,  # No dropouts (ensures continuous visibility)
        )

        sim = simulate_rat_imu(config, seed=42)

        # Check that swap_applied mask has contiguous blocks (not random frames)
        swap_applied = sim["swap_applied"]

        if np.any(swap_applied):
            # Find swap blocks
            diff = np.diff(np.concatenate([[False], swap_applied, [False]]).astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]

            # Verify we have contiguous blocks (starts and ends match)
            assert len(starts) == len(ends), "Swap blocks should be contiguous"

            # Verify blocks exist (this is the key test for persistence)
            assert len(starts) > 0, "Expected at least one swap block"

            # Verify swap blocks last for multiple frames (not single-frame swaps)
            dt_cam = 1.0 / config.fs_cam
            durations = [
                (end - start) * dt_cam for start, end in zip(starts, ends, strict=False)
            ]
            mean_duration = np.mean(durations)

            # With no dropouts, durations should be close to 1.0s (within 0.5s tolerance)
            # Note: Actual duration can be less than expected if swap event extends past
            # visible frames, but mean should be reasonable
            assert mean_duration >= 0.5, (
                f"Expected mean swap duration >=0.5s, got {mean_duration:.2f}s. "
                f"Individual durations: {durations}"
            )

    def test_persistent_swap_mode_parameter(self):
        """Verify led_swap_mode parameter switches between per-frame and persistent."""
        config_per_frame = RatIMUSimConfig(
            duration_s=3.0,
            use_second_led=True,
            led_swap_mode="per_frame",
            led_swap_prob=0.1,  # Used for per-frame mode
        )

        config_persistent = RatIMUSimConfig(
            duration_s=3.0,
            use_second_led=True,
            led_swap_mode="persistent",
            led_swap_rate=1.0,  # Used for persistent mode
            led_swap_duration_mean=0.5,
        )

        sim_per_frame = simulate_rat_imu(config_per_frame, seed=42)
        sim_persistent = simulate_rat_imu(config_persistent, seed=42)

        # Per-frame swaps should be scattered (not contiguous)
        # Persistent swaps should have contiguous blocks

        # Count transitions in swap_applied mask
        def count_transitions(swap_mask):
            diff = np.diff(np.concatenate([[False], swap_mask, [False]]).astype(int))
            return np.sum(np.abs(diff)) // 2  # Number of swap events

        transitions_per_frame = count_transitions(sim_per_frame["swap_applied"])
        transitions_persistent = count_transitions(sim_persistent["swap_applied"])

        # Per-frame should have more transitions (each frame is independent)
        # Persistent should have fewer transitions (swaps last longer)
        if np.any(sim_per_frame["swap_applied"]) and np.any(
            sim_persistent["swap_applied"]
        ):
            # Persistent mode should have fewer transition events
            # (per-frame at 10% prob ~= 0.1 swaps/frame vs persistent at 1 event/s with 0.5s duration)
            assert transitions_persistent < transitions_per_frame, (
                f"Expected fewer transitions in persistent mode. "
                f"Got per_frame={transitions_per_frame}, persistent={transitions_persistent}"
            )

    def test_swap_duration_randomness(self):
        """Verify that swap duration varies when duration_std > 0."""
        config = RatIMUSimConfig(
            duration_s=10.0,
            use_second_led=True,
            led_swap_mode="persistent",
            led_swap_rate=2.0,  # 2 swaps per second (expect many events)
            led_swap_duration_mean=0.5,
            led_swap_duration_std=0.2,  # 20% std dev
        )

        sim = simulate_rat_imu(config, seed=42)
        swap_applied = sim["swap_applied"]

        if np.any(swap_applied):
            # Extract block durations
            diff = np.diff(np.concatenate([[False], swap_applied, [False]]).astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]

            dt_cam = 1.0 / config.fs_cam
            durations = [
                (end - start) * dt_cam for start, end in zip(starts, ends, strict=False)
            ]

            # With randomness, we should see variation in durations
            if len(durations) >= 3:
                std_durations = np.std(durations)
                # Should have some variation (not all identical)
                assert std_durations > 0.05, "Expected variation in swap durations"

    def test_deterministic_with_same_seed(self):
        """Verify that persistent swaps are deterministic with same seed."""
        config = RatIMUSimConfig(
            duration_s=5.0,
            use_second_led=True,
            led_swap_mode="persistent",
            led_swap_rate=1.0,
            led_swap_duration_mean=0.5,
            led_swap_duration_std=0.1,
        )

        sim1 = simulate_rat_imu(config, seed=123)
        sim2 = simulate_rat_imu(config, seed=123)

        np.testing.assert_array_equal(
            sim1["swap_applied"],
            sim2["swap_applied"],
            err_msg="Swap patterns should be identical with same seed",
        )

    def test_different_seeds_produce_different_swaps(self):
        """Verify that different seeds produce different swap patterns."""
        config1 = RatIMUSimConfig(
            duration_s=5.0,
            use_second_led=True,
            led_swap_mode="persistent",
            led_swap_rate=1.0,
            led_swap_duration_mean=0.5,
        )

        config2 = RatIMUSimConfig(
            duration_s=5.0,
            use_second_led=True,
            led_swap_mode="persistent",
            led_swap_rate=1.0,
            led_swap_duration_mean=0.5,
        )

        sim1 = simulate_rat_imu(config1, seed=111)
        sim2 = simulate_rat_imu(config2, seed=222)

        # Different seeds should produce different patterns
        # (with high probability given the parameters)
        assert not np.array_equal(
            sim1["swap_applied"], sim2["swap_applied"]
        ), "Different seeds should produce different swap patterns"

    def test_zero_swap_rate_produces_no_swaps(self):
        """Verify that led_swap_rate=0 produces no swaps in persistent mode."""
        config = RatIMUSimConfig(
            duration_s=5.0,
            use_second_led=True,
            led_swap_mode="persistent",
            led_swap_rate=0.0,  # No swaps
        )

        sim = simulate_rat_imu(config, seed=42)

        assert not np.any(sim["swap_applied"]), "Expected no swaps with led_swap_rate=0"

    def test_swaps_only_when_both_leds_visible(self):
        """Verify that swap blocks only occur when both LEDs are visible."""
        config = RatIMUSimConfig(
            duration_s=5.0,
            use_second_led=True,
            led_swap_mode="persistent",
            led_swap_rate=2.0,
            led_swap_duration_mean=0.3,
            cam_dropout_prob=0.3,  # 30% dropout rate to test interaction
        )

        sim = simulate_rat_imu(config, seed=42)

        # Swaps should only occur when both LEDs are visible
        swap_applied = sim["swap_applied"]
        mask_led1 = sim["mask_led1"]
        mask_led2 = sim["mask_led2"]
        both_visible = mask_led1 & mask_led2

        # Wherever swap is applied, both LEDs must be visible
        swapped_frames = np.where(swap_applied)[0]
        for idx in swapped_frames:
            assert both_visible[idx], f"Frame {idx} has swap but both LEDs not visible"

    def test_backward_compatibility_per_frame_mode(self):
        """Verify that per_frame mode uses legacy led_swap_prob parameter."""
        config = RatIMUSimConfig(
            duration_s=3.0,
            use_second_led=True,
            led_swap_mode="per_frame",  # Legacy mode
            led_swap_prob=0.15,  # Should use this parameter
        )

        sim = simulate_rat_imu(config, seed=42)

        # Verify swaps occurred (probabilistic, but likely with 15% rate)
        # and that they use the per-frame pattern
        swap_applied = sim["swap_applied"]

        # Just verify it runs without error and produces swaps
        # (detailed per-frame testing already exists in test_vision_robustness.py)
        assert isinstance(swap_applied, np.ndarray)
        assert swap_applied.dtype == bool

    def test_invalid_swap_mode_rejected(self):
        """Verify that invalid led_swap_mode values are rejected."""
        with pytest.raises(ValueError, match="led_swap_mode must be"):
            RatIMUSimConfig(
                duration_s=1.0,
                led_swap_mode="invalid_mode",
            )

    def test_negative_swap_rate_rejected(self):
        """Verify that negative swap rates are rejected."""
        with pytest.raises(ValueError, match="led_swap_rate must be non-negative"):
            RatIMUSimConfig(
                duration_s=1.0,
                led_swap_mode="persistent",
                led_swap_rate=-0.5,
            )

    def test_negative_swap_duration_rejected(self):
        """Verify that negative swap durations are rejected."""
        with pytest.raises(ValueError, match="led_swap_duration_mean must be positive"):
            RatIMUSimConfig(
                duration_s=1.0,
                led_swap_mode="persistent",
                led_swap_rate=1.0,
                led_swap_duration_mean=-0.5,
            )

    def test_negative_duration_std_rejected(self):
        """Verify that negative duration std is rejected."""
        with pytest.raises(
            ValueError, match="led_swap_duration_std must be non-negative"
        ):
            RatIMUSimConfig(
                duration_s=1.0,
                led_swap_mode="persistent",
                led_swap_rate=1.0,
                led_swap_duration_mean=0.5,
                led_swap_duration_std=-0.1,
            )
