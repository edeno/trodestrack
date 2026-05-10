"""``from_pose_estimation_container`` and ndx-pose helpers.

Pure container-layer code — does not import ``pynwb`` /
``ndx_pose`` at module load. PoseEstimation attributes are read
duck-typed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from trodestrack.config.schemas import NWBLEDSourceConfig
from trodestrack.io.loaders._shared import PositionPixels
from trodestrack.io.nwb._shared import (
    eager_array,
    read_trodes_pair,
    resolve_coords_mpp,
    resolve_coords_mpp_single,
)


def from_pose_estimation_container(
    pose: Any,
    cfg: NWBLEDSourceConfig,
) -> PositionPixels:
    """Convert an ``ndx_pose.PoseEstimation`` into ``PositionPixels``.

    Reads the named ``PoseEstimationSeries`` for LED1/LED2, applies
    the ``likelihood_threshold`` NaN-mask, and surfaces
    ``source_software`` / ``scorer`` / schema-version diagnostics.

    Reads attributes via fallback so an NWB file written with
    ``ndx-pose`` installed (typed-attribute names
    ``confidence_definition`` / ``source_software_version``) and one
    read without ``ndx-pose`` (on-disk names ``definition`` /
    ``version``) both work.

    For ``cfg.tracking_geometry`` of ``single_led1`` / ``single_led2``
    only the matching ``led{1,2}_bodypart`` is required; the
    unobserved LED is filled with NaN and its confidence column is
    set to a finite neutral value (the all-NaN coordinates already
    drop those frames out of the EKF observation, so the confidence
    value is informational only).

    The Skeleton chain (v0.2.x ``processing["behavior"]["Skeletons"]``
    vs v0.1.x inline ``nodes`` on the ``PoseEstimation``) affects
    diagnostics only — position data comes from
    ``pose.pose_estimation_series[bodypart]`` which works in both
    schemas.
    """

    if cfg.tracking_geometry != "dual_led":
        return _from_pose_estimation_container_single(pose, cfg)

    if cfg.led1_bodypart is None or cfg.led2_bodypart is None:
        raise ValueError(
            "inputs.nwb.led_source.led1_bodypart and led2_bodypart are "
            "required when reading from a PoseEstimation container."
        )

    series_dict = pose.pose_estimation_series
    available = list(series_dict.keys())
    for name in (cfg.led1_bodypart, cfg.led2_bodypart):
        if name not in series_dict:
            raise ValueError(
                f"PoseEstimation series {name!r} not found. Available: {available}."
            )

    series1 = series_dict[cfg.led1_bodypart]
    series2 = series_dict[cfg.led2_bodypart]

    led1, led2, t_cam, scale1, scale2 = read_trodes_pair(series1, series2)

    conf1 = eager_array(series1.confidence, dtype=float)
    conf2 = eager_array(series2.confidence, dtype=float)
    led1[conf1 < cfg.likelihood_threshold] = np.nan
    led2[conf2 < cfg.likelihood_threshold] = np.nan

    confidence = np.column_stack([conf1, conf1, conf2, conf2])
    coords_mpp = resolve_coords_mpp([series1, series2], [scale1, scale2])

    schema_version = _detect_pose_schema_version(pose)
    confidence_definition = getattr(series1, "confidence_definition", None) or getattr(
        series1, "definition", None
    )
    source_software_version = getattr(pose, "source_software_version", None) or getattr(
        pose, "version", None
    )

    return PositionPixels(
        led1_pixels=led1,
        led2_pixels=led2,
        t_cam=t_cam,
        confidence=confidence,
        coords_meters_per_pixel=coords_mpp,
        diagnostics={
            "format": "nwb",
            "container": "ndx_pose",
            "tracking_geometry": "dual_led",
            "led1_bodypart": cfg.led1_bodypart,
            "led2_bodypart": cfg.led2_bodypart,
            "ndx_pose_schema_version": schema_version,
            "source_software": getattr(pose, "source_software", None),
            "source_software_version": source_software_version,
            "confidence_definition": confidence_definition,
            "scorer": getattr(pose, "scorer", None),
            "led1_kept_fraction": float(np.mean(conf1 >= cfg.likelihood_threshold)),
            "led2_kept_fraction": float(np.mean(conf2 >= cfg.likelihood_threshold)),
        },
    )


def _from_pose_estimation_container_single(
    pose: Any, cfg: NWBLEDSourceConfig
) -> PositionPixels:
    """Single-LED PoseEstimation branch of ``from_pose_estimation_container``."""

    if cfg.tracking_geometry == "single_led1":
        observed_name = cfg.led1_bodypart
        observed_field = "led1_bodypart"
    else:
        observed_name = cfg.led2_bodypart
        observed_field = "led2_bodypart"
    if observed_name is None:
        raise ValueError(
            f"inputs.nwb.led_source.{observed_field} is required when "
            f"tracking_geometry={cfg.tracking_geometry!r}."
        )

    series_dict = pose.pose_estimation_series
    if observed_name not in series_dict:
        raise ValueError(
            f"PoseEstimation series {observed_name!r} not found. "
            f"Available: {list(series_dict.keys())}."
        )
    series = series_dict[observed_name]

    data = eager_array(series.data, dtype=float)
    t_cam = eager_array(series.timestamps, dtype=float)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(
            f"PoseEstimationSeries {series.name!r} data shape {data.shape} is "
            "not (n, 2). Native 3-D coordinates are not supported in "
            "v1 of the NWB loader."
        )
    if t_cam.shape != (data.shape[0],):
        raise ValueError(
            f"PoseEstimationSeries {series.name!r} timestamps shape "
            f"{t_cam.shape} does not match data length {data.shape[0]}."
        )

    conf = eager_array(series.confidence, dtype=float)
    data[conf < cfg.likelihood_threshold] = np.nan

    missing = np.full_like(data, np.nan)
    neutral_conf = np.ones_like(conf)
    if cfg.tracking_geometry == "single_led1":
        led1, led2 = data, missing
        confidence = np.column_stack([conf, conf, neutral_conf, neutral_conf])
    else:
        led1, led2 = missing, data
        confidence = np.column_stack([neutral_conf, neutral_conf, conf, conf])

    conversion = float(getattr(series, "conversion", 1.0))
    coords_mpp = resolve_coords_mpp_single(series, conversion)
    schema_version = _detect_pose_schema_version(pose)
    confidence_definition = getattr(series, "confidence_definition", None) or getattr(
        series, "definition", None
    )
    source_software_version = getattr(pose, "source_software_version", None) or getattr(
        pose, "version", None
    )

    return PositionPixels(
        led1_pixels=led1,
        led2_pixels=led2,
        t_cam=t_cam,
        confidence=confidence,
        coords_meters_per_pixel=coords_mpp,
        diagnostics={
            "format": "nwb",
            "container": "ndx_pose",
            "tracking_geometry": cfg.tracking_geometry,
            "observed_bodypart": observed_name,
            "ndx_pose_schema_version": schema_version,
            "source_software": getattr(pose, "source_software", None),
            "source_software_version": source_software_version,
            "confidence_definition": confidence_definition,
            "scorer": getattr(pose, "scorer", None),
            "observed_kept_fraction": float(np.mean(conf >= cfg.likelihood_threshold)),
        },
    )


def _detect_pose_schema_version(pose: Any) -> str:
    """Best-effort ndx-pose schema-version detection.

    v0.2.x stores nodes/edges via a ``Skeleton`` linked from the
    parent processing module's ``Skeletons`` container. v0.1.x stores
    ``nodes`` / ``edges`` inline on the ``PoseEstimation`` itself.

    This is diagnostic-only — position data is read from
    ``pose.pose_estimation_series[bodypart]`` which works in both
    schemas regardless of detection result.
    """

    # A real Skeleton object exposes ``nodes`` with content; a broken
    # link from a v0.1.x file may surface a stub ``skeleton`` whose
    # attributes are missing or empty, so check for actual node data.
    skeleton = getattr(pose, "skeleton", None)
    if skeleton is not None:
        skeleton_nodes = getattr(skeleton, "nodes", None)
        if skeleton_nodes is not None and len(np.asarray(skeleton_nodes)) > 0:
            return "v0.2.x"
    inline_nodes = getattr(pose, "nodes", None)
    if inline_nodes is not None and len(np.asarray(inline_nodes)) > 0:
        return "v0.1.x"
    return "unknown"
