# Phase 3 — `dlc_keypoints` loader

[← back to README](README.md) · [design](designs.md#dlc_keypoints) · [shared contracts](shared-contracts.md)

**Inputs to read first:**

- `/Users/edeno/Documents/GitHub/DeepLabCut/deeplabcut/pose_estimation_pytorch/apis/videos.py` —
  HDF5 saver at lines 728-761; `_meta.pickle` payload at 865-879.
- `/Users/edeno/Documents/GitHub/DeepLabCut/deeplabcut/pose_estimation_tensorflow/predict_videos.py:1053-1069` —
  TF/legacy saver `frame_dimensions=(ny, nx)` ordering.
- [src/trodestrack/config/schemas.py](../../../../src/trodestrack/config/schemas.py) —
  `InputsConfig`.

**Contracts referenced:**

- [`PositionPixels`](shared-contracts.md#positionpixels--loader-private-intermediate)
- [IMU source resolution](shared-contracts.md#imu-source-resolution) —
  same hard-error rule as `trodes_native`.

**Design:** [dlc_keypoints](designs.md#dlc_keypoints).

## Tasks

- `_dlc_keypoints.py` with bodypart selection, likelihood gate,
  meta-pickle parsing, three timestamp sources.
- Multi-animal MultiIndex rejection with a clear error.
- `frame_dimensions` saver-detection (PyTorch `(w, h)` vs TF `(h, w)`).

## Validation slice

| Test | Asserts |
| --- | --- |
| DLC `[dlc]` extra missing | `inputs.format=dlc_keypoints` without `tables` installed raises `ImportError`. |
| DLC multi-animal rejected | `(scorer, individuals, bodyparts, coords)` MultiIndex raises clearly. |
| DLC frame_dimensions ordering | PyTorch `(w, h)` and TF `(h, w)` pickles both normalize to `(width, height)`. |
| DLC likelihood gate | rows with `likelihood < threshold` become NaN; mask_cam reflects. |
| DLC crop offset | non-zero offset adds correctly before pixel→meter. |
| DLC timestamps_source switching | meta_pickle / hw_sync / timestamp_file all produce sensible `t_cam`. |

## Fixtures

Synthesize a tiny single-animal DLC HDF5 in `conftest.py` using
`pandas.DataFrame.to_hdf(..., key="df_with_missing")` with the
`(scorer, bodyparts, coords)` MultiIndex; alongside it a
`*_meta.pickle` containing minimal `{"data": {"fps": ..., "nframes":
...}}`. A multi-animal variant (extra `individuals` level) is
synthesized separately for the rejection test.
