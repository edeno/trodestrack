# Appendix — Reference repos and on-disk format details

[← back to README](README.md)

## Reference repositories (local clones)

The following upstream repositories are checked out locally and are
the authoritative source for format / schema details cited above.
Verify any corner case against these before writing code.

- **`trodes_to_nwb`** — `/Users/edeno/Documents/GitHub/trodes_to_nwb`.
  MIT-licensed (Loren Frank Lab, 2023). Vendor source for the Trodes
  parser. Key files:
  - `src/trodes_to_nwb/convert_position.py:143`
    (`read_trodes_datafile`), `:785` (`get_position_timestamps`),
    `:602-667` (`_get_position_timestamps_ptp`),
    `:1067-1079` (Trodes Position SpatialSeries write).
  - `src/trodes_to_nwb/convert_dios.py:59-65, 108`
    (`behavioral_events` container).
  - `src/trodes_to_nwb/convert_analog.py:89-104, 107-108`
    (analog/IMU container; `_NWB_ANALOG_DATA_PATH`).
  - `src/trodes_to_nwb/spike_gadgets_raw_io.py:953`
    (DIO `int8` 0/1 dtype).
  - `src/trodes_to_nwb/nwb_schema.json` (metadata YAML schema).
- **`trodes`** — `/Users/edeno/Documents/GitHub/trodes`. Upstream
  Trodes (SpikeGadgets). Cross-reference for the on-disk binary
  format. No clear top-level LICENSE; **not** a vendor source.
  - `python/trodes/binary_utils.py:11-32` (`TrodesBinaryReader` —
    `<Start settings>...<End settings>` header, 1000-line cap).
  - `python/trodes/trodes_data.py:661` (regex matches
    `*.videoPositionTracking`), `:700` (matches plain
    `*.videoTimeStamps`), `:719` (single regex matching **both**
    HWSync and HWFrameCount variants:
    `r'^.*\.videoTimeStamps\.(?:cameraHWFrameCount$|cameraHWSync)'`).
  - `doc/basic/CameraModule.rst` (camera module usage).
- **`DeepLabCut`** — `/Users/edeno/Documents/GitHub/DeepLabCut`.
  Reference for HDF5 output schema and metadata pickle. Not a runtime
  dep.
  - `deeplabcut/pose_estimation_pytorch/apis/videos.py:728-761`
    (modern saver — `(scorer, [individuals,] bodyparts, coords)`
    MultiIndex, key `df_with_missing`, table format).
  - `deeplabcut/pose_estimation_pytorch/apis/videos.py:865-879`
    (`_meta.pickle` payload — `fps`, `nframes`, `frame_dimensions=(w, h)`,
    `Scorer`, `cropping_parameters`).
  - `deeplabcut/pose_estimation_tensorflow/predict_videos.py:589-592,
    1053-1069` (TF/legacy saver, `frame_dimensions=(ny, nx)`).
  - `deeplabcut/post_processing/filtering.py:65-72, 268-293`
    (filtered output: median default, window 5).
  - `deeplabcut/create_project/new.py:243, 266, 297`
    (`pcutoff` default 0.6, `multianimalproject` flag).
- **`ndx-pose`** — `/Users/edeno/Documents/GitHub/ndx-pose`. NWB
  extension for pose estimation. Not a runtime dep — `pynwb` reads the
  embedded namespace dynamically.
  - `spec/ndx-pose.extensions.yaml` (`PoseEstimation`,
    `PoseEstimationSeries`, `Skeleton`, `Skeletons` types).
  - `src/pynwb/ndx_pose/pose.py` (typed Python classes — documents
    on-disk attribute names we read by name).
  - `src/pynwb/ndx_pose/io/pose.py:14-25, 33-44, 53-58` (custom IO
    mapping — `definition`/`version` rename and v0.1.x back-compat
    inline-`nodes` handling).
  - `examples/write_pose_estimates_only.py` (representative DLC →
    ndx-pose example).

## On-disk Trodes binary format

- ASCII header bracketed by `<Start settings>\n` ... `<End settings>\n`
  (capped at 1000 lines per `binary_utils.py:26`); each line is
  `key:value`. The `Fields` key declares the per-row dtype.
- Binary record array follows immediately after `<End settings>\n`.
- Pixel origin is **upper left** of the video frame; y increases
  downward.
- Per-row dtype of `.videoPositionTracking`: `xloc`, `yloc`, `xloc2`,
  `yloc2` (uint16 pixel coordinates), plus housekeeping fields
  including a `time` column that is the **Trodes sample-count clock**
  (uint32 sample index, not seconds).

## NWB DIO encoding details

- `nwbfile.processing["behavior"]["behavioral_events"]` is a
  `BehavioralEvents` container.
- One `TimeSeries` per DIO channel, keyed by the metadata YAML's
  semantic `name`; hardware channel id (e.g. `Din1`) lives in
  `TimeSeries.description`.
- `data` is `int8` 0/1 (verified at `spike_gadgets_raw_io.py:953`) —
  `1` = 0→1 transition (rise), `0` = 1→0 (fall). The very first sample
  is the initial level (not a transition); the loader drops it.
- `timestamps` is systime in seconds, same clock as the camera.

## NWB analog/IMU encoding

- One `TimeSeries` named `"analog"` under
  `processing["analog"]["analog"]` containing all ECU + headstage
  analog channels stacked column-wise. Channel ids are stored
  alongside as a list (e.g. `["ECU_Ain1", ..., "Headstage_GyroX",
  "Headstage_GyroY", ...]`).
- Loader matches each `IMUConfig.axis_map` value to a channel-id
  position and pulls those columns; applies the same
  `_convert_imu_to_si` conversion as the parquet path.
