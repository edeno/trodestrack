# Per-format Designs

[← back to README](README.md)

Each section below is the authoritative design for one
`inputs.format`. Phase files implementing a format link directly to
the matching subsection.

- [`trodes_native`](#trodes_native)
- [`dlc_keypoints`](#dlc_keypoints)
- [`nwb` — position](#nwb--position)
- [`nwb` — extras (IMU, DIO)](#nwb--extras-imu-dio)

## `trodes_native`

**Schema** (added to `InputsConfig`):

```python
class TrodesNativeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    position_tracking_file: Path  # *.videoPositionTracking
    # PTP-synced per-frame timestamps. Must be a
    # ``*.videoTimeStamps.cameraHWSync`` file; the loader rejects
    # other variants. v1 supports PTP only.
    camera_timestamps_file: Path
```

**Reader internals** (`src/trodestrack/io/loaders/_trodes_native.py`):

1. **Filename pre-check.** Reject if `cfg.camera_timestamps_file`
   doesn't match the `*.videoTimeStamps.cameraHWSync` suffix; clear
   message points at the v1 PTP-only scope.
2. Read both files via the vendored `read_trodes_datafile`.
3. **Authoritative PTP gate.** After parsing the timestamps file's
   header `Fields` declaration, confirm the resulting record array has
   an `HWTimestamp` column. The vendored `_get_position_timestamps_ptp`
   raises `KeyError` when missing; we wrap that in a clearer
   `ValueError`. The filename is a heuristic; the column is the truth.
4. Join via `get_position_timestamps(..., ptp_enabled=True)`. Note: the
   position record array uses a `time` column that is the **Trodes
   sample-count clock** (uint32 sample index, not seconds); the
   PTP-stitching helper joins on this against `HWframeCount` /
   `HWTimestamp` from the timestamps file to recover seconds.
5. Pull `(xloc, yloc)` and `(xloc2, yloc2)`; build NaN-row mask from
   zero-valued or sentinel rows the Trodes tracker emits when LEDs
   are lost.
6. No confidence (Trodes online tracker doesn't emit per-frame
   likelihood).
7. Diagnostics: PTP pause segments removed, frame count, sample-rate
   estimate.

**No DIO bridge** at the `trodes_native` layer. Trodes stores DIO
multiplexed inside `.rec`; parsing requires `SpikeGadgets RawIO`
(`neo` dep), out of scope for v1. Users who want TTL events from raw
Trodes either run `trodes_to_nwb` and use the `nwb` loader or export
DIO to a parquet upstream and use the existing `ttl_events:` block.

**Vendored parser footprint** — roughly 300–350 LOC pulled from
`trodes_to_nwb/src/trodes_to_nwb/convert_position.py` (MIT, Loren
Frank Lab, 2023):

- `parse_dtype` (~55 LOC, lines 86–140)
- `read_trodes_datafile` (~45 LOC, line 143+)
- `convert_datafile_to_pandas` (~20 LOC, line 203+)
- `get_framerate` (~15 LOC)
- `find_acquisition_timing_pause` (~45 LOC, lines 239–280)
- `find_large_frame_jumps` (~25 LOC)
- `detect_repeat_timestamps` (~17 LOC)
- `detect_trodes_time_repeats_or_frame_jumps` (~55 LOC, lines 329–381)
- `_get_position_timestamps_ptp` (~65 LOC, lines 602–667)
- The PTP branch of `get_position_timestamps` (~80 LOC).

Vendor with a `# Adapted from trodes_to_nwb at <commit>` header.
Required deps (`numpy`, `pandas`, `scipy.ndimage.label`,
`scipy.stats.linregress` per `convert_position.py:18-19`) are already
in trodestrack's base install.

## `dlc_keypoints`

**Schema** (added to `InputsConfig`):

```python
class DLCKeypointsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    h5_file: Path
    led1_bodypart: str
    led2_bodypart: str
    likelihood_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    # Source for camera frame timestamps. ``meta_pickle`` synthesizes
    # from the sibling DLC ``_meta.pickle`` (``fps``, ``nframes``);
    # ``trodes_hw_sync`` joins against an external
    # ``*.videoTimeStamps.cameraHWSync``; ``timestamp_file`` reads a
    # 1-D float array.
    timestamps_source: Literal[
        "meta_pickle", "trodes_hw_sync", "timestamp_file"
    ] = "meta_pickle"
    camera_timestamps_file: Path | None = None  # required if trodes_hw_sync
    timestamp_file: Path | None = None           # required if timestamp_file
    apply_crop_offset: bool = True
```

**Reader internals** (`src/trodestrack/io/loaders/_dlc_keypoints.py`).
Lazy-imports `tables` indirectly via `pandas.read_hdf`; raises a clear
`ImportError` if PyTables is missing, naming `pip install
'trodestrack[dlc]'`:

1. `df = pd.read_hdf(cfg.h5_file, key="df_with_missing")`.
2. Detect MultiIndex level names. Reject if `"individuals"` is present
   (multi-animal — clearer error than slicing individual 0).
3. Drop the `scorer` level: `df.droplevel("scorer", axis=1)`.
4. For each LED bodypart: extract `(x, y, likelihood)`. NaN out rows
   where `likelihood < cfg.likelihood_threshold`.
5. If `cfg.apply_crop_offset` and the sibling `_meta.pickle`'s
   `cropping_parameters` are non-trivial, add the crop offset to (x, y).
6. Build `t_cam` from one of three sources per `cfg.timestamps_source`.
   The `meta_pickle` path's `frame_dimensions` ordering differs by
   saver: PyTorch `(w, h)` (`videos.py:865-879`) vs TF `(ny, nx)` —
   i.e. `(h, w)` (`predict_videos.py:1053-1069`). Loader detects the
   saver via the `Scorer` string format (PyTorch ends in
   `_{snapshot_uid}`; TF ends in `_{iterations}`) and normalizes to
   `(width, height)` before downstream consumers see it.
7. Build `confidence` array `(n, 4)` from `[likelihood1, likelihood1,
   likelihood2, likelihood2]` (per-LED replicated across x/y, matching
   the existing `conf_cam` shape).
8. Diagnostics: scorer string, `fps`, `nframes`, `frame_dimensions`,
   per-LED kept fraction (rows above pcutoff).

DLC outputs are pixels; `coords_meters_per_pixel` left `None`. DLC
filtered output (`*_filtered.h5`, default median window 5) is *not*
preferred — median smoothing interacts badly with abrupt LED occlusions
and we do our own EKF-side smoothing. `cfg.h5_file` may point at either
the raw or filtered HDF5; loader logs which.

## `nwb` — position

**Schema** (added to `InputsConfig`):

```python
class NWBLEDSourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    container: Literal["auto", "trodes_position", "ndx_pose"] = "auto"
    led1_series_name: str | None = None  # for trodes_position
    led2_series_name: str | None = None  # for trodes_position
    led1_bodypart: str | None = None     # for ndx_pose
    led2_bodypart: str | None = None     # for ndx_pose
    likelihood_threshold: float = Field(default=0.6, ge=0.0, le=1.0)


class NWBConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    nwb_file: Path
    led_source: NWBLEDSourceConfig = Field(default_factory=NWBLEDSourceConfig)
    meters_per_pixel_override: float | None = None
    dio_to_ttl: NWBDIOToTTLConfig | None = None
    # IMU is read from the NWB file's processing["analog"] group by
    # default. ``inputs.imu_file`` is **not** required for ``nwb``
    # sessions; if it is provided, it overrides the NWB-stored IMU.
```

**Reader internals** (`src/trodestrack/io/nwb/`,
lazy-imports `pynwb`). Two layers — the public container API
(documented under
[NWB container-layer API](shared-contracts.md#nwb-container-layer-api-public-spyglass-integration-seam))
and a path-based wrapper that delegates to it:

```python
# Public path-based entry. Used by inputs.format=nwb.
def load_nwb_session(cfg: NWBConfig) -> tuple[PositionPixels, NWBSessionExtras]:
    try:
        import pynwb
    except ImportError as e:
        raise ImportError(
            "inputs.format='nwb' requires the [nwb] extra. "
            "Install with: uv pip install 'trodestrack[nwb]'."
        ) from e

    with pynwb.NWBHDF5IO(cfg.nwb_file, mode="r", load_namespaces=True) as io:
        nwbfile = io.read()
        led_container = _detect_led_container(nwbfile, cfg.led_source.container)
        if led_container.neurodata_type == "Position":
            pixels = from_position_container(led_container, cfg.led_source)
        elif led_container.neurodata_type == "PoseEstimation":
            pixels = from_pose_estimation_container(led_container, cfg.led_source)
        else:
            raise ValueError(...)
        extras = _gather_extras(nwbfile, cfg)  # delegates to from_analog_container / from_behavioral_events
        return pixels, extras
    # NWBHDF5IO closed here; pixels/extras hold numpy arrays only.
```

The path wrapper's only job is locate-and-dispatch; all per-container
extraction logic lives in the four public entry points so Spyglass's
`make()` (which already has the containers) can bypass `NWBHDF5IO`
entirely.

**Auto-detection by neurodata type, not container name.** Walk
`nwbfile.processing["behavior"].data_interfaces.values()` matching on
`neurodata_type`: `Position`-typed → Trodes route;
`PoseEstimation`-typed → ndx-pose route. If both exist, the YAML must
disambiguate via `led_source.container`. Default container names
(`Position` / `PoseEstimation`) are the writers' defaults, not
guarantees — never address by container name alone.

**Trodes-position branch.** Read the named SpatialSeries pair
(`led_0_series_{epoch}` / `led_1_series_{epoch}` are the writer
defaults from `convert_position.py:1067-1079`); pull `data` and
`timestamps`; populate `coords_meters_per_pixel` from each series'
`conversion` attribute. `reference_frame` is `"Upper left corner of
video frame"` (no y-flip applied at write time).

**ndx-pose branch.** Read the named `PoseEstimationSeries`; extract
`data` (pixels), `timestamps`, `confidence`, and the per-series
`conversion`. Apply `likelihood_threshold` NaN-mask. Read
`source_software` / `scorer` for diagnostics.

Container ordering under `PoseEstimation` is alphabetical by name
(a `MultiContainerInterface` storage detail); always address by name,
not index.

**ndx-pose v0.1.x compatibility.** Plan targets v0.2.0+. v0.1.x files
don't have `Skeleton` / `Skeletons` containers; `nodes` and `edges` are
stored directly on `PoseEstimation` (back-compat code in
`ndx-pose/src/pynwb/ndx_pose/io/pose.py:33-44, 53-58`). The loader
discovers keypoints with this fallback chain:

1. `processing["behavior"]["Skeletons"]["<skel>"].nodes` (v0.2.x).
2. Inline `nodes` on the `PoseEstimation` object itself (v0.1.x).
3. `pose_estimation_series.keys()` (last-resort discovery).

Diagnostics record the detected schema version.

**Reading without `ndx-pose` installed.** `pynwb.NWBHDF5IO(...,
load_namespaces=True)` reads the embedded namespace from
`/specifications` and surfaces typed containers generically. The
`ndx-pose` package's only role at read time is the custom IO mapper
that renames `definition` → `confidence_definition` and `version` →
`source_software_version`. **Without `ndx-pose` installed, attributes
appear under their on-disk names**; the loader reads `definition` and
`version` regardless of whether `ndx-pose` happens to be installed.

**Conversion=1.0 sentinel fallback.** `SpatialSeries.conversion`
defaults to `1.0` when never set. Many DLC-via-NWB writers (NeuroConv,
SLEAP) leave `conversion=1.0` and `unit="pixels"` — meaning *no
calibration baked in*. The loader treats `conversion == 1.0` and
`unit == "pixels"` as the sentinel, leaving `coords_meters_per_pixel
= None` so the YAML / homography path takes over. If `unit == "meters"`
and `conversion == 1.0`, treat as already-calibrated data
(`coords_meters_per_pixel = 1.0`).

## `nwb` — extras (IMU, DIO)

**Schema:**

```python
class NWBDIOToTTLConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Map semantic name (TimeSeries.name in behavioral_events) to the
    # configured TTLEventsConfig source_id. Edges come straight from
    # the int8 0/1 stream — 1 → "rise", 0 → "fall". The first sample
    # is the initial level (not a transition); the loader drops it.
    name_to_source_id: dict[str, int]
```

**`NWBSessionExtras`** carries optional pieces the loader pulls from
the NWB file when present. Both fields are produced by the
container-layer entry points (`from_analog_container` /
`from_behavioral_events`). Spyglass's `make()` populates them by
calling `fetch_nwb` on `SensorData` / `DIOEvents` (each of which
returns a `list[dict]` of rows with object-id columns replaced by
pynwb objects) and unwrapping into the trodestrack-canonical input
shapes documented per entry below:

- `imu`: `(t_imu, U_imu)` from `from_analog_container(analog_ts,
  imu_cfg)` where `analog_ts` is a single `pynwb.base.TimeSeries`.
  Path-based loader sources it from
  `processing["analog"]["analog"]["analog"]` (the `TimeSeries`
  `convert_analog.py:89-104` writes;
  `_NWB_ANALOG_DATA_PATH = "processing/analog/analog/analog/data"`);
  Spyglass's `make()` unwraps it from `(SensorData & key).fetch_nwb()`
  (typically `[0]["sensor_data"]` per the standard `fetch_nwb`
  list-of-rows shape). Channel
  ids stored alongside the data; the loader matches each
  `IMUConfig.axis_map` value (e.g. `"Headstage_GyroX"`) to a column
  index, applies the same `_convert_imu_to_si` conversion the parquet
  path does. Used when `inputs.imu_file` is not configured. If no
  analog `TimeSeries` is available, fall back to the
  synthetic-vision-only stream documented in
  [IMU source resolution](shared-contracts.md#imu-source-resolution).
- `dio_events`: a flat `(t_evt, source_id, edge)` table from
  `from_behavioral_events(events, cfg.dio_to_ttl)`. Path-based loader
  sources `events` from `processing["behavior"]["behavioral_events"]`.
  The function accepts either a `BehavioralEvents` container or a
  `dict[str, TimeSeries]`. Spyglass's `(DIOEvents & key).fetch_nwb()`
  returns a `list[dict]` with one row per `dio_event_name`
  ([common_dio.py:20](https://github.com/LorenFrankLab/spyglass) and
  [fetch.py:239](https://github.com/LorenFrankLab/spyglass)); Spyglass's
  `make()` is responsible for the trivial assembly
  `{row["dio_event_name"]: row["dio"] for row in
  (DIOEvents & key).fetch_nwb()}` before calling
  `from_behavioral_events`. trodestrack's API stays Spyglass-agnostic.
  `_attach_ttl_events` in `io/session.py` is generalized to accept
  either a parquet path or a pre-built table. DIO data is stored as
  `int8` 0/1 (verified at
  `trodes_to_nwb/src/trodes_to_nwb/spike_gadgets_raw_io.py:953`); the
  loader dropping the first-sample initial level handles the "value
  is the level, not a transition" semantics correctly.

The schema contract for `events_file` and the cross-validators is
documented under
[NWB DIO → TTL events bridge (schema contract)](shared-contracts.md#nwb-dio--ttl-events-bridge-schema-contract).
