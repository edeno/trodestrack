# Development scratchpad

- Use this file to keep notes on ongoing development work.
- When the work is completed, clean it out from this file, so that the contents only reflect ongoing work.

## NOTES

### Current Status (CORRECTED ASSESSMENT)

**✅ ACTUALLY COMPLETED:**
- **Milestone 1**: Complete project setup and infrastructure ✅
  - Basic project structure with modular package layout
  - Main function with CLI entry point (`trodestrack` command)
  - Core dependencies configured (JAX, numpy, scipy, pydantic, etc.)
  - Dev dependencies set up (pytest, black, ruff, mypy, hypothesis)
  - Package structure created: config, io, geom, imu, models, runtime, qa, cli, examples
  - GitHub Actions CI setup would need verification

- **Milestone 2**: Configuration & Data I/O System ✅
  - Comprehensive Pydantic schemas: SessionConfig, MappingConfig, FilterConfig, LEDConfig, OutputConfig, IMUConfig
  - YAML configuration loading and validation with proper error handling
  - Complete CLI framework with all planned subcommands (parsers only)
  - **Data I/O loaders implemented:**
    - `TrodesLEDData`: LED position tracking with confidence values
    - `DLCKeypointData`: DeepLabCut keypoint data handling
    - `SpikeGadgetsIMUData`: IMU data with unit conversions and downsampling
    - `TimestampAlignment`: Hardware-synced timestamp alignment utilities
  - Full test suite with **49 passing tests** covering all I/O and configuration scenarios
  - Basic IMU unit conversions and downsampling

**🚨 CRITICAL GAPS IDENTIFIED:**
- **No actual algorithm implementation**: `models/`, `runtime/`, `geom/`, `imu/` directories are empty
- **No synthetic data generator**: `sim/` module doesn't exist
- **CLI commands are parser-only**: No actual functionality behind the commands
- **No core mathematical algorithms**: No EKF, UKF, pre-integration, homography, etc.

**📋 IMMEDIATE PRIORITIES:**
- Implement synthetic data generator (`sim/` module) to enable testing
- Begin core mathematical components in `models/` and `geom/`
- Focus on getting basic filtering pipeline working

### Development Environment
- Using `uv` package manager for fast dependency resolution
- Python 3.13 with JAX for high-performance computation
- Test framework: pytest with hypothesis for property-based testing
- Code quality: black (formatting), ruff (linting), mypy (typing)
