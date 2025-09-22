# Development scratchpad

- Use this file to keep notes on ongoing development work.
- When the work is completed, clean it out from this file, so that the contents only reflect ongoing work.

## NOTES

### Current Status (Phase 1.2 Configuration System - COMPLETE ✅)

**✅ COMPLETED:**
- **Milestone 1**: Complete project setup and infrastructure
  - Basic project structure with modular package layout
  - Main function with CLI entry point (`trodestrack` command)
  - Core dependencies configured (JAX, numpy, scipy, pydantic, etc.)
  - Dev dependencies set up (pytest, black, ruff, mypy, hypothesis)
  - Package structure created: config, io, geom, imu, models, runtime, qa, cli, examples
  - GitHub Actions CI setup (unit, style, type, property, benchmark tests)

- **Phase 1.2**: Configuration System
  - Comprehensive Pydantic schemas: SessionConfig, MappingConfig, FilterConfig, LEDConfig, OutputConfig, IMUConfig
  - YAML configuration loading and validation with proper error handling
  - Complete CLI framework with all planned subcommands (smooth, online, report, calib-homography)
  - Robust validation with cross-field constraints and file existence checking
  - Full test suite with 29 passing tests covering all configuration scenarios
  - Integration between CLI and configuration system

**📋 NEXT UP:**
- Milestone 2 remaining: Data I/O loaders for Trodes, DeepLabCut, SpikeGadgets
- Synthetic data generator for testing
- Need to implement: `io/` module with video and IMU data loaders

### Development Environment
- Using `uv` package manager for fast dependency resolution
- Python 3.13 with JAX for high-performance computation
- Test framework: pytest with hypothesis for property-based testing
- Code quality: black (formatting), ruff (linting), mypy (typing)
