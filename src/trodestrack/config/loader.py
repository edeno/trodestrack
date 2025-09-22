"""Configuration loading and validation utilities."""

from pathlib import Path
from typing import Any, Dict

import yaml

from ..constants import DEFAULT_PIXEL_PER_CM
from .schemas import SessionConfig


def load_config(config_path: Path) -> SessionConfig:
    """Load and validate configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Validated SessionConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        pydantic.ValidationError: If config validation fails
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Convert string paths to Path objects relative to config file directory
    config_base_dir = config_path.resolve().parent
    config_dict = _convert_paths(config_dict, config_base_dir)

    return SessionConfig(**config_dict)


def save_config(config: SessionConfig, output_path: Path) -> None:
    """Save configuration to YAML file.

    Args:
        config: SessionConfig instance to save
        output_path: Path where to save the config
    """
    # Convert to dict and handle Path objects
    config_dict = config.model_dump()
    config_dict = _convert_paths_to_strings(config_dict)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def _convert_paths(config_dict: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    """Recursively convert string paths to Path objects relative to base directory.

    Args:
        config_dict: Configuration dictionary to process
        base_dir: Base directory for resolving relative paths

    Returns:
        Updated configuration dictionary with Path objects
    """
    path_fields = {"video_file", "imu_file", "output_dir"}

    for key, value in config_dict.items():
        if key in path_fields and isinstance(value, str):
            config_dict[key] = _resolve_path(value, base_dir)
        elif isinstance(value, dict):
            config_dict[key] = _convert_paths(value, base_dir)

    return config_dict


def _resolve_path(path_str: str, base_dir: Path) -> Path:
    """Resolve a path string relative to base directory.

    Args:
        path_str: Path string that may be relative or absolute
        base_dir: Base directory for resolving relative paths

    Returns:
        Resolved absolute Path object
    """
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    else:
        return (base_dir / path).resolve()


def _convert_paths_to_strings(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively convert Path objects to strings for YAML serialization."""
    result: Dict[str, Any] = {}
    for key, value in config_dict.items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, dict):
            result[key] = _convert_paths_to_strings(value)
        else:
            result[key] = value

    return result


def create_default_config(
    video_file: Path, imu_file: Path, output_dir: Path, mapping_type: str = "homography"
) -> SessionConfig:
    """Create a default configuration with minimal required parameters.

    Args:
        video_file: Path to video detection file
        imu_file: Path to IMU data file
        output_dir: Directory for outputs
        mapping_type: Type of coordinate mapping ("homography" or "ruler_scale")

    Returns:
        SessionConfig with default values
    """
    from .schemas import MappingConfig, OutputConfig

    # Create mapping config based on type
    if mapping_type == "homography":
        mapping = MappingConfig(
            type="homography",
            homography_matrix=[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],  # Identity
        )
    else:
        mapping = MappingConfig(
            type="ruler_scale", pixel_per_cm=DEFAULT_PIXEL_PER_CM  # Example: 10 pixels per cm
        )

    output = OutputConfig(output_dir=output_dir)

    return SessionConfig(video_file=video_file, imu_file=imu_file, mapping=mapping, output=output)
