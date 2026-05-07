"""Configuration validation utilities for Point3R-LLM.

This module provides validation to ensure training and evaluation arguments are aligned,
preventing silent performance degradation from mismatched parameters.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ParameterSpec:
    """Specification for a parameter that should match between training and eval."""

    name: str
    severity: str  # "critical", "high", "medium"
    source_key: str  # Key in config.json or preprocessor_config.json
    source_file: str  # "config" or "preprocessor"
    description: str


# Parameters that must match exactly (will cause errors/wrong results if mismatched)
CRITICAL_PARAMS = [
    ParameterSpec(
        "use_pointer_position_encoding",
        "critical",
        "use_pointer_position_encoding",
        "config",
        "Position encoding module presence/absence - weights won't load if mismatched",
    ),
    ParameterSpec(
        "pointer_pos_hidden_dim",
        "critical",
        "pointer_pos_hidden_dim",
        "config",
        "Position encoder weight dimensions - shape mismatch if different",
    ),
    ParameterSpec(
        "merge_memory_feat",
        "critical",
        "merge_memory_feat",
        "config",
        "Memory feature fusion enabled/disabled - changes data flow",
    ),
    ParameterSpec(
        "memory_fusion_method",
        "critical",
        "memory_fusion_method",
        "config",
        "Fusion module architecture - different methods produce different results",
    ),
    ParameterSpec(
        "pointer_memory_size",
        "critical",
        "pointer_memory_size",
        "config",
        "Number of pointer tokens - shape errors if mismatched",
    ),
]

# Parameters that should match for consistency (may affect results quality)
HIGH_PARAMS = [
    ParameterSpec(
        "min_pixels",
        "high",
        "min_pixels",
        "preprocessor",
        "Image preprocessing - affects visual token count and resolution",
    ),
    ParameterSpec(
        "max_pixels",
        "high",
        "max_pixels",
        "preprocessor",
        "Image preprocessing - affects visual token count and resolution",
    ),
]

# Parameters with medium risk (may affect results but less severely)
MEDIUM_PARAMS = [
    ParameterSpec(
        "memory_merger_hidden_dim",
        "medium",
        "memory_merger_hidden_dim",
        "config",
        "Memory merger MLP dimensions",
    ),
    ParameterSpec(
        "memory_merger_type",
        "medium",
        "memory_merger_type",
        "config",
        "Memory merger type (mlp/avg)",
    ),
]


def load_training_config(model_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load config.json and preprocessor_config.json from a trained checkpoint.

    Args:
        model_path: Path to the trained model checkpoint directory

    Returns:
        Tuple of (config dict, preprocessor_config dict)
    """
    config_path = os.path.join(model_path, "config.json")
    preprocessor_path = os.path.join(model_path, "preprocessor_config.json")

    config: Dict[str, Any] = {}
    preprocessor_config: Dict[str, Any] = {}

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        logger.warning(f"config.json not found at {config_path}")

    if os.path.exists(preprocessor_path):
        with open(preprocessor_path, "r") as f:
            preprocessor_config = json.load(f)
    else:
        logger.debug(f"preprocessor_config.json not found at {preprocessor_path}")

    return config, preprocessor_config


def validate_parameters(
    model_path: str,
    eval_args: Dict[str, Any],
    fail_on_critical: bool = True,
    warn_on_high: bool = True,
) -> List[str]:
    """Validate evaluation arguments against training config.

    Args:
        model_path: Path to the trained model checkpoint
        eval_args: Dictionary of evaluation arguments to validate
        fail_on_critical: Raise error on critical mismatches (default: True)
        warn_on_high: Log warnings for high-severity mismatches (default: True)

    Returns:
        List of mismatch messages

    Raises:
        ValueError: If fail_on_critical=True and critical parameters mismatch
    """
    config, preprocessor_config = load_training_config(model_path)

    # If no config found, skip validation with warning
    if not config and not preprocessor_config:
        logger.warning(
            f"No config files found at {model_path}, skipping parameter validation"
        )
        return []

    mismatches: List[str] = []
    critical_mismatches: List[str] = []

    all_params = CRITICAL_PARAMS + HIGH_PARAMS + MEDIUM_PARAMS

    for param in all_params:
        # Get training value from appropriate config file
        if param.source_file == "config":
            train_value = config.get(param.source_key)
        else:
            train_value = preprocessor_config.get(param.source_key)

        # Get eval value
        eval_value = eval_args.get(param.name)

        # Skip if training value not found (parameter not used during training)
        if train_value is None:
            continue

        # Skip if eval value not provided (will use model's saved config)
        if eval_value is None:
            logger.debug(
                f"Parameter '{param.name}' not specified in eval args, "
                "using model's saved config value"
            )
            continue

        # Check for mismatch
        if train_value != eval_value:
            msg = (
                f"[{param.severity.upper()}] Parameter mismatch: {param.name}\n"
                f"  Training config: {train_value}\n"
                f"  Evaluation args: {eval_value}\n"
                f"  Risk: {param.description}"
            )

            if param.severity == "critical":
                critical_mismatches.append(msg)
                logger.error(msg)
            elif param.severity == "high" and warn_on_high:
                logger.warning(msg)
            else:
                logger.info(msg)

            mismatches.append(msg)

    # Fail on critical mismatches if requested
    if fail_on_critical and critical_mismatches:
        error_msg = (
            "Critical parameter mismatches detected between training and evaluation!\n"
            "This will likely cause incorrect model behavior.\n\n"
            + "\n\n".join(critical_mismatches)
            + "\n\nTo proceed anyway, set fail_on_critical=False in validation call."
        )
        raise ValueError(error_msg)

    if not mismatches:
        logger.info(f"Parameter validation passed for checkpoint: {model_path}")

    return mismatches
