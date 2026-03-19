"""
Configuration for the Credit Spread Regime Model (CSRM).

This module contains the canonical default configuration and helper
utilities for safely merging user overrides into the default config.

Design goals:
- deterministic
- explicit
- auditable
- easy to override
- no hidden constants in downstream modules
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "name": "Credit Spread Regime Model",
        "version": "0.1.0",
        "score_min": -1.0,
        "score_max": 1.0,
    },
    "inputs": {
        "required_price_columns": [
            "HYG",
            "JNK",
            "LQD",
            "SHY",
            "IEF",
            "TLT",
            "SPY",
            "VIX",
        ],
        "optional_macro_columns": [
            "HY_OAS",
            "IG_OAS",
            "NFCI",
            "ANFCI",
        ],
        "price_min_history": 252,
        "allow_extra_columns": True,
        "sort_index": True,
        "drop_duplicate_index": True,
    },
    "transforms": {
        "ratios": {
            "hyg_ief": {"numerator": "HYG", "denominator": "IEF"},
            "lqd_ief": {"numerator": "LQD", "denominator": "IEF"},
            "hyg_lqd": {"numerator": "HYG", "denominator": "LQD"},
            "hyg_shy": {"numerator": "HYG", "denominator": "SHY"},
            "jnk_ief": {"numerator": "JNK", "denominator": "IEF"},
            "lqd_shy": {"numerator": "LQD", "denominator": "SHY"},
        },
        "return_method": "pct_change",
        "log_ratio": False,
        "fill_method": None,
    },
    "windows": {
        "short": 5,
        "medium": 21,
        "long": 63,
        "trend_fast": 21,
        "trend_slow": 63,
        "momentum": 21,
        "volatility": 21,
        "zscore": 63,
        "persistence": 10,
        "drawdown": 63,
        "vix_trend": 10,
        "confirmation": 21,
        "divergence": 21,
    },
    "features": {
        "trend": {
            "method": "ma_spread",
            "clip_lower": -1.0,
            "clip_upper": 1.0,
            "use_cross_sectional_average": True,
            "ratio_columns": [
                "hyg_ief",
                "lqd_ief",
                "hyg_lqd",
                "hyg_shy",
                "jnk_ief",
                "lqd_shy",
            ],
        },
        "stress": {
            "zscore_clip": 3.0,
            "drawdown_clip": 0.30,
            "use_oas_if_available": True,
            "oas_zscore_clip": 3.0,
            "nfci_zscore_clip": 3.0,
        },
        "confirmation": {
            "use_spy_trend": True,
            "use_vix_level": True,
            "use_vix_trend": True,
            "use_treasury_confirmation": False,
            "spy_trend_positive_threshold": 0.0,
            "vix_zscore_risk_on_max": 0.5,
            "vix_zscore_risk_off_min": 1.0,
        },
        "divergence": {
            "enabled": True,
            "lookback": 21,
            "large_divergence_threshold": 0.03,
        },
        "persistence": {
            "enabled": True,
            "lookback": 10,
            "risk_on_fraction_threshold": 0.70,
            "risk_off_fraction_threshold": 0.70,
        },
        "vix": {
            "level_zscore_clip": 3.0,
            "trend_clip": 1.0,
            "high_stress_zscore": 1.5,
            "low_stress_zscore": 0.0,
        },
    },
    "scoring": {
        "subscores": {
            "credit_trend_score": {
                "weight": 0.30,
                "clip_lower": -1.0,
                "clip_upper": 1.0,
            },
            "credit_stress_score": {
                "weight": 0.25,
                "clip_lower": -1.0,
                "clip_upper": 1.0,
            },
            "credit_confirmation_score": {
                "weight": 0.20,
                "clip_lower": -1.0,
                "clip_upper": 1.0,
            },
            "credit_divergence_score": {
                "weight": 0.10,
                "clip_lower": -1.0,
                "clip_upper": 1.0,
            },
            "credit_persistence_score": {
                "weight": 0.15,
                "clip_lower": -1.0,
                "clip_upper": 1.0,
            },
        },
        "normalize_weights": True,
        "composite_clip_lower": -1.0,
        "composite_clip_upper": 1.0,
        "neutral_center": 0.0,
    },
    "regime": {
        "risk_on_min": 0.30,
        "risk_off_max": -0.30,
        "labels": {
            "risk_on": "Risk-On",
            "neutral": "Neutral",
            "risk_off": "Risk-Off",
        },
    },
    "flags": {
        "early_warning": {
            "enabled": True,
            "composite_max": -0.10,
            "trend_max": -0.15,
            "stress_min": 0.20,
        },
        "stress_acceleration": {
            "enabled": True,
            "stress_delta_min": 0.15,
            "vix_delta_min": 0.10,
        },
        "recovery": {
            "enabled": True,
            "composite_min": 0.10,
            "trend_min": 0.15,
            "stress_max": -0.10,
        },
        "confirmed_risk_on": {
            "enabled": True,
            "composite_min": 0.30,
            "confirmation_min": 0.20,
            "persistence_min": 0.20,
        },
    },
    "outputs": {
        "include_raw_inputs": False,
        "include_ratios": True,
        "include_features": True,
        "include_subscores": True,
        "include_flags": True,
        "include_diagnostics": True,
        "regime_column": "regime",
        "composite_column": "composite_score",
    },
    "validation": {
        "enforce_numeric": True,
        "reject_nonpositive_prices": True,
        "max_missing_fraction": 0.10,
        "drop_all_na_rows": True,
    },
}


def get_default_config() -> dict[str, Any]:
    """
    Return a deep copy of the default configuration.

    Returns
    -------
    dict[str, Any]
        Independent copy of DEFAULT_CONFIG.
    """
    return deepcopy(DEFAULT_CONFIG)


def merge_config(
    base_config: dict[str, Any],
    override_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Recursively merge an override config into a base config.

    Parameters
    ----------
    base_config : dict[str, Any]
        Base configuration dictionary.
    override_config : dict[str, Any] | None
        User override dictionary. If None, a deep copy of base_config
        is returned.

    Returns
    -------
    dict[str, Any]
        New merged configuration dictionary.

    Notes
    -----
    - Scalars and lists are replaced, not concatenated.
    - Nested dictionaries are merged recursively.
    - Input dictionaries are never mutated.
    """
    merged = deepcopy(base_config)

    if override_config is None:
        return merged

    for key, value in override_config.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = deepcopy(value)

    return merged


def build_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Build the effective model configuration.

    Parameters
    ----------
    config : dict[str, Any] | None, default None
        Optional user overrides.

    Returns
    -------
    dict[str, Any]
        Fully merged configuration dictionary.
    """
    return merge_config(DEFAULT_CONFIG, config)