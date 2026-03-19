"""
Credit Spread Regime Model (CSRM).

Public package interface for the institutional-grade, deterministic
credit spread regime model.

Exposes:
- DEFAULT_CONFIG: canonical model configuration
- run_csrm: top-level model execution function
"""

from credit_spread_regime_model.config import DEFAULT_CONFIG
from credit_spread_regime_model.model import run_csrm

__all__ = [
    "DEFAULT_CONFIG",
    "run_csrm",
]

__version__ = "0.1.0"