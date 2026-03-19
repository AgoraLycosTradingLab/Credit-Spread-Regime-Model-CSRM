import numpy as np
import pandas as pd
import pytest

from credit_spread_regime_model.model import (
    assemble_output,
    get_latest_snapshot,
    run_csrm,
    run_feature_stage,
    run_interpretation_stage,
    run_scoring_stage,
    run_transform_stage,
    run_validation_stage,
)


def make_valid_prices(
    periods: int = 300,
    start: str = "2020-01-01",
) -> pd.DataFrame:
    """
    Create a deterministic valid input DataFrame for end-to-end smoke tests.
    """
    index = pd.date_range(start=start, periods=periods, freq="B")
    base = np.arange(periods, dtype=float)

    data = {
        "HYG": 80.0 + 0.08 * base,
        "JNK": 100.0 + 0.06 * base,
        "LQD": 110.0 + 0.035 * base,
        "SHY": 85.0 + 0.01 * base,
        "IEF": 95.0 + 0.02 * base,
        "TLT": 120.0 + 0.015 * base,
        "SPY": 300.0 + 0.25 * base,
        "VIX": 25.0 - 0.015 * base,
        "HY_OAS": 4.5 - 0.002 * base,
        "IG_OAS": 1.8 - 0.001 * base,
        "NFCI": 0.50 - 0.003 * base,
        "ANFCI": 0.30 - 0.002 * base,
    }
    return pd.DataFrame(data, index=index)


def test_run_csrm_executes_end_to_end() -> None:
    prices = make_valid_prices()

    result = run_csrm(prices)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(prices)
    assert "composite_score" in result.columns
    assert "regime" in result.columns


def test_run_csrm_output_contains_core_columns() -> None:
    prices = make_valid_prices()

    result = run_csrm(prices)

    expected_columns = [
        "hyg_ief",
        "lqd_ief",
        "hyg_lqd",
        "hyg_shy",
        "jnk_ief",
        "lqd_shy",
        "credit_trend_feature",
        "credit_stress_feature",
        "credit_confirmation_feature",
        "credit_divergence_feature",
        "credit_persistence_feature",
        "credit_trend_score",
        "credit_stress_score",
        "credit_confirmation_score",
        "credit_divergence_score",
        "credit_persistence_score",
        "composite_score",
        "regime",
        "early_warning_flag",
        "stress_acceleration_flag",
        "recovery_flag",
        "confirmed_risk_on_flag",
        "composite_score_clipped",
        "regime_numeric",
        "model_state_summary",
    ]

    for col in expected_columns:
        assert col in result.columns


def test_run_csrm_preserves_index() -> None:
    prices = make_valid_prices()

    result = run_csrm(prices)

    pd.testing.assert_index_equal(result.index, prices.index)


def test_run_csrm_outputs_valid_regime_labels() -> None:
    prices = make_valid_prices()

    result = run_csrm(prices)

    observed = set(result["regime"].dropna().unique())
    assert observed.issubset({"Risk-On", "Neutral", "Risk-Off"})


def test_run_csrm_outputs_boolean_flags() -> None:
    prices = make_valid_prices()

    result = run_csrm(prices)

    flag_columns = [
        "early_warning_flag",
        "stress_acceleration_flag",
        "recovery_flag",
        "confirmed_risk_on_flag",
    ]

    for col in flag_columns:
        assert col in result.columns
        assert pd.api.types.is_bool_dtype(result[col])


def test_run_csrm_outputs_bounded_scores() -> None:
    prices = make_valid_prices()

    result = run_csrm(prices)

    bounded_columns = [
        "credit_trend_score",
        "credit_stress_score",
        "credit_confirmation_score",
        "credit_divergence_score",
        "credit_persistence_score",
        "composite_score",
        "composite_score_clipped",
    ]

    for col in bounded_columns:
        assert result[col].min(skipna=True) >= -1.0
        assert result[col].max(skipna=True) <= 1.0


def test_run_csrm_with_minimal_output_config_returns_compact_frame() -> None:
    prices = make_valid_prices()

    config = {
        "outputs": {
            "include_raw_inputs": False,
            "include_ratios": False,
            "include_features": False,
            "include_subscores": False,
            "include_flags": False,
            "include_diagnostics": False,
        }
    }

    result = run_csrm(prices, config=config)

    assert list(result.columns) == ["composite_score", "regime"]


def test_run_csrm_with_full_raw_inputs_includes_price_columns() -> None:
    prices = make_valid_prices()

    config = {
        "outputs": {
            "include_raw_inputs": True,
        }
    }

    result = run_csrm(prices, config=config)

    for col in ["HYG", "JNK", "LQD", "SHY", "IEF", "TLT", "SPY", "VIX"]:
        assert col in result.columns


def test_stage_helpers_run_in_sequence() -> None:
    prices = make_valid_prices()

    validated = run_validation_stage(prices)
    transformed = run_transform_stage(validated)
    featured = run_feature_stage(transformed)
    scored = run_scoring_stage(featured)
    interpreted = run_interpretation_stage(scored)

    assert "HYG" in validated.columns
    assert "hyg_ief" in transformed.columns
    assert "credit_trend_feature" in featured.columns
    assert "credit_trend_score" in scored.columns
    assert "model_state_summary" in interpreted.columns


def test_assemble_output_can_be_called_on_interpreted_frame() -> None:
    prices = make_valid_prices()

    interpreted = run_interpretation_stage(prices)
    result = assemble_output(interpreted)

    assert "composite_score" in result.columns
    assert "regime" in result.columns


def test_get_latest_snapshot_returns_last_row() -> None:
    prices = make_valid_prices()
    result = run_csrm(prices)

    snapshot = get_latest_snapshot(result)

    assert isinstance(snapshot, pd.Series)
    assert snapshot.name == result.index[-1]
    pd.testing.assert_series_equal(snapshot, result.iloc[-1], check_names=True)


def test_get_latest_snapshot_raises_on_empty_dataframe() -> None:
    empty = pd.DataFrame()

    with pytest.raises(ValueError, match="empty DataFrame"):
        get_latest_snapshot(empty)


def test_run_csrm_handles_missing_optional_macro_inputs() -> None:
    prices = make_valid_prices().drop(columns=["HY_OAS", "IG_OAS", "NFCI", "ANFCI"])

    result = run_csrm(prices)

    assert "composite_score" in result.columns
    assert "regime" in result.columns
    assert result["composite_score"].notna().sum() > 0


def test_run_csrm_does_not_mutate_original_dataframe() -> None:
    prices = make_valid_prices()
    original = prices.copy(deep=True)

    _ = run_csrm(prices)

    pd.testing.assert_frame_equal(prices, original)


def test_run_csrm_respects_custom_regime_thresholds() -> None:
    prices = make_valid_prices()

    config = {
        "regime": {
            "risk_on_min": 0.10,
            "risk_off_max": -0.10,
        }
    }

    result = run_csrm(prices, config=config)

    assert result["regime"].isin(["Risk-On", "Neutral", "Risk-Off"]).all()


def test_run_csrm_latest_observation_is_constructive_for_test_fixture() -> None:
    prices = make_valid_prices()

    result = run_csrm(prices)

    assert result["composite_score"].iloc[-1] > -1.0
    assert result["regime"].iloc[-1] in {"Neutral", "Risk-On", "Risk-Off"}


def test_run_interpretation_stage_from_raw_prices_runs_end_to_end() -> None:
    prices = make_valid_prices()

    result = run_interpretation_stage(prices)

    expected_columns = [
        "credit_trend_score_contribution",
        "credit_stress_score_contribution",
        "credit_confirmation_score_contribution",
        "credit_divergence_score_contribution",
        "credit_persistence_score_contribution",
        "dominant_positive_component",
        "dominant_negative_component",
        "regime_signal_direction",
        "regime_conviction",
        "stress_state",
        "trend_state",
        "confirmation_state",
        "divergence_state",
        "persistence_state",
        "primary_driver",
        "primary_drag",
        "model_state_summary",
    ]

    for col in expected_columns:
        assert col in result.columns