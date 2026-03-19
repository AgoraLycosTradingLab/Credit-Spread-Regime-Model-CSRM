import numpy as np
import pandas as pd

from credit_spread_regime_model.config import build_config
from credit_spread_regime_model.features import build_features
from credit_spread_regime_model.scoring import (
    build_scores,
    compute_composite_score,
    get_subscore_weights,
    map_regime_to_numeric,
    map_score_to_regime,
    score_confirmation_feature,
    score_divergence_feature,
    score_persistence_feature,
    score_stress_feature,
    score_trend_feature,
)


def make_valid_prices(
    periods: int = 300,
    start: str = "2020-01-01",
) -> pd.DataFrame:
    """
    Create a deterministic valid input DataFrame for scoring tests.
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


def test_score_trend_feature_preserves_direction_and_bounds() -> None:
    config = build_config()
    feature = pd.Series([-1.5, -0.2, 0.0, 0.4, 1.5])

    result = score_trend_feature(feature=feature, config=config)

    expected = pd.Series([-1.0, -0.2, 0.0, 0.4, 1.0])
    pd.testing.assert_series_equal(result, expected)


def test_score_stress_feature_inverts_feature_and_clips() -> None:
    config = build_config()
    feature = pd.Series([-1.5, -0.4, 0.0, 0.3, 1.5])

    result = score_stress_feature(feature=feature, config=config)

    expected = pd.Series([1.0, 0.4, -0.0, -0.3, -1.0])
    pd.testing.assert_series_equal(result, expected)


def test_score_confirmation_feature_preserves_direction_and_bounds() -> None:
    config = build_config()
    feature = pd.Series([-2.0, -0.5, 0.0, 0.6, 2.0])

    result = score_confirmation_feature(feature=feature, config=config)

    expected = pd.Series([-1.0, -0.5, 0.0, 0.6, 1.0])
    pd.testing.assert_series_equal(result, expected)


def test_score_divergence_feature_inverts_feature_and_clips() -> None:
    config = build_config()
    feature = pd.Series([0.0, 0.2, 0.7, 1.2])

    result = score_divergence_feature(feature=feature, config=config)

    expected = pd.Series([-0.0, -0.2, -0.7, -1.0])
    pd.testing.assert_series_equal(result, expected)


def test_score_persistence_feature_preserves_direction_and_bounds() -> None:
    config = build_config()
    feature = pd.Series([-1.5, -0.3, 0.0, 0.8, 1.5])

    result = score_persistence_feature(feature=feature, config=config)

    expected = pd.Series([-1.0, -0.3, 0.0, 0.8, 1.0])
    pd.testing.assert_series_equal(result, expected)


def test_get_subscore_weights_returns_expected_weights() -> None:
    result = get_subscore_weights()

    expected = pd.Series(
        {
            "credit_trend_score": 0.30,
            "credit_stress_score": 0.25,
            "credit_confirmation_score": 0.20,
            "credit_divergence_score": 0.10,
            "credit_persistence_score": 0.15,
        },
        dtype=float,
    )
    pd.testing.assert_series_equal(result, expected)


def test_compute_composite_score_matches_weighted_average_when_all_present() -> None:
    config = build_config()

    score_frame = pd.DataFrame(
        {
            "credit_trend_score": [0.50],
            "credit_stress_score": [0.20],
            "credit_confirmation_score": [0.10],
            "credit_divergence_score": [-0.30],
            "credit_persistence_score": [0.40],
        }
    )

    result = compute_composite_score(score_frame=score_frame, config=config)

    expected_value = (
        0.30 * 0.50
        + 0.25 * 0.20
        + 0.20 * 0.10
        + 0.10 * (-0.30)
        + 0.15 * 0.40
    ) / (0.30 + 0.25 + 0.20 + 0.10 + 0.15)

    assert np.isclose(result.iloc[0], expected_value)


def test_compute_composite_score_renormalizes_when_scores_missing() -> None:
    config = build_config()

    score_frame = pd.DataFrame(
        {
            "credit_trend_score": [0.60],
            "credit_stress_score": [np.nan],
            "credit_confirmation_score": [0.20],
            "credit_divergence_score": [np.nan],
            "credit_persistence_score": [0.40],
        }
    )

    result = compute_composite_score(score_frame=score_frame, config=config)

    expected_value = (
        0.30 * 0.60
        + 0.20 * 0.20
        + 0.15 * 0.40
    ) / (0.30 + 0.20 + 0.15)

    assert np.isclose(result.iloc[0], expected_value)


def test_compute_composite_score_clips_to_bounds() -> None:
    config = build_config()

    score_frame = pd.DataFrame(
        {
            "credit_trend_score": [1.0],
            "credit_stress_score": [1.0],
            "credit_confirmation_score": [1.0],
            "credit_divergence_score": [1.0],
            "credit_persistence_score": [1.0],
        }
    )

    result = compute_composite_score(score_frame=score_frame, config=config)

    assert result.iloc[0] <= 1.0
    assert result.iloc[0] >= -1.0


def test_map_score_to_regime_returns_expected_labels() -> None:
    config = build_config()
    composite_score = pd.Series([-0.50, -0.10, 0.00, 0.20, 0.35, np.nan])

    result = map_score_to_regime(composite_score=composite_score, config=config)

    expected = pd.Series(
        ["Risk-Off", "Neutral", "Neutral", "Neutral", "Risk-On", np.nan],
        dtype=object,
    )
    pd.testing.assert_series_equal(result, expected)


def test_map_regime_to_numeric_returns_expected_values() -> None:
    config = build_config()
    regime = pd.Series(["Risk-On", "Neutral", "Risk-Off", np.nan], dtype=object)

    result = map_regime_to_numeric(regime=regime, config=config)

    expected = pd.Series([1.0, 0.0, -1.0, np.nan], dtype=float)
    pd.testing.assert_series_equal(result, expected)


def test_build_scores_from_feature_frame_creates_expected_columns() -> None:
    prices = make_valid_prices()
    features = build_features(prices)

    result = build_scores(features)

    expected_columns = [
        "credit_trend_score",
        "credit_stress_score",
        "credit_confirmation_score",
        "credit_divergence_score",
        "credit_persistence_score",
        "composite_score",
        "composite_score_clipped",
        "regime",
        "regime_numeric",
        "risk_on_signal",
        "risk_off_signal",
    ]

    for col in expected_columns:
        assert col in result.columns


def test_build_scores_accepts_raw_prices_and_runs_end_to_end() -> None:
    prices = make_valid_prices()

    result = build_scores(prices)

    assert "credit_trend_score" in result.columns
    assert "credit_stress_score" in result.columns
    assert "credit_confirmation_score" in result.columns
    assert "credit_divergence_score" in result.columns
    assert "credit_persistence_score" in result.columns
    assert "composite_score" in result.columns
    assert "regime" in result.columns


def test_build_scores_outputs_are_bounded_where_expected() -> None:
    prices = make_valid_prices()

    result = build_scores(prices)

    subscore_columns = [
        "credit_trend_score",
        "credit_stress_score",
        "credit_confirmation_score",
        "credit_divergence_score",
        "credit_persistence_score",
        "composite_score",
        "composite_score_clipped",
    ]

    for col in subscore_columns:
        assert result[col].min(skipna=True) >= -1.0
        assert result[col].max(skipna=True) <= 1.0


def test_build_scores_regime_numeric_matches_regime_labels() -> None:
    prices = make_valid_prices()

    result = build_scores(prices)

    mapping = {
        "Risk-On": 1.0,
        "Neutral": 0.0,
        "Risk-Off": -1.0,
    }

    expected = result["regime"].map(mapping).astype(float)
    pd.testing.assert_series_equal(result["regime_numeric"], expected, check_names=False)


def test_build_scores_risk_signals_match_composite_sign() -> None:
    prices = make_valid_prices()

    result = build_scores(prices)

    expected_risk_on = (result["composite_score_clipped"] > 0.0).astype(float)
    expected_risk_off = (result["composite_score_clipped"] < 0.0).astype(float)

    pd.testing.assert_series_equal(result["risk_on_signal"], expected_risk_on, check_names=False)
    pd.testing.assert_series_equal(result["risk_off_signal"], expected_risk_off, check_names=False)


def test_build_scores_constructive_environment_tilts_positive_at_end() -> None:
    prices = make_valid_prices()

    result = build_scores(prices)

    assert result["composite_score_clipped"].iloc[-1] > 0.0
    assert result["regime"].iloc[-1] in {"Neutral", "Risk-On"}


def test_build_scores_does_not_mutate_original_dataframe() -> None:
    prices = make_valid_prices()
    original = prices.copy(deep=True)

    _ = build_scores(prices)

    pd.testing.assert_frame_equal(prices, original)


def test_build_scores_handles_missing_optional_macro_inputs() -> None:
    prices = make_valid_prices().drop(columns=["HY_OAS", "IG_OAS", "NFCI", "ANFCI"])

    result = build_scores(prices)

    assert "composite_score" in result.columns
    assert "regime" in result.columns
    assert result["composite_score"].notna().sum() > 0


def test_build_scores_respects_custom_regime_thresholds() -> None:
    prices = make_valid_prices()

    config = {
        "regime": {
            "risk_on_min": 0.10,
            "risk_off_max": -0.10,
        }
    }

    result = build_scores(prices, config=config)

    assert "regime" in result.columns
    assert result["regime"].isin(["Risk-On", "Neutral", "Risk-Off"]).all()