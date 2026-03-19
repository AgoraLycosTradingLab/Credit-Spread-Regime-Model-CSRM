import numpy as np
import pandas as pd

from credit_spread_regime_model.features import (
    build_confirmation_features,
    build_divergence_features,
    build_features,
    build_persistence_features,
    build_stress_features,
    build_trend_features,
    build_vix_features,
    compute_positive_zscore_feature,
    compute_ratio_stress_feature,
    compute_trend_feature,
)
from credit_spread_regime_model.transforms import build_transforms
from credit_spread_regime_model.config import build_config


def make_valid_prices(
    periods: int = 300,
    start: str = "2020-01-01",
) -> pd.DataFrame:
    """
    Create a deterministic valid input DataFrame for feature tests.
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


def test_compute_trend_feature_returns_series_with_expected_bounds() -> None:
    series = pd.Series(np.linspace(100.0, 130.0, 100))
    result = compute_trend_feature(
        series=series,
        fast_window=5,
        slow_window=20,
        lower=-1.0,
        upper=1.0,
    )

    assert isinstance(result, pd.Series)
    assert result.min(skipna=True) >= -1.0
    assert result.max(skipna=True) <= 1.0


def test_compute_trend_feature_is_positive_for_uptrend() -> None:
    series = pd.Series(np.linspace(100.0, 140.0, 120))
    result = compute_trend_feature(
        series=series,
        fast_window=5,
        slow_window=20,
        lower=-1.0,
        upper=1.0,
    )

    assert result.iloc[-1] > 0.0


def test_compute_ratio_stress_feature_returns_bounded_series() -> None:
    series = pd.Series(
        np.concatenate(
            [
                np.linspace(1.0, 1.1, 40),
                np.linspace(1.1, 0.8, 40),
                np.linspace(0.8, 0.85, 20),
            ]
        )
    )

    result = compute_ratio_stress_feature(
        series=series,
        z_window=20,
        drawdown_window=30,
        zscore_clip=3.0,
        drawdown_clip=0.30,
    )

    assert isinstance(result, pd.Series)
    assert result.min(skipna=True) >= -1.0
    assert result.max(skipna=True) <= 1.0


def test_compute_ratio_stress_feature_rises_when_ratio_weakens() -> None:
    series = pd.Series(
        np.concatenate(
            [
                np.full(40, 1.00),
                np.linspace(1.00, 0.70, 40),
                np.full(20, 0.70),
            ]
        )
    )

    result = compute_ratio_stress_feature(
        series=series,
        z_window=20,
        drawdown_window=30,
        zscore_clip=3.0,
        drawdown_clip=0.30,
    )

    assert result.iloc[-1] > result.iloc[20]


def test_compute_positive_zscore_feature_returns_bounded_series() -> None:
    series = pd.Series(np.linspace(0.0, 5.0, 100))
    result = compute_positive_zscore_feature(
        series=series,
        window=20,
        zscore_clip=3.0,
    )

    assert result.min(skipna=True) >= -1.0
    assert result.max(skipna=True) <= 1.0


def test_build_vix_features_creates_expected_columns() -> None:
    prices = make_valid_prices()
    transformed = build_transforms(prices)
    config = build_config()

    result = build_vix_features(df=transformed, config=config)

    assert list(result.columns) == ["vix_level_zscore", "vix_trend_value"]
    assert result["vix_level_zscore"].min(skipna=True) >= -1.0
    assert result["vix_level_zscore"].max(skipna=True) <= 1.0
    assert result["vix_trend_value"].min(skipna=True) >= -1.0
    assert result["vix_trend_value"].max(skipna=True) <= 1.0


def test_build_trend_features_creates_expected_columns() -> None:
    prices = make_valid_prices()
    transformed = build_transforms(prices)
    config = build_config()

    result = build_trend_features(df=transformed, config=config)

    expected_columns = [
        "trend_hyg_ief",
        "trend_lqd_ief",
        "trend_hyg_lqd",
        "trend_hyg_shy",
        "trend_jnk_ief",
        "trend_lqd_shy",
        "credit_trend_feature",
    ]
    assert list(result.columns) == expected_columns
    assert result["credit_trend_feature"].min(skipna=True) >= -1.0
    assert result["credit_trend_feature"].max(skipna=True) <= 1.0


def test_build_stress_features_creates_expected_columns() -> None:
    prices = make_valid_prices()
    transformed = build_transforms(prices)
    config = build_config()

    result = build_stress_features(df=transformed, config=config)

    expected_columns = [
        "stress_hyg_ief",
        "stress_lqd_ief",
        "stress_hyg_lqd",
        "stress_hyg_shy",
        "stress_jnk_ief",
        "stress_lqd_shy",
        "hy_oas_stress",
        "ig_oas_stress",
        "nfci_stress",
        "anfci_stress",
        "credit_stress_feature",
    ]
    assert list(result.columns) == expected_columns
    assert result["credit_stress_feature"].min(skipna=True) >= -1.0
    assert result["credit_stress_feature"].max(skipna=True) <= 1.0


def test_build_stress_features_handles_missing_optional_macro_columns() -> None:
    prices = make_valid_prices().drop(columns=["HY_OAS", "IG_OAS", "NFCI", "ANFCI"])
    transformed = build_transforms(prices)
    config = build_config()

    result = build_stress_features(df=transformed, config=config)

    assert "hy_oas_stress" in result.columns
    assert "ig_oas_stress" in result.columns
    assert "nfci_stress" in result.columns
    assert "anfci_stress" in result.columns
    assert result["credit_stress_feature"].notna().sum() > 0


def test_build_confirmation_features_creates_expected_columns() -> None:
    prices = make_valid_prices()
    transformed = build_transforms(prices)
    config = build_config()
    vix_frame = build_vix_features(df=transformed, config=config)

    result = build_confirmation_features(
        df=transformed,
        vix_frame=vix_frame,
        config=config,
    )

    expected_columns = [
        "spy_trend_feature",
        "vix_level_feature",
        "vix_trend_feature",
        "credit_confirmation_feature",
    ]
    assert list(result.columns) == expected_columns
    assert result["credit_confirmation_feature"].min(skipna=True) >= -1.0
    assert result["credit_confirmation_feature"].max(skipna=True) <= 1.0


def test_build_divergence_features_creates_expected_columns() -> None:
    prices = make_valid_prices()
    transformed = build_transforms(prices)
    config = build_config()

    trend_frame = build_trend_features(df=transformed, config=config)
    vix_frame = build_vix_features(df=transformed, config=config)
    confirmation_frame = build_confirmation_features(
        df=transformed,
        vix_frame=vix_frame,
        config=config,
    )

    result = build_divergence_features(
        trend_frame=trend_frame,
        confirmation_frame=confirmation_frame,
        config=config,
    )

    expected_columns = [
        "credit_equity_divergence",
        "hy_ig_divergence",
        "credit_divergence_feature",
    ]
    assert list(result.columns) == expected_columns
    assert result["credit_divergence_feature"].min(skipna=True) >= 0.0
    assert result["credit_divergence_feature"].max(skipna=True) <= 1.0


def test_build_persistence_features_creates_expected_columns() -> None:
    prices = make_valid_prices()
    transformed = build_transforms(prices)
    config = build_config()

    trend_frame = build_trend_features(df=transformed, config=config)
    stress_frame = build_stress_features(df=transformed, config=config)

    result = build_persistence_features(
        trend_frame=trend_frame,
        stress_frame=stress_frame,
        config=config,
    )

    expected_columns = [
        "risk_on_persistence",
        "risk_off_persistence",
        "credit_persistence_feature",
    ]
    assert list(result.columns) == expected_columns
    assert result["risk_on_persistence"].min(skipna=True) >= 0.0
    assert result["risk_on_persistence"].max(skipna=True) <= 1.0
    assert result["risk_off_persistence"].min(skipna=True) >= 0.0
    assert result["risk_off_persistence"].max(skipna=True) <= 1.0
    assert result["credit_persistence_feature"].min(skipna=True) >= -1.0
    assert result["credit_persistence_feature"].max(skipna=True) <= 1.0


def test_build_features_from_raw_prices_creates_full_feature_set() -> None:
    prices = make_valid_prices()

    result = build_features(prices)

    expected_columns = [
        "trend_hyg_ief",
        "trend_lqd_ief",
        "trend_hyg_lqd",
        "trend_hyg_shy",
        "trend_jnk_ief",
        "trend_lqd_shy",
        "credit_trend_feature",
        "stress_hyg_ief",
        "stress_lqd_ief",
        "stress_hyg_lqd",
        "stress_hyg_shy",
        "stress_jnk_ief",
        "stress_lqd_shy",
        "credit_stress_feature",
        "hy_oas_stress",
        "ig_oas_stress",
        "nfci_stress",
        "anfci_stress",
        "spy_trend_feature",
        "vix_level_feature",
        "vix_trend_feature",
        "credit_confirmation_feature",
        "credit_equity_divergence",
        "hy_ig_divergence",
        "credit_divergence_feature",
        "risk_on_persistence",
        "risk_off_persistence",
        "credit_persistence_feature",
        "vix_level_zscore",
        "vix_trend_value",
    ]

    for col in expected_columns:
        assert col in result.columns


def test_build_features_accepts_pretransformed_input() -> None:
    prices = make_valid_prices()
    transformed = build_transforms(prices)

    result = build_features(transformed)

    assert "credit_trend_feature" in result.columns
    assert "credit_stress_feature" in result.columns
    assert "credit_confirmation_feature" in result.columns
    assert "credit_divergence_feature" in result.columns
    assert "credit_persistence_feature" in result.columns


def test_build_features_trend_is_positive_for_constructive_credit_environment() -> None:
    prices = make_valid_prices()
    result = build_features(prices)

    assert result["credit_trend_feature"].iloc[-1] > 0.0


def test_build_features_confirmation_is_positive_for_rising_spy_and_falling_vix() -> None:
    prices = make_valid_prices()
    result = build_features(prices)

    assert result["credit_confirmation_feature"].iloc[-1] > 0.0


def test_build_features_does_not_mutate_original_dataframe() -> None:
    prices = make_valid_prices()
    original = prices.copy(deep=True)

    _ = build_features(prices)

    pd.testing.assert_frame_equal(prices, original)