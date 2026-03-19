import numpy as np
import pandas as pd
import pytest

from credit_spread_regime_model.transforms import (
    build_ratio_frame,
    build_return_frame,
    build_transforms,
    compute_log_ratio,
    compute_ratio,
    compute_return_series,
    get_ratio_columns,
    get_return_columns,
)


REQUIRED_COLUMNS = [
    "HYG",
    "JNK",
    "LQD",
    "SHY",
    "IEF",
    "TLT",
    "SPY",
    "VIX",
]


def make_valid_prices(
    periods: int = 300,
    start: str = "2020-01-01",
) -> pd.DataFrame:
    """
    Create a deterministic valid input DataFrame for transform tests.
    """
    index = pd.date_range(start=start, periods=periods, freq="B")
    base = np.arange(periods, dtype=float)

    data = {
        "HYG": 80.0 + 0.05 * base,
        "JNK": 100.0 + 0.04 * base,
        "LQD": 110.0 + 0.03 * base,
        "SHY": 85.0 + 0.01 * base,
        "IEF": 95.0 + 0.02 * base,
        "TLT": 120.0 + 0.025 * base,
        "SPY": 300.0 + 0.20 * base,
        "VIX": 20.0 + 0.01 * base,
    }
    return pd.DataFrame(data, index=index)


def test_compute_ratio_divides_numerator_by_denominator() -> None:
    numerator = pd.Series([10.0, 20.0, 30.0])
    denominator = pd.Series([2.0, 4.0, 5.0])

    result = compute_ratio(numerator=numerator, denominator=denominator)

    expected = pd.Series([5.0, 5.0, 6.0])
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_compute_ratio_sets_zero_denominator_to_nan() -> None:
    numerator = pd.Series([10.0, 20.0, 30.0])
    denominator = pd.Series([2.0, 0.0, 5.0])

    result = compute_ratio(numerator=numerator, denominator=denominator)

    assert np.isclose(result.iloc[0], 5.0)
    assert np.isnan(result.iloc[1])
    assert np.isclose(result.iloc[2], 6.0)


def test_compute_log_ratio_returns_log_for_positive_values() -> None:
    ratio = pd.Series([1.0, np.e, np.e**2])

    result = compute_log_ratio(ratio)

    expected = pd.Series([0.0, 1.0, 2.0])
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_compute_log_ratio_sets_nonpositive_values_to_nan() -> None:
    ratio = pd.Series([1.0, 0.0, -1.0, 2.0])

    result = compute_log_ratio(ratio)

    assert np.isclose(result.iloc[0], 0.0)
    assert np.isnan(result.iloc[1])
    assert np.isnan(result.iloc[2])
    assert np.isclose(result.iloc[3], np.log(2.0))


def test_compute_return_series_pct_change_matches_expected() -> None:
    series = pd.Series([100.0, 110.0, 121.0])

    result = compute_return_series(series=series, method="pct_change")

    expected = pd.Series([np.nan, 0.10, 0.10])
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_compute_return_series_log_return_matches_expected() -> None:
    series = pd.Series([100.0, 110.0, 121.0])

    result = compute_return_series(series=series, method="log_return")

    expected = pd.Series([np.nan, np.log(1.10), np.log(1.10)])
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_compute_return_series_raises_for_unknown_method() -> None:
    series = pd.Series([100.0, 110.0, 121.0])

    with pytest.raises(ValueError, match="Unsupported return_method"):
        compute_return_series(series=series, method="unknown_method")


def test_build_ratio_frame_creates_all_configured_ratios() -> None:
    prices = make_valid_prices()

    ratio_config = {
        "hyg_ief": {"numerator": "HYG", "denominator": "IEF"},
        "lqd_ief": {"numerator": "LQD", "denominator": "IEF"},
        "hyg_lqd": {"numerator": "HYG", "denominator": "LQD"},
    }

    result = build_ratio_frame(df=prices, ratio_config=ratio_config, log_ratio=False)

    assert list(result.columns) == ["hyg_ief", "lqd_ief", "hyg_lqd"]
    assert np.isclose(result.iloc[0]["hyg_ief"], 80.0 / 95.0)
    assert np.isclose(result.iloc[0]["lqd_ief"], 110.0 / 95.0)
    assert np.isclose(result.iloc[0]["hyg_lqd"], 80.0 / 110.0)


def test_build_ratio_frame_optionally_creates_log_ratios() -> None:
    prices = make_valid_prices()

    ratio_config = {
        "hyg_ief": {"numerator": "HYG", "denominator": "IEF"},
    }

    result = build_ratio_frame(df=prices, ratio_config=ratio_config, log_ratio=True)

    assert "hyg_ief" in result.columns
    assert "log_hyg_ief" in result.columns
    assert np.isclose(result.iloc[0]["log_hyg_ief"], np.log(80.0 / 95.0))


def test_build_return_frame_creates_returns_for_assets_and_ratios() -> None:
    prices = make_valid_prices()
    prices["hyg_ief"] = prices["HYG"] / prices["IEF"]
    prices["lqd_ief"] = prices["LQD"] / prices["IEF"]

    result = build_return_frame(
        df=prices,
        ratio_columns=["hyg_ief", "lqd_ief"],
        return_method="pct_change",
    )

    expected_columns = [
        "hyg_ret",
        "jnk_ret",
        "lqd_ret",
        "shy_ret",
        "ief_ret",
        "tlt_ret",
        "spy_ret",
        "vix_ret",
        "hyg_ief_ret",
        "lqd_ief_ret",
    ]
    assert list(result.columns) == expected_columns
    assert np.isnan(result.iloc[0]["hyg_ret"])
    assert np.isnan(result.iloc[0]["hyg_ief_ret"])


def test_build_return_frame_ignores_missing_ratio_columns() -> None:
    prices = make_valid_prices()

    result = build_return_frame(
        df=prices,
        ratio_columns=["hyg_ief", "missing_ratio"],
        return_method="pct_change",
    )

    assert "hyg_ief_ret" not in result.columns
    assert "missing_ratio_ret" not in result.columns
    assert "hyg_ret" in result.columns
    assert "spy_ret" in result.columns


def test_build_transforms_returns_raw_inputs_plus_ratios_and_returns() -> None:
    prices = make_valid_prices()

    result = build_transforms(prices)

    for col in REQUIRED_COLUMNS:
        assert col in result.columns

    for col in ["hyg_ief", "lqd_ief", "hyg_lqd", "hyg_shy", "jnk_ief", "lqd_shy"]:
        assert col in result.columns

    for col in [
        "hyg_ret",
        "jnk_ret",
        "lqd_ret",
        "shy_ret",
        "ief_ret",
        "tlt_ret",
        "spy_ret",
        "vix_ret",
        "hyg_ief_ret",
        "lqd_ief_ret",
        "hyg_lqd_ret",
        "hyg_shy_ret",
        "jnk_ief_ret",
        "lqd_shy_ret",
    ]:
        assert col in result.columns


def test_build_transforms_supports_log_return_override() -> None:
    prices = make_valid_prices()

    config = {
        "transforms": {
            "return_method": "log_return",
        }
    }

    result = build_transforms(prices, config=config)

    assert "hyg_ret" in result.columns
    expected_second = np.log(prices["HYG"].iloc[1] / prices["HYG"].iloc[0])
    assert np.isclose(result["hyg_ret"].iloc[1], expected_second)


def test_build_transforms_supports_log_ratio_override() -> None:
    prices = make_valid_prices()

    config = {
        "transforms": {
            "log_ratio": True,
        }
    }

    result = build_transforms(prices, config=config)

    assert "hyg_ief" in result.columns
    assert "log_hyg_ief" in result.columns
    assert "log_lqd_ief" in result.columns


def test_build_transforms_propagates_nan_when_inputs_missing() -> None:
    prices = make_valid_prices()
    prices.loc[prices.index[20], "HYG"] = np.nan

    result = build_transforms(prices)

    assert np.isnan(result.loc[prices.index[20], "hyg_ief"])
    assert np.isnan(result.loc[prices.index[20], "hyg_lqd"])


def test_build_transforms_ratio_becomes_nan_when_denominator_zero_after_validation_override() -> None:
    prices = make_valid_prices()
    prices.loc[prices.index[50], "IEF"] = 0.0

    config = {
        "validation": {
            "reject_nonpositive_prices": False,
        }
    }

    result = build_transforms(prices, config=config)

    assert np.isnan(result.loc[prices.index[50], "hyg_ief"])
    assert np.isnan(result.loc[prices.index[50], "lqd_ief"])
    assert np.isnan(result.loc[prices.index[50], "jnk_ief"])


def test_get_ratio_columns_returns_configured_ratio_names() -> None:
    result = get_ratio_columns()

    assert result == ["hyg_ief", "lqd_ief", "hyg_lqd", "hyg_shy", "jnk_ief", "lqd_shy"]


def test_get_return_columns_returns_assets_and_ratio_returns() -> None:
    result = get_return_columns()

    expected = [
        "hyg_ret",
        "jnk_ret",
        "lqd_ret",
        "shy_ret",
        "ief_ret",
        "tlt_ret",
        "spy_ret",
        "vix_ret",
        "hyg_ief_ret",
        "lqd_ief_ret",
        "hyg_lqd_ret",
        "hyg_shy_ret",
        "jnk_ief_ret",
        "lqd_shy_ret",
    ]
    assert result == expected


def test_build_transforms_does_not_mutate_original_dataframe() -> None:
    prices = make_valid_prices()
    original = prices.copy(deep=True)

    _ = build_transforms(prices)

    pd.testing.assert_frame_equal(prices, original)