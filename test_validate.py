import numpy as np
import pandas as pd
import pytest

from credit_spread_regime_model.validate import validate_inputs


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
    Create a deterministic valid input DataFrame for validation tests.
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


def test_validate_inputs_returns_dataframe_for_valid_input() -> None:
    prices = make_valid_prices()
    result = validate_inputs(prices)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == REQUIRED_COLUMNS
    assert result.index.is_monotonic_increasing
    assert result.shape == prices.shape


def test_validate_inputs_accepts_optional_macro_columns() -> None:
    prices = make_valid_prices()
    prices["HY_OAS"] = np.linspace(3.0, 4.0, len(prices))
    prices["IG_OAS"] = np.linspace(1.0, 1.5, len(prices))
    prices["NFCI"] = np.linspace(-0.5, 0.5, len(prices))
    prices["ANFCI"] = np.linspace(-0.25, 0.25, len(prices))

    result = validate_inputs(prices)

    for col in ["HY_OAS", "IG_OAS", "NFCI", "ANFCI"]:
        assert col in result.columns


def test_validate_inputs_raises_for_missing_required_columns() -> None:
    prices = make_valid_prices().drop(columns=["HYG"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_inputs(prices)


def test_validate_inputs_sorts_unsorted_index() -> None:
    prices = make_valid_prices().sample(frac=1.0, random_state=7)

    result = validate_inputs(prices)

    assert result.index.is_monotonic_increasing
    pd.testing.assert_index_equal(result.index, prices.index.sort_values())


def test_validate_inputs_drops_duplicate_index_keeping_last() -> None:
    prices = make_valid_prices(periods=300)
    duplicate_row = prices.iloc[[10]].copy()

    duplicated = pd.concat([prices.iloc[:11], duplicate_row, prices.iloc[11:]], axis=0)
    duplicated.index = list(prices.index[:11]) + [prices.index[10]] + list(prices.index[11:])

    duplicated.iloc[11, duplicated.columns.get_loc("HYG")] = 999.0

    result = validate_inputs(duplicated)

    assert result.index.is_unique
    assert result.loc[prices.index[10], "HYG"] == 999.0


def test_validate_inputs_converts_string_index_to_datetime() -> None:
    prices = make_valid_prices()
    prices.index = prices.index.strftime("%Y-%m-%d")

    result = validate_inputs(prices)

    assert pd.api.types.is_datetime64_any_dtype(result.index)


def test_validate_inputs_raises_for_nonconvertible_index() -> None:
    prices = make_valid_prices()
    prices.index = ["not-a-date"] * len(prices)

    with pytest.raises(TypeError, match="datetime-like"):
        validate_inputs(prices)


def test_validate_inputs_coerces_numeric_strings() -> None:
    prices = make_valid_prices().astype(str)

    result = validate_inputs(prices)

    assert all(pd.api.types.is_numeric_dtype(result[col]) for col in result.columns)
    assert np.isclose(result.iloc[0]["HYG"], 80.0)


def test_validate_inputs_raises_when_missing_fraction_exceeds_threshold() -> None:
    prices = make_valid_prices()
    prices.loc[prices.index[:40], "HYG"] = np.nan  # 40 / 300 > 0.10

    with pytest.raises(ValueError, match="maximum missing fraction"):
        validate_inputs(prices)


def test_validate_inputs_allows_missing_fraction_at_threshold() -> None:
    prices = make_valid_prices()
    prices.loc[prices.index[:30], "HYG"] = np.nan  # 30 / 300 == 0.10

    result = validate_inputs(prices)

    assert result["HYG"].isna().sum() == 30


def test_validate_inputs_raises_for_nonpositive_required_prices() -> None:
    prices = make_valid_prices()
    prices.loc[prices.index[50], "IEF"] = 0.0

    with pytest.raises(ValueError, match="Strictly positive price validation failed"):
        validate_inputs(prices)


def test_validate_inputs_raises_for_negative_required_prices() -> None:
    prices = make_valid_prices()
    prices.loc[prices.index[75], "SPY"] = -1.0

    with pytest.raises(ValueError, match="Strictly positive price validation failed"):
        validate_inputs(prices)


def test_validate_inputs_raises_for_insufficient_history() -> None:
    prices = make_valid_prices(periods=200)

    with pytest.raises(ValueError, match="Insufficient history"):
        validate_inputs(prices)


def test_validate_inputs_drops_all_nan_rows() -> None:
    prices = make_valid_prices()
    prices.loc[prices.index[5], :] = np.nan

    result = validate_inputs(prices)

    assert prices.index[5] not in result.index
    assert len(result) == len(prices) - 1


def test_validate_inputs_raises_if_all_rows_become_empty() -> None:
    prices = make_valid_prices()
    prices.loc[:, :] = np.nan

    with pytest.raises(ValueError, match="no usable rows"):
        validate_inputs(prices)


def test_validate_inputs_can_drop_extra_columns_when_configured() -> None:
    prices = make_valid_prices()
    prices["EXTRA_COL"] = 123.0

    config = {
        "inputs": {
            "allow_extra_columns": False,
        }
    }

    result = validate_inputs(prices, config=config)

    assert "EXTRA_COL" not in result.columns
    assert list(result.columns) == REQUIRED_COLUMNS


def test_validate_inputs_keeps_extra_columns_by_default() -> None:
    prices = make_valid_prices()
    prices["EXTRA_COL"] = 123.0

    result = validate_inputs(prices)

    assert "EXTRA_COL" in result.columns


def test_validate_inputs_handles_optional_macro_with_missing_values() -> None:
    prices = make_valid_prices()
    prices["HY_OAS"] = np.nan
    prices["IG_OAS"] = np.linspace(1.0, 1.5, len(prices))

    result = validate_inputs(prices)

    assert "HY_OAS" in result.columns
    assert "IG_OAS" in result.columns
    assert result["HY_OAS"].isna().all()


def test_validate_inputs_does_not_mutate_original_dataframe() -> None:
    prices = make_valid_prices()
    original = prices.copy(deep=True)

    _ = validate_inputs(prices)

    pd.testing.assert_frame_equal(prices, original)


def test_validate_inputs_preserves_required_column_order() -> None:
    prices = make_valid_prices()
    reordered = prices[["SPY", "VIX", "HYG", "JNK", "LQD", "SHY", "IEF", "TLT"]]

    result = validate_inputs(reordered)

    assert list(result.columns) == ["SPY", "VIX", "HYG", "JNK", "LQD", "SHY", "IEF", "TLT"]


def test_validate_inputs_with_config_override_for_min_history() -> None:
    prices = make_valid_prices(periods=100)

    config = {
        "inputs": {
            "price_min_history": 100,
        }
    }

    result = validate_inputs(prices, config=config)

    assert len(result) == 100