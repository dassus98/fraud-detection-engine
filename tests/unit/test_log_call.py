"""Unit tests for `fraud_engine.utils.logging.log_call` and `_describe`.

These are pure-function tests with no I/O. They verify the contract
every decorated data function in the pipeline will rely on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_engine.utils.logging import _describe, log_call


class TestDescribe:
    """Contract tests for the `_describe` shape summariser."""

    def test_dataframe_returns_shape(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        out = _describe(df)
        assert out["type"] == "DataFrame"
        assert out["shape"] == [3, 2]

    def test_ndarray_returns_shape(self) -> None:
        arr = np.zeros((4, 5))
        out = _describe(arr)
        assert out["type"] == "ndarray"
        assert out["shape"] == [4, 5]

    def test_str_returns_length_not_content(self) -> None:
        out = _describe("secrets-should-not-be-logged")
        assert out == {"type": "str", "length": len("secrets-should-not-be-logged")}

    def test_path_returns_path_string(self) -> None:
        out = _describe(Path("/tmp/raw"))
        assert out["type"] == "Path"
        assert out["path"].endswith("raw")

    def test_scalar_returns_value(self) -> None:
        assert _describe(42) == {"type": "int", "value": 42}
        assert _describe(3.14) == {"type": "float", "value": 3.14}
        assert _describe(True) == {"type": "bool", "value": True}
        assert _describe(None) == {"type": "NoneType", "value": None}

    def test_collection_returns_length(self) -> None:
        assert _describe([1, 2, 3]) == {"type": "list", "length": 3}
        assert _describe({"a": 1}) == {"type": "dict", "length": 1}


class TestLogCall:
    """Contract tests for the `log_call` decorator."""

    def test_returns_wrapped_result(self) -> None:
        @log_call
        def doubler(x: int) -> int:
            return x * 2

        assert doubler(5) == 10

    def test_preserves_function_metadata(self) -> None:
        @log_call
        def labelled(x: int) -> int:
            """Doubles its input."""
            return x * 2

        assert labelled.__name__ == "labelled"
        assert labelled.__doc__ == "Doubles its input."

    def test_reraises_exceptions(self) -> None:
        @log_call
        def explodes() -> None:
            raise ValueError("nope")

        with pytest.raises(ValueError, match="nope"):
            explodes()

    def test_passes_kwargs_through(self) -> None:
        @log_call
        def joiner(a: str, *, b: str = "default") -> str:
            return f"{a}-{b}"

        assert joiner("left", b="right") == "left-right"
