from __future__ import annotations

import pytest
from wabizabi.usage import RunUsage


def test_total_tokens_is_derived() -> None:
    usage = RunUsage(input_tokens=4, output_tokens=7)

    assert usage.total_tokens == 11


def test_usage_combine_adds_counts() -> None:
    left = RunUsage(input_tokens=2, output_tokens=3)
    right = RunUsage(input_tokens=5, output_tokens=7)

    assert left.combine(right) == RunUsage(input_tokens=7, output_tokens=10)
    assert left + right == RunUsage(input_tokens=7, output_tokens=10)


def test_usage_rejects_negative_counts() -> None:
    with pytest.raises(ValueError, match="input_tokens must be non-negative"):
        RunUsage(input_tokens=-1)
