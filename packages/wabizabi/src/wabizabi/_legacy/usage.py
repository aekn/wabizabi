"""Usage accounting models."""

from __future__ import annotations

from dataclasses import dataclass
from types import NotImplementedType


@dataclass(frozen=True, slots=True)
class RunUsage:
    """Token usage captured during an agent run."""

    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        if self.input_tokens < 0:
            raise ValueError("input_tokens must be non-negative.")
        if self.output_tokens < 0:
            raise ValueError("output_tokens must be non-negative.")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @classmethod
    def zero(cls) -> RunUsage:
        return cls()

    def combine(self, other: RunUsage) -> RunUsage:
        return RunUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )

    def __add__(self, other: object) -> RunUsage | NotImplementedType:
        if not isinstance(other, RunUsage):
            return NotImplemented
        return self.combine(other)


__all__ = ["RunUsage"]
