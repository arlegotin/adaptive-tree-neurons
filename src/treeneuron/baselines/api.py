"""Shared adapter contract from design specification section 11."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class FitRequest:
    X_fit: Any
    y_fit: np.ndarray
    X_select: Any
    y_select: np.ndarray
    seed: int
    config: dict[str, Any]
    budget: dict[str, Any]
    artifact_dir: Path


class Predictor(Protocol):
    def fit(self, request: FitRequest) -> 'Predictor': ...
    def predict(self, X: Any, *, batch_size: int | None = None) -> np.ndarray: ...
    def save(self, path: Path) -> None: ...
    def resource_summary(self) -> dict[str, Any]: ...
