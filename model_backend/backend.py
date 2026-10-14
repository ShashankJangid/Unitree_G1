from __future__ import annotations
import abc
from typing import Any, Dict, Optional
import numpy as np

class ModelBackend(abc.ABC):
    @abc.abstractmethod
    def load_model(self, model_path: str) -> bool: ...

    @abc.abstractmethod
    def infer(self, observation: np.ndarray) -> np.ndarray: ...

    @abc.abstractmethod
    def reset(self) -> None: ...
