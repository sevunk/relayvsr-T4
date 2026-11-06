"""
Base Engine Interface (`engines/base_engine.py`).
Matches the `ai-studio-t4` engine contract so any engine built here can be imported
directly into `ai-studio-t4/engines/` without modification.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from config.catalog import ModelFileSpec, SPEED_PRESETS


class BaseEngine(ABC):
    """Abstract base class for AI Studio T4 engines."""

    engine_id: str = "base"
    display_name: str = "Base Engine"
    category: str = "upscaler"

    @abstractmethod
    def get_required_files(self, variant_id: str = "") -> Tuple[ModelFileSpec, ...]:
        """Returns the required model file specifications for this engine."""
        raise NotImplementedError

    @abstractmethod
    def build_workflow(self, params: Dict[str, object]) -> Dict[str, object]:
        """Builds a ComfyUI API workflow graph dictionary from user parameters."""
        raise NotImplementedError

    def get_presets(self) -> Dict[str, Dict[str, object]]:
        """Returns T4 speed/quality presets."""
        return dict(SPEED_PRESETS)
