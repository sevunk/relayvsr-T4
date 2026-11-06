"""
Engine Registry (`engines/__init__.py`).
Follows the exact `ENGINE_CLASSES`, `MAIN_ENGINE_IDS`, and `resolve_engine_id()`
contract from `ai-studio-t4/engines/__init__.py` so integrating RelayVSR into
`ai-studio-t4` only requires registering `RelayVSREngine` here.
"""

from typing import Dict, Type

from engines.base_engine import BaseEngine
from engines.relayvsr_engine import RelayVSREngine

ENGINE_CLASSES: Dict[str, Type[BaseEngine]] = {
    "relayvsr": RelayVSREngine,
    "relayvsr_x2_int8": RelayVSREngine,
    "relayvsr_x4_int8": RelayVSREngine,
    "relayvsr_x2_fp16": RelayVSREngine,
    "relayvsr_x4_fp16": RelayVSREngine,
}

MAIN_ENGINE_IDS = (
    "relayvsr_x2_int8",
    "relayvsr_x4_int8",
    "relayvsr_x2_fp16",
    "relayvsr_x4_fp16",
)


def resolve_engine_id(label_or_id: str) -> str:
    text = (label_or_id or "").strip().lower()
    if text in ENGINE_CLASSES:
        return text
    if "4× int8" in text or "4x int8" in text or text.startswith("2."):
        return "relayvsr_x4_int8"
    if "2× full" in text or "2x full" in text or text.startswith("3."):
        return "relayvsr_x2_fp16"
    if "4× official" in text or "4x official" in text or text.startswith("4."):
        return "relayvsr_x4_fp16"
    return "relayvsr_x2_int8"


__all__ = [
    "BaseEngine",
    "RelayVSREngine",
    "ENGINE_CLASSES",
    "MAIN_ENGINE_IDS",
    "resolve_engine_id",
]
