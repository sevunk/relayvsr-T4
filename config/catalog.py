"""
Model & Engine Catalog for RelayVSR T4 Studio.
Designed with 100% schema compatibility for drop-in integration into `ai-studio-t4`.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ModelFileSpec:
    """Specification for a single model weight file on Hugging Face."""
    repo_id: str
    filename: str
    local_subdir: str
    local_filename: Optional[str] = None
    size_gb: float = 0.0
    sha256: Optional[str] = None
    gated: bool = False
    shared_with: Tuple[str, ...] = ()
    alias_paths: Tuple[str, ...] = ()

    @property
    def target_name(self) -> str:
        return self.local_filename or Path(self.filename).name

    def resolve_path(self, models_root: Path) -> Path:
        return models_root / self.local_subdir / self.target_name


@dataclass(frozen=True)
class EngineCatalogEntry:
    """Catalog metadata for an AI Studio Engine."""
    engine_id: str
    label: str
    short_name: str
    category: str
    native_scale: int
    supported_scales: Tuple[int, ...]
    precision: str
    vram_estimate_gb: float
    description: str
    files: Tuple[ModelFileSpec, ...] = field(default_factory=tuple)
    default_dtype: str = "fp16"
    default_chunk_len: int = 16
    default_offload: bool = True


# 4 Verified RelayVSR Model Bundles for Tesla T4 (15 GB VRAM / 12.7 GB System RAM)
RELAYVSR_BUNDLES: Dict[str, EngineCatalogEntry] = {
    "relayvsr_x2_int8": EngineCatalogEntry(
        engine_id="relayvsr_x2_int8",
        label="1. RelayVSR 2× INT8 ConvRot [Comfy-Org / LightX2V — 5.29 GB, Rekomendasi Utama T4]",
        short_name="RelayVSR 2× (INT8 ConvRot)",
        category="upscaler_video_image",
        native_scale=2,
        supported_scales=(2,),
        precision="int8_convrot + fp16",
        vram_estimate_gb=6.8,
        description=(
            "Versi kuantisasi resmi INT8 ConvRot (5.16 GB) + Flash 2× Decoder (128 MB) dari Comfy-Org/RelayVSR. "
            "Mendukung Tensor Core INT8 pada Tesla T4 (Turing sm_75), ~6.2× lebih cepat dari BF16, bebas temporal flicker, "
            "dan sangat aman untuk batas VRAM 15 GB & System RAM 12.7 GB di Google Colab."
        ),
        files=(
            ModelFileSpec(
                repo_id="Comfy-Org/RelayVSR",
                filename="diffusion_models/relayvsr_x2_lightx2v_int8_convrot.safetensors",
                local_subdir="relayvsr/RelayVSR_X2_int8_convrot",
                local_filename="wan_x2_int8_convrot.safetensors",
                size_gb=5.16,
                sha256="240e75e0c17cf2845d559b25254a230ff7458f81449d7a8c5e7e3fb8275c7b3c",
                shared_with=("relayvsr_x2_int8",),
                alias_paths=(
                    "diffusion_models/relayvsr_x2_lightx2v_int8_convrot.safetensors",
                ),
            ),
            ModelFileSpec(
                repo_id="Comfy-Org/RelayVSR",
                filename="vae/relayvsr_flash_vae_x2_fp32.safetensors",
                local_subdir="relayvsr/RelayVSR_X2_int8_convrot",
                local_filename="flash_x2.safetensors",
                size_gb=0.128,
                shared_with=("relayvsr_x2_int8", "relayvsr_x2_fp16"),
                alias_paths=(
                    "vae/relayvsr_flash_vae_x2_fp32.safetensors",
                ),
            ),
        ),
        default_dtype="fp16",
        default_chunk_len=16,
        default_offload=True,
    ),
    "relayvsr_x4_int8": EngineCatalogEntry(
        engine_id="relayvsr_x4_int8",
        label="2. RelayVSR 4× INT8 ConvRot [Comfy-Org / Kopper — 5.29 GB, Upscale 4× & 2× di T4]",
        short_name="RelayVSR 4× (INT8 ConvRot)",
        category="upscaler_video_image",
        native_scale=4,
        supported_scales=(2, 4),
        precision="int8_convrot + fp16",
        vram_estimate_gb=7.9,
        description=(
            "Versi kuantisasi resmi INT8 ConvRot (5.16 GB) + Flash 4× Decoder (128 MB) dari Comfy-Org/RelayVSR. "
            "Menghasilkan super-resolusi penuh 4× (atau 2× via Lanczos/bicubic downscale) untuk gambar tunggal maupun video."
        ),
        files=(
            ModelFileSpec(
                repo_id="Comfy-Org/RelayVSR",
                filename="diffusion_models/relayvsr_x4_int8_convrot.safetensors",
                local_subdir="relayvsr/RelayVSR_X4_int8_convrot",
                local_filename="wan_int8_convrot.safetensors",
                size_gb=5.16,
                sha256="21303b3c5f9ab4e1fd3fe00df5dd2e0b343f0cf2e3d269e03a9872000a313a2e",
                shared_with=("relayvsr_x4_int8",),
                alias_paths=(
                    "diffusion_models/relayvsr_x4_int8_convrot.safetensors",
                ),
            ),
            ModelFileSpec(
                repo_id="Comfy-Org/RelayVSR",
                filename="vae/relayvsr_flash_vae_x4_fp32.safetensors",
                local_subdir="relayvsr/RelayVSR_X4_int8_convrot",
                local_filename="flash.safetensors",
                size_gb=0.128,
                shared_with=("relayvsr_x4_int8", "relayvsr_x4_fp16"),
                alias_paths=(
                    "vae/relayvsr_flash_vae_x4_fp32.safetensors",
                ),
            ),
        ),
        default_dtype="fp16",
        default_chunk_len=12,
        default_offload=True,
    ),
    "relayvsr_x2_fp16": EngineCatalogEntry(
        engine_id="relayvsr_x2_fp16",
        label="3. RelayVSR 2× Full Weights [LightX2V Official — 10.23 GB, Wajib Block Offload di T4]",
        short_name="RelayVSR 2× (Full 10.2GB)",
        category="upscaler_video_image",
        native_scale=2,
        supported_scales=(2,),
        precision="fp16 (from bf16)",
        vram_estimate_gb=11.5,
        description=(
            "Checkpoint penuh LightX2V RelayVSR_X2 (wan_x2.safetensors 10.1 GB + flash_x2.safetensors 127 MB). "
            "Dieksekusi dalam FP16 dengan per-block Wan GPU offload agar tetap muat di Tesla T4."
        ),
        files=(
            ModelFileSpec(
                repo_id="lightx2v/RelayVSR_X2",
                filename="wan_x2.safetensors",
                local_subdir="relayvsr/RelayVSR_X2",
                local_filename="wan_x2.safetensors",
                size_gb=10.1,
            ),
            ModelFileSpec(
                repo_id="lightx2v/RelayVSR_X2",
                filename="flash_x2.safetensors",
                local_subdir="relayvsr/RelayVSR_X2",
                local_filename="flash_x2.safetensors",
                size_gb=0.127,
            ),
        ),
        default_dtype="fp16",
        default_chunk_len=12,
        default_offload=True,
    ),
    "relayvsr_x4_fp16": EngineCatalogEntry(
        engine_id="relayvsr_x4_fp16",
        label="4. RelayVSR 4× Official [Kopper Original — 10.23 GB, Wajib Block Offload di T4]",
        short_name="RelayVSR 4× (Official 10.2GB)",
        category="upscaler_video_image",
        native_scale=4,
        supported_scales=(2, 4),
        precision="fp16 (from bf16)",
        vram_estimate_gb=12.4,
        description=(
            "Checkpoint asli kopper/RelayVSR (wan.safetensors 10.1 GB + flash.safetensors 128 MB). "
            "Mendukung baik Comfyui-Relay-VSR maupun CLI resmi kopperx/RelayVSR (inference.py --dtype fp16)."
        ),
        files=(
            ModelFileSpec(
                repo_id="kopper/RelayVSR",
                filename="wan.safetensors",
                local_subdir="relayvsr/RelayVSR_X4",
                local_filename="wan.safetensors",
                size_gb=10.07,
                sha256="638acb6a27c17ba87b2e66a1770f17e61a99ad87cfff6fa3d3e05c2d57f2eb98",
            ),
            ModelFileSpec(
                repo_id="kopper/RelayVSR",
                filename="flash.safetensors",
                local_subdir="relayvsr/RelayVSR_X4",
                local_filename="flash.safetensors",
                size_gb=0.128,
            ),
        ),
        default_dtype="fp16",
        default_chunk_len=8,
        default_offload=True,
    ),
}

# Unified ENGINE_CATALOG alias for plug-and-play merging into `ai-studio-t4/config/catalog.py`
ENGINE_CATALOG: Dict[str, EngineCatalogEntry] = dict(RELAYVSR_BUNDLES)
ENGINE_CATALOG["relayvsr"] = RELAYVSR_BUNDLES["relayvsr_x2_int8"]

# Preset configurations matching `ai-studio-t4` convention (⚡ Extreme Turbo, 🚀 Speed, 🎯 High Quality)
SPEED_PRESETS: Dict[str, Dict[str, object]] = {
    "⚡ Extreme Turbo (T4 Safe)": {
        "chunk_len": 8,
        "chunk_overlap": 2,
        "dtype": "fp16",
        "offload": True,
        "spatial_grid_cols": 1,
        "spatial_grid_rows": 1,
        "spatial_overlap": 48,
        "auto_tile_threshold": 1024,
        "description": "Window 8 frame, FP16 + Block Offload aktif. Paling ringan VRAM (~6.5 GB) & bebas OOM di T4.",
    },
    "🚀 Speed (Balanced Default)": {
        "chunk_len": 16,
        "chunk_overlap": 4,
        "dtype": "fp16",
        "offload": True,
        "spatial_grid_cols": 1,
        "spatial_grid_rows": 1,
        "spatial_overlap": 64,
        "auto_tile_threshold": 1280,
        "description": "Window 16 frame (interval keyframe Δ=15 sesuai paper), FP16 + Block Offload. Keseimbangan terbaik.",
    },
    "🎯 High Quality (2K/4K Tiled)": {
        "chunk_len": 16,
        "chunk_overlap": 4,
        "dtype": "fp16",
        "offload": True,
        "spatial_grid_cols": 2,
        "spatial_grid_rows": 2,
        "spatial_overlap": 64,
        "auto_tile_threshold": 960,
        "description": "Window 16 frame dengan 2×2 Spatial Tiling + Linear Crossfade 64px untuk input resolusi tinggi.",
    },
}


def get_engine_entry(engine_id_or_label: str) -> EngineCatalogEntry:
    key = (engine_id_or_label or "").strip()
    if key in ENGINE_CATALOG:
        return ENGINE_CATALOG[key]
    for entry in RELAYVSR_BUNDLES.values():
        if key == entry.label or key == entry.short_name or key.startswith(entry.label[:18]):
            return entry
    return RELAYVSR_BUNDLES["relayvsr_x2_int8"]


def get_bundle_files(bundle_id: str) -> Tuple[ModelFileSpec, ...]:
    entry = get_engine_entry(bundle_id)
    return entry.files


def check_bundle_installed(bundle_id: str, models_root: Path) -> Tuple[bool, List[Path]]:
    entry = get_engine_entry(bundle_id)
    missing: List[Path] = []
    for fspec in entry.files:
        p = fspec.resolve_path(models_root)
        if not p.exists() or p.stat().st_size < 1024 * 100:
            missing.append(p)
    return (len(missing) == 0, missing)
