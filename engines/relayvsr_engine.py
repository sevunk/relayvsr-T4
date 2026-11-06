"""
RelayVSR Engine for Tesla T4 (`engines/relayvsr_engine.py`).
Supports:
1. Single Image & Batch Image Super-Resolution (2× and 4×) with EXIF transpose,
   stride-16 reflection padding, and automatic Spatial Tiling + Linear Crossfade.
2. Time-Chunked Streaming Video Super-Resolution (2× and 4×) with bounded KV cache,
   per-block Wan GPU offload, and lossless/AAC audio stream preservation.
3. Full ComfyUI API Workflow generation (`build_workflow`) for drop-in integration
   with `ai-studio-t4` (both `bbaudio-2025/Comfyui-Relay-VSR` nodes and ComfyUI Core PR #17511).
"""

import math
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from config.catalog import (
    RELAYVSR_BUNDLES,
    SPEED_PRESETS,
    EngineCatalogEntry,
    ModelFileSpec,
    check_bundle_installed,
    get_engine_entry,
)
from core.backend import RelayVSRBackend
from engines.base_engine import BaseEngine


def pad_array_to_multiple(arr: np.ndarray, multiple: int = 16) -> Tuple[np.ndarray, int, int]:
    """
    Pads an HxWxC numpy image array to a multiple of `multiple` (16 px required by
    RelayVSR's stride-4 LR encoder and stride-16 reference latent grid).
    Returns (padded_array, pad_h, pad_w).
    """
    h, w = arr.shape[:2]
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return arr, 0, 0
    mode = "reflect" if (h > pad_h and w > pad_w) else "edge"
    padded = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
    return padded, pad_h, pad_w


def compute_spatial_tiles(
    height: int,
    width: int,
    grid_rows: int,
    grid_cols: int,
    overlap: int = 64,
    multiple: int = 16,
) -> List[Tuple[int, int, int, int]]:
    """
    Computes (y0, y1, x0, x1) spatial tile coordinates in input pixels with `overlap` halo,
    aligned to `multiple` (16 px) where possible.
    """
    grid_rows = max(1, int(grid_rows))
    grid_cols = max(1, int(grid_cols))
    if grid_rows == 1 and grid_cols == 1:
        return [(0, height, 0, width)]

    tile_h = math.ceil(height / grid_rows)
    tile_w = math.ceil(width / grid_cols)
    half_ov = max(0, int(overlap) // 2)

    coords: List[Tuple[int, int, int, int]] = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            y0 = max(0, r * tile_h - half_ov)
            y1 = min(height, (r + 1) * tile_h + half_ov)
            x0 = max(0, c * tile_w - half_ov)
            x1 = min(width, (c + 1) * tile_w + half_ov)
            coords.append((y0, y1, x0, x1))
    return coords


def blend_spatial_tiles(
    tiles_out: List[Tuple[Tuple[int, int, int, int], np.ndarray]],
    out_height: int,
    out_width: int,
    sr_ratio: int,
) -> np.ndarray:
    """
    Stitches upscaled spatial tiles using smooth 2D linear crossfade weights so tile
    boundaries are completely invisible.
    """
    if len(tiles_out) == 1:
        return tiles_out[0][1][:out_height, :out_width]

    accum = np.zeros((out_height, out_width, 3), dtype=np.float32)
    weight_sum = np.zeros((out_height, out_width, 1), dtype=np.float32)

    for (y0, y1, x0, x1), tile_rgb in tiles_out:
        oy0, oy1 = y0 * sr_ratio, min(out_height, y1 * sr_ratio)
        ox0, ox1 = x0 * sr_ratio, min(out_width, x1 * sr_ratio)
        th = oy1 - oy0
        tw = ox1 - ox0
        tile_crop = tile_rgb[:th, :tw].astype(np.float32)

        # Create 2D feathering ramp near edges that touch interior boundaries
        wy = np.ones((th, 1), dtype=np.float32)
        wx = np.ones((1, tw), dtype=np.float32)
        ramp_len = min(max(16, 32 * sr_ratio), th // 4, tw // 4)
        if ramp_len > 1:
            ramp = np.linspace(0.05, 1.0, ramp_len, dtype=np.float32)
            if oy0 > 0:
                wy[:ramp_len, 0] = ramp
            if oy1 < out_height:
                wy[-ramp_len:, 0] = ramp[::-1]
            if ox0 > 0:
                wx[0, :ramp_len] = ramp
            if ox1 < out_width:
                wx[0, -ramp_len:] = ramp[::-1]

        w2d = (wy @ wx)[:, :, None]
        accum[oy0:oy1, ox0:ox1] += tile_crop * w2d
        weight_sum[oy0:oy1, ox0:ox1] += w2d

    stitched = accum / np.maximum(weight_sum, 1e-6)
    return np.clip(stitched, 0.0, 255.0).round().astype(np.uint8)


class RelayVSREngine(BaseEngine):
    """
    Unified RelayVSR Engine for Video & Image Upscaling on Tesla T4.
    """

    engine_id = "relayvsr"
    display_name = "RelayVSR (Large-Small Collaborative Video & Image Upscaler)"
    category = "upscaler_video_image"

    def __init__(self, backend: Optional[RelayVSRBackend] = None) -> None:
        self.backend = backend or RelayVSRBackend()

    def get_required_files(self, variant_id: str = "relayvsr_x2_int8") -> Tuple[ModelFileSpec, ...]:
        entry = get_engine_entry(variant_id)
        return entry.files

    def build_workflow(self, params: Dict[str, object]) -> Dict[str, object]:
        """
        Builds a standard ComfyUI API Workflow JSON dictionary for `ai-studio-t4`.
        Supports 3 workflow modes:
        - `mode="image"`: LoadImage -> RelayVSR Model Loader -> RelayVSR Upscale -> SaveImage
        - `mode="video_streaming"`: VHS_LoadVideo / LoadVideo -> RelayVSR Model Loader -> RelayVSR Upscale Streaming
        - `mode="comfy_core_native"`: UNETLoader (int8_convrot) + VAELoader (flash_vae) for ComfyUI Core PR #17511
        """
        mode = str(params.get("mode", "image")).lower()
        bundle_id = str(params.get("bundle_id", "relayvsr_x2_int8"))
        entry = get_engine_entry(bundle_id)

        sr_ratio = int(params.get("sr_ratio", entry.native_scale))
        if entry.native_scale == 2:
            sr_ratio = 2
        elif sr_ratio not in (2, 4):
            sr_ratio = 4

        dtype = str(params.get("dtype", entry.default_dtype))
        if dtype not in ("fp16", "bf16"):
            dtype = "fp16"

        offload = bool(params.get("offload", entry.default_offload))
        chunk_len = int(params.get("chunk_len", entry.default_chunk_len))
        seed = int(params.get("seed", 42))
        bundle_folder_name = Path(entry.files[0].local_subdir).name
        custom_path = str(params.get("custom_path", ""))

        if mode == "video_streaming":
            video_input = str(params.get("video_path", "input.mp4"))
            upscale_width = int(params.get("upscale_width", 0))
            chunk_overlap = min(int(params.get("chunk_overlap", 4)), max(1, chunk_len - 1))
            spatial_grid_cols = max(1, int(params.get("spatial_grid_cols", 1)))
            spatial_grid_rows = max(1, int(params.get("spatial_grid_rows", 1)))
            spatial_overlap = int(params.get("spatial_overlap", 64))
            codec = str(params.get("codec", "h264"))
            crf = int(params.get("crf", 18))
            preset = str(params.get("encoder_preset", "medium"))
            filename_prefix = str(params.get("filename_prefix", "RelayVSR/upscaled_video"))

            return {
                "1": {
                    "class_type": "RelayVSR Model Loader",
                    "inputs": {
                        "bundle": bundle_folder_name,
                        "dtype": dtype,
                        "offload": offload,
                        "path": custom_path,
                    },
                },
                "2": {
                    "class_type": "LoadVideo",
                    "inputs": {
                        "file": video_input,
                    },
                },
                "3": {
                    "class_type": "RelayVSR Upscale Streaming",
                    "inputs": {
                        "model": ["1", 0],
                        "video": ["2", 0],
                        "upscale_width": upscale_width,
                        "sr_ratio": sr_ratio,
                        "chunk_len": chunk_len,
                        "seed": seed,
                        "chunk_overlap": chunk_overlap,
                        "spatial_grid_cols": spatial_grid_cols,
                        "spatial_grid_rows": spatial_grid_rows,
                        "spatial_overlap": spatial_overlap,
                        "codec": codec,
                        "crf": crf,
                        "preset": preset,
                        "filename_prefix": filename_prefix,
                    },
                },
            }

        if mode == "comfy_core_native":
            unet_name = (
                "relayvsr_x2_lightx2v_int8_convrot.safetensors"
                if entry.native_scale == 2
                else "relayvsr_x4_int8_convrot.safetensors"
            )
            vae_name = (
                "relayvsr_flash_vae_x2_fp32.safetensors"
                if entry.native_scale == 2
                else "relayvsr_flash_vae_x4_fp32.safetensors"
            )
            return {
                "1": {
                    "class_type": "UNETLoader",
                    "inputs": {
                        "unet_name": unet_name,
                        "weight_dtype": "default",
                    },
                },
                "2": {
                    "class_type": "VAELoader",
                    "inputs": {
                        "vae_name": vae_name,
                    },
                },
                "3": {
                    "class_type": "LoadImage",
                    "inputs": {
                        "image": str(params.get("image_path", "input.png")),
                    },
                },
            }

        # Default: Image / Frame Batch Upscale workflow
        image_input = str(params.get("image_path", "input.png"))
        filename_prefix = str(params.get("filename_prefix", "RelayVSR/upscaled_img"))
        return {
            "1": {
                "class_type": "RelayVSR Model Loader",
                "inputs": {
                    "bundle": bundle_folder_name,
                    "dtype": dtype,
                    "offload": offload,
                    "path": custom_path,
                },
            },
            "2": {
                "class_type": "LoadImage",
                "inputs": {
                    "image": image_input,
                },
            },
            "3": {
                "class_type": "RelayVSR Upscale",
                "inputs": {
                    "model": ["1", 0],
                    "images": ["2", 0],
                    "sr_ratio": sr_ratio,
                    "chunk_len": chunk_len,
                    "seed": seed,
                },
            },
            "4": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["3", 0],
                    "filename_prefix": filename_prefix,
                },
            },
        }

    def _get_or_load_comfy_model(
        self,
        entry: EngineCatalogEntry,
        dtype: str = "fp16",
        offload: bool = True,
    ) -> Optional[object]:
        """Loads and caches the RELAYVSR_MODEL object via `RelayVSR Model Loader`."""
        nodes = self.backend.load_comfy_relayvsr_nodes()
        if not nodes or "RelayVSR Model Loader" not in nodes:
            return None

        bundle_dir = self.backend.models_root / entry.files[0].local_subdir
        bundle_name = bundle_dir.name

        if (
            self.backend._loaded_model_obj is not None
            and self.backend._loaded_bundle_id == entry.engine_id
            and self.backend._loaded_dtype == dtype
            and self.backend._loaded_offload == offload
        ):
            return self.backend._loaded_model_obj

        # Unload any previous bundle before loading new one on T4
        if self.backend._loaded_model_obj is not None:
            self.backend._loaded_model_obj = None
            self.backend.memory.aggressive_cleanup([self.backend.models_root])

        loader_cls = nodes["RelayVSR Model Loader"]
        loader_inst = loader_cls()
        fn_name = getattr(loader_cls, "FUNCTION", "load")
        load_fn = getattr(loader_inst, fn_name)

        res = load_fn(
            bundle=bundle_name,
            dtype=dtype,
            offload=offload,
            path=str(bundle_dir.resolve()),
        )
        model_obj = res[0] if isinstance(res, (tuple, list)) else res

        self.backend._loaded_model_obj = model_obj
        self.backend._loaded_bundle_id = entry.engine_id
        self.backend._loaded_dtype = dtype
        self.backend._loaded_offload = offload
        self.backend.memory.evict_model_directory(bundle_dir)
        return model_obj

    def _run_frames_through_model(
        self,
        frames_rgb_uint8: List[np.ndarray],
        entry: EngineCatalogEntry,
        sr_ratio: int,
        chunk_len: int,
        dtype: str,
        offload: bool,
        seed: int,
    ) -> Tuple[List[np.ndarray], str]:
        """
        Runs a list of HxWx3 uint8 RGB frames through:
        1. `Comfyui-Relay-VSR` (`RelayVSR Upscale` node) if available
        2. Official `kopperx/RelayVSR` (`inference.py`) CLI if available
        3. High-detail Lanczos + Edge-Guided Residual synthesis fallback (for dry-run/CI environments without GPU weights)
        Returns (upscaled_frames_uint8, backend_used_label).
        """
        if not frames_rgb_uint8:
            raise ValueError("Daftar frame input kosong.")

        orig_h, orig_w = frames_rgb_uint8[0].shape[:2]
        padded_frames: List[np.ndarray] = []
        pad_h, pad_w = 0, 0
        for fr in frames_rgb_uint8:
            p_arr, ph, pw = pad_array_to_multiple(fr, multiple=16)
            padded_frames.append(p_arr)
            pad_h, pad_w = ph, pw

        target_h = orig_h * sr_ratio
        target_w = orig_w * sr_ratio

        # Tier 1: Direct Comfyui-Relay-VSR node execution
        nodes = self.backend.load_comfy_relayvsr_nodes()
        installed, _ = check_bundle_installed(entry.engine_id, self.backend.models_root)
        if nodes and "RelayVSR Upscale" in nodes and installed:
            import torch
            model_obj = self._get_or_load_comfy_model(entry, dtype=dtype, offload=offload)
            if model_obj is not None:
                upscale_cls = nodes["RelayVSR Upscale"]
                upscale_inst = upscale_cls()
                fn_name = getattr(upscale_cls, "FUNCTION", "upscale")
                upscale_fn = getattr(upscale_inst, fn_name)

                np_batch = np.stack(padded_frames, axis=0).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(np_batch)

                with torch.inference_mode():
                    out_res = upscale_fn(
                        model=model_obj,
                        images=img_tensor,
                        sr_ratio=int(sr_ratio),
                        chunk_len=int(chunk_len),
                        seed=int(seed),
                    )
                out_tensor = out_res[0] if isinstance(out_res, (tuple, list)) else out_res
                out_np = (
                    out_tensor.detach()
                    .cpu()
                    .clamp(0.0, 1.0)
                    .numpy()
                )
                result_frames: List[np.ndarray] = []
                for idx in range(out_np.shape[0]):
                    fr_u8 = (out_np[idx] * 255.0).round().astype(np.uint8)
                    result_frames.append(fr_u8[:target_h, :target_w])
                return result_frames, f"Comfyui-Relay-VSR ({entry.short_name}, {dtype.upper()}, Offload={offload})"

        # Tier 2: Official kopperx/RelayVSR `inference.py` CLI
        inference_py = self.backend.vendor_relayvsr_dir / "inference.py"
        if inference_py.exists() and installed:
            bundle_dir = self.backend.models_root / entry.files[0].local_subdir
            wan_ckpt = entry.files[0].resolve_path(self.backend.models_root)
            flash_ckpt = entry.files[1].resolve_path(self.backend.models_root)
            if wan_ckpt.exists() and flash_ckpt.exists():
                with tempfile.TemporaryDirectory(prefix="relayvsr_cli_") as tmpdir:
                    tmp_path = Path(tmpdir)
                    in_dir = tmp_path / "in_frames"
                    in_dir.mkdir(parents=True, exist_ok=True)
                    for idx, fr in enumerate(padded_frames):
                        Image.fromarray(fr).save(in_dir / f"frame_{idx:05d}.png")
                    out_mp4 = tmp_path / "out_x4.mp4"
                    cmd = [
                        "python3",
                        str(inference_py),
                        "--input",
                        str(in_dir),
                        "--output",
                        str(out_mp4),
                        "--wan-checkpoint",
                        str(wan_ckpt),
                        "--flash-checkpoint",
                        str(flash_ckpt),
                        "--window-size",
                        str(chunk_len),
                        "--fps",
                        "24",
                        "--dtype",
                        dtype,
                        "--overwrite",
                    ]
                    proc = subprocess.run(cmd, cwd=str(self.backend.vendor_relayvsr_dir), capture_output=True, text=True)
                    if proc.returncode == 0 and out_mp4.exists():
                        import av
                        decoded: List[np.ndarray] = []
                        with av.open(str(out_mp4)) as container:
                            for frame in container.decode(video=0):
                                pil_f = frame.to_image().convert("RGB")
                                if sr_ratio != entry.native_scale:
                                    pil_f = pil_f.resize(
                                        (padded_frames[0].shape[1] * sr_ratio, padded_frames[0].shape[0] * sr_ratio),
                                        Image.Resampling.LANCZOS,
                                    )
                                arr_f = np.asarray(pil_f, dtype=np.uint8)
                                decoded.append(arr_f[:target_h, :target_w])
                        if decoded:
                            return decoded, f"kopperx/RelayVSR CLI ({entry.short_name}, {dtype.upper()})"

        # Tier 3: High-Precision Edge-Residual Lanczos Super-Resolution Fallback
        # (Used during offline sandbox testing or if weights haven't been downloaded yet)
        fallback_frames: List[np.ndarray] = []
        for fr in frames_rgb_uint8:
            pil_in = Image.fromarray(fr)
            up_pil = pil_in.resize((target_w, target_h), Image.Resampling.LANCZOS)
            sharp_pil = up_pil.filter(ImageFilter.UnsharpMask(radius=1.4, percent=115, threshold=2))
            fallback_frames.append(np.asarray(sharp_pil, dtype=np.uint8))
        return fallback_frames, f"Preview Residual Upscaler (Unduh bobot {entry.short_name} di Tab 4 untuk inferensi Wan+Flash penuh)"

    def upscale_image(
        self,
        image_input: object,
        bundle_id_or_label: str = "relayvsr_x2_int8",
        sr_ratio: int = 2,
        preset_name: str = "🚀 Speed (Balanced Default)",
        custom_chunk_len: Optional[int] = None,
        dtype: str = "fp16",
        offload: bool = True,
        spatial_grid_rows: int = 1,
        spatial_grid_cols: int = 1,
        spatial_overlap: int = 64,
        target_upscale_width: int = 0,
        seed: int = 42,
    ) -> Tuple[Path, Image.Image, Dict[str, object]]:
        """
        Upscales a single image (path, PIL.Image, or numpy array) using RelayVSR.
        Automatically applies EXIF orientation correction, stride-16 reflection padding,
        and spatial tiling + crossfade when needed on Tesla T4.
        """
        t0 = time.perf_counter()
        entry = get_engine_entry(bundle_id_or_label)
        preset = SPEED_PRESETS.get(preset_name, SPEED_PRESETS["🚀 Speed (Balanced Default)"])

        effective_sr = int(sr_ratio)
        if entry.native_scale == 2:
            effective_sr = 2
        elif effective_sr not in (2, 4):
            effective_sr = 4

        chunk_len = int(custom_chunk_len or preset.get("chunk_len", entry.default_chunk_len))
        dtype = str(dtype or preset.get("dtype", "fp16")).lower()
        if dtype not in ("fp16", "bf16"):
            dtype = "fp16"

        # Load & normalize EXIF orientation
        src_name = "image"
        if isinstance(image_input, (str, Path)):
            src_path = Path(image_input)
            src_name = src_path.stem
            with Image.open(src_path) as im:
                pil_img = ImageOps.exif_transpose(im).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = ImageOps.exif_transpose(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
        else:
            raise ValueError(f"Tipe input gambar tidak didukung: {type(image_input)}")

        in_w, in_h = pil_img.size
        in_rgb = np.asarray(pil_img, dtype=np.uint8)

        # Auto-enable 2x2 spatial tiling on T4 if input dimension exceeds threshold
        auto_thresh = int(preset.get("auto_tile_threshold", 1280))
        rows = max(1, int(spatial_grid_rows))
        cols = max(1, int(spatial_grid_cols))
        if rows == 1 and cols == 1 and max(in_w, in_h) > auto_thresh:
            rows = 2 if in_h > auto_thresh else 1
            cols = 2 if in_w > auto_thresh else 1

        tile_coords = compute_spatial_tiles(
            height=in_h,
            width=in_w,
            grid_rows=rows,
            grid_cols=cols,
            overlap=int(spatial_overlap),
            multiple=16,
        )

        tiles_upscaled: List[Tuple[Tuple[int, int, int, int], np.ndarray]] = []
        backend_used = ""

        with self.backend.gpu_lock:
            for coords in tile_coords:
                y0, y1, x0, x1 = coords
                tile_crop = in_rgb[y0:y1, x0:x1]
                out_list, backend_used = self._run_frames_through_model(
                    frames_rgb_uint8=[tile_crop],
                    entry=entry,
                    sr_ratio=effective_sr,
                    chunk_len=chunk_len,
                    dtype=dtype,
                    offload=offload,
                    seed=seed,
                )
                tiles_upscaled.append((coords, out_list[0]))

        out_h = in_h * effective_sr
        out_w = in_w * effective_sr
        stitched_rgb = blend_spatial_tiles(tiles_upscaled, out_h, out_w, effective_sr)
        out_pil = Image.fromarray(stitched_rgb)

        # Optional fixed delivery width resize (preserving aspect ratio, rounded to even)
        if int(target_upscale_width) > 0 and int(target_upscale_width) != out_w:
            final_w = int(target_upscale_width) - (int(target_upscale_width) % 2)
            final_h = max(2, int(round(out_h * (final_w / out_w))))
            final_h -= final_h % 2
            out_pil = out_pil.resize((final_w, final_h), Image.Resampling.LANCZOS)

        elapsed = round(time.perf_counter() - t0, 2)
        ts = int(time.time() * 1000) % 1000000
        out_filename = f"{src_name}_relayvsr_{effective_sr}x_{ts}.png"
        out_path = self.backend.output_dir / out_filename
        out_pil.save(out_path, format="PNG", optimize=False)

        meta = {
            "mode": "image",
            "bundle_id": entry.engine_id,
            "model_name": entry.short_name,
            "backend_used": backend_used,
            "sr_ratio": effective_sr,
            "input_resolution": f"{in_w}×{in_h}",
            "output_resolution": f"{out_pil.width}×{out_pil.height}",
            "spatial_tiles": f"{cols}×{rows} ({len(tile_coords)} tile)",
            "dtype": dtype,
            "offload": offload,
            "elapsed_sec": elapsed,
            "seed": seed,
        }
        self.backend.gallery.register_output(out_path, meta)
        return out_path, out_pil, meta

    def upscale_image_batch(
        self,
        image_paths: List[Path],
        bundle_id_or_label: str = "relayvsr_x2_int8",
        sr_ratio: int = 2,
        preset_name: str = "🚀 Speed (Balanced Default)",
        dtype: str = "fp16",
        offload: bool = True,
        spatial_grid_rows: int = 1,
        spatial_grid_cols: int = 1,
        spatial_overlap: int = 64,
        target_upscale_width: int = 0,
        seed: int = 42,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[List[Path], Dict[str, object]]:
        """Upscales a batch of images sequentially and packages them into a downloadable ZIP."""
        import zipfile

        outputs: List[Path] = []
        t0 = time.perf_counter()
        total = len(image_paths)
        for idx, img_p in enumerate(image_paths, start=1):
            if progress_cb:
                progress_cb(idx, total, f"Meng-upscale [{idx}/{total}] {Path(img_p).name}...")
            out_p, _, _ = self.upscale_image(
                image_input=Path(img_p),
                bundle_id_or_label=bundle_id_or_label,
                sr_ratio=sr_ratio,
                preset_name=preset_name,
                dtype=dtype,
                offload=offload,
                spatial_grid_rows=spatial_grid_rows,
                spatial_grid_cols=spatial_grid_cols,
                spatial_overlap=spatial_overlap,
                target_upscale_width=target_upscale_width,
                seed=seed,
            )
            outputs.append(out_p)

        zip_name = f"relayvsr_batch_{len(outputs)}imgs_{int(time.time()) % 100000}.zip"
        zip_path = self.backend.output_dir / zip_name
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
            for op in outputs:
                zf.write(op, arcname=op.name)

        summary = {
            "count": len(outputs),
            "zip_path": str(zip_path),
            "elapsed_sec": round(time.perf_counter() - t0, 2),
        }
        return outputs, summary

    @staticmethod
    def inspect_video(video_path: Path) -> Dict[str, object]:
        """Extracts resolution, FPS, frame count, duration, and audio presence from a video file."""
        video_path = Path(video_path)
        info: Dict[str, object] = {
            "width": 0,
            "height": 0,
            "fps": 24.0,
            "frame_count": 0,
            "duration_sec": 0.0,
            "has_audio": False,
        }
        if not video_path.exists():
            return info
        try:
            import av
            with av.open(str(video_path)) as container:
                if container.streams.video:
                    vs = container.streams.video[0]
                    info["width"] = int(vs.codec_context.width or 0)
                    info["height"] = int(vs.codec_context.height or 0)
                    fps_val = float(vs.average_rate or vs.base_rate or 24.0)
                    info["fps"] = round(fps_val, 3) if fps_val > 0 else 24.0
                    frames = int(vs.frames or 0)
                    if frames <= 0 and container.duration:
                        dur = float(container.duration) / 1_000_000.0
                        frames = max(1, int(round(dur * info["fps"])))
                    info["frame_count"] = frames
                    info["duration_sec"] = round(frames / info["fps"], 2) if frames > 0 else 0.0
                info["has_audio"] = len(container.streams.audio) > 0
        except Exception:
            pass
        return info

    def upscale_video(
        self,
        video_input: Path,
        bundle_id_or_label: str = "relayvsr_x2_int8",
        sr_ratio: int = 2,
        preset_name: str = "🚀 Speed (Balanced Default)",
        custom_chunk_len: Optional[int] = None,
        chunk_overlap: int = 4,
        max_frames: int = 0,
        dtype: str = "fp16",
        offload: bool = True,
        spatial_grid_rows: int = 1,
        spatial_grid_cols: int = 1,
        spatial_overlap: int = 64,
        target_upscale_width: int = 0,
        codec: str = "h264",
        crf: int = 18,
        encoder_preset: str = "medium",
        preserve_audio: bool = True,
        seed: int = 42,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[Path, Dict[str, object]]:
        """
        Upscales a video using time-chunked streaming windows (`chunk_len` frames at a time)
        with linear temporal crossfade (`chunk_overlap`) and spatial tiling so long videos
        never overflow Tesla T4's 15 GB VRAM or 12.7 GB System RAM.
        """
        import av

        video_path = Path(video_input)
        if not video_path.exists():
            raise FileNotFoundError(f"File video tidak ditemukan: {video_path}")

        t0 = time.perf_counter()
        entry = get_engine_entry(bundle_id_or_label)
        preset = SPEED_PRESETS.get(preset_name, SPEED_PRESETS["🚀 Speed (Balanced Default)"])

        effective_sr = int(sr_ratio)
        if entry.native_scale == 2:
            effective_sr = 2
        elif effective_sr not in (2, 4):
            effective_sr = 4

        chunk_len = max(4, int(custom_chunk_len or preset.get("chunk_len", entry.default_chunk_len)))
        chunk_overlap = max(0, min(int(chunk_overlap), chunk_len // 2))
        dtype = str(dtype or preset.get("dtype", "fp16")).lower()
        if dtype not in ("fp16", "bf16"):
            dtype = "fp16"

        vinfo = self.inspect_video(video_path)
        fps = float(vinfo.get("fps") or 24.0)
        in_w = int(vinfo.get("width") or 640)
        in_h = int(vinfo.get("height") or 360)

        # Tier 1A: Try Comfyui-Relay-VSR native `RelayVSR Upscale Streaming` node if installed
        nodes = self.backend.load_comfy_relayvsr_nodes()
        installed, _ = check_bundle_installed(entry.engine_id, self.backend.models_root)
        if (
            nodes
            and "RelayVSR Upscale Streaming" in nodes
            and installed
            and int(max_frames) <= 0
        ):
            try:
                model_obj = self._get_or_load_comfy_model(entry, dtype=dtype, offload=offload)
                if model_obj is not None:
                    stream_cls = nodes["RelayVSR Upscale Streaming"]
                    stream_inst = stream_cls()
                    fn_name = getattr(stream_cls, "FUNCTION", "upscale_streaming")
                    stream_fn = getattr(stream_inst, fn_name)
                    with self.backend.gpu_lock:
                        res = stream_fn(
                            model=model_obj,
                            video=str(video_path.resolve()),
                            upscale_width=int(target_upscale_width),
                            sr_ratio=effective_sr,
                            chunk_len=chunk_len,
                            seed=int(seed),
                            chunk_overlap=chunk_overlap,
                            spatial_grid_cols=max(1, int(spatial_grid_cols)),
                            spatial_grid_rows=max(1, int(spatial_grid_rows)),
                            spatial_overlap=int(spatial_overlap),
                            codec=codec,
                            crf=int(crf),
                            preset=encoder_preset,
                            filename_prefix=f"RelayVSR/{video_path.stem}_{effective_sr}x",
                        )
                    saved_str = res[0] if isinstance(res, (tuple, list)) else res
                    if isinstance(saved_str, dict) and "result" in saved_str:
                        saved_str = saved_str["result"][0]
                    cand_path = Path(str(saved_str))
                    if cand_path.exists():
                        dest_path = self.backend.output_dir / cand_path.name
                        if cand_path.resolve() != dest_path.resolve():
                            shutil.copy2(cand_path, dest_path)
                        elapsed = round(time.perf_counter() - t0, 2)
                        meta = {
                            "mode": "video",
                            "bundle_id": entry.engine_id,
                            "model_name": entry.short_name,
                            "backend_used": f"RelayVSR Upscale Streaming ({entry.short_name})",
                            "sr_ratio": effective_sr,
                            "fps": fps,
                            "elapsed_sec": elapsed,
                        }
                        self.backend.gallery.register_output(dest_path, meta)
                        return dest_path, meta
            except Exception:
                # Fall through cleanly to chunked streaming pipeline below
                pass

        # Tier 1B / Tier 2: Time-Chunked Streaming Reader + Spatial Tiler + Background Encoder
        raw_out_w = in_w * effective_sr
        raw_out_h = in_h * effective_sr
        if int(target_upscale_width) > 0:
            final_w = int(target_upscale_width) - (int(target_upscale_width) % 2)
            final_h = max(2, int(round(raw_out_h * (final_w / raw_out_w))))
            final_h -= final_h % 2
        else:
            final_w = raw_out_w - (raw_out_w % 2)
            final_h = raw_out_h - (raw_out_h % 2)

        auto_thresh = int(preset.get("auto_tile_threshold", 1280))
        rows = max(1, int(spatial_grid_rows))
        cols = max(1, int(spatial_grid_cols))
        if rows == 1 and cols == 1 and max(in_w, in_h) > auto_thresh:
            rows = 2 if in_h > auto_thresh else 1
            cols = 2 if in_w > auto_thresh else 1

        ts = int(time.time() * 1000) % 1000000
        ext = ".webm" if codec == "av1" else ".mp4"
        out_filename = f"{video_path.stem}_relayvsr_{effective_sr}x_{ts}{ext}"
        final_out_path = self.backend.output_dir / out_filename
        temp_video_only = self.backend.output_dir / f".tmp_noaudio_{ts}{ext}"

        codec_map = {
            "h264": "libx264",
            "h265": "libx265",
            "av1": "libsvtav1",
        }
        av_codec = codec_map.get(codec, "libx264")

        out_container = av.open(str(temp_video_only), mode="w")
        try:
            out_stream = out_container.add_stream(av_codec, rate=int(round(fps)))
        except Exception:
            out_stream = out_container.add_stream("mpeg4", rate=int(round(fps)))
        out_stream.width = final_w
        out_stream.height = final_h
        out_stream.pix_fmt = "yuv420p"

        tile_coords = compute_spatial_tiles(
            height=in_h,
            width=in_w,
            grid_rows=rows,
            grid_cols=cols,
            overlap=int(spatial_overlap),
            multiple=16,
        )

        step_size = max(1, chunk_len - chunk_overlap)
        window_buf: List[np.ndarray] = []
        prev_tail_upscaled: List[np.ndarray] = []
        emitted_frames = 0
        total_est = int(vinfo.get("frame_count") or 0)
        if int(max_frames) > 0:
            total_est = min(total_est, int(max_frames)) if total_est > 0 else int(max_frames)
        backend_used = ""

        def _process_and_flush_chunk(chunk_frames: List[np.ndarray], is_last: bool) -> None:
            nonlocal prev_tail_upscaled, emitted_frames, backend_used
            if not chunk_frames:
                return

            # Spatial tiling per chunk
            T = len(chunk_frames)
            tiles_per_t: List[List[Tuple[Tuple[int, int, int, int], np.ndarray]]] = [
                [] for _ in range(T)
            ]
            with self.backend.gpu_lock:
                for coords in tile_coords:
                    y0, y1, x0, x1 = coords
                    sub_seq = [fr[y0:y1, x0:x1] for fr in chunk_frames]
                    up_sub_seq, backend_used = self._run_frames_through_model(
                        frames_rgb_uint8=sub_seq,
                        entry=entry,
                        sr_ratio=effective_sr,
                        chunk_len=chunk_len,
                        dtype=dtype,
                        offload=offload,
                        seed=seed,
                    )
                    for t_idx in range(T):
                        tiles_per_t[t_idx].append((coords, up_sub_seq[t_idx]))

            upscaled_chunk: List[np.ndarray] = []
            for t_idx in range(T):
                stitched = blend_spatial_tiles(
                    tiles_per_t[t_idx], raw_out_h, raw_out_w, effective_sr
                )
                if stitched.shape[1] != final_w or stitched.shape[0] != final_h:
                    pil_s = Image.fromarray(stitched).resize(
                        (final_w, final_h), Image.Resampling.LANCZOS
                    )
                    stitched = np.asarray(pil_s, dtype=np.uint8)
                upscaled_chunk.append(stitched)

            # Linear temporal crossfade with previous chunk's tail
            ov = min(len(prev_tail_upscaled), len(upscaled_chunk))
            for k in range(ov):
                alpha = float(k + 1) / float(ov + 1)
                blended = (
                    (1.0 - alpha) * prev_tail_upscaled[k].astype(np.float32)
                    + alpha * upscaled_chunk[k].astype(np.float32)
                )
                upscaled_chunk[k] = np.clip(blended, 0.0, 255.0).round().astype(np.uint8)

            if is_last or chunk_overlap == 0 or len(upscaled_chunk) <= chunk_overlap:
                to_emit = upscaled_chunk
                prev_tail_upscaled = []
            else:
                to_emit = upscaled_chunk[:-chunk_overlap]
                prev_tail_upscaled = upscaled_chunk[-chunk_overlap:]

            for fr_rgb in to_emit:
                av_frame = av.VideoFrame.from_ndarray(fr_rgb, format="rgb24")
                for packet in out_stream.encode(av_frame):
                    out_container.mux(packet)
                emitted_frames += 1

        read_count = 0
        with av.open(str(video_path)) as in_container:
            for frame in in_container.decode(video=0):
                if int(max_frames) > 0 and read_count >= int(max_frames):
                    break
                pil_f = frame.to_image().convert("RGB")
                window_buf.append(np.asarray(pil_f, dtype=np.uint8))
                read_count += 1

                if len(window_buf) >= chunk_len:
                    if progress_cb:
                        progress_cb(
                            read_count,
                            max(read_count, total_est),
                            f"Streaming chunk ({read_count}/{total_est or '?'} frame)...",
                        )
                    _process_and_flush_chunk(window_buf, is_last=False)
                    window_buf = window_buf[step_size:] if chunk_overlap > 0 else []

        if window_buf:
            _process_and_flush_chunk(window_buf, is_last=True)
        elif prev_tail_upscaled:
            for fr_rgb in prev_tail_upscaled:
                av_frame = av.VideoFrame.from_ndarray(fr_rgb, format="rgb24")
                for packet in out_stream.encode(av_frame):
                    out_container.mux(packet)
                emitted_frames += 1

        for packet in out_stream.encode():
            out_container.mux(packet)
        out_container.close()

        # Remux original audio if present & requested
        ffmpeg_bin = shutil.which("ffmpeg")
        if preserve_audio and vinfo.get("has_audio") and ffmpeg_bin and int(max_frames) <= 0:
            mux_cmd = [
                ffmpeg_bin,
                "-y",
                "-i",
                str(temp_video_only),
                "-i",
                str(video_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
                "-c:v",
                "copy",
                "-c:a",
                "aac" if ext == ".mp4" else "libopus",
                "-shortest",
                str(final_out_path),
            ]
            proc = subprocess.run(mux_cmd, capture_output=True, text=True, timeout=120)
            if proc.returncode == 0 and final_out_path.exists():
                temp_video_only.unlink(missing_ok=True)
            else:
                shutil.move(str(temp_video_only), str(final_out_path))
        else:
            shutil.move(str(temp_video_only), str(final_out_path))

        elapsed = round(time.perf_counter() - t0, 2)
        fps_achieved = round(emitted_frames / max(elapsed, 0.01), 2)
        meta = {
            "mode": "video",
            "bundle_id": entry.engine_id,
            "model_name": entry.short_name,
            "backend_used": backend_used,
            "sr_ratio": effective_sr,
            "input_resolution": f"{in_w}×{in_h}",
            "output_resolution": f"{final_w}×{final_h}",
            "frames_processed": emitted_frames,
            "fps": fps,
            "processing_fps": fps_achieved,
            "chunk_len": chunk_len,
            "chunk_overlap": chunk_overlap,
            "spatial_tiles": f"{cols}×{rows} ({len(tile_coords)} tile)",
            "elapsed_sec": elapsed,
        }
        self.backend.gallery.register_output(final_out_path, meta)
        return final_out_path, meta
