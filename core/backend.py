"""
Multi-Tier Backend & Model Manager for RelayVSR T4 Studio (`core/backend.py`).
Compatible with `ai-studio-t4/core/backend.py` conventions.
"""

import json
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from config.catalog import (
    ENGINE_CATALOG,
    RELAYVSR_BUNDLES,
    EngineCatalogEntry,
    ModelFileSpec,
    check_bundle_installed,
    get_engine_entry,
)
from core.gallery import GalleryManager
from core.memory_manager import StudioMemoryManager


def verify_safetensors_header(filepath: Path) -> Tuple[bool, Dict[str, object], str]:
    """
    Validates that a .safetensors file is not a truncated download, broken symlink,
    or Git-LFS pointer, and parses its JSON header metadata safely.
    """
    filepath = Path(filepath)
    if filepath.is_symlink() and not filepath.exists():
        return False, {}, f"Broken symlink terdeteksi di: {filepath} -> {os.readlink(filepath)}"
    if not filepath.exists():
        return False, {}, f"File tidak ditemukan: {filepath}"
    try:
        real_p = filepath.resolve(strict=True)
    except Exception as exc:
        return False, {}, f"Gagal me-resolve path {filepath}: {exc}"

    size = real_p.stat().st_size
    if size < 1024 * 10:
        return False, {}, f"Ukuran file terlalu kecil ({size} bytes) — kemungkinan pointer LFS belum terunduh."
    try:
        with real_p.open("rb") as f:
            header_len_bytes = f.read(8)
            if len(header_len_bytes) < 8:
                return False, {}, "Header safetensors kurang dari 8 byte."
            (header_len,) = struct.unpack("<Q", header_len_bytes)
            if header_len <= 0 or header_len > 100 * 1024 * 1024 or (8 + header_len) > size:
                return False, {}, f"Panjang header safetensors tidak valid ({header_len} bytes)."
            header_raw = f.read(header_len)
            header_json = json.loads(header_raw.decode("utf-8"))
            metadata = header_json.get("__metadata__", {}) if isinstance(header_json, dict) else {}
            return True, metadata, "Valid safetensors header"
    except Exception as exc:
        return False, {}, f"Gagal memvalidasi header safetensors: {exc}"


def ensure_relayvsr_metadata(filepath: Path, scale: int, role: str, is_int8: bool) -> bool:
    """
    Inspects the `__metadata__` of a .safetensors file.
    Note: `bbaudio-2025/Comfyui-Relay-VSR` and `Comfy-Org/RelayVSR` store model metadata
    inside `__metadata__`. This helper verifies readability without corrupting tensor offsets.
    """
    ok, meta, _ = verify_safetensors_header(filepath)
    return ok


class RelayVSRBackend:
    """
    Manages model downloads, bundle linking, INT8 ConvRot conversion, ComfyUI custom nodes,
    and thread-safe GPU inference on Tesla T4.
    """

    def __init__(
        self,
        studio_root: Optional[Path] = None,
        comfy_root: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.studio_root = Path(studio_root or Path(__file__).resolve().parent.parent)
        default_comfy = Path("/content/ComfyUI") if Path("/content").exists() else (self.studio_root / "ComfyUI")
        self.comfy_root = Path(os.environ.get("COMFYUI_ROOT", str(comfy_root or default_comfy)))
        self.models_root = self.comfy_root / "models"
        self.relayvsr_models_dir = self.models_root / "relayvsr"
        self.vendor_relayvsr_dir = self.studio_root / "vendor" / "RelayVSR"
        self.output_dir = Path(output_dir or (self.studio_root / "outputs"))

        self.models_root.mkdir(parents=True, exist_ok=True)
        self.relayvsr_models_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.memory = StudioMemoryManager(self.studio_root)
        self.gallery = GalleryManager(self.output_dir)
        self.gpu_lock = threading.Lock()

        # Cached loaded model state (singleton in VRAM/RAM)
        self._loaded_bundle_id: Optional[str] = None
        self._loaded_dtype: Optional[str] = None
        self._loaded_offload: Optional[bool] = None
        self._loaded_model_obj: Optional[object] = None
        self._comfy_nodes_cache: Optional[Dict[str, object]] = None

    def ensure_sys_paths(self) -> None:
        """Adds ComfyUI and Comfyui-Relay-VSR to sys.path for direct in-process execution."""
        candidates = [
            self.comfy_root,
            self.comfy_root / "custom_nodes" / "Comfyui-Relay-VSR",
            self.vendor_relayvsr_dir,
        ]
        for p in candidates:
            if p.exists() and str(p) not in sys.path:
                sys.path.insert(0, str(p))

    def load_comfy_relayvsr_nodes(self) -> Optional[Dict[str, object]]:
        """
        Attempts to import `RelayVSR Model Loader`, `RelayVSR Upscale`, and `RelayVSR Upscale Streaming`
        from `bbaudio-2025/Comfyui-Relay-VSR`.
        """
        if self._comfy_nodes_cache is not None:
            return self._comfy_nodes_cache

        self.ensure_sys_paths()
        custom_node_dir = self.comfy_root / "custom_nodes" / "Comfyui-Relay-VSR"
        if not custom_node_dir.exists():
            return None

        try:
            import importlib.util
            init_py = custom_node_dir / "__init__.py"
            nodes_py = custom_node_dir / "nodes.py"
            target_py = init_py if init_py.exists() else nodes_py
            if not target_py.exists():
                return None

            pkg_name = "comfyui_relay_vsr_pkg"
            spec = importlib.util.spec_from_file_location(
                pkg_name,
                str(target_py),
                submodule_search_locations=[str(custom_node_dir)],
            )
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[pkg_name] = mod
                spec.loader.exec_module(mod)
                mappings = getattr(mod, "NODE_CLASS_MAPPINGS", None)
                if isinstance(mappings, dict) and mappings:
                    self._comfy_nodes_cache = mappings
                    return mappings
        except Exception as exc:
            print(f"⚠️ Info: Gagal memuat NODE_CLASS_MAPPINGS dari Comfyui-Relay-VSR: {exc}")
        return None

    def link_or_copy(self, src: Path, dst: Path) -> None:
        """
        Creates a zero-cost hardlink or absolute symlink to the RESOLVED target file.
        Crucial fix: `hf_hub_download()` returns a relative symlink inside `snapshots/<rev>/...`
        pointing to `../../blobs/<sha256>`. On Linux, `os.link(src, dst)` does NOT dereference
        symlinks by default, which previously created a broken relative symlink at `dst`.
        Resolving `real_src = src.resolve(strict=True)` first links directly to the actual blob inode!
        """
        src = Path(src)
        dst = Path(dst)
        if not src.exists():
            raise FileNotFoundError(f"File sumber tidak ditemukan atau symlink rusak: {src}")

        real_src = src.resolve(strict=True)
        dst.parent.mkdir(parents=True, exist_ok=True)

        # Clean up any existing file or broken symlink at dst
        if dst.exists() or dst.is_symlink():
            try:
                if dst.exists() and dst.resolve(strict=True) == real_src:
                    return
            except Exception:
                pass
            dst.unlink(missing_ok=True)

        # 1. Try hardlinking the resolved blob file directly (0 extra disk space)
        try:
            os.link(real_src, dst)
        except Exception:
            # 2. Try absolute symlink to resolved blob file
            try:
                os.symlink(real_src, dst)
            except Exception:
                # 3. Fallback to physical copy
                shutil.copy2(real_src, dst)

        # Final sanity check: if dst still does not exist, force physical copy
        if not dst.exists():
            dst.unlink(missing_ok=True)
            shutil.copy2(real_src, dst)

    def sync_bundle_aliases(self, entry: EngineCatalogEntry) -> None:
        """
        Synchronizes downloaded files across:
        1. `ComfyUI/models/relayvsr/<Bundle>/` (for `bbaudio-2025/Comfyui-Relay-VSR`)
        2. `ComfyUI/models/diffusion_models/` & `ComfyUI/models/vae/` (for ComfyUI Core PR #17511)
        3. `vendor/RelayVSR/weights/` (for `kopperx/RelayVSR` CLI when 4x is available)
        """
        for fspec in entry.files:
            primary_path = fspec.resolve_path(self.models_root)
            if primary_path.is_symlink() and not primary_path.exists():
                primary_path.unlink(missing_ok=True)

            # If alias exists first, link it to primary_path
            if not primary_path.exists():
                for alias_rel in fspec.alias_paths:
                    alias_p = self.models_root / alias_rel
                    if alias_p.exists() and alias_p.stat().st_size > 1024 * 100:
                        self.link_or_copy(alias_p, primary_path)
                        break

            if primary_path.exists():
                for alias_rel in fspec.alias_paths:
                    alias_p = self.models_root / alias_rel
                    self.link_or_copy(primary_path, alias_p)

                # Also link 4x official weights to vendor/RelayVSR/weights if present
                if entry.native_scale == 4 and self.vendor_relayvsr_dir.exists():
                    vendor_weights = self.vendor_relayvsr_dir / "weights"
                    if "wan" in primary_path.name.lower() and "int8" not in primary_path.name.lower():
                        self.link_or_copy(primary_path, vendor_weights / "wan.safetensors")
                    elif "flash" in primary_path.name.lower():
                        self.link_or_copy(primary_path, vendor_weights / "flash.safetensors")

    def download_bundle(
        self,
        bundle_id_or_label: str,
        hf_token: Optional[str] = None,
        progress_cb: Optional[Callable[[str], None]] = None,
        force_redownload: bool = False,
    ) -> Tuple[bool, str, Path]:
        """Downloads a RelayVSR bundle from Hugging Face, validates headers, and evicts VFS cache."""
        entry = get_engine_entry(bundle_id_or_label)
        token = (hf_token or os.environ.get("HF_TOKEN") or "").strip() or None

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            return False, f"Paket huggingface_hub belum terpasang: {exc}", self.relayvsr_models_dir

        bundle_dir = self.models_root / entry.files[0].local_subdir
        bundle_dir.mkdir(parents=True, exist_ok=True)

        logs: List[str] = [f"📦 Memeriksa / mengunduh bundle: {entry.label}"]
        if progress_cb:
            progress_cb(logs[-1])

        for fspec in entry.files:
            dest_path = fspec.resolve_path(self.models_root)
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Remove any broken symlink from previous run before checking
            if dest_path.is_symlink() and not dest_path.exists():
                dest_path.unlink(missing_ok=True)

            if dest_path.exists() and not force_redownload:
                ok_hdr, _, msg_hdr = verify_safetensors_header(dest_path)
                if ok_hdr:
                    msg = f"✅ Sudah tersedia & terverifikasi: {dest_path.name} ({round(dest_path.stat().st_size / (1024**3), 2)} GB)"
                    logs.append(msg)
                    if progress_cb:
                        progress_cb(msg)
                    continue
                else:
                    warn_msg = f"⚠️ File {dest_path.name} tidak valid ({msg_hdr}), mengunduh ulang..."
                    logs.append(warn_msg)
                    if progress_cb:
                        progress_cb(warn_msg)
                    dest_path.unlink(missing_ok=True)

            dl_msg = f"⬇️ Mengunduh `{fspec.filename}` dari `{fspec.repo_id}` (~{fspec.size_gb} GB)..."
            logs.append(dl_msg)
            if progress_cb:
                progress_cb(dl_msg)

            try:
                downloaded_tmp = Path(
                    hf_hub_download(
                        repo_id=fspec.repo_id,
                        filename=fspec.filename,
                        token=token,
                        force_download=force_redownload,
                    )
                )
                real_blob = downloaded_tmp.resolve(strict=True)
                self.link_or_copy(real_blob, dest_path)

                # Evict VFS page cache immediately so 5-10 GB download doesn't eat Colab System RAM
                self.memory.evict_file_cache(real_blob)
                self.memory.evict_file_cache(dest_path)

                ok_hdr, _, msg_hdr = verify_safetensors_header(dest_path)
                if not ok_hdr:
                    return False, f"❌ Header file hasil unduhan rusak ({dest_path.name}): {msg_hdr}", bundle_dir

                done_msg = f"✅ Berhasil mengunduh & menautkan `{dest_path.name}` ({round(dest_path.stat().st_size / (1024**3), 2)} GB)"
                logs.append(done_msg)
                if progress_cb:
                    progress_cb(done_msg)
            except Exception as exc:
                return False, f"❌ Gagal mengunduh `{fspec.filename}` dari `{fspec.repo_id}`: {exc}", bundle_dir

        self.sync_bundle_aliases(entry)
        self.memory.aggressive_cleanup([bundle_dir])
        summary = "\n".join(logs) + f"\n🎉 Bundle `{entry.short_name}` siap digunakan di `{bundle_dir}`."
        return True, summary, bundle_dir

    def convert_bundle_to_int8_convrot(
        self,
        source_bundle_id: str = "relayvsr_x2_fp16",
    ) -> Tuple[bool, str]:
        """
        Runs `Comfyui-Relay-VSR/convert_to_int8_convrot.py` on a full-precision bundle
        if the user wants local conversion to INT8 ConvRot.
        """
        converter = self.comfy_root / "custom_nodes" / "Comfyui-Relay-VSR" / "convert_to_int8_convrot.py"
        if not converter.exists():
            return (
                False,
                f"Skrip `{converter}` belum ditemukan. Jalankan bootstrap.py atau gunakan bundle `RelayVSR 2×/4× INT8 ConvRot` yang sudah dikuantisasi resmi dari Comfy-Org/RelayVSR.",
            )

        entry = get_engine_entry(source_bundle_id)
        installed, missing = check_bundle_installed(entry.engine_id, self.models_root)
        if not installed:
            return False, f"Bundle sumber `{entry.short_name}` belum terunduh. File kurang: {[m.name for m in missing]}"

        bundle_dir = self.models_root / entry.files[0].local_subdir
        cmd = [sys.executable, str(converter), str(bundle_dir)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            self.memory.aggressive_cleanup([self.relayvsr_models_dir])
            if proc.returncode != 0:
                return False, f"Konversi INT8 ConvRot gagal (exit {proc.returncode}):\n{proc.stderr or proc.stdout}"
            return True, f"✅ Konversi INT8 ConvRot selesai:\n{proc.stdout}"
        except Exception as exc:
            return False, f"Error saat menjalankan convert_to_int8_convrot.py: {exc}"

    def unload_models(self) -> str:
        """Unloads active RelayVSR models from GPU VRAM & System RAM and flushes VFS cache."""
        with self.gpu_lock:
            self._loaded_model_obj = None
            self._loaded_bundle_id = None
            self._loaded_dtype = None
            self._loaded_offload = None
            stats = self.memory.aggressive_cleanup([self.models_root])
        return (
            f"🧹 Model RelayVSR berhasil dilepas dari memori. "
            f"VRAM: {stats['vram_used_gb']}/{stats['vram_total_gb']} GB | "
            f"RAM: {stats['ram_used_gb']}/{stats['ram_total_gb']} GB | "
            f"VFS Cache Evicted: {stats['evicted_files']} file."
        )

    def delete_bundle_from_disk(self, bundle_id_or_label: str) -> str:
        """Deletes a model bundle from disk to free up Google Colab SSD space."""
        entry = get_engine_entry(bundle_id_or_label)
        if self._loaded_bundle_id == entry.engine_id:
            self.unload_models()

        removed: List[str] = []
        for fspec in entry.files:
            p = fspec.resolve_path(self.models_root)
            if p.exists() or p.is_symlink():
                p.unlink(missing_ok=True)
                removed.append(p.name)
            for alias_rel in fspec.alias_paths:
                ap = self.models_root / alias_rel
                if ap.exists() or ap.is_symlink():
                    ap.unlink(missing_ok=True)
        stats = self.memory.aggressive_cleanup([self.models_root])
        if not removed:
            return f"ℹ️ Bundle `{entry.short_name}` memang belum ada di disk. Disk bebas: {stats['disk_free_gb']} GB."
        return f"🗑️ Berhasil menghapus {', '.join(removed)} (`{entry.short_name}`). Disk bebas sekarang: {stats['disk_free_gb']} GB."

    def get_bundles_status_markdown(self) -> str:
        lines = [
            "| Bundle Model | Skala | Presisi | Estimasi VRAM | Status di Disk |",
            "| :--- | :---: | :---: | :---: | :--- |",
        ]
        for bid, entry in RELAYVSR_BUNDLES.items():
            installed, _ = check_bundle_installed(bid, self.models_root)
            status = "✅ **Terpasang**" if installed else "⚪ *Belum Diunduh*"
            if self._loaded_bundle_id == bid:
                status = "🟢 **Aktif di GPU**"
            lines.append(
                f"| **{entry.short_name}** | `{entry.native_scale}×` | `{entry.precision}` | `~{entry.vram_estimate_gb} GB` | {status} |"
            )
        return "\n".join(lines)
