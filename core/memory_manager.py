"""
T4 VRAM, System RAM & VFS Page-Cache Manager (`core/memory_manager.py`).
Compatible with `ai-studio-t4/core/memory_manager.py`.
Uses `psutil` if available, or falls back directly to `/proc/meminfo` and `shutil.disk_usage`
for 100% dependency-free telemetry on Linux/Colab.
"""

import ctypes
import gc
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Optional

try:
    import psutil  # type: ignore
except ImportError:
    psutil = None


def _get_system_ram_gb() -> tuple[float, float, float]:
    """Returns (used_gb, total_gb, percent) using psutil or /proc/meminfo."""
    if psutil is not None:
        vm = psutil.virtual_memory()
        used = round((vm.total - vm.available) / (1024 ** 3), 2)
        tot = round(vm.total / (1024 ** 3), 2)
        return used, tot, round(float(vm.percent), 1)
    try:
        meminfo: Dict[str, int] = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val_kb = int(parts[1].strip().split()[0])
                    meminfo[key] = val_kb
        total_kb = meminfo.get("MemTotal", 0)
        avail_kb = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
        if total_kb > 0:
            used_kb = max(0, total_kb - avail_kb)
            tot_gb = round(total_kb / (1024 ** 2), 2)
            used_gb = round(used_kb / (1024 ** 2), 2)
            pct = round((used_kb / total_kb) * 100.0, 1)
            return used_gb, tot_gb, pct
    except Exception:
        pass
    return 0.0, 12.7, 0.0


class StudioMemoryManager:
    """Manages CUDA VRAM, Python GC, glibc heap, and Linux VFS page cache on Colab T4."""

    def __init__(self, root_dir: Optional[Path] = None) -> None:
        self.root_dir = Path(root_dir or Path(__file__).resolve().parent.parent)
        self.cpp_source = self.root_dir / "core" / "studio_native_mem.cpp"
        self.so_path = self.root_dir / "core" / "libstudio_mem.so"
        self._lib: Optional[ctypes.CDLL] = None
        self._ensure_native_lib()

    def _ensure_native_lib(self) -> None:
        if not self.so_path.exists() and self.cpp_source.exists():
            gpp = shutil.which("g++")
            if gpp:
                try:
                    subprocess.run(
                        [
                            gpp,
                            "-O3",
                            "-shared",
                            "-fPIC",
                            str(self.cpp_source),
                            "-o",
                            str(self.so_path),
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=25,
                    )
                except Exception:
                    pass
        if self.so_path.exists():
            try:
                self._lib = ctypes.CDLL(str(self.so_path))
                self._lib.studio_mem_init.restype = ctypes.c_int
                self._lib.studio_mem_trim.restype = ctypes.c_int
                self._lib.studio_evict_file_cache.argtypes = [ctypes.c_char_p]
                self._lib.studio_evict_file_cache.restype = ctypes.c_int
                self._lib.studio_mem_init()
            except Exception:
                self._lib = None

    def evict_file_cache(self, path: Path) -> bool:
        if self._lib is None or not path.exists():
            return False
        try:
            rc = self._lib.studio_evict_file_cache(str(path.resolve()).encode("utf-8"))
            return rc == 0
        except Exception:
            return False

    def evict_model_directory(self, directory: Path) -> int:
        if not directory.exists():
            return 0
        count = 0
        for ext in ("*.safetensors", "*.gguf", "*.pth", "*.pt", "*.bin"):
            for fpath in directory.rglob(ext):
                if fpath.is_file() and self.evict_file_cache(fpath):
                    count += 1
        return count

    def aggressive_cleanup(self, evict_dirs: Optional[Iterable[Path]] = None) -> Dict[str, object]:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

        evicted_files = 0
        if evict_dirs:
            for d in evict_dirs:
                evicted_files += self.evict_model_directory(Path(d))

        trimmed = False
        if self._lib is not None:
            try:
                trimmed = bool(self._lib.studio_mem_trim())
            except Exception:
                trimmed = False

        stats = self.get_telemetry()
        stats["evicted_files"] = evicted_files
        stats["glibc_trimmed"] = trimmed
        return stats

    def get_telemetry(self) -> Dict[str, object]:
        ram_used_gb, ram_total_gb, ram_percent = _get_system_ram_gb()
        target_disk_path = str(self.root_dir if self.root_dir.exists() else Path.cwd())
        du = shutil.disk_usage(target_disk_path)

        vram_used_gb = 0.0
        vram_total_gb = 0.0
        gpu_name = "CPU / No CUDA GPU"
        compute_cap = "N/A"

        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                gpu_name = props.name
                compute_cap = f"sm_{props.major}{props.minor}"
                vram_total_gb = round(props.total_memory / (1024 ** 3), 2)
                free_b, total_b = torch.cuda.mem_get_info(0)
                vram_used_gb = round((total_b - free_b) / (1024 ** 3), 2)
        except Exception:
            pass

        return {
            "gpu_name": gpu_name,
            "compute_capability": compute_cap,
            "vram_used_gb": vram_used_gb,
            "vram_total_gb": vram_total_gb,
            "vram_percent": round((vram_used_gb / vram_total_gb) * 100, 1) if vram_total_gb > 0 else 0.0,
            "ram_used_gb": ram_used_gb,
            "ram_total_gb": ram_total_gb,
            "ram_percent": ram_percent,
            "disk_used_gb": round(du.used / (1024 ** 3), 2),
            "disk_free_gb": round(du.free / (1024 ** 3), 2),
            "disk_total_gb": round(du.total / (1024 ** 3), 2),
            "native_cpp_active": self._lib is not None,
        }

    def format_telemetry_markdown(self) -> str:
        t = self.get_telemetry()
        cpp_badge = "🟢 C++ VFS Cleaner Aktif" if t["native_cpp_active"] else "🟡 Python Fallback"
        return (
            f"**GPU:** `{t['gpu_name']}` (`{t['compute_capability']}`) · "
            f"**VRAM:** `{t['vram_used_gb']} / {t['vram_total_gb']} GB ({t['vram_percent']}%)` · "
            f"**System RAM:** `{t['ram_used_gb']} / {t['ram_total_gb']} GB ({t['ram_percent']}%)` · "
            f"**Disk Bebas:** `{t['disk_free_gb']} GB` · {cpp_badge}"
        )
