#!/usr/bin/env python3
"""
Bootstrap & Model Downloader for RelayVSR T4 Studio (`bootstrap.py`).
CLI arguments are 100% compatible with `ai-studio-t4/bootstrap.py`.
Includes self-healing ComfyUI git clone when `/content/ComfyUI` already exists.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

STUDIO_ROOT = Path(__file__).resolve().parent
if str(STUDIO_ROOT) not in sys.path:
    sys.path.insert(0, str(STUDIO_ROOT))

from config.catalog import RELAYVSR_BUNDLES
from core.backend import RelayVSRBackend


def run_cmd(cmd: list, cwd: Path = STUDIO_ROOT, check: bool = False) -> int:
    print(f"🔧 Menjalankan: {' '.join(str(c) for c in cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd))
    if check and proc.returncode != 0:
        raise RuntimeError(f"Perintah gagal (exit {proc.returncode}): {' '.join(str(c) for c in cmd)}")
    return proc.returncode


def ensure_python_deps() -> None:
    """Installs missing lightweight packages without touching Colab's pre-installed PyTorch/CUDA."""
    required_pkgs = []
    for mod_name, pip_spec in [
        ("safetensors", "safetensors>=0.4.5"),
        ("huggingface_hub", "huggingface_hub>=0.26.0"),
        ("av", "av>=12.0.0"),
        ("gradio", "gradio>=5.0.0,<6.0.0"),
        ("psutil", "psutil>=6.0.0"),
    ]:
        try:
            __import__(mod_name)
        except ImportError:
            required_pkgs.append(pip_spec)

    if required_pkgs:
        print(f"📦 Memasang paket pendukung: {', '.join(required_pkgs)}...")
        run_cmd([sys.executable, "-m", "pip", "install", "-q", *required_pkgs])


def ensure_comfy_and_nodes(comfy_root: Path, vendor_dir: Path) -> None:
    """
    Clones ComfyUI, Comfyui-Relay-VSR custom node, and kopperx/RelayVSR CLI repo.
    Handles non-empty `/content/ComfyUI` directories gracefully (self-healing clone).
    """
    comfy_root = Path(comfy_root)
    if not (comfy_root / "main.py").exists():
        print(f"📥 Menyiapkan Comfy-Org/ComfyUI di {comfy_root}...")
        comfy_root.parent.mkdir(parents=True, exist_ok=True)

        if not comfy_root.exists() or not any(comfy_root.iterdir()):
            run_cmd(
                ["git", "clone", "--depth", "1", "https://github.com/Comfy-Org/ComfyUI.git", str(comfy_root)],
                check=True,
            )
        else:
            # Self-healing clone when /content/ComfyUI already exists with models/ or custom_nodes/
            print(f"ℹ️ Direktori {comfy_root} sudah ada (non-empty). Melakukan self-healing clone tanpa menghapus isi yang ada...")
            tmp_clone = comfy_root.parent / f"_tmp_comfyui_clone_{os.getpid()}"
            if tmp_clone.exists():
                shutil.rmtree(tmp_clone, ignore_errors=True)
            run_cmd(
                ["git", "clone", "--depth", "1", "https://github.com/Comfy-Org/ComfyUI.git", str(tmp_clone)],
                check=True,
            )
            for item in tmp_clone.iterdir():
                dest_item = comfy_root / item.name
                if not dest_item.exists():
                    shutil.move(str(item), str(dest_item))
                elif item.is_dir():
                    shutil.copytree(str(item), str(dest_item), dirs_exist_ok=True)
            shutil.rmtree(tmp_clone, ignore_errors=True)
            print(f"✅ Comfy-Org/ComfyUI berhasil dipasang di {comfy_root}.")

        req_txt = comfy_root / "requirements.txt"
        if req_txt.exists():
            run_cmd([sys.executable, "-m", "pip", "install", "-q", "-r", str(req_txt)])
        run_cmd([sys.executable, "-m", "pip", "install", "-q", "comfy-kitchen", "einops", "torchsde"])
    else:
        print(f"✅ ComfyUI sudah terpasang di {comfy_root}.")
        run_cmd([sys.executable, "-m", "pip", "install", "-q", "comfy-kitchen", "einops", "torchsde"])

    custom_nodes_dir = comfy_root / "custom_nodes"
    custom_nodes_dir.mkdir(parents=True, exist_ok=True)
    relay_node_dir = custom_nodes_dir / "Comfyui-Relay-VSR"
    if not (relay_node_dir / "nodes.py").exists():
        if relay_node_dir.exists():
            shutil.rmtree(relay_node_dir, ignore_errors=True)
        print("📥 Meng-clone custom node bbaudio-2025/Comfyui-Relay-VSR...")
        run_cmd(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/bbaudio-2025/Comfyui-Relay-VSR.git",
                str(relay_node_dir),
            ]
        )
    else:
        print("✅ Custom node Comfyui-Relay-VSR sudah terpasang.")

    if not (vendor_dir / "inference.py").exists():
        if vendor_dir.exists():
            shutil.rmtree(vendor_dir, ignore_errors=True)
        print("📥 Meng-clone repositori resmi kopperx/RelayVSR (fallback CLI)...")
        vendor_dir.parent.mkdir(parents=True, exist_ok=True)
        run_cmd(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/kopperx/RelayVSR.git",
                str(vendor_dir),
            ]
        )
    else:
        print("✅ Repositori kopperx/RelayVSR sudah tersedia.")


def main() -> int:
    parser = argparse.ArgumentParser(description="RelayVSR T4 Studio Bootstrap & Model Downloader")
    parser.add_argument(
        "--engines",
        type=str,
        default="relayvsr_x2_int8",
        help="Daftar bundle dipisahkan koma (relayvsr_x2_int8, relayvsr_x4_int8, relayvsr_x2_fp16, relayvsr_x4_fp16, all, atau none)",
    )
    parser.add_argument("--comfy-root", type=str, default="", help="Path opsional ke direktori ComfyUI")
    parser.add_argument("--hf-token", type=str, default="", help="Hugging Face Token opsional")
    parser.add_argument("--skip-git", action="store_true", help="Lewati git clone ComfyUI/custom_nodes (mode offline/CI)")
    args = parser.parse_args()

    if args.hf_token.strip():
        os.environ["HF_TOKEN"] = args.hf_token.strip()

    print("=" * 72)
    print("🚀 Memulai Bootstrap RelayVSR T4 Studio (Tesla T4 CUDA Edition)")
    print("=" * 72)

    ensure_python_deps()

    default_comfy = Path("/content/ComfyUI") if Path("/content").exists() else (STUDIO_ROOT / "ComfyUI")
    comfy_path = Path(args.comfy_root) if args.comfy_root else Path(os.environ.get("COMFYUI_ROOT", str(default_comfy)))
    vendor_dir = STUDIO_ROOT / "vendor" / "RelayVSR"

    # IMPORTANT: Clone ComfyUI BEFORE RelayVSRBackend creates /content/ComfyUI/models subdirectories
    if not args.skip_git:
        ensure_comfy_and_nodes(comfy_path, vendor_dir)

    backend = RelayVSRBackend(studio_root=STUDIO_ROOT, comfy_root=comfy_path)
    print(backend.memory.format_telemetry_markdown())

    raw_engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    if not raw_engines or raw_engines == ["none"]:
        print("ℹ️ Tidak ada bundle awal yang dipilih (--engines none). Anda bisa mengunduh kapan saja dari Tab 4 UI.")
        return 0

    if "all" in raw_engines:
        selected_bundles = list(RELAYVSR_BUNDLES.keys())
    else:
        selected_bundles = [e for e in raw_engines if e in RELAYVSR_BUNDLES]

    for bid in selected_bundles:
        ok, summary, _ = backend.download_bundle(
            bundle_id_or_label=bid,
            hf_token=args.hf_token.strip() or None,
            progress_cb=lambda m: print(f"   {m}"),
        )
        if not ok:
            print(summary)
            return 1

    print("=" * 72)
    print("✅ Bootstrap RelayVSR T4 Studio selesai dengan sukses!")
    print(backend.memory.format_telemetry_markdown())
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
