"""
Non-blocking Gallery & History Inspector (`core/gallery.py`).
Generates lightweight JPEG thumbnails (~20 KB) for both upscaled Images and Videos,
and stores sidecar JSON metadata so the UI never lags or blocks during GPU rendering.
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image


class GalleryManager:
    """Manages output images, videos, lightweight thumbnails, and sidecar metadata."""

    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
    VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv"}

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.thumbs_dir = self.output_dir / ".thumbs"
        self.meta_dir = self.output_dir / ".meta"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    def register_output(
        self,
        media_path: Path,
        metadata: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        media_path = Path(media_path)
        if not media_path.exists():
            raise FileNotFoundError(f"Output file not found: {media_path}")

        stem = media_path.stem
        thumb_path = self.thumbs_dir / f"{stem}.thumb.jpg"
        meta_path = self.meta_dir / f"{stem}.json"

        self._ensure_thumbnail(media_path, thumb_path)

        record = {
            "filename": media_path.name,
            "media_path": str(media_path.resolve()),
            "thumb_path": str(thumb_path.resolve()) if thumb_path.exists() else str(media_path.resolve()),
            "media_type": "video" if media_path.suffix.lower() in self.VIDEO_EXTS else "image",
            "size_mb": round(media_path.stat().st_size / (1024 * 1024), 2),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": time.time(),
        }
        if metadata:
            record.update(metadata)

        meta_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        return record

    def _ensure_thumbnail(self, media_path: Path, thumb_path: Path) -> None:
        if thumb_path.exists() and thumb_path.stat().st_size > 256:
            return
        ext = media_path.suffix.lower()
        try:
            if ext in self.IMAGE_EXTS:
                with Image.open(media_path) as im:
                    im_rgb = im.convert("RGB")
                    im_rgb.thumbnail((384, 384), Image.Resampling.LANCZOS)
                    im_rgb.save(thumb_path, format="JPEG", quality=82, optimize=True)
            elif ext in self.VIDEO_EXTS:
                # Try PyAV first, then ffmpeg, then fallback placeholder
                extracted = False
                try:
                    import av
                    with av.open(str(media_path)) as container:
                        for frame in container.decode(video=0):
                            pil_img = frame.to_image().convert("RGB")
                            pil_img.thumbnail((384, 384), Image.Resampling.LANCZOS)
                            pil_img.save(thumb_path, format="JPEG", quality=82, optimize=True)
                            extracted = True
                            break
                except Exception:
                    pass
                if not extracted:
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-i",
                            str(media_path),
                            "-ss",
                            "00:00:00",
                            "-vframes",
                            "1",
                            "-vf",
                            "scale=384:-1",
                            str(thumb_path),
                        ],
                        capture_output=True,
                        timeout=10,
                    )
        except Exception:
            pass

    def list_items(self, limit: int = 100) -> List[Dict[str, object]]:
        items: List[Dict[str, object]] = []
        all_files = sorted(
            [
                p
                for p in self.output_dir.iterdir()
                if p.is_file() and (p.suffix.lower() in self.IMAGE_EXTS or p.suffix.lower() in self.VIDEO_EXTS)
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for fpath in all_files[:limit]:
            stem = fpath.stem
            meta_path = self.meta_dir / f"{stem}.json"
            thumb_path = self.thumbs_dir / f"{stem}.thumb.jpg"
            if not thumb_path.exists():
                self._ensure_thumbnail(fpath, thumb_path)
            if meta_path.exists():
                try:
                    rec = json.loads(meta_path.read_text(encoding="utf-8"))
                    rec["media_path"] = str(fpath.resolve())
                    rec["thumb_path"] = str(thumb_path.resolve()) if thumb_path.exists() else str(fpath.resolve())
                    items.append(rec)
                    continue
                except Exception:
                    pass
            items.append(
                {
                    "filename": fpath.name,
                    "media_path": str(fpath.resolve()),
                    "thumb_path": str(thumb_path.resolve()) if thumb_path.exists() else str(fpath.resolve()),
                    "media_type": "video" if fpath.suffix.lower() in self.VIDEO_EXTS else "image",
                    "size_mb": round(fpath.stat().st_size / (1024 * 1024), 2),
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(fpath.stat().st_mtime)),
                }
            )
        return items

    def get_gradio_gallery_tuples(self, limit: int = 60) -> List[Tuple[str, str]]:
        """Returns (thumbnail_path, caption_label) tuples for instant Gradio Gallery rendering."""
        tuples: List[Tuple[str, str]] = []
        for item in self.list_items(limit=limit):
            badge = "🎬 VIDEO" if item.get("media_type") == "video" else "🖼️ IMAGE"
            scale = item.get("sr_ratio", "")
            scale_str = f" {scale}×" if scale else ""
            tuples.append(( str(item["thumb_path"]), f"{badge}{scale_str} | {item['filename']}" ))
        return tuples

    def delete_item(self, filename: str) -> bool:
        target = self.output_dir / Path(filename).name
        if not target.exists():
            return False
        stem = target.stem
        for p in (target, self.thumbs_dir / f"{stem}.thumb.jpg", self.meta_dir / f"{stem}.json"):
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
        return True

    def clear_all(self) -> int:
        items = self.list_items(limit=10000)
        count = 0
        for it in items:
            if self.delete_item(str(it["filename"])):
                count += 1
        return count
