#!/usr/bin/env python3
"""
Entrypoint to launch RelayVSR T4 Studio (`run_studio.py`).
Compatible with `ai-studio-t4/run_studio.py`.
"""

import argparse
import os
import sys
from pathlib import Path

STUDIO_ROOT = Path(__file__).resolve().parent
if str(STUDIO_ROOT) not in sys.path:
    sys.path.insert(0, str(STUDIO_ROOT))

from core.backend import RelayVSRBackend
from ui.fooocus_app import create_studio_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch RelayVSR T4 Gradio Studio")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", default=True)
    parser.add_argument("--no-share", action="store_false", dest="share")
    parser.add_argument("--comfy-root", type=str, default="")
    args = parser.parse_args()

    comfy_path = Path(args.comfy_root) if args.comfy_root else None
    backend = RelayVSRBackend(studio_root=STUDIO_ROOT, comfy_root=comfy_path)
    demo = create_studio_app(backend=backend)
    demo.queue(default_concurrency_limit=8)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[str(backend.output_dir.resolve())],
    )


if __name__ == "__main__":
    main()
