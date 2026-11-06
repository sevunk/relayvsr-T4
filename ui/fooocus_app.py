"""
Gradio Studio Interface (`ui/fooocus_app.py`) for RelayVSR T4 Studio.
Matches `ai-studio-t4/ui/fooocus_app.py` layout and non-blocking gallery architecture.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr

from config.catalog import RELAYVSR_BUNDLES, SPEED_PRESETS, get_engine_entry
from core.backend import RelayVSRBackend
from engines.relayvsr_engine import RelayVSREngine


CUSTOM_CSS = """
.studio-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: #f8fafc;
    padding: 16px 22px;
    border-radius: 12px;
    border: 1px solid #334155;
    margin-bottom: 10px;
}
.studio-header h1 {
    margin: 0 0 6px 0;
    font-size: 1.45rem;
    font-weight: 700;
    color: #38bdf8;
}
.telemetry-bar {
    background: #0f172a;
    color: #e2e8f0;
    padding: 10px 14px;
    border-radius: 8px;
    border: 1px solid #1e293b;
    font-size: 0.9rem;
}
"""


def create_studio_app(backend: Optional[RelayVSRBackend] = None) -> gr.Blocks:
    backend = backend or RelayVSRBackend()
    engine = RelayVSREngine(backend=backend)

    bundle_choices = [entry.label for entry in RELAYVSR_BUNDLES.values()]
    default_bundle_label = RELAYVSR_BUNDLES["relayvsr_x2_int8"].label
    preset_choices = list(SPEED_PRESETS.keys())
    default_preset = "🚀 Speed (Balanced Default)"

    def _on_bundle_change(bundle_label: str):
        entry = get_engine_entry(bundle_label)
        if entry.native_scale == 2:
            return gr.update(choices=[2], value=2, interactive=False)
        return gr.update(choices=[2, 4], value=4, interactive=True)

    def _on_preset_change(preset_name: str) -> Tuple[int, int, str, bool, int, int, int]:
        p = SPEED_PRESETS.get(preset_name, SPEED_PRESETS[default_preset])
        return (
            int(p["chunk_len"]),
            int(p["chunk_overlap"]),
            str(p["dtype"]),
            bool(p["offload"]),
            int(p["spatial_grid_cols"]),
            int(p["spatial_grid_rows"]),
            int(p["spatial_overlap"]),
        )

    def _run_image_upscale(
        single_img_path: Optional[str],
        batch_files: Optional[List[object]],
        bundle_label: str,
        sr_ratio: int,
        preset_name: str,
        dtype: str,
        offload: bool,
        grid_cols: int,
        grid_rows: int,
        spatial_overlap: int,
        target_width: int,
        seed: int,
        progress: gr.Progress = gr.Progress(track_tqdm=False),
    ) -> Tuple[Optional[str], Optional[str], str, str]:
        entry = get_engine_entry(bundle_label)
        installed, _ = (
            backend.models_root / entry.files[0].local_subdir / entry.files[0].target_name
        ).exists(), None

        # Check batch first if files uploaded
        valid_batch: List[Path] = []
        if batch_files:
            for bf in batch_files:
                fpath = Path(getattr(bf, "name", str(bf)))
                if fpath.exists():
                    valid_batch.append(fpath)

        if valid_batch:
            def _prog(i: int, tot: int, msg: str) -> None:
                progress((i - 1) / max(tot, 1), desc=msg)

            out_paths, summary = engine.upscale_image_batch(
                image_paths=valid_batch,
                bundle_id_or_label=bundle_label,
                sr_ratio=int(sr_ratio),
                preset_name=preset_name,
                dtype=dtype,
                offload=bool(offload),
                spatial_grid_rows=int(grid_rows),
                spatial_grid_cols=int(grid_cols),
                spatial_overlap=int(spatial_overlap),
                target_upscale_width=int(target_width),
                seed=int(seed),
                progress_cb=_prog,
            )
            first_preview = str(out_paths[0]) if out_paths else None
            zip_file = str(summary["zip_path"])
            return (
                first_preview,
                zip_file,
                json.dumps(summary, indent=2, ensure_ascii=False),
                backend.memory.format_telemetry_markdown(),
            )

        if not single_img_path or not Path(single_img_path).exists():
            raise gr.Error("Silakan unggah 1 gambar atau beberapa file gambar terlebih dahulu.")

        progress(0.15, desc=f"Menjalankan {entry.short_name} ({sr_ratio}×)...")
        out_path, _, meta = engine.upscale_image(
            image_input=Path(single_img_path),
            bundle_id_or_label=bundle_label,
            sr_ratio=int(sr_ratio),
            preset_name=preset_name,
            dtype=dtype,
            offload=bool(offload),
            spatial_grid_rows=int(grid_rows),
            spatial_grid_cols=int(grid_cols),
            spatial_overlap=int(spatial_overlap),
            target_upscale_width=int(target_width),
            seed=int(seed),
        )
        progress(1.0, desc="Selesai!")
        return (
            str(out_path),
            str(out_path),
            json.dumps(meta, indent=2, ensure_ascii=False),
            backend.memory.format_telemetry_markdown(),
        )

    def _inspect_uploaded_video(video_path: Optional[str]) -> str:
        if not video_path or not Path(video_path).exists():
            return "*Belum ada video yang diunggah.*"
        info = engine.inspect_video(Path(video_path))
        audio_badge = "🔊 Ada Audio" if info["has_audio"] else "🔇 Tanpa Audio"
        return (
            f"🎬 **Input Video:** `{info['width']}×{info['height']}` · "
            f"`{info['fps']} FPS` · `{info['frame_count']} frame` (`~{info['duration_sec']} dtk`) · {audio_badge}"
        )

    def _run_video_upscale(
        video_path: Optional[str],
        bundle_label: str,
        sr_ratio: int,
        preset_name: str,
        chunk_len: int,
        chunk_overlap: int,
        max_frames: int,
        dtype: str,
        offload: bool,
        grid_cols: int,
        grid_rows: int,
        spatial_overlap: int,
        target_width: int,
        codec: str,
        crf: int,
        encoder_preset: str,
        preserve_audio: bool,
        seed: int,
        progress: gr.Progress = gr.Progress(track_tqdm=False),
    ) -> Tuple[Optional[str], Optional[str], str, str]:
        if not video_path or not Path(video_path).exists():
            raise gr.Error("Silakan unggah file video (.mp4 / .webm / .mov) terlebih dahulu.")

        def _vprog(cur: int, tot: int, msg: str) -> None:
            progress(min(0.98, cur / max(tot, 1)), desc=msg)

        out_video, meta = engine.upscale_video(
            video_input=Path(video_path),
            bundle_id_or_label=bundle_label,
            sr_ratio=int(sr_ratio),
            preset_name=preset_name,
            custom_chunk_len=int(chunk_len),
            chunk_overlap=int(chunk_overlap),
            max_frames=int(max_frames),
            dtype=dtype,
            offload=bool(offload),
            spatial_grid_rows=int(grid_rows),
            spatial_grid_cols=int(grid_cols),
            spatial_overlap=int(spatial_overlap),
            target_upscale_width=int(target_width),
            codec=codec,
            crf=int(crf),
            encoder_preset=encoder_preset,
            preserve_audio=bool(preserve_audio),
            seed=int(seed),
            progress_cb=_vprog,
        )
        progress(1.0, desc="Video selesai di-upscale!")
        return (
            str(out_video),
            str(out_video),
            json.dumps(meta, indent=2, ensure_ascii=False),
            backend.memory.format_telemetry_markdown(),
        )

    def _refresh_gallery() -> List[Tuple[str, str]]:
        return backend.gallery.get_gradio_gallery_tuples(limit=60)

    def _on_gallery_select(evt: gr.SelectData) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
        items = backend.gallery.list_items(limit=60)
        idx = int(evt.index) if evt and evt.index is not None else 0
        if idx < 0 or idx >= len(items):
            return None, None, None, "{}"
        rec = items[idx]
        mpath = str(rec["media_path"])
        if rec.get("media_type") == "video":
            return None, mpath, mpath, json.dumps(rec, indent=2, ensure_ascii=False)
        return mpath, None, mpath, json.dumps(rec, indent=2, ensure_ascii=False)

    def _clear_gallery() -> Tuple[List[Tuple[str, str]], str]:
        removed = backend.gallery.clear_all()
        return [], f"🗑️ Berhasil menghapus {removed} item dari galeri."

    def _download_bundle_ui(bundle_label: str, hf_token: str, force: bool) -> Tuple[str, str, str]:
        _, msg, _ = backend.download_bundle(
            bundle_id_or_label=bundle_label,
            hf_token=hf_token,
            force_redownload=bool(force),
        )
        return (
            msg,
            backend.get_bundles_status_markdown(),
            backend.memory.format_telemetry_markdown(),
        )

    def _convert_int8_ui(bundle_label: str) -> Tuple[str, str, str]:
        entry = get_engine_entry(bundle_label)
        _, msg = backend.convert_bundle_to_int8_convrot(entry.engine_id)
        return (
            msg,
            backend.get_bundles_status_markdown(),
            backend.memory.format_telemetry_markdown(),
        )

    def _unload_vram_ui() -> Tuple[str, str, str]:
        msg = backend.unload_models()
        return (
            msg,
            backend.get_bundles_status_markdown(),
            backend.memory.format_telemetry_markdown(),
        )

    def _delete_bundle_ui(bundle_label: str) -> Tuple[str, str, str]:
        msg = backend.delete_bundle_from_disk(bundle_label)
        return (
            msg,
            backend.get_bundles_status_markdown(),
            backend.memory.format_telemetry_markdown(),
        )

    def _export_workflow_ui(mode: str, bundle_label: str, sr_ratio: int, chunk_len: int) -> str:
        entry = get_engine_entry(bundle_label)
        wf = engine.build_workflow(
            {
                "mode": mode,
                "bundle_id": entry.engine_id,
                "sr_ratio": sr_ratio,
                "chunk_len": chunk_len,
                "dtype": "fp16",
                "offload": True,
            }
        )
        return json.dumps(wf, indent=2, ensure_ascii=False)

    with gr.Blocks(title="RelayVSR T4 Studio — Video & Image Super-Resolution", css=CUSTOM_CSS) as demo:
        gr.HTML(
            """
            <div class="studio-header">
                <h1>⚡ RelayVSR T4 Studio — Collaborative 2× & 4× Video & Image Super-Resolution</h1>
                <div>Arsitektur <b>Wan2.2-5B Sparse Keyframe Relay + 31.9M Dual-Memory Flash Transformer</b> ·
                Dioptimalkan untuk <b>Google Colab Tesla T4 (INT8 ConvRot + FP16 + C++ VFS Page-Cache Cleaner)</b> ·
                <i>Siap Integrasi Plug-and-Play ke <code>ai-studio-t4</code></i></div>
            </div>
            """
        )

        with gr.Row():
            telemetry_md = gr.Markdown(
                value=backend.memory.format_telemetry_markdown(),
                elem_classes=["telemetry-bar"],
            )
            btn_quick_refresh = gr.Button("🔄 Cek VRAM/RAM", scale=0, min_width=140)
            btn_quick_unload = gr.Button("🧹 Unload VRAM", scale=0, min_width=140)

        with gr.Tabs():
            # ==================== TAB 1: IMAGE UPSCALER ====================
            with gr.Tab("🖼️ Image Upscaler (Single & Batch)"):
                with gr.Row():
                    with gr.Column(scale=5):
                        img_input = gr.Image(
                            label="Unggah Gambar Tunggal (PNG / JPG / WEBP)",
                            type="filepath",
                            height=320,
                        )
                        batch_img_input = gr.File(
                            label="Atau Unggah Banyak Gambar Sekaligus (Batch Upscale → ZIP)",
                            file_count="multiple",
                            file_types=["image"],
                        )
                        img_bundle = gr.Dropdown(
                            choices=bundle_choices,
                            value=default_bundle_label,
                            label="Pilih Bundle Model RelayVSR",
                        )
                        with gr.Row():
                            img_sr_ratio = gr.Radio(
                                choices=[2, 4],
                                value=2,
                                label="Skala Output (sr_ratio)",
                                interactive=False,
                            )
                            img_preset = gr.Dropdown(
                                choices=preset_choices,
                                value=default_preset,
                                label="Preset Optimasi T4",
                            )

                        with gr.Accordion("⚙️ Pengaturan Lanjutan T4 (Spatial Tiling, Offload & Resolusi Target)", open=False):
                            with gr.Row():
                                img_dtype = gr.Radio(
                                    choices=["fp16", "bf16"],
                                    value="fp16",
                                    label="Presisi Floating-Point (Wajib fp16 di Tesla T4)",
                                )
                                img_offload = gr.Checkbox(
                                    value=True,
                                    label="Aktifkan Per-Block Wan GPU Offload (Hemat VRAM)",
                                )
                            with gr.Row():
                                img_grid_cols = gr.Slider(1, 4, value=1, step=1, label="Spatial Grid Cols")
                                img_grid_rows = gr.Slider(1, 4, value=1, step=1, label="Spatial Grid Rows")
                                img_spatial_overlap = gr.Slider(16, 128, value=64, step=16, label="Tile Crossfade Overlap (px)")
                            with gr.Row():
                                img_target_width = gr.Number(
                                    value=0,
                                    precision=0,
                                    label="Target Delivery Width (0 = Resolusi Asli × Skala)",
                                )
                                img_seed = gr.Number(value=42, precision=0, label="Seed (Noise Keyframe)")

                        btn_upscale_img = gr.Button("🚀 Upscale Gambar Sekarang", variant="primary", size="lg")

                    with gr.Column(scale=5):
                        img_output_preview = gr.Image(label="Hasil Upscale RelayVSR", type="filepath", height=420)
                        img_download_file = gr.File(label="Unduh File Hasil (PNG Resolusi Penuh / Batch ZIP)")
                        img_meta_box = gr.Code(label="Telemetri & Detail Eksekusi", language="json", lines=8)

            # ==================== TAB 2: VIDEO STREAMING UPSCALER ====================
            with gr.Tab("🎬 Video Streaming Upscaler (2× / 4×)"):
                with gr.Row():
                    with gr.Column(scale=5):
                        vid_input = gr.Video(label="Unggah Video Input (MP4 / WebM / MOV)", height=300)
                        vid_info_md = gr.Markdown("*Belum ada video yang diunggah.*")
                        vid_bundle = gr.Dropdown(
                            choices=bundle_choices,
                            value=default_bundle_label,
                            label="Pilih Bundle Model RelayVSR",
                        )
                        with gr.Row():
                            vid_sr_ratio = gr.Radio(
                                choices=[2, 4],
                                value=2,
                                label="Skala Output Video (sr_ratio)",
                                interactive=False,
                            )
                            vid_preset = gr.Dropdown(
                                choices=preset_choices,
                                value=default_preset,
                                label="Preset Streaming T4",
                            )
                        with gr.Row():
                            vid_chunk_len = gr.Slider(
                                4, 32, value=16, step=2,
                                label="Streaming Window (chunk_len / keyframe interval)",
                            )
                            vid_chunk_overlap = gr.Slider(
                                0, 8, value=4, step=1,
                                label="Temporal Crossfade Overlap (frame)",
                            )
                            vid_max_frames = gr.Number(
                                value=0,
                                precision=0,
                                label="Batas Frame Uji Coba (0 = Proses Seluruh Video)",
                            )

                        with gr.Accordion("⚙️ Pengaturan Tiling, Codec & Audio Remuxing", open=False):
                            with gr.Row():
                                vid_dtype = gr.Radio(choices=["fp16", "bf16"], value="fp16", label="Presisi (fp16 untuk T4)")
                                vid_offload = gr.Checkbox(value=True, label="Per-Block Wan GPU Offload")
                                vid_preserve_audio = gr.Checkbox(value=True, label="Pertahankan Audio Asli (Remux/AAC)")
                            with gr.Row():
                                vid_grid_cols = gr.Slider(1, 4, value=1, step=1, label="Spatial Grid Cols")
                                vid_grid_rows = gr.Slider(1, 4, value=1, step=1, label="Spatial Grid Rows")
                                vid_spatial_overlap = gr.Slider(16, 128, value=64, step=16, label="Spatial Overlap (px)")
                            with gr.Row():
                                vid_target_width = gr.Number(
                                    value=0,
                                    precision=0,
                                    label="Target Delivery Width (0=Native, 1920=1080p, 2560=1440p, 3840=4K)",
                                )
                                vid_codec = gr.Dropdown(choices=["h264", "h265", "av1"], value="h264", label="Video Codec")
                                vid_crf = gr.Slider(10, 30, value=18, step=1, label="Kualitas CRF (Lebih kecil = Lebih tajam)")
                                vid_enc_preset = gr.Dropdown(
                                    choices=["ultrafast", "fast", "medium", "slow"],
                                    value="medium",
                                    label="Preset Encoder",
                                )
                                vid_seed = gr.Number(value=42, precision=0, label="Seed")

                        btn_upscale_vid = gr.Button("🎬 Mulai Streaming Video Upscale", variant="primary", size="lg")

                    with gr.Column(scale=5):
                        vid_output_player = gr.Video(label="Pratinjau Video Hasil Upscale", height=380)
                        vid_download_file = gr.File(label="Unduh Video Hasil (.mp4 / .webm)")
                        vid_meta_box = gr.Code(label="Telemetri Streaming & Kecepatan FPS", language="json", lines=8)

            # ==================== TAB 3: NON-BLOCKING GALLERY ====================
            with gr.Tab("🗂️ Galeri & History Inspector"):
                with gr.Row():
                    btn_gal_refresh = gr.Button("🔄 Refresh Galeri", variant="secondary")
                    btn_gal_clear = gr.Button("💥 Hapus Semua History", variant="stop")
                    gal_status_md = gr.Markdown("")
                with gr.Row():
                    with gr.Column(scale=6):
                        history_gallery = gr.Gallery(
                            value=_refresh_gallery(),
                            label="Riwayat Hasil Upscale (Klik Thumbnail untuk Inspeksi Resolusi Penuh)",
                            columns=4,
                            height=480,
                            object_fit="contain",
                        )
                    with gr.Column(scale=4):
                        gal_img_view = gr.Image(label="Inspeksi Gambar Penuh", type="filepath", visible=True)
                        gal_vid_view = gr.Video(label="Putar Video Hasil", visible=True)
                        gal_download = gr.File(label="Unduh File Terpilih")
                        gal_meta_json = gr.Code(label="Metadata Parameter", language="json", lines=8)

            # ==================== TAB 4: MODELS, INT8 & AI-STUDIO-T4 EXPORT ====================
            with gr.Tab("📦 Models, INT8 ConvRot & Integrasi AI-Studio-T4"):
                bundles_table_md = gr.Markdown(value=backend.get_bundles_status_markdown())
                with gr.Row():
                    with gr.Column(scale=6):
                        mgr_bundle = gr.Dropdown(
                            choices=bundle_choices,
                            value=default_bundle_label,
                            label="Pilih Bundle untuk Diunduh / Dikelola",
                        )
                        with gr.Row():
                            mgr_hf_token = gr.Textbox(
                                label="Hugging Face Token (Opsional)",
                                placeholder="hf_...",
                                type="password",
                            )
                            mgr_force = gr.Checkbox(value=False, label="Paksa Unduh Ulang")
                        with gr.Row():
                            btn_dl_bundle = gr.Button("⬇️ Download Bundle Ini Sekarang", variant="primary")
                            btn_conv_int8 = gr.Button("⚡ Konversi Bundle Full ke INT8 ConvRot")
                            btn_del_bundle = gr.Button("🗑️ Hapus Bundle dari Disk", variant="stop")
                        mgr_log_box = gr.Textbox(label="Log Operasi Model & Memori", lines=6)
                    with gr.Column(scale=4):
                        gr.Markdown("### 🔌 Generator Workflow ComfyUI (`ai-studio-t4` Ready)")
                        wf_mode = gr.Radio(
                            choices=["image", "video_streaming", "comfy_core_native"],
                            value="image",
                            label="Tipe Graph Workflow",
                        )
                        btn_gen_wf = gr.Button("📄 Generate JSON Workflow ComfyUI")
                        wf_json_box = gr.Code(label="ComfyUI API Graph JSON", language="json", lines=10)

        # Event Wiring (Non-blocking & Queue-optimized)
        img_bundle.change(_on_bundle_change, inputs=[img_bundle], outputs=[img_sr_ratio], queue=False)
        vid_bundle.change(_on_bundle_change, inputs=[vid_bundle], outputs=[vid_sr_ratio], queue=False)

        img_preset.change(
            lambda p: _on_preset_change(p)[2:],
            inputs=[img_preset],
            outputs=[img_dtype, img_offload, img_grid_cols, img_grid_rows, img_spatial_overlap],
            queue=False,
        )
        vid_preset.change(
            _on_preset_change,
            inputs=[vid_preset],
            outputs=[vid_chunk_len, vid_chunk_overlap, vid_dtype, vid_offload, vid_grid_cols, vid_grid_rows, vid_spatial_overlap],
            queue=False,
        )
        vid_input.change(_inspect_uploaded_video, inputs=[vid_input], outputs=[vid_info_md], queue=False)

        btn_upscale_img.click(
            _run_image_upscale,
            inputs=[
                img_input,
                batch_img_input,
                img_bundle,
                img_sr_ratio,
                img_preset,
                img_dtype,
                img_offload,
                img_grid_cols,
                img_grid_rows,
                img_spatial_overlap,
                img_target_width,
                img_seed,
            ],
            outputs=[img_output_preview, img_download_file, img_meta_box, telemetry_md],
        ).then(
            _refresh_gallery,
            outputs=[history_gallery],
            queue=False,
            show_progress="hidden",
        )

        btn_upscale_vid.click(
            _run_video_upscale,
            inputs=[
                vid_input,
                vid_bundle,
                vid_sr_ratio,
                vid_preset,
                vid_chunk_len,
                vid_chunk_overlap,
                vid_max_frames,
                vid_dtype,
                vid_offload,
                vid_grid_cols,
                vid_grid_rows,
                vid_spatial_overlap,
                vid_target_width,
                vid_codec,
                vid_crf,
                vid_enc_preset,
                vid_preserve_audio,
                vid_seed,
            ],
            outputs=[vid_output_player, vid_download_file, vid_meta_box, telemetry_md],
        ).then(
            _refresh_gallery,
            outputs=[history_gallery],
            queue=False,
            show_progress="hidden",
        )

        btn_gal_refresh.click(_refresh_gallery, outputs=[history_gallery], queue=False)
        btn_gal_clear.click(_clear_gallery, outputs=[history_gallery, gal_status_md], queue=False)
        history_gallery.select(
            _on_gallery_select,
            outputs=[gal_img_view, gal_vid_view, gal_download, gal_meta_json],
            queue=False,
        )

        btn_dl_bundle.click(
            _download_bundle_ui,
            inputs=[mgr_bundle, mgr_hf_token, mgr_force],
            outputs=[mgr_log_box, bundles_table_md, telemetry_md],
        )
        btn_conv_int8.click(
            _convert_int8_ui,
            inputs=[mgr_bundle],
            outputs=[mgr_log_box, bundles_table_md, telemetry_md],
        )
        btn_del_bundle.click(
            _delete_bundle_ui,
            inputs=[mgr_bundle],
            outputs=[mgr_log_box, bundles_table_md, telemetry_md],
            queue=False,
        )
        btn_quick_refresh.click(
            lambda: (backend.memory.format_telemetry_markdown(), backend.get_bundles_status_markdown()),
            outputs=[telemetry_md, bundles_table_md],
            queue=False,
        )
        btn_quick_unload.click(
            _unload_vram_ui,
            outputs=[mgr_log_box, bundles_table_md, telemetry_md],
            queue=False,
        )
        btn_gen_wf.click(
            _export_workflow_ui,
            inputs=[wf_mode, mgr_bundle, img_sr_ratio, vid_chunk_len],
            outputs=[wf_json_box],
            queue=False,
        )

    return demo
