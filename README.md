# ⚡ RelayVSR T4 Studio — Collaborative 2× & 4× Video & Image Super-Resolution

Proyek mandiri siap unggah ke GitHub untuk menjalankan **[RelayVSR (`arXiv:2609.37850`)](https://arxiv.org/html/2609.37850v1)** ([`kopperx/RelayVSR`](https://github.com/kopperx/RelayVSR), [`lightx2v/RelayVSR_X2`](https://huggingface.co/collections/lightx2v/minimax-h3-lightx2v), dan [`Comfy-Org/RelayVSR`](https://huggingface.co/Comfy-Org/RelayVSR)) pada GPU **Google Colab Tesla T4 (15 GB VRAM, 12.7 GB System RAM, arsitektur Turing `sm_75`)**.

Dirancang dengan struktur modul yang **100% kompatibel dengan `ai-studio-t4`** (`config/catalog.py`, `core/studio_native_mem.cpp`, `core/memory_manager.py`, `core/gallery.py`, `engines/relayvsr_engine.py`, `ui/fooocus_app.py`) sehingga dapat langsung digabungkan ke `ai-studio-t4` kapan saja tanpa perubahan arsitektur.

## Fitur Utama

* **Dukungan Penuh `INT8 ConvRot` (`5.29 GB`) untuk Tesla T4**: Memanfaatkan [`Comfy-Org/RelayVSR`](https://huggingface.co/Comfy-Org/RelayVSR/tree/main/diffusion_models) (`relayvsr_x2_lightx2v_int8_convrot.safetensors` & `relayvsr_x4_int8_convrot.safetensors` `5.16 GB` + `relayvsr_flash_vae_x2/x4_fp32.safetensors` `128 MB`) yang mempercepat inferensi di GPU Turing `sm_75` hingga ~6.2× dibanding BF16 dan menghemat ~5 GB VRAM.
* **Upscaling Gambar Tunggal & Batch (`Tab 1`)**:
  * Koreksi orientasi otomatis `ImageOps.exif_transpose()`.
  * *Reflection padding* otomatis ke kelipatan `16 px` (mencegah error dimensi pada *stride-4 LR encoder* dan *stride-16 reference grid*).
  * *Spatial Tiling* otomatis dengan *2D Linear Crossfade* (`64 px`) untuk gambar resolusi besar agar tidak pernah mengalami *CUDA Out of Memory*.
* **Streaming Video Upscaling Anti-Flicker (`Tab 2`)**:
  * Memproses video per jendela waktu (`chunk_len = 16`, `chunk_overlap = 4`) sehingga video berdurasi panjang tidak menumpuk di RAM/VRAM.
  * Menyalin atau mengenkode ulang *audio stream* asli (`AAC` / `Opus`) secara otomatis.
* **Native C++ RAM & VFS Page-Cache Cleaner (`core/studio_native_mem.cpp`)**: Menguras *Linux VFS page cache* (`posix_fadvise(..., POSIX_FADV_DONTNEED)`) dan `malloc_trim(0)` setiap selesai mengunduh atau memuat file `.safetensors` agar kuota System RAM 12.7 GB di Colab tidak pernah penuh.

## Struktur Repositori

```text
relayvsr-t4-studio/
├── bootstrap.py                                 # Installer ComfyUI + Comfyui-Relay-VSR + C++ RAM Cleaner + Downloader
├── run_studio.py                                # Peluncur utama Gradio Studio
├── requirements.txt                             # Daftar pustaka Python
├── 01_RelayVSR_T4_Studio.ipynb                  # Notebook Colab 2-Sel (siap pakai)
├── notebooks/
│   └── RelayVSR_T4_Colab_Launcher.ipynb         # Salinan Notebook Colab di folder notebooks/
├── config/
│   ├── __init__.py
│   └── catalog.py                               # Katalog 4 Bundle RelayVSR (2x/4x INT8 ConvRot & Full FP16)
├── core/
│   ├── __init__.py
│   ├── studio_native_mem.cpp                    # Modul C++ pembersih glibc heap & Linux VFS page-cache
│   ├── memory_manager.py                        # Telemetri VRAM/RAM/Disk & pembungkus ctypes C++
│   ├── gallery.py                               # Galeri non-blocking dengan cache thumbnail JPEG ringan
│   └── backend.py                               # Manajer unduhan, validasi header safetensors, & pemuat node
├── engines/
│   ├── __init__.py                              # Registrasi ENGINE_CLASSES & resolve_engine_id()
│   ├── base_engine.py                           # Kontrak BaseEngine kompatibel ai-studio-t4
│   └── relayvsr_engine.py                       # Engine RelayVSR (Image, Batch, Video Streaming, & ComfyUI Graph)
└── ui/
    ├── __init__.py
    └── fooocus_app.py                           # Antarmuka 4-Tab Gradio Studio
```

## Cara Menggabungkan ke `ai-studio-t4` Nantinya

Saat Anda siap menyatukan proyek ini ke dalam repositori `ai-studio-t4`:
1. Salin `engines/relayvsr_engine.py` ke dalam folder `ai-studio-t4/engines/relayvsr_engine.py`.
2. Tambahkan isi `RELAYVSR_BUNDLES` dari `config/catalog.py` ke `ai-studio-t4/config/catalog.py`.
3. Daftarkan `"relayvsr": RelayVSREngine` pada `ENGINE_CLASSES` di `ai-studio-t4/engines/__init__.py`.
