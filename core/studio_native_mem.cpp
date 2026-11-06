// core/studio_native_mem.cpp
// Native Linux glibc & VFS Page-Cache Memory Manager for Google Colab Tesla T4 (12.7 GB System RAM)
// Identical ABI to ai-studio-t4/core/studio_native_mem.cpp for zero-friction integration.

#include <fcntl.h>
#include <malloc.h>
#include <stdint.h>
#include <sys/stat.h>
#include <unistd.h>

extern "C" {

// Configure glibc allocator to release large tensor buffers immediately via munmap
int studio_mem_init() {
#ifdef __GLIBC__
    mallopt(M_MMAP_THRESHOLD, 128 * 1024); // 128 KB threshold
    mallopt(M_TRIM_THRESHOLD, 64 * 1024);
    mallopt(M_ARENA_MAX, 2);
    return 1;
#else
    return 0;
#endif
}

// Force glibc heap trim back to OS kernel
int studio_mem_trim() {
#ifdef __GLIBC__
    return malloc_trim(0);
#else
    return 0;
#endif
}

// Evict model weight files (.safetensors / .gguf / .pth) from Linux VFS page cache
// Prevents 5-10 GB model reads from consuming Colab's 12.7 GB System RAM
int studio_evict_file_cache(const char* filepath) {
    if (!filepath) return -1;
    int fd = open(filepath, O_RDONLY);
    if (fd < 0) return -2;

    struct stat st;
    int rc = 0;
    if (fstat(fd, &st) == 0 && st.st_size > 0) {
#if defined(_POSIX_ADVISORY_INFO) && (_POSIX_ADVISORY_INFO > 0)
        rc = posix_fadvise(fd, 0, st.st_size, POSIX_FADV_DONTNEED);
#endif
    }
    close(fd);
    return rc;
}

} // extern "C"
