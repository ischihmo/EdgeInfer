//
// EdgeInfer - Custom Image Structure (decouples OpenCV)
// Reference: Design Doc v4.0 Section 4.1
//

#ifndef EDGEINFER_IMAGE_H
#define EDGEINFER_IMAGE_H

#include <vector>
#include <cstdint>
#include <string>
#include <cstring>

#ifdef EDGEINFER_WITH_OPENCV
#include <opencv2/core/mat.hpp>
#endif

// Visibility (mirrors EdgeInfer.hpp — image.h may be included standalone)
#if defined(_WIN32)
  #define EDGEINFER_IMAGE_API __declspec(dllexport)
#else
  #define EDGEINFER_IMAGE_API __attribute__((visibility("default")))
#endif

namespace edgeinfer {

// ============================================================
// Pixel Format Enum
// ============================================================

enum class PixelFormat : uint8_t {
    BGR = 0,
    RGB = 1,
    GRAY = 2,
    NV12 = 3,   // Y + UV interleaved (U first)
    NV21 = 6,   // Y + VU interleaved (V first) — Android camera / Novatek NPU
    RGBA = 4,
    BGRA = 5
};

// ============================================================
// Memory Layout Enum
// ============================================================

enum class MemoryLayout : uint8_t {
    HWC = 0,   // Height-Width-Channel (OpenCV default)
    CHW = 1    // Channel-Height-Width (deep learning default)
};

// ============================================================
// Lightweight Image Struct
// ============================================================

struct EDGEINFER_IMAGE_API Image {
    std::vector<uint8_t> data;
    int width = 0;
    int height = 0;
    int channels = 3;
    PixelFormat format = PixelFormat::BGR;
    MemoryLayout layout = MemoryLayout::HWC;

    // --- Constructors ---
    Image() = default;
    Image(int w, int h, int c, PixelFormat fmt = PixelFormat::BGR);
    Image(const uint8_t* src, int w, int h, int c,
          PixelFormat fmt = PixelFormat::BGR);

    // --- Factory ---
    static Image Create(int w, int h, int c, PixelFormat fmt = PixelFormat::BGR);
    static Image FromFile(const std::string& path);

    // --- Queries ---
    bool Empty() const {
        return data.empty() || width <= 0 || height <= 0;
    }
    size_t Size() const {
        // NV12/NV21 are YUV420 semi-planar: Y plane + interleaved UV/VU
        if (format == PixelFormat::NV12 || format == PixelFormat::NV21)
            return static_cast<size_t>(width) * height * 3 / 2;
        return static_cast<size_t>(width) * height * channels;
    }
    size_t DataSize() const { return data.size(); }

    // --- Format Conversion ---
    bool ConvertFormat(PixelFormat target);
    bool ConvertLayout(MemoryLayout target);

    // --- OpenCV Interop (optional) ---
#ifdef EDGEINFER_WITH_OPENCV
    static Image FromCvMat(const cv::Mat& mat);
    cv::Mat ToCvMat() const;
#endif
};

// ============================================================
// Inline Implementations
// ============================================================

inline Image::Image(int w, int h, int c, PixelFormat fmt)
    : width(w), height(h), channels(c), format(fmt)
    , layout(MemoryLayout::HWC) {
    data.resize(Size());
}

inline Image::Image(const uint8_t* src, int w, int h, int c, PixelFormat fmt)
    : width(w), height(h), channels(c), format(fmt)
    , layout(MemoryLayout::HWC) {
    data.assign(src, src + Size());
}

inline Image Image::Create(int w, int h, int c, PixelFormat fmt) {
    return Image(w, h, c, fmt);
}

// Implemented in src/types/image.cpp

} // namespace edgeinfer

#endif // EDGEINFER_IMAGE_H
