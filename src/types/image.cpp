//
// EdgeInfer - Image Format & Layout Conversion Implementation
//

#include "../../include/types/image.h"
#include <algorithm>
#include <cstring>

namespace edgeinfer {

// ============================================================
// Helpers
// ============================================================

namespace {

// BT.601 limited-range YUV → RGB
inline uint8_t Clamp(int v) {
    return static_cast<uint8_t>(std::max(0, std::min(255, v)));
}

inline void YuvToRgb(int y, int u, int v,
                     uint8_t& r, uint8_t& g, uint8_t& b) {
    int c = y - 16;
    int d = u - 128;
    int e = v - 128;
    r = Clamp((298 * c + 409 * e + 128) >> 8);
    g = Clamp((298 * c - 100 * d - 208 * e + 128) >> 8);
    b = Clamp((298 * c + 516 * d + 128) >> 8);
}

// Swap channels 0 and 2 (BGR ↔ RGB)
void SwapRgbChannels(Image& img) {
    for (size_t i = 0; i + 2 < img.data.size(); i += 3) {
        std::swap(img.data[i], img.data[i + 2]);
    }
    img.format = (img.format == PixelFormat::BGR) ? PixelFormat::RGB
                                                  : PixelFormat::BGR;
}

// BGR/RGB → GRAY: Y = 0.299R + 0.587G + 0.114B
void RgbToGray(Image& img) {
    int c_in = img.channels;
    std::vector<uint8_t> gray(static_cast<size_t>(img.width) * img.height);
    for (size_t y = 0; y < static_cast<size_t>(img.height); ++y) {
        const uint8_t* src = img.data.data() + y * img.width * c_in;
        uint8_t*       dst = gray.data()         + y * img.width;
        for (int x = 0; x < img.width; ++x) {
            int r, g, b;
            if (img.format == PixelFormat::BGR || img.format == PixelFormat::BGRA) {
                b = src[0]; g = src[1]; r = src[2];
            } else {
                r = src[0]; g = src[1]; b = src[2];
            }
            dst[x] = static_cast<uint8_t>(
                (static_cast<int>(r) * 77  +
                 static_cast<int>(g) * 150 +
                 static_cast<int>(b) * 29) >> 8);
            src += c_in;
        }
    }
    img.data = std::move(gray);
    img.channels = 1;
    img.format = PixelFormat::GRAY;
}

// NV12 → BGR or RGB
bool Nv12ToRgb(Image& img, PixelFormat target) {
    int w = img.width, h = img.height;
    if ((w & 1) || (h & 1)) return false;  // must be even
    if (img.format != PixelFormat::NV12) return false;

    size_t y_size  = static_cast<size_t>(w) * h;
    size_t uv_size = y_size / 2;  // interleaved UV
    const uint8_t* y_plane  = img.data.data();
    const uint8_t* uv_plane = img.data.data() + y_size;

    int out_c = 3;
    std::vector<uint8_t> out(static_cast<size_t>(w) * h * out_c);

    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            int yy = y_plane[row * w + col];
            // UV subsampled 2×2: each UV pair covers 2×2 Y block
            int uv_idx = (row / 2) * (w / 2) + (col / 2);
            int u = uv_plane[uv_idx * 2];
            int v = uv_plane[uv_idx * 2 + 1];

            uint8_t r, g, b;
            YuvToRgb(yy, u, v, r, g, b);

            size_t out_idx = (static_cast<size_t>(row) * w + col) * 3;
            if (target == PixelFormat::BGR) {
                out[out_idx] = b; out[out_idx + 1] = g; out[out_idx + 2] = r;
            } else {
                out[out_idx] = r; out[out_idx + 1] = g; out[out_idx + 2] = b;
            }
        }
    }

    img.data = std::move(out);
    img.channels = 3;
    img.format = target;
    return true;
}

} // anonymous namespace

// ============================================================
// ConvertFormat
// ============================================================

bool Image::ConvertFormat(PixelFormat target) {
    if (format == target) return true;

    // NV12 → BGR/RGB
    if (format == PixelFormat::NV12) {
        if (target == PixelFormat::BGR || target == PixelFormat::RGB) {
            return Nv12ToRgb(*this, target);
        }
        return false;  // NV12 → other unsupported
    }

    // BGR/RGB → NV12 unsupported
    if (target == PixelFormat::NV12) return false;

    // BGR ↔ RGB
    if ((format == PixelFormat::BGR  && target == PixelFormat::RGB) ||
        (format == PixelFormat::RGB  && target == PixelFormat::BGR)) {
        SwapRgbChannels(*this);
        return true;
    }

    // BGR/RGB → GRAY
    if ((format == PixelFormat::BGR  || format == PixelFormat::RGB) &&
         target == PixelFormat::GRAY) {
        RgbToGray(*this);
        return true;
    }

    // RGBA/BGRA → BGR/RGB (drop alpha)
    if ((format == PixelFormat::RGBA || format == PixelFormat::BGRA) &&
        (target == PixelFormat::BGR || target == PixelFormat::RGB)) {
        // Simply strip alpha: RGBA → RGB, then swap if needed
        int w = width, h = height;
        std::vector<uint8_t> out(static_cast<size_t>(w) * h * 3);
        for (size_t y = 0; y < static_cast<size_t>(h); ++y) {
            const uint8_t* src = data.data() + y * w * 4;
            uint8_t*       dst = out.data()    + y * w * 3;
            for (int x = 0; x < w; ++x) {
                dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
                src += 4; dst += 3;
            }
        }
        data = std::move(out);
        channels = 3;

        // After stripping alpha, data keeps the original channel order
        PixelFormat intermediate = (format == PixelFormat::RGBA)
                                       ? PixelFormat::RGB
                                       : PixelFormat::BGR;
        format = intermediate;

        if (format != target) {
            SwapRgbChannels(*this);  // RGB ↔ BGR
        }
        return true;
    }

    return false;  // unsupported conversion
}

// ============================================================
// ConvertLayout
// ============================================================

bool Image::ConvertLayout(MemoryLayout target) {
    if (layout == target) return true;
    if (data.empty()) return false;

    size_t total = static_cast<size_t>(width) * height * channels;
    std::vector<uint8_t> reordered(total);

    if (layout == MemoryLayout::HWC && target == MemoryLayout::CHW) {
        // HWC → CHW: pixel[y][x][c] → channel[c][y][x]
        for (int c = 0; c < channels; ++c) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    size_t hwc_idx = (static_cast<size_t>(y) * width + x) * channels + c;
                    size_t chw_idx = (static_cast<size_t>(c) * height + y) * width + x;
                    reordered[chw_idx] = data[hwc_idx];
                }
            }
        }
    } else if (layout == MemoryLayout::CHW && target == MemoryLayout::HWC) {
        // CHW → HWC: channel[c][y][x] → pixel[y][x][c]
        for (int c = 0; c < channels; ++c) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    size_t chw_idx = (static_cast<size_t>(c) * height + y) * width + x;
                    size_t hwc_idx = (static_cast<size_t>(y) * width + x) * channels + c;
                    reordered[hwc_idx] = data[chw_idx];
                }
            }
        }
    } else {
        return false;
    }

    data = std::move(reordered);
    layout = target;
    return true;
}

} // namespace edgeinfer

// ============================================================
// OpenCV Interop
// ============================================================

#ifdef EDGEINFER_WITH_OPENCV
#include <opencv2/opencv.hpp>

namespace edgeinfer {

Image Image::FromCvMat(const cv::Mat& mat) {
    Image img;
    if (mat.empty()) return img;

    img.width    = mat.cols;
    img.height   = mat.rows;
    img.channels = mat.channels();
    img.layout   = MemoryLayout::HWC;

    // Map OpenCV type → PixelFormat
    switch (mat.channels()) {
        case 1:  img.format = PixelFormat::GRAY; break;
        case 3:  img.format = PixelFormat::BGR;  break;
        case 4:  img.format = PixelFormat::BGRA; break;
        default: img.format = PixelFormat::BGR;  break;
    }

    if (mat.isContinuous()) {
        img.data.assign(mat.data, mat.data + mat.total() * mat.channels());
    } else {
        img.data.resize(img.Size());
        for (int y = 0; y < img.height; ++y) {
            ::memcpy(img.data.data() + static_cast<size_t>(y) * img.width * img.channels,
                     mat.ptr(y), static_cast<size_t>(img.width) * img.channels);
        }
    }
    return img;
}

cv::Mat Image::ToCvMat() const {
    int cvType = CV_8UC(channels);
    cv::Mat mat(height, width, cvType);

    if (data.size() >= Size()) {
        ::memcpy(mat.data, data.data(), Size());
    }
    return mat;
}
} // namespace edgeinfer
#endif

// ============================================================
// FromFile — unified image loader
// ============================================================

namespace edgeinfer {

static bool ReadPPM(const std::string& path, std::vector<uint8_t>& data,
                    int& w, int& h, int& c) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[3];
    if (fscanf(f, "%2s", magic) != 1 || magic[0] != 'P' || magic[1] != '6')
        { fclose(f); return false; }
    if (fscanf(f, "%d %d", &w, &h) != 2) { fclose(f); return false; }
    int maxval;
    if (fscanf(f, "%d", &maxval) != 1 || maxval != 255)
        { fclose(f); return false; }
    fgetc(f);
    c = 3;
    data.resize(static_cast<size_t>(w) * h * c);
    size_t n = fread(data.data(), 1, data.size(), f);
    fclose(f);
    return n == data.size();
}

// Parse WxH from filename e.g. "cat_1280x720.nv21.yuv" → 1280, 720
static bool ParseNv21Dims(const std::string& path, int& w, int& h) {
    // Find the last occurrence of NNNNxNNNN pattern before the extension
    const char* p = path.c_str();
    const char* last_x = nullptr;
    for (const char* s = p; *s; ++s) {
        if (*s == 'x' || *s == 'X') last_x = s;
    }
    if (!last_x) return false;

    // Scan digits before 'x'
    const char* pre = last_x - 1;
    while (pre >= p && *pre >= '0' && *pre <= '9') --pre;
    ++pre;

    // Scan digits after 'x'
    const char* post = last_x + 1;
    while (*post >= '0' && *post <= '9') ++post;

    w = atoi(pre);
    h = atoi(last_x + 1);
    return w > 0 && h > 0;
}

static bool ReadRawNV21(const std::string& path, std::vector<uint8_t>& data,
                        int& w, int& h, int& c) {
    if (!ParseNv21Dims(path, w, h)) return false;
    size_t expect = static_cast<size_t>(w) * h * 3 / 2;

    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (static_cast<size_t>(sz) != expect) { fclose(f); return false; }

    data.resize(sz);
    size_t n = fread(data.data(), 1, sz, f);
    fclose(f);
    if (n != static_cast<size_t>(sz)) return false;

    c = 2;
    return true;
}

Image Image::FromFile(const std::string& path) {
#ifdef EDGEINFER_WITH_OPENCV
    cv::Mat m = cv::imread(path, cv::IMREAD_COLOR);
    if (!m.empty()) {
        Image img = FromCvMat(m);
        return img;
    }
#endif
    // Try NV21 raw YUV (parse WxH from filename)
    {
        std::vector<uint8_t> data;
        int w = 0, h = 0, c = 0;
        if (ReadRawNV21(path, data, w, h, c))
            return Image(data.data(), w, h, c, PixelFormat::NV21);
    }
    // Try PPM
    {
        std::vector<uint8_t> data;
        int w = 0, h = 0, c = 0;
        if (ReadPPM(path, data, w, h, c))
            return Image(data.data(), w, h, c, PixelFormat::RGB);
    }
    return Image();
}

} // namespace edgeinfer
