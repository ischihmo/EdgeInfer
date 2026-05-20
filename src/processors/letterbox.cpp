//
// EdgeInfer - LetterBox PreProcessor Implementation
//

#include "../../include/processors/letterbox.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace edgeinfer {

LetterBoxProcessor::LetterBoxProcessor()
    : target_w_(640), target_h_(640), max_stride_(32), pad_value_(114) {}

void LetterBoxProcessor::SetParams(
    const std::unordered_map<std::string, double>& params) {
    if (params.count("target_width"))
        target_w_ = static_cast<int>(params.at("target_width"));
    if (params.count("target_height"))
        target_h_ = static_cast<int>(params.at("target_height"));
    if (params.count("max_stride"))
        max_stride_ = static_cast<int>(params.at("max_stride"));
    if (params.count("pad_value"))
        pad_value_ = static_cast<uint8_t>(params.at("pad_value"));
}

void LetterBoxProcessor::BilinearResize(const Image& src, Image& dst,
                                         int dw, int dh) {
    int sw = src.width, sh = src.height;
    int sc = src.channels;
    dst = Image(dw, dh, sc, src.format);

    float scale_x = static_cast<float>(sw) / dw;
    float scale_y = static_cast<float>(sh) / dh;

    for (int y = 0; y < dh; ++y) {
        float src_y = (y + 0.5f) * scale_y - 0.5f;
        int y0 = std::max(0, static_cast<int>(src_y));
        int y1 = std::min(sh - 1, y0 + 1);
        float fy = src_y - y0;

        uint8_t* drow = dst.data.data() + static_cast<size_t>(y) * dw * sc;

        for (int x = 0; x < dw; ++x) {
            float src_x = (x + 0.5f) * scale_x - 0.5f;
            int x0 = std::max(0, static_cast<int>(src_x));
            int x1 = std::min(sw - 1, x0 + 1);
            float fx = src_x - x0;

            const uint8_t* s00 = src.data.data() +
                (static_cast<size_t>(y0) * sw + x0) * sc;
            const uint8_t* s01 = src.data.data() +
                (static_cast<size_t>(y0) * sw + x1) * sc;
            const uint8_t* s10 = src.data.data() +
                (static_cast<size_t>(y1) * sw + x0) * sc;
            const uint8_t* s11 = src.data.data() +
                (static_cast<size_t>(y1) * sw + x1) * sc;

            uint8_t* dp = drow + static_cast<size_t>(x) * sc;
            for (int c = 0; c < sc; ++c) {
                float v00 = s00[c], v01 = s01[c];
                float v10 = s10[c], v11 = s11[c];
                float v0 = v00 + (v01 - v00) * fx;
                float v1 = v10 + (v11 - v10) * fx;
                dp[c] = static_cast<uint8_t>(v0 + (v1 - v0) * fy + 0.5f);
            }
        }
    }
}

void LetterBoxProcessor::Process(const Image& input, Image& output) {
    if (input.Empty()) return;

    orig_w_ = input.width;
    orig_h_ = input.height;

    // Aspect-ratio-preserving scale
    float w_ratio = static_cast<float>(target_w_) / orig_w_;
    float h_ratio = static_cast<float>(target_h_) / orig_h_;
    scale_ = std::min(w_ratio, h_ratio);

    int new_w = static_cast<int>(std::round(orig_w_ * scale_));
    int new_h = static_cast<int>(std::round(orig_h_ * scale_));

    // Stride-aligned padding (dynamic)
    int pad_w = ((new_w + max_stride_ - 1) / max_stride_) * max_stride_ - new_w;
    int pad_h = ((new_h + max_stride_ - 1) / max_stride_) * max_stride_ - new_h;

    pad_left_ = pad_w / 2;
    pad_top_  = pad_h / 2;

    // Step 1: bilinear resize
    Image resized;
    BilinearResize(input, resized, new_w, new_h);

    // Step 2: create padded canvas
    int out_w = new_w + pad_w;
    int out_h = new_h + pad_h;
    output = Image(out_w, out_h, resized.channels, resized.format);
    std::memset(output.data.data(), pad_value_, output.data.size());

    // Step 3: copy resized into center of canvas
    for (int y = 0; y < new_h; ++y) {
        const uint8_t* src_row = resized.data.data() +
            static_cast<size_t>(y) * new_w * resized.channels;
        uint8_t* dst_row = output.data.data() +
            static_cast<size_t>(y + pad_top_) * out_w * output.channels +
            static_cast<size_t>(pad_left_) * output.channels;
        std::memcpy(dst_row, src_row,
                    static_cast<size_t>(new_w) * resized.channels);
    }
}

bool LetterBoxProcessor::GetLetterboxInfo(
    float& scale, int& pad_left, int& pad_top,
    int& orig_w, int& orig_h) const {
    scale    = scale_;
    pad_left = pad_left_;
    pad_top  = pad_top_;
    orig_w   = orig_w_;
    orig_h   = orig_h_;
    return true;
}

} // namespace edgeinfer
