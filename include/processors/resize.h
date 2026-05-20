//
// EdgeInfer - Resize PreProcessor
//

#ifndef EDGEINFER_RESIZE_PROCESSOR_H
#define EDGEINFER_RESIZE_PROCESSOR_H

#include "../core/processor.h"
#include <cmath>
#include <algorithm>

namespace edgeinfer {

class ResizeProcessor : public IPreProcessor {
public:
    ResizeProcessor() : target_width_(640), target_height_(640), keep_ratio_(false) {}

    void Process(const Image& input, Image& output) override {
        if (input.Empty()) return;

        int new_w = target_width_;
        int new_h = target_height_;

        if (keep_ratio_) {
            float ratio = std::min(
                static_cast<float>(target_width_) / input.width,
                static_cast<float>(target_height_) / input.height);
            new_w = static_cast<int>(input.width * ratio);
            new_h = static_cast<int>(input.height * ratio);
        }

        // TODO: bilinear interpolation resize on raw pixels
        output = Image(new_w, new_h, input.channels, input.format);
    }

    std::string Name() const override { return "Resize"; }

    void SetParams(const std::unordered_map<std::string, double>& params) override {
        if (params.count("width")) target_width_ = static_cast<int>(params.at("width"));
        if (params.count("height")) target_height_ = static_cast<int>(params.at("height"));
        if (params.count("keep_ratio")) keep_ratio_ = params.at("keep_ratio") > 0.5;
    }

private:
    int target_width_;
    int target_height_;
    bool keep_ratio_;
};

REGISTER_PREPROCESSOR("Resize", ResizeProcessor);

} // namespace edgeinfer

#endif // EDGEINFER_RESIZE_PROCESSOR_H
