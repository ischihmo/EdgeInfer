//
// EdgeInfer - YOLOv5 Decode + NMS PostProcessor
//

#ifndef EDGEINFER_YOLO_DECODE_PROCESSOR_H
#define EDGEINFER_YOLO_DECODE_PROCESSOR_H

#include "../core/processor.h"
#include "../types/types.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cmath>

namespace edgeinfer {

// ============================================================
// YoloDecodeProcessor — YOLOv5 post-processing
// ============================================================

class YoloDecodeProcessor : public IPostProcessor {
public:
    YoloDecodeProcessor();

    void Process(const std::vector<Tensor>& outputs, void* results) override;
    std::string Name() const override { return "YoloDecode"; }
    void SetParams(const std::unordered_map<std::string, double>& params) override;

    // Passed by pipeline after letterbox
    void SetLetterboxInfo(float scale, int pad_left, int pad_top,
                          int orig_w, int orig_h) override;

private:
    void GenerateProposals(const Tensor& feat, int stride,
                           const std::vector<float>& anchor,
                           std::vector<Boxf>& proposals);
    void ApplyNms(std::vector<Boxf>& proposals,
                  std::vector<Boxf>& results);
    void RemapCoordinates(std::vector<Boxf>& boxes);

    int   num_classes_ = 80;
    float conf_threshold_ = 0.4f;
    float nms_threshold_ = 0.45f;

    std::vector<int> strides_ = {8, 16, 32};
    std::vector<std::vector<float>> anchors_ = {
        {10.0f, 13.0f, 16.0f, 30.0f, 33.0f, 23.0f},
        {30.0f, 61.0f, 62.0f, 45.0f, 59.0f, 119.0f},
        {116.0f, 90.0f, 156.0f, 198.0f, 373.0f, 326.0f}
    };

    // Letterbox info for coordinate remapping
    bool  has_letterbox_info_ = false;
    float scale_ = 1.0f;
    int   pad_left_ = 0;
    int   pad_top_ = 0;
    int   orig_w_ = 0;
    int   orig_h_ = 0;
};

// Small helpers
inline float SigmoidF(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline float IoU(const Boxf& a, const Boxf& b) {
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);
    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (area_a + area_b - inter + 1e-6f);
}

} // namespace edgeinfer

#endif // EDGEINFER_YOLO_DECODE_PROCESSOR_H
