//
// EdgeInfer - NMS PostProcessor
//

#ifndef EDGEINFER_NMS_PROCESSOR_H
#define EDGEINFER_NMS_PROCESSOR_H

#include "../core/processor.h"
#include "../core/postprocess.hpp"
#include <vector>
#include <algorithm>

namespace edgeinfer {

class NmsProcessor : public IPostProcessor {
public:
    NmsProcessor()
        : conf_threshold_(0.45f), iou_threshold_(0.45f) {}

    void Process(const std::vector<Tensor>& /*outputs*/,
                 void* results) override {
        auto* boxes = static_cast<std::vector<Boxf>*>(results);
        // TODO: decode raw output and apply NMS
        // 1. Decode bounding boxes from output tensors
        // 2. Filter by conf_threshold_
        // 3. Apply NMS with iou_threshold_
        boxes->clear();
    }

    std::string Name() const override { return "NMS"; }

    void SetParams(const std::unordered_map<std::string, double>& params) override {
        if (params.count("conf_threshold")) conf_threshold_ = static_cast<float>(params.at("conf_threshold"));
        if (params.count("iou_threshold")) iou_threshold_ = static_cast<float>(params.at("iou_threshold"));
    }

private:
    float conf_threshold_;
    float iou_threshold_;
};

REGISTER_POSTPROCESSOR("NMS", NmsProcessor);

} // namespace edgeinfer

#endif // EDGEINFER_NMS_PROCESSOR_H
