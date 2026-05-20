//
// EdgeInfer - Detection Model
//

#ifndef EDGEINFER_DETECTOR_H
#define EDGEINFER_DETECTOR_H

#include "base_model.h"
#include "../types/types.h"
#include <vector>

namespace edgeinfer {

class Detector : public BaseModel {
public:
    Detector() = default;

    std::string TaskType() const override { return "detection"; }

    // Run detection on an Image, returns bounding boxes
    void Detect(const Image& input, std::vector<Boxf>& results) {
        if (pipeline_) {
            pipeline_->Run(input, &results);
        }
    }

#ifdef EDGEINFER_WITH_OPENCV
    void Detect(const class cv::Mat& input, std::vector<Boxf>& results) {
        if (pipeline_) {
            pipeline_->Run(input, &results);
        }
    }
#endif
};

} // namespace edgeinfer

#endif // EDGEINFER_DETECTOR_H
