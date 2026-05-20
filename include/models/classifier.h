//
// EdgeInfer - Classification Model
//

#ifndef EDGEINFER_CLASSIFIER_H
#define EDGEINFER_CLASSIFIER_H

#include "base_model.h"
#include "../types/types.h"
#include <vector>

namespace edgeinfer {

class Classifier : public BaseModel {
public:
    Classifier() = default;

    std::string TaskType() const override { return "classification"; }

    // Run classification on an Image, returns scores and labels
    void Classify(const Image& input, ClassificationResult& results) {
        if (pipeline_) {
            pipeline_->Run(input, &results);
        }
    }

#ifdef EDGEINFER_WITH_OPENCV
    void Classify(const class cv::Mat& input, ClassificationResult& results) {
        if (pipeline_) {
            pipeline_->Run(input, &results);
        }
    }
#endif
};

} // namespace edgeinfer

#endif // EDGEINFER_CLASSIFIER_H
