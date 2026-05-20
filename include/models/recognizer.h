//
// EdgeInfer - Recognition Model (e.g. face recognition)
//

#ifndef EDGEINFER_RECOGNIZER_H
#define EDGEINFER_RECOGNIZER_H

#include "base_model.h"
#include "../types/types.h"
#include <vector>

namespace edgeinfer {

class Recognizer : public BaseModel {
public:
    Recognizer() = default;

    std::string TaskType() const override { return "recognition"; }

    // Run recognition, extracts feature embedding
    void Extract(const Image& input, EmbeddingResult& results) {
        if (pipeline_) {
            pipeline_->Run(input, &results);
        }
    }

#ifdef EDGEINFER_WITH_OPENCV
    void Extract(const class cv::Mat& input, EmbeddingResult& results) {
        if (pipeline_) {
            pipeline_->Run(input, &results);
        }
    }
#endif
};

} // namespace edgeinfer

#endif // EDGEINFER_RECOGNIZER_H
