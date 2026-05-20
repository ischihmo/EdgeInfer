//
// EdgeInfer - Normalize PreProcessor
//

#ifndef EDGEINFER_NORMALIZE_PROCESSOR_H
#define EDGEINFER_NORMALIZE_PROCESSOR_H

#include "../core/processor.h"
#include <vector>

namespace edgeinfer {

class NormalizeProcessor : public IPreProcessor {
public:
    NormalizeProcessor()
        : mean_({0, 0, 0})
        , std_({1.0 / 255, 1.0 / 255, 1.0 / 255}) {}

    void Process(const Image& input, Image& output) override {
        // TODO: apply (pixel - mean) / std normalization
        output = input;
    }

    std::string Name() const override { return "Normalize"; }

    void SetParams(const std::unordered_map<std::string, double>& params) override {
        // params can include "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b"
        // TODO: parse normalized params
    }

private:
    std::vector<float> mean_;
    std::vector<float> std_;
};

REGISTER_PREPROCESSOR("Normalize", NormalizeProcessor);

} // namespace edgeinfer

#endif // EDGEINFER_NORMALIZE_PROCESSOR_H
