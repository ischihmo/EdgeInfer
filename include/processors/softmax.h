//
// EdgeInfer - Softmax PostProcessor
//

#ifndef EDGEINFER_SOFTMAX_PROCESSOR_H
#define EDGEINFER_SOFTMAX_PROCESSOR_H

#include "../core/processor.h"
#include "../types/types.h"
#include <vector>
#include <string>
#include <unordered_map>

namespace edgeinfer {

class SoftmaxProcessor : public IPostProcessor {
public:
    SoftmaxProcessor() : top_k_(5) {}

    void Process(const std::vector<Tensor>& outputs, void* results) override;
    std::string Name() const override { return "Softmax"; }

    void SetParams(const std::unordered_map<std::string, double>& params) override {
        if (params.count("top_k"))
            top_k_ = static_cast<int>(params.at("top_k"));
    }

private:
    int top_k_;
};

REGISTER_POSTPROCESSOR("Softmax", SoftmaxProcessor);

} // namespace edgeinfer

#endif // EDGEINFER_SOFTMAX_PROCESSOR_H
