//
// EdgeInfer - Softmax PostProcessor Implementation
//

#include "../../include/processors/softmax.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace edgeinfer {

void SoftmaxProcessor::Process(const std::vector<Tensor>& outputs,
                                void* results) {
    auto* result = static_cast<ClassificationResult*>(results);
    if (outputs.empty()) return;

    const auto& tensor = outputs[0];
    std::vector<float> scores = tensor.data;

    // Softmax: exp(x - max) / sum(exp(x - max))
    float max_val = *std::max_element(scores.begin(), scores.end());
    float sum = 0.0f;
    for (auto& s : scores) {
        s = std::exp(s - max_val);
        sum += s;
    }
    for (auto& s : scores) s /= sum;

    // Top-K
    std::vector<size_t> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);
    int k = std::min(top_k_, static_cast<int>(scores.size()));
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
        [&scores](size_t a, size_t b) { return scores[a] > scores[b]; });

    result->scores.resize(k);
    result->labels.resize(k);
    for (int i = 0; i < k; ++i) {
        result->scores[i] = scores[indices[i]];
        result->labels[i] = static_cast<uint32_t>(indices[i]);
    }
    result->flag = true;
}

} // namespace edgeinfer
