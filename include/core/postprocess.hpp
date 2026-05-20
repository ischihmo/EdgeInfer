//
// EdgeInfer - PostProcessor inline utilities
//

#ifndef EDGEINFER_POSTPROCESS_HPP
#define EDGEINFER_POSTPROCESS_HPP

#include "../types/types.h"
#include <algorithm>
#include <cmath>

namespace edgeinfer {

// ============================================================
// Small utility functions (inline)
// ============================================================

inline float Sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline float BoxIou(const Boxf& a, const Boxf& b) {
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);

    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);

    return inter / (area_a + area_b - inter + 1e-6f);
}

// PostProcessorChain implementations in src/core/processor.cpp

} // namespace edgeinfer

#endif // EDGEINFER_POSTPROCESS_HPP
