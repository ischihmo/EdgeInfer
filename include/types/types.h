//
// EdgeInfer - Unified Types Definition
// Reference: Design Doc v4.0 Section 4.2
//

#ifndef EDGEINFER_TYPES_H
#define EDGEINFER_TYPES_H

#include "image.h"
#include <cstdint>
#include <vector>
#include <string>

namespace edgeinfer {

// ============================================================
// Inference Backend Enum
// ============================================================

enum class Backend : uint8_t {
    MNN = 0,
    NCNN = 1,
    CVDNN = 2,
    ONNXRUNTIME = 3,
    NTCNN = 4,
    AUTO = 255
};

// ============================================================
// Data Type Enum
// ============================================================

enum class DataType : uint8_t {
    FP32 = 0,
    FP16 = 1,
    INT8 = 2,
    INT32 = 3,
    UINT8 = 4
};

// ============================================================
// Device Type Enum
// ============================================================

enum class DeviceType : uint8_t {
    CPU = 0,
    GPU = 1,
    NPU = 2,
    VULKAN = 3
};

// ============================================================
// Shape
// ============================================================

struct Shape {
    int batch = 1;
    int channels = 3;
    int height = 0;
    int width = 0;

    Shape() = default;
    Shape(int b, int c, int h, int w)
        : batch(b), channels(c), height(h), width(w) {}

    size_t Size() const {
        return static_cast<size_t>(batch) * channels * height * width;
    }

    bool operator==(const Shape& other) const {
        return batch == other.batch && channels == other.channels
            && height == other.height && width == other.width;
    }
    bool operator!=(const Shape& other) const { return !(*this == other); }
};

// ============================================================
// Tensor
// ============================================================

struct Tensor {
    std::vector<float> data;
    Shape shape;
    DataType dtype = DataType::FP32;

    Tensor() = default;
    explicit Tensor(const Shape& s)
        : shape(s), dtype(DataType::FP32) {
        data.resize(s.Size());
    }

    size_t Size() const { return shape.Size(); }
    bool Empty() const { return data.empty(); }
};

// ============================================================
// BoundingBox (generic detection box)
// ============================================================

template<typename T = float>
struct BoundingBox {
    T x1 = 0;
    T y1 = 0;
    T x2 = 0;
    T y2 = 0;
    float score = 0.0f;
    uint32_t label = 0;
    std::string label_text;
    bool flag = false;

    BoundingBox() = default;

    T Width() const { return x2 - x1 + static_cast<T>(1); }
    T Height() const { return y2 - y1 + static_cast<T>(1); }
    T Area() const { return Width() * Height(); }

#ifdef EDGEINFER_WITH_OPENCV
    cv::Rect ToCvRect() const {
        return cv::Rect(static_cast<int>(x1), static_cast<int>(y1),
                        static_cast<int>(Width()),
                        static_cast<int>(Height()));
    }
#endif
};

using Boxf = BoundingBox<float>;
using Boxi = BoundingBox<int>;
using Boxd = BoundingBox<double>;

// ============================================================
// Task Result Types
// ============================================================

struct ClassificationResult {
    std::vector<float> scores;
    std::vector<uint32_t> labels;
    std::vector<std::string> texts;
    bool flag = false;

    ClassificationResult() = default;
};

struct EmbeddingResult {
    std::vector<float> embedding;
    bool flag = false;

    EmbeddingResult() = default;
};

// ============================================================
// Inference Configuration
// ============================================================

struct InferConfig {
    Backend backend = Backend::AUTO;
    std::string model_path;
    int num_threads = 1;
    int input_width = 640;
    int input_height = 640;
    float conf_threshold = 0.45f;
    float nms_threshold = 0.45f;
    std::vector<float> mean = {0, 0, 0};
    std::vector<float> scale = {1.0f / 255, 1.0f / 255, 1.0f / 255};

    // Letterbox params — set by LetterBox preprocessor, read by decoder
    float letterbox_scale = 1.0f;
    int   letterbox_pad_left = 0;
    int   letterbox_pad_top = 0;
    int   letterbox_orig_width = 0;
    int   letterbox_orig_height = 0;
};

struct DetectConfig : InferConfig {
    int num_classes = 80;
    int max_stride = 32;
    std::vector<int> strides = {8, 16, 32};
    std::vector<std::vector<float>> anchors = {
        {10.0f, 13.0f, 16.0f, 30.0f, 33.0f, 23.0f},
        {30.0f, 61.0f, 62.0f, 45.0f, 59.0f, 119.0f},
        {116.0f, 90.0f, 156.0f, 198.0f, 373.0f, 326.0f}
    };
};

struct ClassifyConfig : InferConfig {
    int top_k = 5;
};

} // namespace edgeinfer

#endif // EDGEINFER_TYPES_H
