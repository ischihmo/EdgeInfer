//
// EdgeInfer - C Public API Implementation
//

#include "../../include/EdgeInfer.h"
#include "../../include/EdgeInfer.hpp"
#include "../../include/types/image.h"
#include "../../include/types/config.h"
#include <new>
#include <cstring>

using edgeinfer::EdgeInfer;
using edgeinfer::Image;

// Each handle holds an independent EdgeInfer instance via unique_ptr
typedef std::unique_ptr<EdgeInfer> EnginePtr;

inline EnginePtr* ToPtr(EdgeInferHandle h) {
    return static_cast<EnginePtr*>(h);
}

inline EdgeInfer* ToEngine(EdgeInferHandle h) {
    return static_cast<EnginePtr*>(h)->get();
}

inline Image MakeImage(const uint8_t* data, int32_t w, int32_t h, int32_t c) {
    return Image(data, w, h, c, edgeinfer::PixelFormat::BGR);
}

// ============================================================
// Lifecycle
// ============================================================

EdgeInferHandle EdgeInferCreate(const char* config_path) {
    if (!config_path) return nullptr;

    auto engine = EdgeInfer::Create();
    if (!engine) return nullptr;

    if (!engine->Init(std::string(config_path))) {
        return nullptr;
    }

    auto* ptr = new (std::nothrow) EnginePtr(std::move(engine));
    if (!ptr) return nullptr;

    return static_cast<EdgeInferHandle>(ptr);
}

EdgeInferHandle EdgeInferCreateFromJson(const char* json_config) {
    if (!json_config) return nullptr;

    auto engine = EdgeInfer::Create();
    if (!engine) return nullptr;

    auto config = edgeinfer::ConfigParser::ParseString(json_config);
    if (!engine->Init(config)) {
        return nullptr;
    }

    auto* ptr = new (std::nothrow) EnginePtr(std::move(engine));
    if (!ptr) return nullptr;

    return static_cast<EdgeInferHandle>(ptr);
}

void EdgeInferDestroy(EdgeInferHandle handle) {
    if (!handle) return;
    auto* ptr = ToPtr(handle);
    delete ptr;
}

// ============================================================
// Single Image Inference
// ============================================================

int32_t EdgeInferDetect(EdgeInferHandle handle,
                         const uint8_t* img_data, int32_t w, int32_t h, int32_t c,
                         EdgeInferDetectResult* result) {
    if (!handle || !img_data || !result) return -1;

    auto* engine = ToEngine(handle);
    Image img = MakeImage(img_data, w, h, c);

    std::vector<edgeinfer::Boxf> boxes;
    engine->Detect(img, boxes);

    result->count = static_cast<int32_t>(boxes.size());
    if (boxes.empty()) {
        result->boxes = nullptr;
        return 0;
    }

    result->boxes = static_cast<EdgeInferBox*>(
        std::malloc(boxes.size() * sizeof(EdgeInferBox)));
    if (!result->boxes) {
        result->count = 0;
        return -1;
    }

    for (size_t i = 0; i < boxes.size(); ++i) {
        result->boxes[i] = {boxes[i].x1, boxes[i].y1, boxes[i].x2, boxes[i].y2,
                            boxes[i].score,
                            static_cast<int32_t>(boxes[i].label)};
    }
    return 0;
}

int32_t EdgeInferClassify(EdgeInferHandle handle,
                           const uint8_t* img_data, int32_t w, int32_t h, int32_t c,
                           EdgeInferClassifyResult* result) {
    if (!handle || !img_data || !result) return -1;

    auto* engine = ToEngine(handle);
    Image img = MakeImage(img_data, w, h, c);

    edgeinfer::ClassificationResult cls;
    engine->Classify(img, cls);

    result->count = static_cast<int32_t>(cls.scores.size());
    if (cls.scores.empty()) {
        result->scores = nullptr;
        result->labels = nullptr;
        return 0;
    }

    result->scores = static_cast<float*>(
        std::malloc(cls.scores.size() * sizeof(float)));
    result->labels = static_cast<int32_t*>(
        std::malloc(cls.labels.size() * sizeof(int32_t)));
    if (!result->scores || !result->labels) {
        std::free(result->scores);
        std::free(result->labels);
        result->count = 0;
        return -1;
    }

    std::memcpy(result->scores, cls.scores.data(), cls.scores.size() * sizeof(float));
    for (size_t i = 0; i < cls.labels.size(); ++i) {
        result->labels[i] = static_cast<int32_t>(cls.labels[i]);
    }
    return 0;
}

int32_t EdgeInferExtract(EdgeInferHandle handle,
                          const uint8_t* img_data, int32_t w, int32_t h, int32_t c,
                          EdgeInferEmbeddingResult* result) {
    if (!handle || !img_data || !result) return -1;

    auto* engine = ToEngine(handle);
    Image img = MakeImage(img_data, w, h, c);

    edgeinfer::EmbeddingResult emb;
    engine->Extract(img, emb);

    result->size = static_cast<int32_t>(emb.embedding.size());
    if (emb.embedding.empty()) {
        result->data = nullptr;
        return 0;
    }

    result->data = static_cast<float*>(
        std::malloc(emb.embedding.size() * sizeof(float)));
    if (!result->data) {
        result->size = 0;
        return -1;
    }

    std::memcpy(result->data, emb.embedding.data(),
                emb.embedding.size() * sizeof(float));
    return 0;
}

// ============================================================
// Batch Detection
// ============================================================

int32_t EdgeInferDetectBatch(EdgeInferHandle handle,
                              const uint8_t* const* img_data_array,
                              const int32_t* widths,
                              const int32_t* heights,
                              int32_t batch_size, int32_t c,
                              EdgeInferDetectResult* results) {
    if (!handle || !img_data_array || !widths || !heights || !results || batch_size <= 0) {
        return -1;
    }

    auto* engine = ToEngine(handle);

    for (int32_t i = 0; i < batch_size; ++i) {
        Image img = MakeImage(img_data_array[i], widths[i], heights[i], c);

        std::vector<edgeinfer::Boxf> boxes;
        engine->Detect(img, boxes);

        auto& r = results[i];
        r.count = static_cast<int32_t>(boxes.size());
        if (boxes.empty()) {
            r.boxes = nullptr;
            continue;
        }

        r.boxes = static_cast<EdgeInferBox*>(
            std::malloc(boxes.size() * sizeof(EdgeInferBox)));
        if (!r.boxes) {
            r.count = 0;
            continue;
        }

        for (size_t j = 0; j < boxes.size(); ++j) {
            r.boxes[j] = {boxes[j].x1, boxes[j].y1, boxes[j].x2, boxes[j].y2,
                          boxes[j].score,
                          static_cast<int32_t>(boxes[j].label)};
        }
    }
    return 0;
}

// ============================================================
// Result Cleanup
// ============================================================

void EdgeInferFreeDetectResult(EdgeInferDetectResult* result) {
    if (!result) return;
    std::free(result->boxes);
    result->boxes = nullptr;
    result->count = 0;
}

void EdgeInferFreeClassifyResult(EdgeInferClassifyResult* result) {
    if (!result) return;
    std::free(result->scores);
    std::free(result->labels);
    result->scores = nullptr;
    result->labels = nullptr;
    result->count = 0;
}

void EdgeInferFreeEmbeddingResult(EdgeInferEmbeddingResult* result) {
    if (!result) return;
    std::free(result->data);
    result->data = nullptr;
    result->size = 0;
}

// ============================================================
// Version
// ============================================================

const char* EdgeInferVersion(void) {
    return "1.0.0";
}
