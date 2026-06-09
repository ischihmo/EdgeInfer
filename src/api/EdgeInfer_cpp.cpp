//
// EdgeInfer - C++ Public API Implementation
//

#include "../../include/EdgeInfer.hpp"
#include "../../include/core/pipeline.h"

namespace edgeinfer {

EdgeInfer::~EdgeInfer() = default;

// ============================================================
// Factory
// ============================================================

std::unique_ptr<EdgeInfer> EdgeInfer::Create() {
    return std::unique_ptr<EdgeInfer>(new EdgeInfer());
}

// ============================================================
// Move Semantics (std::mutex is not movable — use fresh mutex)
// ============================================================

EdgeInfer::EdgeInfer(EdgeInfer&& other) noexcept
    : mutex_()
    , pipeline_(std::move(other.pipeline_))
    , task_type_(std::move(other.task_type_))
    , initialized_(other.initialized_) {
    other.initialized_ = false;
}

EdgeInfer& EdgeInfer::operator=(EdgeInfer&& other) noexcept {
    if (this != &other) {
        pipeline_ = std::move(other.pipeline_);
        task_type_ = std::move(other.task_type_);
        initialized_ = other.initialized_;
        other.initialized_ = false;
    }
    return *this;
}

// ============================================================
// Lifecycle
// ============================================================

bool EdgeInfer::Init(const std::string& config_path) {
    auto cfg = ConfigParser::Parse(config_path);
    return Init(cfg);
}

bool EdgeInfer::Init(const PipelineConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Release previous pipeline first
    pipeline_.reset();

    pipeline_ = InferencePipeline::Create(config);
    if (!pipeline_ || !pipeline_->IsReady()) {
        pipeline_.reset();
        initialized_ = false;
        return false;
    }

    task_type_   = config.task_type;
    initialized_ = true;
    return true;
}

bool EdgeInfer::IsReady() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return initialized_ && pipeline_ && pipeline_->IsReady();
}

void EdgeInfer::Release() {
    std::lock_guard<std::mutex> lock(mutex_);
    pipeline_.reset();
    initialized_ = false;
    task_type_.clear();
}

std::string EdgeInfer::TaskType() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return task_type_;
}

// ============================================================
// Internal helpers
// ============================================================

void EdgeInfer::RunPipeline(const Image& img, void* results) {
    pipeline_->Run(img, results);
}

std::vector<Boxf> EdgeInfer::DetectOne(const Image& img) {
    std::vector<Boxf> results;
    RunPipeline(img, &results);
    return results;
}

ClassificationResult EdgeInfer::ClassifyOne(const Image& img) {
    ClassificationResult result;
    RunPipeline(img, &result);
    return result;
}

EmbeddingResult EdgeInfer::ExtractOne(const Image& img) {
    EmbeddingResult result;
    RunPipeline(img, &result);
    return result;
}

// ============================================================
// Single Image API
// ============================================================

void EdgeInfer::Detect(const Image& img, std::vector<Boxf>& results) {
    std::lock_guard<std::mutex> lock(mutex_);
    results = DetectOne(img);
}

void EdgeInfer::Classify(const Image& img, ClassificationResult& result) {
    std::lock_guard<std::mutex> lock(mutex_);
    result = ClassifyOne(img);
}

void EdgeInfer::Extract(const Image& img, EmbeddingResult& result) {
    std::lock_guard<std::mutex> lock(mutex_);
    result = ExtractOne(img);
}

// ============================================================
// Batch Image API
// ============================================================

void EdgeInfer::Detect(const std::vector<Image>& images,
                       std::vector<std::vector<Boxf>>& results) {
    std::lock_guard<std::mutex> lock(mutex_);
    results.clear();
    results.reserve(images.size());
    for (const auto& img : images) {
        results.push_back(DetectOne(img));
    }
}

void EdgeInfer::Classify(const std::vector<Image>& images,
                          std::vector<ClassificationResult>& results) {
    std::lock_guard<std::mutex> lock(mutex_);
    results.clear();
    results.reserve(images.size());
    for (const auto& img : images) {
        results.push_back(ClassifyOne(img));
    }
}

void EdgeInfer::Extract(const std::vector<Image>& images,
                         std::vector<EmbeddingResult>& results) {
    std::lock_guard<std::mutex> lock(mutex_);
    results.clear();
    results.reserve(images.size());
    for (const auto& img : images) {
        results.push_back(ExtractOne(img));
    }
}

} // namespace edgeinfer
