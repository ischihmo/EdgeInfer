//
// EdgeInfer - Model Base Class
//

#ifndef EDGEINFER_BASE_MODEL_H
#define EDGEINFER_BASE_MODEL_H

#include "../core/pipeline.h"
#include "../types/config.h"
#include <memory>
#include <string>

namespace edgeinfer {

// ============================================================
// BaseModel - wraps an InferencePipeline for task-specific models
// ============================================================

class BaseModel {
public:
    BaseModel() = default;
    virtual ~BaseModel() = default;

    virtual bool Load(const std::string& config_path);
    virtual bool Load(const PipelineConfig& config);

    virtual bool IsReady() const { return pipeline_ != nullptr && pipeline_->IsReady(); }
    virtual std::string TaskType() const = 0;

    // Reload from new config (for remote updates)
    virtual bool Reload(const std::string& config_path);

    // Disallow copy
    BaseModel(const BaseModel&) = delete;
    BaseModel& operator=(const BaseModel&) = delete;

protected:
    std::unique_ptr<InferencePipeline> pipeline_;
};

// ============================================================
// Inline Implementation
// ============================================================

inline bool BaseModel::Load(const std::string& config_path) {
    pipeline_ = InferencePipeline::Create(config_path);
    return pipeline_ != nullptr && pipeline_->IsReady();
}

inline bool BaseModel::Load(const PipelineConfig& config) {
    pipeline_ = InferencePipeline::Create(config);
    return pipeline_ != nullptr && pipeline_->IsReady();
}

inline bool BaseModel::Reload(const std::string& config_path) {
    pipeline_ = InferencePipeline::Create(config_path);
    return pipeline_ != nullptr;
}

} // namespace edgeinfer

#endif // EDGEINFER_BASE_MODEL_H
