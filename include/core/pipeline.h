//
// EdgeInfer - Inference Pipeline (Core Orchestrator)
// Reference: Design Doc v4.0 Section 4.7
//

#ifndef EDGEINFER_PIPELINE_H
#define EDGEINFER_PIPELINE_H

#include "../types/types.h"
#include "../types/config.h"
#include "processor.h"
#include "engine.h"
#include <memory>
#include <string>

namespace edgeinfer {

// ============================================================
// InferencePipeline
// ============================================================

class InferencePipeline {
public:
    InferencePipeline();
    ~InferencePipeline() = default;

    // --- Static Factory (called by PipelineFactory) ---
    static std::unique_ptr<InferencePipeline> Create(
        const std::string& config_path);
    static std::unique_ptr<InferencePipeline> Create(
        const PipelineConfig& config);

    // --- Execute Inference ---
    // Primary interface: accepts Image
    void Run(const Image& input, void* results);

    // OpenCV compatibility (optional)
#ifdef EDGEINFER_WITH_OPENCV
    void Run(const class cv::Mat& input, void* results);
#endif

    // --- Configuration Management ---
    const PipelineConfig& GetConfig() const { return config_; }
    bool Reload(const std::string& config_path);

    // --- Status ---
    bool IsReady() const { return ready_; }

    // Disallow copy
    InferencePipeline(const InferencePipeline&) = delete;
    InferencePipeline& operator=(const InferencePipeline&) = delete;

    // Allow move
    InferencePipeline(InferencePipeline&&) = default;
    InferencePipeline& operator=(InferencePipeline&&) = default;

private:
    void BuildFromConfig(const PipelineConfig& config);
    Backend ParseBackend(const std::string& name);
    void BuildPreProcessors(const std::vector<PreProcessItem>& items);
    void BuildPostProcessors(const std::vector<PostProcessItem>& items);

    void Preprocess(const Image& input, Tensor& output);
    void Forward(const Tensor& input, std::vector<Tensor>& outputs);
    void Postprocess(const std::vector<Tensor>& outputs, void* results);

    Backend backend_ = Backend::AUTO;
    std::unique_ptr<Engine> engine_;
    std::unique_ptr<PreProcessorChain> pre_processor_;
    std::unique_ptr<PostProcessorChain> post_processor_;
    Image preprocessed_;   // reused across Preprocess() calls
    PipelineConfig config_;
    bool ready_ = false;
};

} // namespace edgeinfer

#endif // EDGEINFER_PIPELINE_H
