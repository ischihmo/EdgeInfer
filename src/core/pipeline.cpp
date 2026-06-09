//
// EdgeInfer - Inference Pipeline Implementation
//

#include "../../include/core/pipeline.h"
#include "../../include/core/engine.h"
#include "../../include/core/processor.h"
#include "../../include/core/factory.h"

namespace edgeinfer {

InferencePipeline::InferencePipeline()
    : backend_(Backend::AUTO), ready_(false) {}

// --- Static Factory ---

std::unique_ptr<InferencePipeline>
InferencePipeline::Create(const std::string& config_path) {
    auto config = ConfigParser::Parse(config_path);
    return Create(config);
}

std::unique_ptr<InferencePipeline>
InferencePipeline::Create(const PipelineConfig& config) {
    auto pipeline = std::unique_ptr<InferencePipeline>(
        new InferencePipeline());
    pipeline->BuildFromConfig(config);
    return pipeline;
}

// --- Build from Config ---

void InferencePipeline::BuildFromConfig(const PipelineConfig& config) {
    config_  = config;
    backend_ = ParseBackend(config.model.backend);

    engine_ = std::make_unique<Engine>();
    InferConfig infer_config;
    infer_config.backend    = backend_;
    infer_config.model_path = config_.model.path;
    infer_config.num_threads = config_.model.num_threads;

    ready_ = engine_->Initialize(infer_config);

    pre_processor_  = std::make_unique<PreProcessorChain>();
    post_processor_ = std::make_unique<PostProcessorChain>();

    BuildPreProcessors(config_.preprocess);
    BuildPostProcessors(config_.postprocess);
}

Backend InferencePipeline::ParseBackend(const std::string& name) {
    if (name == "MNN")           return Backend::MNN;
    if (name == "NCNN")          return Backend::NCNN;
    if (name == "CVDNN")         return Backend::CVDNN;
    if (name == "ONNX"
        || name == "ONNXRUNTIME") return Backend::ONNXRUNTIME;
    if (name == "NTCNN")         return Backend::NTCNN;
    return Backend::AUTO;
}

void InferencePipeline::BuildPreProcessors(
    const std::vector<PreProcessItem>& items) {
    for (const auto& item : items) {
        auto proc = ProcessorFactory::CreatePre(item.type);
        if (proc) {
            proc->SetParams(item.params);
            pre_processor_->Add(std::move(proc));
        }
    }
}

void InferencePipeline::BuildPostProcessors(
    const std::vector<PostProcessItem>& items) {
    for (const auto& item : items) {
        auto proc = ProcessorFactory::CreatePost(item.type);
        if (proc) {
            proc->SetParams(item.params);
            post_processor_->Add(std::move(proc));
        }
    }
}

// --- Run ---

void InferencePipeline::Run(const Image& input, void* results) {
    Tensor tensor;
    Preprocess(input, tensor);

    // --- Pass letterbox info from pre- to post-processors ---
    if (pre_processor_) {
        for (const auto& step : pre_processor_->Steps()) {
            float s; int pl, pt, ow, oh;
            if (step->GetLetterboxInfo(s, pl, pt, ow, oh)) {
                if (post_processor_) {
                    for (auto& ps : post_processor_->Steps()) {
                        ps->SetLetterboxInfo(s, pl, pt, ow, oh);
                    }
                }
                break;
            }
        }
    }

    std::vector<Tensor> outputs;
    Forward(tensor, outputs);

    Postprocess(outputs, results);
}

bool InferencePipeline::Reload(const std::string& config_path) {
    auto new_pipeline = Create(config_path);
    if (!new_pipeline || !new_pipeline->IsReady()) {
        return false;
    }
    *this = std::move(*new_pipeline);
    return true;
}

// --- Pipeline Stages ---

void InferencePipeline::Preprocess(const Image& input, Tensor& output) {
    // Step 1: preprocessor chain — Image → Image (optional)
    if (pre_processor_ && !pre_processor_->Empty()) {
        pre_processor_->Process(input, preprocessed_);
        if (engine_) {
            output = engine_->ImageToTensor(preprocessed_);
        }
    } else if (engine_) {
        // Fast path: no preprocessor — ImageToTensor directly from input
        output = engine_->ImageToTensor(input);
    }
}

void InferencePipeline::Forward(const Tensor& input,
                                  std::vector<Tensor>& outputs) {
    if (engine_ && engine_->IsReady()) {
        engine_->Forward(input, outputs);
    }
}

void InferencePipeline::Postprocess(const std::vector<Tensor>& outputs,
                                       void* results) {
    if (post_processor_ && !post_processor_->Empty()) {
        post_processor_->Process(outputs, results);
    }
}

// ============================================================
// PipelineFactory
// ============================================================

std::unique_ptr<InferencePipeline>
PipelineFactory::Create(const std::string& config_path) {
    return InferencePipeline::Create(config_path);
}

std::unique_ptr<InferencePipeline>
PipelineFactory::Create(const PipelineConfig& config) {
    return InferencePipeline::Create(config);
}

} // namespace edgeinfer
