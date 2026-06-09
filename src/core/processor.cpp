//
// EdgeInfer - Processor Core Implementation
// PreProcessorChain / PostProcessorChain / ProcessorFactory
//

#include "../../include/core/processor.h"
#include "../../include/core/preprocess.hpp"
#include "../../include/core/postprocess.hpp"

namespace edgeinfer {

// ============================================================
// PreProcessorChain
// ============================================================

void PreProcessorChain::Add(std::unique_ptr<IPreProcessor> step) {
    steps_.push_back(std::move(step));
}

void PreProcessorChain::Process(const Image& input, Image& output) {
    // Single-step fast path: skip scratch buffers
    if (steps_.size() == 1) {
        steps_[0]->Process(input, output);
        return;
    }

    // Ping-pong between two scratch buffers so input ≠ output always
    const Image* src = &input;
    for (size_t i = 0; i < steps_.size(); ++i) {
        if (i + 1 == steps_.size()) {
            steps_[i]->Process(*src, output);
        } else {
            Image* dst = &buf_[i & 1];
            steps_[i]->Process(*src, *dst);
            src = dst;
        }
    }
}

// ============================================================
// PostProcessorChain
// ============================================================

void PostProcessorChain::Add(std::unique_ptr<IPostProcessor> step) {
    steps_.push_back(std::move(step));
}

void PostProcessorChain::Process(const std::vector<Tensor>& outputs,
                                  void* results) {
    for (auto& step : steps_) {
        step->Process(outputs, results);
    }
}

// ============================================================
// ProcessorFactory
// ============================================================

std::unordered_map<std::string, ProcessorFactory::PreCreator>&
ProcessorFactory::PreCreators() {
    static std::unordered_map<std::string, PreCreator> registry;
    return registry;
}

std::unordered_map<std::string, ProcessorFactory::PostCreator>&
ProcessorFactory::PostCreators() {
    static std::unordered_map<std::string, PostCreator> registry;
    return registry;
}

void ProcessorFactory::RegisterPreProcessor(const std::string& type,
                                             PreCreator creator) {
    PreCreators()[type] = std::move(creator);
}

void ProcessorFactory::RegisterPostProcessor(const std::string& type,
                                               PostCreator creator) {
    PostCreators()[type] = std::move(creator);
}

std::unique_ptr<IPreProcessor>
ProcessorFactory::CreatePre(const std::string& type) {
    auto& registry = PreCreators();
    auto it = registry.find(type);
    if (it != registry.end()) {
        return it->second();
    }
    return nullptr;
}

std::unique_ptr<IPostProcessor>
ProcessorFactory::CreatePost(const std::string& type) {
    auto& registry = PostCreators();
    auto it = registry.find(type);
    if (it != registry.end()) {
        return it->second();
    }
    return nullptr;
}

bool ProcessorFactory::IsPreRegistered(const std::string& type) {
    return PreCreators().count(type) > 0;
}

bool ProcessorFactory::IsPostRegistered(const std::string& type) {
    return PostCreators().count(type) > 0;
}

} // namespace edgeinfer
