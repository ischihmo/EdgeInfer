//
// EdgeInfer - Engine Implementation
//

#include "../../include/core/engine.h"
#include "../../include/adapters/base_adapter.h"

namespace edgeinfer {

Engine::Engine() : backend_(Backend::AUTO), ready_(false) {}

bool Engine::Initialize(const InferConfig& config) {
    config_  = config;
    backend_ = config.backend;

    adapter_ = AdapterFactory::Create(backend_);
    if (!adapter_) {
        return false;
    }

    ready_ = adapter_->Initialize(config);
    return ready_;
}

void Engine::Forward(const Tensor& input, std::vector<Tensor>& outputs) {
    if (adapter_ && ready_) {
        adapter_->Forward(input, outputs);
    }
}

Tensor Engine::ImageToTensor(const Image& img) {
    if (adapter_) {
        return adapter_->ImageToTensor(img);
    }
    return Tensor();
}

} // namespace edgeinfer
