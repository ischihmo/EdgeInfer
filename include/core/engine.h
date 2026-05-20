//
// EdgeInfer - Engine Core Abstraction
//

#ifndef EDGEINFER_ENGINE_H
#define EDGEINFER_ENGINE_H

#include "../types/types.h"
#include "../adapters/base_adapter.h"
#include <memory>
#include <string>

namespace edgeinfer {

// ============================================================
// Engine : bundles an adapter with its configuration
// Responsible for the forward() step of the pipeline
// ============================================================

class Engine {
public:
    Engine();
    ~Engine() = default;

    // --- Initialization ---
    bool Initialize(const InferConfig& config);
    bool IsReady() const { return ready_; }

    // --- Inference ---
    void Forward(const Tensor& input, std::vector<Tensor>& outputs);

    // --- Accessors ---
    Backend GetBackend() const { return backend_; }
    const InferConfig& GetConfig() const { return config_; }

    // --- Tensor Conversion ---
    Tensor ImageToTensor(const Image& img);

    // Disallow copy
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // Allow move
    Engine(Engine&&) = default;
    Engine& operator=(Engine&&) = default;

private:
    Backend backend_ = Backend::AUTO;
    std::unique_ptr<IAdapter> adapter_;
    InferConfig config_;
    bool ready_ = false;
};

} // namespace edgeinfer

#endif // EDGEINFER_ENGINE_H
