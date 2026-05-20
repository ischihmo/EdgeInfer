//
// EdgeInfer - Adapter Base Class & Factory
// Reference: Design Doc v4.0 Section 4.6
//

#ifndef EDGEINFER_BASE_ADAPTER_H
#define EDGEINFER_BASE_ADAPTER_H

#include "../types/types.h"
#include "../types/image.h"
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>

namespace edgeinfer {

// ============================================================
// Adapter Interface
// ============================================================

class IAdapter {
public:
    virtual ~IAdapter() = default;

    virtual Backend Type() const = 0;
    virtual bool Initialize(const InferConfig& config) = 0;
    virtual void Forward(const Tensor& input,
                         std::vector<Tensor>& outputs) = 0;
    virtual void Release() = 0;
    virtual bool IsReady() const = 0;
    virtual Tensor ImageToTensor(const Image& img) = 0;
};

// ============================================================
// Adapter Factory
// ============================================================

class AdapterFactory {
public:
    using Creator = std::function<std::unique_ptr<IAdapter>()>;

    static void RegisterAdapter(Backend type, Creator creator);
    static std::unique_ptr<IAdapter> Create(Backend type);
    static bool IsRegistered(Backend type);

private:
    static std::unordered_map<Backend, Creator>& Creators();
};

} // namespace edgeinfer

// ============================================================
// Registration Macro
// ============================================================

#define REGISTER_ADAPTER(backend_type, class_name)                         \
    static bool _reg_adapter_##class_name = []() {                         \
        edgeinfer::AdapterFactory::RegisterAdapter(                        \
            backend_type,                                                  \
            []() { return std::make_unique<class_name>(); });              \
        return true;                                                       \
    }()

#endif // EDGEINFER_BASE_ADAPTER_H
