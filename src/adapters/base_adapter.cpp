//
// EdgeInfer - Base Adapter Factory Implementation
//

#include "../../include/adapters/base_adapter.h"

namespace edgeinfer {

std::unordered_map<Backend, AdapterFactory::Creator>&
AdapterFactory::Creators() {
    static std::unordered_map<Backend, Creator> registry;
    return registry;
}

void AdapterFactory::RegisterAdapter(Backend type, Creator creator) {
    Creators()[type] = std::move(creator);
}

std::unique_ptr<IAdapter> AdapterFactory::Create(Backend type) {
    auto& registry = Creators();
    auto it = registry.find(type);
    if (it != registry.end()) {
        return it->second();
    }
    return nullptr;
}

bool AdapterFactory::IsRegistered(Backend type) {
    return Creators().count(type) > 0;
}

} // namespace edgeinfer
