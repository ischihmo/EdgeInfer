//
// EdgeInfer - Pipeline Factory
// Creates InferencePipeline from JSON/YAML config
//

#ifndef EDGEINFER_FACTORY_H
#define EDGEINFER_FACTORY_H

#include "../types/config.h"
#include <memory>
#include <string>

namespace edgeinfer {

class InferencePipeline;

// ============================================================
// PipelineFactory
// ============================================================

class PipelineFactory {
public:
    static std::unique_ptr<InferencePipeline> Create(
        const std::string& config_path);

    static std::unique_ptr<InferencePipeline> Create(
        const PipelineConfig& config);
};

} // namespace edgeinfer

#endif // EDGEINFER_FACTORY_H
