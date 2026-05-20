//
// EdgeInfer - JSON/YAML Configuration Structures
// Reference: Design Doc v4.0 Section 4.3
//

#ifndef EDGEINFER_CONFIG_H
#define EDGEINFER_CONFIG_H

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace edgeinfer {

// ============================================================
// Model Configuration
// ============================================================

struct ModelConfig {
    std::string path;            // model file path
    std::string backend = "AUTO"; // MNN / NCNN / ONNX / CVDNN / NPU
    std::string device = "CPU";   // CPU / GPU / VULKAN / NPU
    int num_threads = 1;
};

// ============================================================
// PreProcess / PostProcess Configuration Items
// ============================================================

struct PreProcessItem {
    std::string type;                                // e.g. Resize / Normalize / LetterBox
    std::unordered_map<std::string, double> params;  // key-value parameters
};

struct PostProcessItem {
    std::string type;                                // e.g. NMS / Softmax / Decode
    std::unordered_map<std::string, double> params;
};

// ============================================================
// Complete Pipeline Configuration
// ============================================================

struct PipelineConfig {
    ModelConfig model;
    std::vector<PreProcessItem> preprocess;
    std::vector<PostProcessItem> postprocess;
    std::string task_type;  // detection / classification / recognition
};

// ============================================================
// Config Parser (stub - implementation deferred)
// ============================================================

class ConfigParser {
public:
    static PipelineConfig Parse(const std::string& path);
    static PipelineConfig ParseString(const std::string& content);

private:
    static PipelineConfig ParseJsonFile(const std::string& path);
    static PipelineConfig ParseYamlFile(const std::string& path);
    static PipelineConfig ParseJsonString(const std::string& json);
    static PipelineConfig ParseYamlString(const std::string& yaml);
    static std::string ReadFile(const std::string& path);
};

} // namespace edgeinfer

#endif // EDGEINFER_CONFIG_H
