//
// EdgeInfer - Configuration Parser Implementation
//

#include "../../include/types/config.h"
#include "../../include/tools/json.h"
#include "../../include/tools/yaml.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>

namespace edgeinfer {

PipelineConfig ConfigParser::Parse(const std::string& path) {
    std::string lower = path;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower.size() >= 5 && lower.compare(lower.size() - 5, 5, ".json") == 0) {
        return ParseJsonFile(path);
    }
    if (lower.size() >= 5 && (lower.compare(lower.size() - 5, 5, ".yaml") == 0
                           || lower.compare(lower.size() - 4, 4, ".yml") == 0)) {
        return ParseYamlFile(path);
    }

    // Fallback: try JSON first, then YAML
    try {
        return ParseJsonFile(path);
    } catch (...) {
        return ParseYamlFile(path);
    }
}

PipelineConfig ConfigParser::ParseString(const std::string& content) {
    size_t start = content.find_first_not_of(" \t\r\n");
    if (start != std::string::npos && content[start] == '{') {
        return ParseJsonString(content);
    }
    return ParseYamlString(content);
}

std::string ConfigParser::ReadFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("ConfigParser: cannot open " + path);
    }
    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}

// ============================================================
// JSON Parsing
// ============================================================

PipelineConfig ConfigParser::ParseJsonFile(const std::string& path) {
    return ParseJsonString(ReadFile(path));
}

PipelineConfig ConfigParser::ParseJsonString(const std::string& json) {
    using tools::Json;
    auto root = Json::Parse(json);
    PipelineConfig config;

    auto model = root.GetObject("model");
    if (!model.IsNull()) {
        config.model.path        = model.GetString("path", "");
        config.model.backend     = model.GetString("backend", "AUTO");
        config.model.device      = model.GetString("device", "CPU");
        config.model.num_threads = static_cast<int>(model.GetInt("num_threads", 1));
    }

    auto pre_items = root.GetArray("preprocess");
    for (auto& item : pre_items.ArrayItems()) {
        PreProcessItem pp;
        pp.type = item.GetString("type", "");
        auto params = item.GetObject("params");
        if (!params.IsNull()) {
            params.ForEach([&pp](const std::string& key, const Json& val) {
                pp.params[key] = val.GetDouble();
            });
        }
        if (!pp.type.empty()) {
            config.preprocess.push_back(std::move(pp));
        }
    }

    auto post_items = root.GetArray("postprocess");
    for (auto& item : post_items.ArrayItems()) {
        PostProcessItem pp;
        pp.type = item.GetString("type", "");
        auto params = item.GetObject("params");
        if (!params.IsNull()) {
            params.ForEach([&pp](const std::string& key, const Json& val) {
                pp.params[key] = val.GetDouble();
            });
        }
        if (!pp.type.empty()) {
            config.postprocess.push_back(std::move(pp));
        }
    }

    config.task_type = root.GetString("task_type", "detection");
    return config;
}

// ============================================================
// YAML Parsing
// ============================================================

PipelineConfig ConfigParser::ParseYamlFile(const std::string& path) {
    return ParseYamlString(ReadFile(path));
}

PipelineConfig ConfigParser::ParseYamlString(const std::string& yaml) {
    using tools::Yaml;
    auto root = Yaml::Parse(yaml);
    PipelineConfig config;

    auto model = root.GetMap("model");
    if (!model.IsNull()) {
        config.model.path        = model.GetString("path", "");
        config.model.backend     = model.GetString("backend", "AUTO");
        config.model.device      = model.GetString("device", "CPU");
        config.model.num_threads = static_cast<int>(model.GetInt("num_threads", 1));
    }

    auto pre_seq = root.GetSeq("preprocess");
    for (auto& item : pre_seq.SeqItems()) {
        PreProcessItem pp;
        pp.type = item.GetString("type", "");
        auto params = item.GetMap("params");
        if (!params.IsNull()) {
            params.ForEach([&pp](const std::string& key, const Yaml& val) {
                pp.params[key] = val.GetDouble();
            });
        }
        if (!pp.type.empty()) {
            config.preprocess.push_back(std::move(pp));
        }
    }

    auto post_seq = root.GetSeq("postprocess");
    for (auto& item : post_seq.SeqItems()) {
        PostProcessItem pp;
        pp.type = item.GetString("type", "");
        auto params = item.GetMap("params");
        if (!params.IsNull()) {
            params.ForEach([&pp](const std::string& key, const Yaml& val) {
                pp.params[key] = val.GetDouble();
            });
        }
        if (!pp.type.empty()) {
            config.postprocess.push_back(std::move(pp));
        }
    }

    config.task_type = root.GetString("task_type", "detection");
    return config;
}

} // namespace edgeinfer
