//
// EdgeInfer - Processor Base Classes & Factory
// Reference: Design Doc v4.0 Section 4.4
//

#ifndef EDGEINFER_PROCESSOR_H
#define EDGEINFER_PROCESSOR_H

#include "../types/image.h"
#include "../types/types.h"
#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include <functional>

namespace edgeinfer {

// ============================================================
// PreProcessor Interface
// ============================================================

class IPreProcessor {
public:
    virtual ~IPreProcessor() = default;

    virtual void Process(const Image& input, Image& output) = 0;
    virtual std::string Name() const = 0;
    virtual void SetParams(
        const std::unordered_map<std::string, double>& params) = 0;

    // Letterbox info for coordinate remap (default: not a letterbox)
    virtual bool GetLetterboxInfo(
        float& /*scale*/, int& /*pad_left*/, int& /*pad_top*/,
        int& /*orig_w*/, int& /*orig_h*/) const { return false; }
};

// ============================================================
// PostProcessor Interface
// ============================================================

class IPostProcessor {
public:
    virtual ~IPostProcessor() = default;

    virtual void Process(const std::vector<Tensor>& outputs,
                         void* results) = 0;
    virtual std::string Name() const = 0;
    virtual void SetParams(
        const std::unordered_map<std::string, double>& params) = 0;

    // Receive letterbox info for coordinate remap (default: no-op)
    virtual void SetLetterboxInfo(
        float /*scale*/, int /*pad_left*/, int /*pad_top*/,
        int /*orig_w*/, int /*orig_h*/) {}
};

// ============================================================
// PreProcessor Chain
// ============================================================

class PreProcessorChain {
public:
    void Add(std::unique_ptr<IPreProcessor> step);
    void Process(const Image& input, Image& output);
    size_t Size() const { return steps_.size(); }
    bool Empty() const { return steps_.empty(); }

    const std::vector<std::unique_ptr<IPreProcessor>>& Steps() const {
        return steps_;
    }

private:
    std::vector<std::unique_ptr<IPreProcessor>> steps_;
    Image buf_[2];  // ping-pong scratch buffers (never overlap input/output)
};

// ============================================================
// PostProcessor Chain
// ============================================================

class PostProcessorChain {
public:
    void Add(std::unique_ptr<IPostProcessor> step);
    void Process(const std::vector<Tensor>& outputs, void* results);
    size_t Size() const { return steps_.size(); }
    bool Empty() const { return steps_.empty(); }

    const std::vector<std::unique_ptr<IPostProcessor>>& Steps() const {
        return steps_;
    }

private:
    std::vector<std::unique_ptr<IPostProcessor>> steps_;
};

// ============================================================
// Processor Factory (registration-based)
// ============================================================

class ProcessorFactory {
public:
    using PreCreator = std::function<std::unique_ptr<IPreProcessor>()>;
    using PostCreator = std::function<std::unique_ptr<IPostProcessor>()>;

    static void RegisterPreProcessor(const std::string& type,
                                     PreCreator creator);
    static void RegisterPostProcessor(const std::string& type,
                                      PostCreator creator);

    static std::unique_ptr<IPreProcessor> CreatePre(const std::string& type);
    static std::unique_ptr<IPostProcessor> CreatePost(const std::string& type);

    static bool IsPreRegistered(const std::string& type);
    static bool IsPostRegistered(const std::string& type);

private:
    static std::unordered_map<std::string, PreCreator>& PreCreators();
    static std::unordered_map<std::string, PostCreator>& PostCreators();
};

} // namespace edgeinfer

// ============================================================
// Registration Macros
// ============================================================

#define REGISTER_PREPROCESSOR(type_name, class_name)                       \
    static bool _reg_pre_##class_name = []() {                             \
        edgeinfer::ProcessorFactory::RegisterPreProcessor(                 \
            type_name, []() { return std::make_unique<class_name>(); });   \
        return true;                                                       \
    }()

#define REGISTER_POSTPROCESSOR(type_name, class_name)                      \
    static bool _reg_post_##class_name = []() {                            \
        edgeinfer::ProcessorFactory::RegisterPostProcessor(                \
            type_name, []() { return std::make_unique<class_name>(); });   \
        return true;                                                       \
    }()

#endif // EDGEINFER_PROCESSOR_H
