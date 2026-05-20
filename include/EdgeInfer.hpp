//
// EdgeInfer - Public C++ API
//
// Thread-safe singleton interface for inference.
//
// Usage:
//   #include <EdgeInfer.hpp>
//   auto& engine = EdgeInfer::GetInstance();
//   engine.Init("config.json");
//   std::vector<Boxf> boxes;
//   engine.Detect(image, boxes);
//

#ifndef EDGEINFER_HPP
#define EDGEINFER_HPP

// Visibility control for shared library
#if defined(_WIN32)
  #define EDGEINFER_EXPORT __declspec(dllexport)
  #define EDGEINFER_IMPORT __declspec(dllimport)
#else
  #define EDGEINFER_EXPORT __attribute__((visibility("default")))
  #define EDGEINFER_IMPORT __attribute__((visibility("default")))
#endif

#ifdef EDGEINFER_BUILDING_SO
  #define EDGEINFER_API EDGEINFER_EXPORT
#else
  #define EDGEINFER_API EDGEINFER_IMPORT
#endif

#include "types/image.h"
#include "types/types.h"
#include "types/config.h"
#include <vector>
#include <string>
#include <mutex>
#include <memory>

namespace edgeinfer {

// Forward declaration
class InferencePipeline;

// ============================================================
// EdgeInfer - Thread-Safe Singleton Engine
// ============================================================

class EDGEINFER_API EdgeInfer {
public:
    // --- Singleton ---
    static EdgeInfer& GetInstance();

    // --- Factory (non-singleton, used by C API) ---
    static std::unique_ptr<EdgeInfer> Create();

    // --- Lifecycle ---
    bool Init(const std::string& config_path);
    bool Init(const PipelineConfig& config);
    bool IsReady() const;
    void Release();

    // --- Single Image: Detection ---
    void Detect(const Image& img, std::vector<Boxf>& results);

    // --- Single Image: Classification ---
    void Classify(const Image& img, ClassificationResult& result);

    // --- Single Image: Feature Extraction ---
    void Extract(const Image& img, EmbeddingResult& result);

    // --- Batch Image: Detection ---
    void Detect(const std::vector<Image>& images,
                std::vector<std::vector<Boxf>>& results);

    // --- Batch Image: Classification ---
    void Classify(const std::vector<Image>& images,
                  std::vector<ClassificationResult>& results);

    // --- Batch Image: Feature Extraction ---
    void Extract(const std::vector<Image>& images,
                 std::vector<EmbeddingResult>& results);

    // --- Query ---
    std::string TaskType() const;

    // Disallow copy / move
    EdgeInfer(const EdgeInfer&) = delete;
    EdgeInfer& operator=(const EdgeInfer&) = delete;
    EdgeInfer(EdgeInfer&&) = delete;
    EdgeInfer& operator=(EdgeInfer&&) = delete;

private:
    EdgeInfer() = default;

public:
    ~EdgeInfer();

private:
    std::vector<Boxf> DetectOne(const Image& img);
    ClassificationResult ClassifyOne(const Image& img);
    EmbeddingResult ExtractOne(const Image& img);
    void RunPipeline(const Image& img, void* results);

    mutable std::mutex mutex_;
    std::unique_ptr<InferencePipeline> pipeline_;
    std::string task_type_;
    bool initialized_ = false;
};

} // namespace edgeinfer

#endif // EDGEINFER_HPP
