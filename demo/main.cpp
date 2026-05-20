//
// EdgeInfer Demo — YOLOv5 object detection via EdgeInfer C++ API
//
// Build: see build.sh
// Usage: ./demo <image_path> [config_path]
//

#include "../include/EdgeInfer.hpp"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <string>
#include <algorithm>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <image.ppm> [config.json|config.yaml]\n",
                argv[0]);
        return 1;
    }

    std::string image_path = argv[1];
    std::string config_path = (argc >= 3) ? argv[2] : "config.json";

    // --- Load image (supports JPEG/PNG/BMP/PPM via internal OpenCV or PPM fallback) ---
    edgeinfer::Image img = edgeinfer::Image::FromFile(image_path);
    if (img.Empty()) {
        fprintf(stderr, "Failed to read image: %s\n", image_path.c_str());
        return 1;
    }
    fprintf(stdout, "Image: %dx%dx%d (%zu bytes)\n",
            img.width, img.height, img.channels, img.DataSize());

    // --- Initialize EdgeInfer ---
    auto& engine = edgeinfer::EdgeInfer::GetInstance();

    auto t0 = std::chrono::steady_clock::now();
    if (!engine.Init(config_path)) {
        fprintf(stderr, "Failed to initialize from: %s\n", config_path.c_str());
        return 1;
    }
    auto t1 = std::chrono::steady_clock::now();
    auto init_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    fprintf(stdout, "Init: %ld ms  Task: %s\n", init_ms, engine.TaskType().c_str());

    // --- Run detection ---
    std::vector<edgeinfer::Boxf> boxes;
    auto t2 = std::chrono::steady_clock::now();
    engine.Detect(img, boxes);
    auto t3 = std::chrono::steady_clock::now();
    auto infer_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();

    fprintf(stdout, "Inference: %ld ms  Detections: %zu\n", infer_ms, boxes.size());

    // --- Sort by score descending, keep top 5 ---
    std::sort(boxes.begin(), boxes.end(),
              [](const edgeinfer::Boxf& a, const edgeinfer::Boxf& b) {
                  return a.score > b.score;
              });

    // --- Print top-5 results ---
    const char* kCocoNames[] = {
        "person","bicycle","car","motorcycle","airplane","bus","train","truck",
        "boat","traffic light","fire hydrant","stop sign","parking meter",
        "bench","bird","cat","dog","horse","sheep","cow","elephant","bear",
        "zebra","giraffe","backpack","umbrella","handbag","tie","suitcase",
        "frisbee","skis","snowboard","sports ball","kite","baseball bat",
        "baseball glove","skateboard","surfboard","tennis racket","bottle",
        "wine glass","cup","fork","knife","spoon","bowl","banana","apple",
        "sandwich","orange","broccoli","carrot","hot dog","pizza","donut",
        "cake","chair","couch","potted plant","bed","dining table","toilet",
        "tv","laptop","mouse","remote","keyboard","cell phone","microwave",
        "oven","toaster","sink","refrigerator","book","clock","vase",
        "scissors","teddy bear","hair drier","toothbrush"
    };

    int topk = std::min(5, static_cast<int>(boxes.size()));
    for (int i = 0; i < topk; ++i) {
        const auto& b = boxes[i];
        const char* name = (b.label < 80) ? kCocoNames[b.label] : "?";
        fprintf(stdout, "  [%d] %s (cls=%-3u): rect=[%.0f, %.0f, %.0f, %.0f] score=%.3f\n",
                i, name, b.label,
                b.x1, b.y1, b.x2, b.y2, b.score);
    }

    // --- Benchmark: 5 rounds × 100 inferences ---
    const int kRounds = 5;
    const int kPerRound = 100;
    fprintf(stdout, "\nBenchmark: %d rounds × %d inferences\n", kRounds, kPerRound);
    fprintf(stdout, "%-6s %10s %12s\n", "Round", "Total(ms)", "Avg(ms)");
    for (int r = 0; r < kRounds; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        for (int n = 0; n < kPerRound; ++n) {
            std::vector<edgeinfer::Boxf> b;
            engine.Detect(img, b);
        }
        auto t1 = std::chrono::steady_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        fprintf(stdout, "%-6d %10.1f %12.2f\n", r + 1, total_ms, total_ms / kPerRound);
    }

    // --- Cleanup ---
    engine.Release();
    fprintf(stdout, "Done.\n");
    return 0;
}
