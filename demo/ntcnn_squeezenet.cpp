//
// EdgeInfer NTCNN Demo — SqueezeNet fog classification (NV21 input)
//
// Target: Novatek NT98331 (ARM Cortex-A53)
// Build:  EDGEINFER_ENABLE_NTCNN=ON ./build.sh
// Usage:  ./ntcnn_squeezenet <image.nv21> [config.json]
//
// The NV21 image must be 1280×720 YUV420 semi-planar raw data.
//

#include "../include/EdgeInfer.hpp"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <string>
#include <algorithm>

// ============================================================
// Read raw NV21 file
// ============================================================
static bool ReadNV21(const std::string& path,
                     std::vector<uint8_t>& data,
                     int expect_w, int expect_h) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "Cannot open: %s\n", path.c_str());
        return false;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);

    size_t expect_sz = static_cast<size_t>(expect_w) * expect_h * 3 / 2;
    if (static_cast<size_t>(sz) != expect_sz) {
        fprintf(stderr, "NV21 size mismatch: file=%ld expected=%zux%zux3/2=%zu\n",
                sz, (size_t)expect_w, (size_t)expect_h, expect_sz);
        fclose(f);
        return false;
    }

    data.resize(sz);
    size_t n = fread(data.data(), 1, sz, f);
    fclose(f);
    return n == static_cast<size_t>(sz);
}

// ============================================================
// Main
// ============================================================
int main(int argc, char* argv[]) {
    setvbuf(stdout, NULL, _IONBF, 0);  // unbuffered — prints appear immediately

    if (argc < 3) {
        fprintf(stderr, "Usage: %s <image.nv21> [config.json]\n", argv[0]);
        return 1;
    }

    std::string image_path = argv[1];
    std::string config_path = (argc >= 3) ? argv[2] : "config_ntcnn.json";

    fprintf(stdout, "[DEMO] start, argc=%d\n", argc);

    // --- Load NV21 image (1280×720 YUV420) ---
    const int kInputW = 1280;
    const int kInputH = 720;
    std::vector<uint8_t> img_data;
    if (!ReadNV21(image_path, img_data, kInputW, kInputH)) {
        return 1;
    }
    fprintf(stdout, "Image: %dx%d NV21 (%zu bytes)\n", kInputW, kInputH, img_data.size());

    // --- Create EdgeInfer Image (NV21, 2 channels) ---
    edgeinfer::Image img(img_data.data(), kInputW, kInputH, 2,
                         edgeinfer::PixelFormat::NV21);

    // --- Init engine ---
    auto engine = edgeinfer::EdgeInfer::Create();

    auto t0 = std::chrono::steady_clock::now();
    if (!engine->Init(config_path)) {
        fprintf(stderr, "FAILED to init from: %s\n", config_path.c_str());
        return 1;
    }
    auto t1 = std::chrono::steady_clock::now();
    auto init_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    fprintf(stdout, "Init: %ld ms  Task: %s\n", init_ms, engine->TaskType().c_str());

    // --- Run classification ---
    fprintf(stdout, "\n=== First Inference ===\n");
    edgeinfer::ClassificationResult result;
    auto t2 = std::chrono::steady_clock::now();
    engine->Classify(img, result);
    auto t3 = std::chrono::steady_clock::now();
    auto infer_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();

    fprintf(stdout, "Inference: %ld ms  Top-%u\n", infer_ms, result.scores.size());

    for (size_t i = 0; i < result.scores.size(); ++i) {
        fprintf(stdout, "  [%zu] class=%-5u  score=%.4f\n",
                i, result.labels[i], result.scores[i]);
    }

    // --- Benchmark: 10 rounds × 100 inferences ---
    fprintf(stdout, "\nBenchmark: 10 rounds × 100 inferences\n");
    fprintf(stdout, "%-6s %10s %12s\n", "Round", "Total(ms)", "Avg(ms)");
    for (int r = 0; r < 10; ++r) {
        auto b0 = std::chrono::steady_clock::now();
        for (int n = 0; n < 100; ++n) {
            edgeinfer::ClassificationResult r2;
            engine->Classify(img, r2);
        }
        auto b1 = std::chrono::steady_clock::now();
        double total = std::chrono::duration<double, std::milli>(b1 - b0).count();
        fprintf(stdout, "%-6d %10.1f %12.2f\n", r + 1, total, total / 100.0);
    }

    engine->Release();
    fprintf(stdout, "Done.\n");
    return 0;
}

// {
//     "model": {
//         "path": "model/NT98331_squeezenet_fog.bin",
//         "backend": "NTCNN",
//         "device": "NPU",
//         "num_threads": 1
//     },
//     "preprocess": [],
//     "postprocess": [
//         {
//             "type": "Softmax",
//             "params": { "top_k": 5 }
//         }
//     ],
//     "task_type": "classification"
// }