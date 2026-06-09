# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

All backends default to OFF — enable explicitly via env var.

```bash
cd EdgeInfer

# x86 native: NCNN + OpenCV(CVDNN) + demo
EDGEINFER_ENABLE_NCNN=ON EDGEINFER_ENABLE_OPENCV=ON ./build.sh

# x86: library only, no demo
EDGEINFER_ENABLE_NCNN=ON EDGEINFER_BUILD_DEMO=OFF ./build.sh

# ARM cross-compile (Novatek NPU)
EDGEINFER_ENABLE_NTCNN=ON EDGEINFER_BUILD_DEMO=ON ./build.sh

# ARM: custom toolchain path
TOOLCHAIN_DIR=/opt/my-tc/bin EDGEINFER_ENABLE_NTCNN=ON ./build.sh

# Debug build
BUILD_TYPE=Debug EDGEINFER_ENABLE_NCNN=ON ./build.sh
```

Output in `install/`:
- `lib/libedgeinfer.so` + third-party `.so` (MNN, OpenCV if enabled)
- `include/EdgeInfer/` (4 public headers: EdgeInfer.h, EdgeInfer.hpp, image.h, types.h)
- `bin/demo` (regular backends) or `bin/ntcnn_squeezenet` (NTCNN)

Toolchain for NTCNN defaults to `~/cross-compilier/arm-ca53-linux-gnueabihf-8.4/bin/`.
Build script auto-detects `THIRD_PARTY_ARCH` from compiler name (`arm-ca53` vs `x86`).

There are no tests. The demo is the only validation target:
- `demo/main.cpp` — YOLOv5n detection (PPM/JPEG input, NCNN/MNN/CVDNN)
- `demo/ntcnn_squeezenet.cpp` — SqueezeNet classification (NV21 input, NTCNN only)

CMake selects the demo source automatically: `EDGEINFER_WITH_NTCNN=ON` → `ntcnn_squeezenet.cpp`, else → `main.cpp`.

## Architecture

The pipeline runs: **PreProcessor chain (Image→Image) → Adapter (Image→Tensor + Forward) → PostProcessor chain (Tensor[]→results)**.

### Key invariant: normalisation path

`PreProcessorChain::Process` is `Image→Image` only. Image→Tensor conversion (with mean/scale normalisation) happens **only** in `Adapter::ImageToTensor()`. Pipeline::Preprocess always calls both: chain first (if configured), then adapter.ImageToTensor. There is no alternate path that bypasses normalisation. When adding new preprocessors or adapters, never add a direct Image→Tensor conversion outside the adapter.

### Three registration mechanisms

| Mechanism | Where defined | Where invoked |
|-----------|--------------|---------------|
| `REGISTER_PREPROCESSOR(name, class)` | `core/processor.h` | processor header (e.g. `letterbox.h`) |
| `REGISTER_POSTPROCESSOR(name, class)` | `core/processor.h` | processor header (e.g. `yolo_decode.h`) |
| `REGISTER_ADAPTER(backend, class)` | `adapters/base_adapter.h` | adapter .cpp (e.g. `ncnn_adapter.cpp`) |

Registrations fire via static initialisation when the .so loads. Adapter registrations are in `.cpp` files; processor registrations are in header files (so including the header auto-registers the processor).

### Letterbox info handoff

LetterBox preprocessor stores scale/pad/orig during Process(). Pipeline::Run() extracts them via `GetLetterboxInfo()` and passes them to postprocessors via `SetLetterboxInfo()`. This lets YoloDecode remap model-space coordinates back to original image coordinates. Both are virtual methods on IPreProcessor/IPostProcessor with default no-op implementations.

### Config-driven pipeline assembly

`PipelineConfig` (from JSON/YAML) lists preprocess/postprocess items by type name. Pipeline::BuildFromConfig creates processors via `ProcessorFactory::CreatePre/Post(type)`, then calls `SetParams()` on each. Parameters flow: JSON/YAML → `unordered_map<string,double>` → processor member variables.

### Public API

C++ API uses factory pattern (not singleton):
```cpp
auto engine = edgeinfer::EdgeInfer::Create();   // unique_ptr, independent instance
engine->Init("config.json");
engine->Detect(img, boxes);
```

C API uses opaque handles (each handle = independent instance):
```c
EdgeInferHandle h = EdgeInferCreate("config.json");
EdgeInferDetect(h, img_data, w, h, c, &result);
EdgeInferDestroy(h);
```

## Naming conventions

- Types: PascalCase (`InferConfig`, `LetterBoxProcessor`)
- Functions: PascalCase (`Process()`, `GetScale()`)
- Variables: snake_case, members have trailing underscore (`target_w_`, `scale_`)
- Files: lowercase with underscores (`letterbox.h`, `ncnn_adapter.cpp`)

## Third-party libraries

All in `third_party/`, organised by target architecture:

| Library | x86 | arm-ca53 | Type | Include |
|---------|-----|----------|------|---------|
| ncnn | `x86/ncnn/` | `arm-ca53/ncnn/` | static `.a` | `<net.h>`, `<mat.h>` |
| MNN | `x86/mnn/` | `arm-ca53/mnn/` | shared `.so` | `<ImageProcess.hpp>`, `<Interpreter.hpp>` |
| OpenCV | `x86/opencv/` | — | shared `.so` | OpenCV headers |
| NTCNN SDK | — | `arm-ca53/ntcnn/` | shared `.so` | `<hdal.h>`, `<vendor_ai.h>`, `<vendor_ai_cpu/vendor_ai_cpu.h>`, `<vendor_ai_util.h>` |
| nlohmann/json | `nlohmann/` (shared) | — | header | `<json.hpp>` |
| rapidyaml | `rapidyaml/` (shared) | — | header | `<rapidyaml.hpp>` |

nlohmann/json and rapidyaml are header-only, shared across architectures.

### NTCNN adapter notes

- `ntcnn_adapter.h` defines `UINT32` as `unsigned long` — must match the SDK's type (SDK uses `unsigned long` on ARM32, NOT `unsigned int`). Do NOT change the typedef without checking the SDK.
- SDK headers must be wrapped in `extern "C" {}`.
- String literals passed to SDK functions (e.g. `hd_common_mem_alloc`) must use local `char[]` arrays — SDK APIs take non-const `CHAR*`.
- SDK computed param IDs (e.g. `VENDOR_AI_NET_PARAM_IN(0,0)`) need explicit `(VENDOR_AI_NET_PARAM_ID)` cast in C++.
- NTCNN SDK shared libs (libhdal.so, libvendor_ai2.so, etc.) need `-Wl,-rpath-link` at build time and must be present at runtime.

### rapidyaml

Define `RYML_SINGLE_HDR_DEFINE_NOW` in exactly one TU (`src/tools/yaml.cpp`). Types are in `c4::yml::` namespace; `ryml` namespace is a using-directive alias.

## Key types

- `Image` (`types/image.h`) — Owns raw pixel data (`vector<uint8_t>`). Has width/height/channels + PixelFormat + MemoryLayout enums. `ConvertFormat()` handles BGR↔RGB, NV12→RGB, RGBA→RGB. `ConvertLayout()` handles HWC↔CHW. `FromFile()` loads JPEG/PNG/BMP/PPM (OpenCV priority, PPM fallback).
- `Tensor` (`types/types.h`) — Owns float data + Shape (batch/channels/height/width) + DataType. Always FP32, always NCHW layout after leaving the adapter.
- `InferConfig` — Backend, model_path, threads, input size, mean/scale normalisation values, letterbox info fields.
- `PipelineConfig` (`types/config.h`) — ModelConfig + vector<PreProcessItem> + vector<PostProcessItem> + task_type. Deserialised from JSON/YAML by ConfigParser.
