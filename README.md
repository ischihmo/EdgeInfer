# EdgeInfer

轻量级 C/C++ 边缘推理引擎，配置驱动、后端可插拔。内置 OpenCV 互操作，`Image::FromFile()` 一行加载 JPEG/PNG/BMP/PPM。

## 特性

- **双 API** — C 接口（稳定 ABI，opaque handle）+ C++ 接口（单例 / 线程安全）
- **配置驱动** — JSON / YAML 描述模型 + 预处理 + 后处理管道
- **后端可插拔** — 适配器注册机制，支持 NCNN / MNN / CVDNN (OpenCV DNN)
- **多格式图片** — `Image::FromFile()` 支持 JPEG / PNG / BMP / PPM，OpenCV 优先 + PPM 回退
- **交叉编译友好** — `build.sh` 统一控制编译器、构建选项

## 目录结构

```
EdgeInfer/
├── include/
│   ├── EdgeInfer.h          ← C 公 API
│   ├── EdgeInfer.hpp        ← C++ 公 API
│   ├── EdgeInferInternal.h  ← 内部聚合头
│   ├── types/
│   │   ├── image.h          ← Image 结构体 + PixelFormat
│   │   ├── types.h          ← Tensor / Boxf / InferConfig …
│   │   └── config.h         ← PipelineConfig / ConfigParser 声明
│   ├── core/                ← Engine / Pipeline / Processor 工厂
│   ├── adapters/            ← IAdapter + NCNN / MNN / CVDNN 实现
│   ├── processors/          ← Resize / LetterBox / Normalize / NMS / Softmax / YoloDecode
│   ├── models/              ← Detector / Classifier / Recognizer
│   └── tools/               ← Json / Yaml 包装器
├── src/                     ← .cpp 实现 (core / adapters / processors / tools / api)
├── demo/                    ← YOLOv5n 推理示例 + 配置文件
├── third_party/
│   ├── ncnn/                ← NCNN 头文件 + libncnn.a
│   ├── mnn/                 ← MNN 头文件 + libMNN.so
│   ├── opencv/              ← OpenCV 头文件 + libopencv_world.so
│   ├── nlohmann/            ← json.hpp (单头文件)
│   └── rapidyaml/           ← rapidyaml.hpp (单头文件)
├── CMakeLists.txt
├── build.sh
└── README.md
```

## 快速开始

### 构建

```bash
cd EdgeInfer

# 默认 gcc/g++, Release, NCNN + OpenCV(CVDNN), 带 demo
./build.sh

# 启用 MNN 后端
EDGEINFER_ENABLE_MNN=ON ./build.sh

# 禁用 OpenCV (纯 PPM 图片, 无外部图像依赖)
EDGEINFER_ENABLE_OPENCV=OFF ./build.sh

# 禁用 demo
EDGEINFER_BUILD_DEMO=OFF ./build.sh

# ARM 交叉编译
CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ ./build.sh
```

产物在 `install/` 目录：

```
install/
├── bin/demo
├── include/EdgeInfer/       ← 调用方需要的头文件
├── lib/libedgeinfer.so      ← 动态库
└── share/EdgeInfer/         ← 示例配置 + 模型文件
```

### C++ 接口

```cpp
#include <EdgeInfer.hpp>

// 单例, 线程安全
auto& engine = edgeinfer::EdgeInfer::GetInstance();

// 从配置文件初始化
engine.Init("config.json");

// 加载图片 (JPEG/PNG/BMP/PPM — OpenCV 优先, PPM 回退)
edgeinfer::Image img = edgeinfer::Image::FromFile("dog.jpg");

// 检测
std::vector<edgeinfer::Boxf> boxes;
engine.Detect(img, boxes);

for (auto& b : boxes) {
    printf("[%u] rect=[%.0f %.0f %.0f %.0f] score=%.3f\n",
           b.label, b.x1, b.y1, b.x2, b.y2, b.score);
}

// 分类
edgeinfer::ClassificationResult cls;
engine.Classify(img, cls);

// 特征提取
edgeinfer::EmbeddingResult emb;
engine.Extract(img, emb);

// 批量检测
std::vector<edgeinfer::Image> images = {img1, img2, img3};
std::vector<std::vector<edgeinfer::Boxf>> batch_results;
engine.Detect(images, batch_results);

engine.Release();
```

### C 接口

```c
#include <EdgeInfer.h>

// 创建引擎
EdgeInferHandle h = EdgeInferCreate("config.json");

// 检测
EdgeInferDetectResult result;
EdgeInferDetect(h, img_data, 640, 640, 3, &result);

for (int32_t i = 0; i < result.count; ++i) {
    EdgeInferBox* b = &result.boxes[i];
    printf("[%d] rect=[%.0f %.0f %.0f %.0f] score=%.3f\n",
           b->label, b->x1, b->y1, b->x2, b->y2, b->score);
}
EdgeInferFreeDetectResult(&result);

// 批量
EdgeInferDetectBatch(h, img_ptrs, widths, heights, batch_size, 3, results);

EdgeInferDestroy(h);
```

### 配置文件格式

JSON:

```json
{
  "model": {
    "path": "model/yolov5n",
    "backend": "NCNN",
    "device": "CPU",
    "num_threads": 4
  },
  "preprocess": [
    {
      "type": "LetterBox",
      "params": {
        "target_width": 640,
        "target_height": 640,
        "max_stride": 32,
        "pad_value": 114
      }
    }
  ],
  "postprocess": [
    {
      "type": "YoloDecode",
      "params": {
        "num_classes": 80,
        "conf_threshold": 0.4,
        "nms_threshold": 0.45
      }
    }
  ],
  "task_type": "detection"
}
```

YAML 格式等效。切换后端只需改 `backend` 字段（`"NCNN"` / `"MNN"` / `"CVDNN"`）和 `path` 指向对应模型文件（`.param` / `.mnn` / `.onnx`）。

### Benchmark

Demo 首次推理后自动运行 5×100 次推理并输出耗时对比：

```
Benchmark: 5 rounds × 100 inferences
Round   Total(ms)      Avg(ms)
1         14199.2       141.99
2         14074.7       140.75
3         14312.9       143.13
4         14388.6       143.89
5         13963.1       139.63
```

### 处理器列表

| 类型 | 阶段 | 说明 |
|------|------|------|
| `LetterBox` | Pre | 等比例缩放 + stride 对齐填充 |
| `Resize` | Pre | 双线性插值缩放 |
| `Normalize` | Pre | mean/std 归一化 |
| `NV12ToBGR` | Pre | NV12 格式转 BGR |
| `YoloDecode` | Post | YOLOv5 解码 + NMS + 坐标映射 |
| `NMS` | Post | 非极大值抑制 |
| `Softmax` | Post | Softmax + Top-K 分类 |

## 构建脚本变量

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `CC` / `CXX` | `gcc` / `g++` | C/C++ 编译器 |
| `EDGEINFER_BUILD_DEMO` | `ON` | 是否编译 demo |
| `EDGEINFER_ENABLE_NCNN` | `ON` | 是否启用 NCNN 后端 |
| `EDGEINFER_ENABLE_MNN` | `OFF` | 是否启用 MNN 后端 |
| `EDGEINFER_ENABLE_OPENCV` | `ON` | 是否启用 OpenCV（含 CVDNN 后端 + 多格式图片加载）|
| `BUILD_TYPE` | `Release` | Debug / Release |
| `INSTALL_PREFIX` | `./install` | 安装目标目录 |
| `JOBS` | `nproc` | 并行编译数 |

## 依赖库

| 库 | 版本 | 用途 |
|----|------|------|
| ncnn | 20260113 | NCNN 推理后端 |
| MNN | 3.5.0 | MNN 推理后端 |
| OpenCV | 4.13.0 | CVDNN 后端 + 图片加载 |
| nlohmann/json | 3.12.0 | JSON 配置解析 |
| rapidyaml | 0.12.0 | YAML 配置解析 |

## 许可证

MIT
