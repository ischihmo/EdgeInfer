# 添加新模型

以 YOLOv5 目标检测为例，说明如何接入一个新模型。

## 1. 确定推理管道

EdgeInfer 的推理链路固定为：

```
PreProcessChain (Image→Image) → Adapter::ImageToTensor (Image→Tensor) → Forward (推理) → PostProcessChain (Tensor[]→结果)
```

接入新模型时，先梳理模型需要的预处理和后处理步骤：

| 阶段 | YOLOv5 做了什么 | 对应组件 |
|------|----------------|----------|
| 预处理 | LetterBox 缩放 + 填充 114 | `LetterBox` (已有) |
| 归一化 | 像素 / 255 | Adapter `ImageToTensor` 自动处理 |
| 推理 | 3 个输出头 (stride 8/16/32) | 框架 Adapter |
| 后处理 | 解码 + NMS + 坐标映射 | `YoloDecode` (已有) |

## 2. 编写配置文件

`demo/config.json`：

```json
{
  "model": {
    "path": "model/model_name",
    "backend": "NCNN",
    "device": "CPU",
    "num_threads": 4
  },
  "preprocess": [
    { "type": "LetterBox", "params": { "target_width": 640, "target_height": 640, "max_stride": 32, "pad_value": 114 } }
  ],
  "postprocess": [
    { "type": "YoloDecode", "params": { "num_classes": 80, "conf_threshold": 0.4, "nms_threshold": 0.45 } }
  ],
  "task_type": "detection"
}
```

配置文件通过 `ConfigParser::Parse()` 解析为 `PipelineConfig`，再由 `InferencePipeline::BuildFromConfig()` 组装管道。

## 3. 判断是否需要新处理器

### 情况 A：现有处理器够用

直接写配置文件即可，零代码。当前内置处理器：

| 类型 | 阶段 | 说明 |
|------|------|------|
| `LetterBox` | Pre | 等比例缩放 + stride 对齐填充 114 |
| `Resize` | Pre | 直接缩放到目标尺寸 |
| `Normalize` | Pre | mean/std 归一化 (通常由 Adapter 处理) |
| `NV12ToBGR` | Pre | NV12 格式转 BGR |
| `YoloDecode` | Post | YOLOv5 解码 + NMS + 坐标映射 |
| `NMS` | Post | 非极大值抑制 |
| `Softmax` | Post | Softmax + Top-K 分类 |

### 情况 B：需要新处理器

以分类模型为例，需要实现 `Softmax` 后处理：

**Step 1：创建处理器头文件** `include/processors/softmax.h`

```cpp
class SoftmaxProcessor : public IPostProcessor {
public:
    void Process(const std::vector<Tensor>& outputs, void* results) override;
    std::string Name() const override { return "Softmax"; }
    void SetParams(const std::unordered_map<std::string, double>& params) override;
private:
    int top_k_ = 5;
};

// 注册 —— 必须, 否则配置中找不到
REGISTER_POSTPROCESSOR("Softmax", SoftmaxProcessor);
```

**Step 2：实现** `src/processors/softmax.cpp`

```cpp
void SoftmaxProcessor::Process(const std::vector<Tensor>& outputs, void* results) {
    auto* result = static_cast<ClassificationResult*>(results);
    // 1. Softmax: exp(x-max) / sum(exp(x-max))
    // 2. Top-K 排序
    // 3. 填充 result->scores, result->labels
    result->flag = true;
}

void SoftmaxProcessor::SetParams(const std::unordered_map<std::string, double>& params) {
    if (params.count("top_k")) top_k_ = static_cast<int>(params.at("top_k"));
}
```

**Step 3：添加到构建**

`CMakeLists.txt`：
```cmake
set(EDGEINFER_SOURCES
    ...
    src/processors/softmax.cpp
)
```

**Step 4：在配置中引用**

```json
"postprocess": [
    { "type": "Softmax", "params": { "top_k": 5 } }
]
```

## 4. 输入数据处理

### PreProcessor 接口

```cpp
class IPreProcessor {
    virtual void Process(const Image& input, Image& output) = 0;
    virtual std::string Name() const = 0;
    virtual void SetParams(const std::unordered_map<std::string, double>& params) = 0;

    // 如果处理器需要向 PostProcessor 传递信息 (如 LetterBox 的 scale/pad)：
    virtual bool GetLetterboxInfo(float& scale, int& pad_left, int& pad_top,
                                  int& orig_w, int& orig_h) const { return false; }
};
```

### PostProcessor 接口

```cpp
class IPostProcessor {
    virtual void Process(const std::vector<Tensor>& outputs, void* results) = 0;
    virtual std::string Name() const = 0;
    virtual void SetParams(const std::unordered_map<std::string, double>& params) = 0;

    // 接收来自 PreProcessor 的信息：
    virtual void SetLetterboxInfo(float scale, int pad_left, int pad_top,
                                  int orig_w, int orig_h) {}
};
```

`results` 的实际类型由 `task_type` 决定：

| task_type | results 类型 |
|-----------|-------------|
| `detection` | `std::vector<Boxf>*` |
| `classification` | `ClassificationResult*` |
| `recognition` | `EmbeddingResult*` |

## 5. 添加模型文件

将模型文件放入 `demo/model/`，配置文件中的 `path` 相对于运行目录。YOLOv5 示例：

```
demo/model/
├── yolov5n.param      ← NCNN 模型参数
├── yolov5n.bin        ← NCNN 模型权重
├── yolov5n.mnn        ← MNN 模型
└── yolov5n.onnx       ← ONNX 模型 (OpenCV DNN)
```

## 6. 配置参数传递链路

```
config.json
  → ConfigParser::Parse()
    → PipelineConfig
      → BuildFromConfig()
        ├─ preprocess[]  → ProcessorFactory::CreatePre(type)  → SetParams(params)
        ├─ postprocess[] → ProcessorFactory::CreatePost(type) → SetParams(params)
        └─ model         → InferConfig → Adapter::Initialize()
```

`params` 中所有的 `double` 值通过 `SetParams()` 传给处理器，处理器自行转换为需要的类型。
