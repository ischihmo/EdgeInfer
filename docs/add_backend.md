# 添加推理后端

以 NCNN 适配器为例。

## 1. 创建适配器头文件

`include/adapters/xxx_adapter.h`：

```cpp
#ifndef EDGEINFER_XXX_ADAPTER_H
#define EDGEINFER_XXX_ADAPTER_H

#include "base_adapter.h"

namespace edgeinfer {

class XxxAdapter : public IAdapter {
public:
    XxxAdapter();
    ~XxxAdapter() override;

    Backend Type() const override { return Backend::XXX; }    // ← 在 types.h 添加枚举值

    bool    Initialize(const InferConfig& config) override;
    void    Forward(const Tensor& input, std::vector<Tensor>& outputs) override;
    void    Release() override;
    bool    IsReady() const override { return ready_; }
    Tensor  ImageToTensor(const Image& img) override;

    XxxAdapter(const XxxAdapter&) = delete;
    XxxAdapter& operator=(const XxxAdapter&) = delete;

private:
    InferConfig config_;
    bool        ready_ = false;
};

} // namespace edgeinfer

#endif
```

## 2. 创建适配器实现

`src/adapters/xxx_adapter.cpp`：

```cpp
#include "../../include/adapters/xxx_adapter.h"

namespace edgeinfer {

XxxAdapter::XxxAdapter()  = default;
XxxAdapter::~XxxAdapter() { Release(); }

bool XxxAdapter::Initialize(const InferConfig& config) {
    config_ = config;
    // 1. 加载模型文件 (config_.model_path)
    // 2. 配置推理选项 (线程数、精度等)
    ready_ = true;
    return true;
}

void XxxAdapter::Release() {
    ready_ = false;
    // 释放后端资源
}

Tensor XxxAdapter::ImageToTensor(const Image& img) {
    Tensor tensor;
    // 1. 将 Image (uint8_t[]) 转为框架的输入格式
    // 2. 应用 config_.mean / config_.scale 做归一化
    // 3. 转换为 EdgeInfer Tensor (NCHW, FP32)
    tensor.shape = Shape(1, img.channels, img.height, img.width);
    tensor.data.resize(tensor.shape.Size());
    // ... 填充数据 ...
    return tensor;
}

void XxxAdapter::Forward(const Tensor& input, std::vector<Tensor>& outputs) {
    outputs.clear();
    // 1. EdgeInfer Tensor → 框架输入
    // 2. 执行推理
    // 3. 框架输出 → EdgeInfer Tensor
}

// 注册 —— 必须, 否则工厂找不到
REGISTER_ADAPTER(Backend::XXX, XxxAdapter);

} // namespace edgeinfer
```

## 3. 添加 Backend 枚举值

`include/types/types.h`：

```cpp
enum class Backend : uint8_t {
    MNN = 0,
    NCNN = 1,
    CVDNN = 2,
    ONNXRUNTIME = 3,
    NPU = 4,
    XXX = 5,     // ← 新增
    AUTO = 255
};
```

## 4. 更新构建系统

`CMakeLists.txt` 添加后端开关和源文件：

```cmake
set(EDGEINFER_WITH_XXX OFF CACHE BOOL "Enable XXX backend")

if(EDGEINFER_WITH_XXX)
    list(APPEND EDGEINFER_SOURCES src/adapters/xxx_adapter.cpp)
endif()

if(EDGEINFER_WITH_XXX)
    # include / link 路径
    target_include_directories(edgeinfer PRIVATE ${XXX_ROOT}/include)
    target_link_directories(edgeinfer PRIVATE ${XXX_ROOT}/lib)
    target_link_libraries(edgeinfer PRIVATE xxx)
endif()
```

`build.sh` 添加环境变量：

```bash
ENABLE_XXX="${EDGEINFER_ENABLE_XXX:-OFF}"
# ...
-DEDGEINFER_WITH_XXX="$ENABLE_XXX"
```

`include/EdgeInferInternal.h` 添加 include：

```cpp
#include "adapters/xxx_adapter.h"
```

## 5. 接口约定

| 方法 | 职责 | 要点 |
|------|------|------|
| `Initialize` | 加载模型 | 阻塞调用, 失败返回 false |
| `ImageToTensor` | Image → Tensor | **必须**应用 mean/scale 归一化, 输出 NCHW FP32 |
| `Forward` | 推理 | 输入一个 Tensor, 输出多个 Tensor |
| `Release` | 释放资源 | 幂等, 可重复调用 |
| `IsReady` | 状态查询 | Initialize 成功后返回 true |

**归一化公式**：`output = (pixel - mean) * scale`，其中 scale 默认为 `1/255`。

## 6. 验证

编译后使用 JSON 配置切换后端：

```json
{
  "model": {
    "path": "model/your_model",
    "backend": "XXX",
    "num_threads": 4
  }
}
```

运行 demo 验证推理结果正确。
