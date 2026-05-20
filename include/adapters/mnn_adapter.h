//
// EdgeInfer - MNN Adapter (Module API + ImageProcess)
//

#ifndef EDGEINFER_MNN_ADAPTER_H
#define EDGEINFER_MNN_ADAPTER_H

#include "base_adapter.h"
#include <memory>
#include <vector>

// Forward declare MNN types (avoid pulling in full headers here)
namespace MNN {
namespace CV    { class ImageProcess; }
namespace Express {
    class Module;
    class Executor;
}
}

namespace edgeinfer {

class MnnAdapter : public IAdapter {
public:
    MnnAdapter();
    ~MnnAdapter() override;

    Backend Type() const override { return Backend::MNN; }

    bool Initialize(const InferConfig& config) override;
    void Forward(const Tensor& input, std::vector<Tensor>& outputs) override;
    void Release() override;
    bool IsReady() const override { return ready_; }
    Tensor ImageToTensor(const Image& img) override;

    MnnAdapter(const MnnAdapter&) = delete;
    MnnAdapter& operator=(const MnnAdapter&) = delete;

private:
    static int MapPixelFormat(PixelFormat fmt);
    void CreateImageProcess();

    InferConfig                                         config_;
    std::shared_ptr<MNN::Express::Module>               module_;
    std::unique_ptr<MNN::CV::ImageProcess>              preprocess_;
    int                                                 input_width_  = 640;
    int                                                 input_height_ = 640;
    bool                                                ready_ = false;
};

} // namespace edgeinfer

#endif // EDGEINFER_MNN_ADAPTER_H
