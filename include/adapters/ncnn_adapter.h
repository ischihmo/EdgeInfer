//
// EdgeInfer - NCNN Adapter
//
// Model files: xxx.param + xxx.bin (model_path = "xxx")
// Include path: add -I third_party/ncnn/include to your build.
//

#ifndef EDGEINFER_NCNN_ADAPTER_H
#define EDGEINFER_NCNN_ADAPTER_H

#include "base_adapter.h"
#include <memory>
#include <vector>

#include "net.h"

namespace edgeinfer {

class NcnnAdapter : public IAdapter {
public:
    NcnnAdapter() = default;
    ~NcnnAdapter() override;

    Backend Type() const override { return Backend::NCNN; }

    bool Initialize(const InferConfig& config) override;
    void Forward(const Tensor& input, std::vector<Tensor>& outputs) override;
    void Release() override;
    bool IsReady() const override { return ready_; }
    Tensor ImageToTensor(const Image& img) override;

    NcnnAdapter(const NcnnAdapter&) = delete;
    NcnnAdapter& operator=(const NcnnAdapter&) = delete;

private:
    static int MapPixelFormat(PixelFormat fmt);

    InferConfig                 config_;
    std::unique_ptr<ncnn::Net>  net_;
    int                         input_index_ = 0;
    std::vector<int>            output_indexes_;
    bool                        ready_ = false;
};

} // namespace edgeinfer

#endif // EDGEINFER_NCNN_ADAPTER_H
