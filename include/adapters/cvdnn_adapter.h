//
// EdgeInfer - OpenCV DNN Adapter (blobFromImage + dnn::Net)
//

#ifndef EDGEINFER_CVDNN_ADAPTER_H
#define EDGEINFER_CVDNN_ADAPTER_H

#include "base_adapter.h"
#include <opencv2/dnn.hpp>
#include <memory>

namespace edgeinfer {

class CvDnnAdapter : public IAdapter {
public:
    CvDnnAdapter();
    ~CvDnnAdapter() override;

    Backend Type() const override { return Backend::CVDNN; }

    bool Initialize(const InferConfig& config) override;
    void Forward(const Tensor& input, std::vector<Tensor>& outputs) override;
    void Release() override;
    bool IsReady() const override { return ready_; }
    Tensor ImageToTensor(const Image& img) override;

    CvDnnAdapter(const CvDnnAdapter&) = delete;
    CvDnnAdapter& operator=(const CvDnnAdapter&) = delete;

private:
    InferConfig                             config_;
    std::unique_ptr<cv::dnn::Net>           net_;
    int                                     input_width_  = 640;
    int                                     input_height_ = 640;
    bool                                    ready_ = false;
};

} // namespace edgeinfer

#endif // EDGEINFER_CVDNN_ADAPTER_H
