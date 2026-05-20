//
// EdgeInfer - OpenCV DNN Adapter Implementation
//

#include "../../include/adapters/cvdnn_adapter.h"
#include <opencv2/imgproc.hpp>

namespace edgeinfer {

CvDnnAdapter::CvDnnAdapter()  = default;
CvDnnAdapter::~CvDnnAdapter() { Release(); }

// ============================================================
// Initialize
// ============================================================

bool CvDnnAdapter::Initialize(const InferConfig& config) {
    config_       = config;
    input_width_  = config_.input_width;
    input_height_ = config_.input_height;

    try {
        auto net = std::make_unique<cv::dnn::Net>(
            cv::dnn::readNet(config_.model_path));
        net->setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        net->setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        cv::setNumThreads(std::max(1, config_.num_threads));
        net_ = std::move(net);
    } catch (const cv::Exception& e) {
        return false;
    }

    ready_ = true;
    return true;
}

void CvDnnAdapter::Release() {
    ready_ = false;
    net_.reset();
}

// ============================================================
// Image → Tensor (blobFromImage handles BGR→RGB + normalize + NCHW)
// ============================================================

Tensor CvDnnAdapter::ImageToTensor(const Image& img) {
    Tensor tensor;
    if (!ready_ || img.Empty()) return tensor;

    // Convert EdgeInfer Image → cv::Mat
    int cvType = CV_8UC(img.channels);
    cv::Mat cvImg(img.height, img.width, cvType,
                  const_cast<uint8_t*>(img.data.data()));

    // Resize to fixed model input size (ONNX graphs require exact dimensions)
    cv::Mat resized;
    cv::resize(cvImg, resized, cv::Size(input_width_, input_height_),
               0, 0, cv::INTER_LINEAR);

    // blobFromImage: scale + mean + swapRB + NCHW in one call
    cv::Scalar mean(config_.mean.size() >= 3 ? config_.mean[0] : 0.0,
                    config_.mean.size() >= 3 ? config_.mean[1] : 0.0,
                    config_.mean.size() >= 3 ? config_.mean[2] : 0.0);
    double scale = config_.scale.size() >= 1 ? config_.scale[0] : 1.0;

    cv::Mat blob = cv::dnn::blobFromImage(
        resized, scale, cv::Size(input_width_, input_height_),
        mean, true, false);  // swapRB=true: BGR→RGB, crop=false

    // blob is NCHW float Mat (4D) — copy to Tensor
    int n = blob.size[0], c = blob.size[1],
        h = blob.size[2], w = blob.size[3];
    tensor.shape = Shape(n, c, h, w);
    tensor.data.resize(tensor.shape.Size());
    ::memcpy(tensor.data.data(), blob.ptr<float>(), tensor.data.size() * sizeof(float));

    return tensor;
}

// ============================================================
// Forward
// ============================================================

void CvDnnAdapter::Forward(const Tensor& input, std::vector<Tensor>& outputs) {
    outputs.clear();
    if (!ready_ || input.Empty()) return;

    // Wrap input tensor data as cv::Mat 4D blob
    int sizes[] = {input.shape.batch, input.shape.channels,
                   input.shape.height, input.shape.width};
    cv::Mat blob(4, sizes, CV_32F, const_cast<float*>(input.data.data()));

    net_->setInput(blob);

    std::vector<cv::String> outNames = net_->getUnconnectedOutLayersNames();
    std::vector<cv::Mat> cvOuts;
    net_->forward(cvOuts, outNames);

    // Sort outputs by spatial size (descending) — stride 8, 16, 32
    std::sort(cvOuts.begin(), cvOuts.end(),
              [](const cv::Mat& a, const cv::Mat& b) {
                  return (a.size[2] * a.size[3]) > (b.size[2] * b.size[3]);
              });

    // Convert to EdgeInfer Tensor
    for (auto& cvOut : cvOuts) {
        Tensor t;
        t.shape = Shape(cvOut.size[0], cvOut.size[1], cvOut.size[2], cvOut.size[3]);
        t.data.resize(t.shape.Size());
        ::memcpy(t.data.data(), cvOut.ptr<float>(), t.data.size() * sizeof(float));
        outputs.push_back(std::move(t));
    }
}

REGISTER_ADAPTER(Backend::CVDNN, CvDnnAdapter);

} // namespace edgeinfer
