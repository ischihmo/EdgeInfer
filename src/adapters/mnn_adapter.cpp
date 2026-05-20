//
// EdgeInfer - MNN Adapter Implementation (Module API + ImageProcess)
//

#include "../../include/adapters/mnn_adapter.h"
#include <ImageProcess.hpp>
#include <expr/Module.hpp>
#include <expr/ExprCreator.hpp>
#include <Tensor.hpp>
#include <MNNForwardType.h>

namespace edgeinfer {

// ============================================================
// Helpers
// ============================================================

// ============================================================
// Lifecycle
// ============================================================

MnnAdapter::MnnAdapter()  = default;
MnnAdapter::~MnnAdapter() { Release(); }

int MnnAdapter::MapPixelFormat(PixelFormat fmt) {
    switch (fmt) {
        case PixelFormat::BGR:  return MNN::CV::BGR;
        case PixelFormat::RGB:  return MNN::CV::RGB;
        case PixelFormat::GRAY: return MNN::CV::GRAY;
        case PixelFormat::RGBA: return MNN::CV::RGBA;
        case PixelFormat::BGRA: return MNN::CV::BGRA;
        default:                return MNN::CV::RGB;
    }
}

void MnnAdapter::CreateImageProcess() {
    MNN::CV::ImageProcess::Config ipConfig;
    ipConfig.sourceFormat = MNN::CV::RGB;     // input from our Image is already RGB
    ipConfig.destFormat   = MNN::CV::RGB;
    ipConfig.filterType   = MNN::CV::BILINEAR;

    // mean / normal from InferConfig
    for (int i = 0; i < 3; ++i) {
        ipConfig.mean[i]   = (i < static_cast<int>(config_.mean.size()))  ? config_.mean[i]  : 0.0f;
        ipConfig.normal[i] = (i < static_cast<int>(config_.scale.size())) ? config_.scale[i] : 1.0f;
    }
    ipConfig.mean[3]   = 0.0f;
    ipConfig.normal[3] = 1.0f;

    preprocess_.reset(MNN::CV::ImageProcess::create(ipConfig));
}

// ============================================================
// Initialize
// ============================================================

bool MnnAdapter::Initialize(const InferConfig& config) {
    config_       = config;
    input_width_  = config_.input_width;
    input_height_ = config_.input_height;

    // --- ScheduleConfig ---
    MNN::ScheduleConfig sConfig;
    sConfig.type      = MNN_FORWARD_AUTO;
    sConfig.numThread = config_.num_threads;

    // --- RuntimeManager ---
    auto* rtMgr = MNN::Express::Executor::RuntimeManager::createRuntimeManager({sConfig});
    if (!rtMgr) return false;
    auto spRtMgr = std::shared_ptr<MNN::Express::Executor::RuntimeManager>(rtMgr);

    // --- Load module ---
    MNN::Express::Module::Config mConfig;
    mConfig.shapeMutable = true;

    auto* m = MNN::Express::Module::load({}, {}, config_.model_path.c_str(), spRtMgr, &mConfig);
    if (!m) return false;

    module_.reset(m);

    // --- Create ImageProcess ---
    CreateImageProcess();

    ready_ = true;
    return true;
}

void MnnAdapter::Release() {
    ready_ = false;
    preprocess_.reset();
    module_.reset();
}

// ============================================================
// Image → Tensor
// ============================================================

Tensor MnnAdapter::ImageToTensor(const Image& img) {
    Tensor tensor;
    if (!ready_ || img.Empty()) return tensor;

    int dstW = input_width_;
    int dstH = input_height_;
    int dstC = img.channels;

    // --- Setup affine transform (resize to model input size) ---
    MNN::CV::Matrix trans;
    trans.setScale(static_cast<float>(img.width)  / static_cast<float>(dstW),
                   static_cast<float>(img.height) / static_cast<float>(dstH));
    preprocess_->setMatrix(trans);

    // --- Create dest tensor (NCHW layout) ---
    std::unique_ptr<MNN::Tensor> dstTensor(
        MNN::Tensor::create<float>({1, dstC, dstH, dstW}, nullptr, MNN::Tensor::CAFFE));

    // --- Convert: uint8 source → float normalized NCHW ---
    preprocess_->convert(img.data.data(), img.width, img.height,
                          0, dstTensor.get());

    // --- Copy to EdgeInfer Tensor ---
    tensor.shape = Shape(1, dstC, dstH, dstW);
    size_t total = tensor.shape.Size();
    tensor.data.resize(total);
    ::memcpy(tensor.data.data(), dstTensor->host<float>(), total * sizeof(float));

    return tensor;
}

// ============================================================
// Forward
// ============================================================

void MnnAdapter::Forward(const Tensor& input, std::vector<Tensor>& outputs) {
    outputs.clear();
    if (!ready_ || input.Empty()) return;

    int n = input.shape.batch;
    int c = input.shape.channels;
    int h = input.shape.height;
    int w = input.shape.width;

    // --- Create _Input VARP (NCHW) ---
    auto inp = MNN::Express::_Input({n, c, h, w}, MNN::Express::NCHW);

    // --- Write data ---
    ::memcpy(inp->writeMap<float>(), input.data.data(),
             input.data.size() * sizeof(float));

    // --- Forward ---
    auto results = module_->onForward({inp});

    // --- Read outputs ---
    outputs.reserve(results.size());
    for (auto& r : results) {
        auto* info = r->getInfo();
        if (!info || info->dim.empty()) continue;

        Tensor t;
        if (info->dim.size() == 4) {
            t.shape = Shape(info->dim[0], info->dim[1], info->dim[2], info->dim[3]);
        } else if (info->dim.size() == 3) {
            t.shape = Shape(1, info->dim[0], info->dim[1], info->dim[2]);
        } else if (info->dim.size() == 2) {
            t.shape = Shape(1, 1, info->dim[0], info->dim[1]);
        } else {
            t.shape = Shape(1, 1, 1, static_cast<int>(info->dim[0]));
        }

        size_t total = t.shape.Size();
        t.data.resize(total);
        ::memcpy(t.data.data(), r->readMap<float>(), total * sizeof(float));

        outputs.push_back(std::move(t));
    }
}

REGISTER_ADAPTER(Backend::MNN, MnnAdapter);

} // namespace edgeinfer
