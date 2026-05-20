//
// EdgeInfer - NCNN Adapter Implementation
//

#include "../../include/adapters/ncnn_adapter.h"
#include <cstring>

namespace edgeinfer {

NcnnAdapter::~NcnnAdapter() { Release(); }

// static
int NcnnAdapter::MapPixelFormat(PixelFormat fmt) {
    switch (fmt) {
        case PixelFormat::BGR:  return ncnn::Mat::PIXEL_BGR2RGB;
        case PixelFormat::RGB:  return ncnn::Mat::PIXEL_RGB;
        case PixelFormat::GRAY: return ncnn::Mat::PIXEL_GRAY;
        case PixelFormat::RGBA: return ncnn::Mat::PIXEL_RGBA2RGB;
        case PixelFormat::BGRA: return ncnn::Mat::PIXEL_BGRA2RGB;
        default:                return ncnn::Mat::PIXEL_BGR2RGB;
    }
}

bool NcnnAdapter::Initialize(const InferConfig& config) {
    config_ = config;
    net_ = std::make_unique<ncnn::Net>();

    // --- Configure options ---
    net_->opt.num_threads = config_.num_threads;
    net_->opt.use_fp16_packed = false;
    net_->opt.use_fp16_storage = false;
    net_->opt.use_fp16_arithmetic = false;

    // --- Load model (.param + .bin) ---
    std::string param_path = config_.model_path + ".param";
    std::string bin_path   = config_.model_path + ".bin";

    if (net_->load_param(param_path.c_str()) != 0) {
        net_.reset();
        return false;
    }
    if (net_->load_model(bin_path.c_str()) != 0) {
        net_.reset();
        return false;
    }

    // --- Cache I/O indexes ---
    const auto& input_indexes  = net_->input_indexes();
    const auto& output_indexes = net_->output_indexes();

    if (input_indexes.empty() || output_indexes.empty()) {
        net_.reset();
        return false;
    }

    input_index_    = input_indexes[0];
    output_indexes_ = output_indexes;

    ready_ = true;
    return true;
}

void NcnnAdapter::Forward(const Tensor& input, std::vector<Tensor>& outputs) {
    outputs.clear();
    if (!ready_ || !net_) return;

    // --- Convert Tensor -> ncnn::Mat ---
    const auto& shape = input.shape;
    ncnn::Mat ncnn_input;
    if (shape.batch == 1) {
        ncnn_input = ncnn::Mat(shape.width, shape.height, shape.channels);
    } else {
        ncnn_input = ncnn::Mat(shape.width, shape.height,
                                shape.channels * shape.batch);
    }
    std::memcpy(ncnn_input.data, input.data.data(),
                input.data.size() * sizeof(float));

    // --- Create Extractor & run ---
    ncnn::Extractor ex = net_->create_extractor();
    ex.set_light_mode(true);
    ex.input(input_index_, ncnn_input);

    // --- Extract outputs ---
    outputs.resize(output_indexes_.size());
    for (size_t i = 0; i < output_indexes_.size(); ++i) {
        ncnn::Mat ncnn_output;
        ex.extract(output_indexes_[i], ncnn_output);

        Tensor& t = outputs[i];
        if (ncnn_output.dims == 3) {
            t.shape = Shape(1, ncnn_output.c, ncnn_output.h, ncnn_output.w);
        } else if (ncnn_output.dims == 2) {
            t.shape = Shape(1, 1, ncnn_output.h, ncnn_output.w);
        } else {
            t.shape = Shape(1, 1, 1, ncnn_output.w);
        }

        size_t total = t.shape.Size();
        t.data.resize(total);
        std::memcpy(t.data.data(), ncnn_output.data, total * sizeof(float));
    }
}

void NcnnAdapter::Release() {
    ready_ = false;
    output_indexes_.clear();
    if (net_) {
        net_->clear();
        net_.reset();
    }
}

Tensor NcnnAdapter::ImageToTensor(const Image& img) {
    if (img.Empty()) return Tensor();

    Tensor tensor;
    int pix_type = MapPixelFormat(img.format);

    // Use ncnn's built-in from_pixels (automatic HWC->CHW layout conversion)
    ncnn::Mat ncnn_mat;
    if (img.width != config_.input_width || img.height != config_.input_height) {
        ncnn_mat = ncnn::Mat::from_pixels_resize(
            img.data.data(), pix_type,
            img.width, img.height,
            config_.input_width, config_.input_height);
    } else {
        ncnn_mat = ncnn::Mat::from_pixels(
            img.data.data(), pix_type,
            img.width, img.height);
    }

    // --- Apply mean/scale normalization ---
    // ncnn formula: output = (input - mean) * norm
    // Our config: output = (input - mean) * scale  (scale defaults to 1/255)
    if (!config_.mean.empty() && !config_.scale.empty()) {
        float mean_vals[3] = {0, 0, 0};
        float norm_vals[3] = {1.0f, 1.0f, 1.0f};
        for (int i = 0; i < 3 && i < static_cast<int>(config_.mean.size()); ++i) {
            mean_vals[i] = config_.mean[i];
            norm_vals[i] = (i < static_cast<int>(config_.scale.size()))
                               ? config_.scale[i] : 1.0f;
        }
        ncnn_mat.substract_mean_normalize(mean_vals, norm_vals);
    }

    // --- Convert to Tensor ---
    tensor.shape = Shape(1, ncnn_mat.c, ncnn_mat.h, ncnn_mat.w);
    tensor.data.resize(tensor.shape.Size());
    std::memcpy(tensor.data.data(), ncnn_mat.data,
                tensor.data.size() * sizeof(float));

    return tensor;
}

// Auto-register
REGISTER_ADAPTER(Backend::NCNN, NcnnAdapter);

} // namespace edgeinfer
