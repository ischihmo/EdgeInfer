//
// EdgeInfer - YOLOv5 Decode + NMS Implementation
//

#include "../../include/processors/yolo_decode.h"
#include <cfloat>
#include <cstring>

namespace edgeinfer {

YoloDecodeProcessor::YoloDecodeProcessor() = default;

void YoloDecodeProcessor::SetParams(
    const std::unordered_map<std::string, double>& params) {
    if (params.count("num_classes"))
        num_classes_ = static_cast<int>(params.at("num_classes"));
    if (params.count("conf_threshold"))
        conf_threshold_ = static_cast<float>(params.at("conf_threshold"));
    if (params.count("nms_threshold"))
        nms_threshold_ = static_cast<float>(params.at("nms_threshold"));
}

void YoloDecodeProcessor::SetLetterboxInfo(
    float scale, int pad_left, int pad_top, int orig_w, int orig_h) {
    scale_   = scale;
    pad_left_ = pad_left;
    pad_top_  = pad_top;
    orig_w_   = orig_w;
    orig_h_   = orig_h;
    has_letterbox_info_ = true;
}

void YoloDecodeProcessor::GenerateProposals(
    const Tensor& feat, int stride,
    const std::vector<float>& anchor,
    std::vector<Boxf>& proposals) {

    // ncnn Permute 0=3 transforms conv output (w=G, h=G, c=255)
    // into (w=255, h=G, c=G). Our Tensor stores this as NCHW:
    //   Shape(1, c=G, h=G, w=255)
    // where G = spatial grid size, c dim = spatial rows, h dim = spatial cols.
    int grid_h   = feat.shape.channels;   // spatial rows
    int grid_w   = feat.shape.height;     // spatial cols
    int feat_dim = feat.shape.width;      // num_anchors * (5+num_classes) = 255
    int num_anchors = 3;
    int feat_len = 5 + num_classes_;

    for (int y = 0; y < grid_h; ++y) {
        for (int x = 0; x < grid_w; ++x) {
            for (int k = 0; k < num_anchors; ++k) {
                int base_feat = k * feat_len;

                auto get_val = [&](int f) -> float {
                    size_t idx = static_cast<size_t>(y) * grid_w * feat_dim
                               + static_cast<size_t>(x) * feat_dim
                               + static_cast<size_t>(base_feat + f);
                    return feat.data[idx];
                };

                float dx = get_val(0);
                float dy = get_val(1);
                float dw_val = get_val(2);
                float dh_val = get_val(3);
                float obj_conf = SigmoidF(get_val(4));

                if (obj_conf < 0.001f) continue;

                // Find max class score
                int max_cls = 0;
                float max_cls_score = -FLT_MAX;
                for (int c = 0; c < num_classes_; ++c) {
                    float cls_score = SigmoidF(get_val(5 + c));
                    if (cls_score > max_cls_score) {
                        max_cls_score = cls_score;
                        max_cls = c;
                    }
                }

                float score = obj_conf * max_cls_score;
                if (score < conf_threshold_) continue;

                // YOLOv5 decode
                float cx = (SigmoidF(dx) * 2.0f - 0.5f + static_cast<float>(x))
                            * static_cast<float>(stride);
                float cy = (SigmoidF(dy) * 2.0f - 0.5f + static_cast<float>(y))
                            * static_cast<float>(stride);
                float bw = std::pow(SigmoidF(dw_val) * 2.0f, 2.0f)
                            * anchor[static_cast<size_t>(k) * 2];
                float bh = std::pow(SigmoidF(dh_val) * 2.0f, 2.0f)
                            * anchor[static_cast<size_t>(k) * 2 + 1];

                Boxf box;
                box.x1 = cx - bw * 0.5f;
                box.y1 = cy - bh * 0.5f;
                box.x2 = cx + bw * 0.5f;
                box.y2 = cy + bh * 0.5f;
                box.score = score;
                box.label = static_cast<uint32_t>(max_cls);
                box.flag = true;
                proposals.push_back(box);
            }
        }
    }
}

void YoloDecodeProcessor::ApplyNms(std::vector<Boxf>& proposals,
                                    std::vector<Boxf>& results) {
    results.clear();

    // Sort by score descending
    std::sort(proposals.begin(), proposals.end(),
              [](const Boxf& a, const Boxf& b) { return a.score > b.score; });

    std::vector<bool> suppressed(proposals.size(), false);

    for (size_t i = 0; i < proposals.size(); ++i) {
        if (suppressed[i]) continue;
        results.push_back(proposals[i]);

        for (size_t j = i + 1; j < proposals.size(); ++j) {
            if (suppressed[j]) continue;
            // NMS: suppress same-class boxes with high IoU
            if (proposals[i].label == proposals[j].label) {
                if (IoU(proposals[i], proposals[j]) > nms_threshold_) {
                    suppressed[j] = true;
                }
            }
        }
    }
}

void YoloDecodeProcessor::RemapCoordinates(std::vector<Boxf>& boxes) {
    if (!has_letterbox_info_) return;

    for (auto& box : boxes) {
        box.x1 = (box.x1 - static_cast<float>(pad_left_)) / scale_;
        box.y1 = (box.y1 - static_cast<float>(pad_top_)) / scale_;
        box.x2 = (box.x2 - static_cast<float>(pad_left_)) / scale_;
        box.y2 = (box.y2 - static_cast<float>(pad_top_)) / scale_;

        box.x1 = std::max(0.0f, std::min(box.x1,
                            static_cast<float>(orig_w_)));
        box.y1 = std::max(0.0f, std::min(box.y1,
                            static_cast<float>(orig_h_)));
        box.x2 = std::max(0.0f, std::min(box.x2,
                            static_cast<float>(orig_w_)));
        box.y2 = std::max(0.0f, std::min(box.y2,
                            static_cast<float>(orig_h_)));
    }
}

void YoloDecodeProcessor::Process(const std::vector<Tensor>& outputs,
                                   void* results) {
    auto* boxes = static_cast<std::vector<Boxf>*>(results);
    boxes->clear();

    // 1. Generate proposals from each feature map
    std::vector<Boxf> proposals;
    size_t num_strides = std::min(strides_.size(), outputs.size());
    for (size_t i = 0; i < num_strides; ++i) {
        if (i >= anchors_.size()) break;
        GenerateProposals(outputs[i], strides_[i], anchors_[i], proposals);
    }

    // 2. NMS
    ApplyNms(proposals, *boxes);

    // 3. Remap coordinates from letterbox space to original image space
    RemapCoordinates(*boxes);
}

// Register
REGISTER_POSTPROCESSOR("YoloDecode", YoloDecodeProcessor);

} // namespace edgeinfer
