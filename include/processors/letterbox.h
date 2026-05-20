//
// EdgeInfer - LetterBox PreProcessor
// Aspect-ratio-preserving resize with stride-aligned padding.
//

#ifndef EDGEINFER_LETTERBOX_PROCESSOR_H
#define EDGEINFER_LETTERBOX_PROCESSOR_H

#include "../core/processor.h"
#include "../types/types.h"
#include <string>
#include <unordered_map>

namespace edgeinfer {

class LetterBoxProcessor : public IPreProcessor {
public:
    LetterBoxProcessor();

    void Process(const Image& input, Image& output) override;
    std::string Name() const override { return "LetterBox"; }
    void SetParams(const std::unordered_map<std::string, double>& params) override;

    bool GetLetterboxInfo(float& scale, int& pad_left, int& pad_top,
                          int& orig_w, int& orig_h) const override;

    float GetScale() const { return scale_; }
    int   GetPadLeft() const { return pad_left_; }
    int   GetPadTop() const  { return pad_top_; }
    int   GetTargetW() const { return target_w_; }
    int   GetTargetH() const { return target_h_; }

private:
    void BilinearResize(const Image& src, Image& dst, int dw, int dh);

    int     target_w_ = 640;
    int     target_h_ = 640;
    int     max_stride_ = 32;
    uint8_t pad_value_ = 114;

    // State after Process
    float scale_ = 1.0f;
    int   pad_left_ = 0;
    int   pad_top_ = 0;
    int   orig_w_ = 0;
    int   orig_h_ = 0;
};

REGISTER_PREPROCESSOR("LetterBox", LetterBoxProcessor);

} // namespace edgeinfer

#endif // EDGEINFER_LETTERBOX_PROCESSOR_H
