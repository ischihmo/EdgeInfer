//
// EdgeInfer - NV12 to BGR Conversion PreProcessor
// Supports hardware-decoded NV12 format commonly used in embedded cameras
//

#ifndef EDGEINFER_NV12_TO_BGR_PROCESSOR_H
#define EDGEINFER_NV12_TO_BGR_PROCESSOR_H

#include "../core/processor.h"

namespace edgeinfer {

class Nv12ToBgrProcessor : public IPreProcessor {
public:
    void Process(const Image& input, Image& output) override {
        if (input.format != PixelFormat::NV12) {
            return; // error: expected NV12 input
        }

        // NV12: Y plane (WxH) followed by UV interleaved plane (WxH/2)
        // TODO: implement YUV→BGR conversion
        // For now, create placeholder BGR output
        output = Image(input.width, input.height, 3, PixelFormat::BGR);
    }

    std::string Name() const override { return "NV12ToBGR"; }

    void SetParams(const std::unordered_map<std::string, double>& /*params*/) override {}
};

REGISTER_PREPROCESSOR("NV12ToBGR", Nv12ToBgrProcessor);

} // namespace edgeinfer

#endif // EDGEINFER_NV12_TO_BGR_PROCESSOR_H
