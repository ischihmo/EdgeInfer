//
// EdgeInfer - Novatek NT98331 NPU Adapter (ntcnn / vendor_ai SDK)
//
// Target: ARM Cortex-A53 cross-compile only
// Toolchain: arm-ca53-linux-gnueabihf-gcc / g++
//

#ifndef EDGEINFER_NTCNN_ADAPTER_H
#define EDGEINFER_NTCNN_ADAPTER_H

#include "base_adapter.h"
#include <memory>
#include <vector>



namespace edgeinfer {
// Must match SDK UINT32 type (unsigned long on ARM32)
typedef unsigned long UINT32;
class NtcnnAdapter : public IAdapter {
public:
    NtcnnAdapter();
    ~NtcnnAdapter() override;

    Backend Type() const override { return Backend::NTCNN; }

    bool    Initialize(const InferConfig& config) override;
    void    Forward(const Tensor& input, std::vector<Tensor>& outputs) override;
    void    Release() override;
    bool    IsReady() const override { return ready_; }
    Tensor  ImageToTensor(const Image& img) override;

    NtcnnAdapter(const NtcnnAdapter&) = delete;
    NtcnnAdapter& operator=(const NtcnnAdapter&) = delete;

private:
    // Map EdgeInfer PixelFormat → Novatek HD_VIDEO_PXLFMT
    static UINT32 MapPixelFormat(PixelFormat fmt);
    bool CheckFormat(PixelFormat fmt) const;

    InferConfig config_;

    // SDK state (in init order; Release checks before teardown)
    bool sdk_inited_   = false;   // vendor_ai_init      succeeded
    bool net_opened_   = false;   // vendor_ai_net_open  succeeded
    bool net_started_  = false;   // vendor_ai_net_start succeeded
    bool ready_        = false;

    UINT32 proc_id_ = 0;

    // Model / work buffers (physically-contiguous, pa = UINT32 on 32-bit ARM)
    UINT32 model_pa_   = 0;
    void*  model_va_   = nullptr;
    UINT32 model_size_ = 0;
    UINT32 work_pa_    = 0;
    void*  work_va_    = nullptr;
    UINT32 work_size_  = 0;

    // Input buffer
    UINT32 input_pa_   = 0;
    void*  input_va_   = nullptr;
    UINT32 input_size_ = 0;

    // Output layer info
    UINT32   out_buf_cnt_    = 0;
    UINT32*  out_path_list_  = nullptr;
    void*    out_bufs_vp_    = nullptr;   // VENDOR_AI_BUF array

};

} // namespace edgeinfer

#endif // EDGEINFER_NTCNN_ADAPTER_H
