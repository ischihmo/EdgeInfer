//
// EdgeInfer - Novatek NT98331 NPU Adapter Implementation
//
// Uses the Novatek vendor_ai SDK (libvendor_ai2, libhdal, etc.)
// Reference: 参考开源库/ntcnn/demo.cc  (AI SDK API conventions)
//

#include "../../include/adapters/ntcnn_adapter.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif
#include "hdal.h"
#include "hd_type.h"
#include "vendor_ai.h"
#include "vendor_ai_cpu/vendor_ai_cpu.h"
#include "vendor_ai_util.h"
#ifdef __cplusplus
}
#endif

namespace edgeinfer {

// ============================================================
// Helpers
// ============================================================

static bool ReadBinaryFile(const std::string& path,
                           std::vector<char>& data) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f.seekg(0, std::ios::end);
    data.resize(f.tellg());
    f.seekg(0, std::ios::beg);
    f.read(data.data(), data.size());
    return f.good();
}

bool NtcnnAdapter::CheckFormat(PixelFormat fmt) const {
    if (fmt != PixelFormat::NV21) {
        fprintf(stderr, "[NtcnnAdapter] unsupported pixel format %d (expect NV21)\n",
                static_cast<int>(fmt));
        return false;
    }
    return true;
}

// ============================================================
// Lifecycle
// ============================================================

NtcnnAdapter::NtcnnAdapter()  = default;
NtcnnAdapter::~NtcnnAdapter() { Release(); }

// ============================================================
// Initialize  (follows 参考开源库/ntcnn/demo.cc init / network_start)
// ============================================================

bool NtcnnAdapter::Initialize(const InferConfig& config) {
    config_ = config;

    // --- Step 1: Get model size ---
    std::vector<char> model_data;
    if (!ReadBinaryFile(config_.model_path, model_data)) {
        fprintf(stderr, "[NTCNN] FAIL: read model file\n");
        return false;
    }
    model_size_ = static_cast<UINT32>(model_data.size());

    // --- Step 4: Mount CPU/DSP plugin engine ---
    vendor_ai_cfg_set(VENDOR_AI_CFG_PLUGIN_ENGINE, vendor_ai_cpu1_get_engine());

    // --- Step 5: Configure scheduler (fair) ---
    UINT32 schd = VENDOR_AI_PROC_SCHD_FAIR;
    vendor_ai_cfg_set(VENDOR_AI_CFG_PROC_SCHD, &schd);

    // --- Step 6: Enable multi-thread ---
    UINT32 multi_thread = 1;
    vendor_ai_cfg_set(VENDOR_AI_CFG_MULTI_THREAD, &multi_thread);

    // --- Step 7: Enable multi-process ---
    UINT32 multi_proc = 1;
    vendor_ai_cfg_set(VENDOR_AI_CFG_MULTI_PROCESS, &multi_proc);

    // --- Step 8: Init AI engine ---
    if (vendor_ai_init() != HD_OK) {
        fprintf(stderr, "[NTCNN] FAIL: vendor_ai_init\n");
        Release(); return false;
    }
    sdk_inited_ = true;

    // --- Step 9: Get proc ID ---
    if (vendor_ai_get_id(&proc_id_) != HD_OK) {
        fprintf(stderr, "[NTCNN] FAIL: vendor_ai_get_id\n");
        Release(); return false;
    }

    // --- Step 10: Allocate model buffer & load ---
    { char name[] = "ai_model_buf";
      if (hd_common_mem_alloc(name, &model_pa_, &model_va_,
                              model_size_, DDR_ID0) != HD_OK) {
          fprintf(stderr, "[NTCNN] FAIL: alloc model buf\n");
          Release(); return false;
      } }
    ::memcpy(model_va_, model_data.data(), model_size_);
    hd_common_mem_flush_cache(model_va_, model_size_);

    VENDOR_AI_NET_CFG_MODEL cfg_model;
    cfg_model.pa   = model_pa_;
    cfg_model.va   = reinterpret_cast<uintptr_t>(model_va_);
    cfg_model.size = model_size_;
    vendor_ai_net_set(proc_id_, VENDOR_AI_NET_PARAM_CFG_MODEL, &cfg_model);

    // --- Step 11: Buffer / Job options ---
    VENDOR_AI_NET_CFG_BUF_OPT cfg_buf_opt;
    ::memset(&cfg_buf_opt, 0, sizeof(cfg_buf_opt));
    cfg_buf_opt.method = VENDOR_AI_NET_BUF_OPT_SHRINK;
    cfg_buf_opt.ddr_id = DDR_ID0;
    vendor_ai_net_set(proc_id_, VENDOR_AI_NET_PARAM_CFG_BUF_OPT, &cfg_buf_opt);

    VENDOR_AI_NET_CFG_JOB_OPT cfg_job_opt;
    ::memset(&cfg_job_opt, 0, sizeof(cfg_job_opt));
    cfg_job_opt.method    = VENDOR_AI_NET_JOB_OPT_LINEAR_O1;
    cfg_job_opt.wait_ms   = 0;
    cfg_job_opt.schd_parm = VENDOR_AI_FAIR_CORE_ALL;
    vendor_ai_net_set(proc_id_, VENDOR_AI_NET_PARAM_CFG_JOB_OPT, &cfg_job_opt);

    // --- Step 12: Open network ---
    if (vendor_ai_net_open(proc_id_) != HD_OK) {
        fprintf(stderr, "[NTCNN] FAIL: vendor_ai_net_open\n");
        Release(); return false;
    }
    net_opened_ = true;

    // --- Query and allocate work buffer ---
    VENDOR_AI_NET_CFG_WORKBUF wbuf;
    ::memset(&wbuf, 0, sizeof(wbuf));
    vendor_ai_net_get(proc_id_, VENDOR_AI_NET_PARAM_CFG_WORKBUF, &wbuf);
    work_size_ = wbuf.size;

    { char name[] = "ai_io_buf";
      if (hd_common_mem_alloc(name, &work_pa_, &work_va_,
                               work_size_, DDR_ID0) != HD_OK) {
          fprintf(stderr, "[NTCNN] FAIL: alloc work buf\n");
          Release(); return false;
      } }
    wbuf.pa   = work_pa_;
    wbuf.va   = reinterpret_cast<uintptr_t>(work_va_);
    wbuf.size = work_size_;
    vendor_ai_net_set(proc_id_, VENDOR_AI_NET_PARAM_CFG_WORKBUF, &wbuf);

    // --- Query IO info ---
    VENDOR_AI_NET_INFO net_info;
    ::memset(&net_info, 0, sizeof(net_info));
    vendor_ai_net_get(proc_id_, VENDOR_AI_NET_PARAM_INFO, &net_info);
    out_buf_cnt_ = net_info.out_buf_cnt;

    UINT32 path_list_size = sizeof(UINT32) * out_buf_cnt_;
    out_path_list_ = static_cast<UINT32*>(::malloc(path_list_size));
    ::memset(out_path_list_, 0, path_list_size);
    vendor_ai_net_get(proc_id_, VENDOR_AI_NET_PARAM_OUT_PATH_LIST, out_path_list_);

    UINT32 out_bufs_size = sizeof(VENDOR_AI_BUF) * out_buf_cnt_;
    out_bufs_vp_ = ::malloc(out_bufs_size);
    ::memset(out_bufs_vp_, 0, out_bufs_size);

    // --- Start network ---
    if (vendor_ai_net_start(proc_id_) != HD_OK) {
        fprintf(stderr, "[NTCNN] FAIL: vendor_ai_net_start\n");
        Release(); return false;
    }
    net_started_ = true;

    ready_ = true;
    fprintf(stdout, "[NTCNN] init ok  proc_id=%lu  model=%u bytes\n",
            (unsigned long)proc_id_, model_size_);
    return true;
}

// ============================================================
// Release — guarded teardown (follows demo.cc uninit order)
// ============================================================

void NtcnnAdapter::Release() {
    ready_ = false;

    if (net_started_) { vendor_ai_net_stop(proc_id_); }
    net_started_ = false;

    ::free(out_path_list_); out_path_list_ = nullptr;
    ::free(out_bufs_vp_);   out_bufs_vp_   = nullptr;

    if (input_va_) { hd_common_mem_free(input_pa_, input_va_); input_pa_ = 0; input_va_ = nullptr; }
    if (work_va_)  { hd_common_mem_free(work_pa_,  work_va_);  work_pa_  = 0; work_va_  = nullptr; }
    if (model_va_) { hd_common_mem_free(model_pa_, model_va_); model_pa_ = 0; model_va_ = nullptr; }

    if (net_opened_) { vendor_ai_net_close(proc_id_); }
    net_opened_ = false;

    vendor_ai_release_id(proc_id_);

    if (sdk_inited_) { vendor_ai_uninit(); }
    sdk_inited_ = false;

    fprintf(stdout, "[NTCNN] Release complete\n");
}

// ============================================================
// Image → Tensor (validate format, copy to NPU input buffer)
// ============================================================

Tensor NtcnnAdapter::ImageToTensor(const Image& img) {
    Tensor tensor;
    if (!ready_ || img.Empty()) return tensor;

    if (!CheckFormat(img.format)) {
        return tensor;
    }

    UINT32 img_sz = static_cast<UINT32>(img.data.size());

    tensor.shape = Shape(1, img.channels, img.height, img.width);
    tensor.data.resize(tensor.shape.Size());

    // Store raw image as float for the pipeline contract;
    // the actual NPU input is set in Forward().
    const uint8_t* src = img.data.data();
    for (size_t i = 0; i < tensor.data.size() && i < img.data.size(); ++i)
        tensor.data[i] = static_cast<float>(src[i]);

    // Pre-copy to NPU input buffer
    if (input_size_ != img_sz)
    {
        if (input_va_) 
        {
            hd_common_mem_free(input_pa_, input_va_); input_pa_ = 0; input_va_ = nullptr;
        }
        input_size_ = static_cast<UINT32>(img_sz);
        { char name[] = "ai_in_buf";
        if (hd_common_mem_alloc(name, &input_pa_, &input_va_,
                                input_size_, DDR_ID0) != HD_OK) {
            fprintf(stderr, "[NTCNN] FAIL: alloc input buf\n");
            Release(); return tensor;
        } }
        fprintf(stdout, "[NTCNN] input buf alloc      ok  size=%u pa=0x%lx\n",
                input_size_, (unsigned long)input_pa_);
    }
    UINT32 copy_sz = img_sz < input_size_ ? img_sz : input_size_;
    ::memcpy(input_va_, img.data.data(), input_size_);
    hd_common_mem_flush_cache(input_va_, input_size_);

    return tensor;
}

// ============================================================
// Forward — push input, execute, extract outputs
// (follows 参考开源库/ntcnn/demo.cc predict / inference.cc inference)
// ============================================================

void NtcnnAdapter::Forward(const Tensor& input,
                            std::vector<Tensor>& outputs) {
    outputs.clear();
    if (!ready_) {
        fprintf(stderr, "[NTCNN] Forward: not ready\n");
        return;
    }

    // --- Push input to NPU ---
    VENDOR_AI_BUF src_img;
    ::memset(&src_img, 0, sizeof(src_img));
    src_img.sign       = MAKEFOURCC('A', 'B', 'U', 'F');
    src_img.pa         = input_pa_;
    src_img.va         = reinterpret_cast<uintptr_t>(input_va_);
    src_img.width      = static_cast<UINT32>(input.shape.width);
    src_img.height     = static_cast<UINT32>(input.shape.height);
    src_img.channel    = static_cast<UINT32>(input.shape.channels);
    src_img.fmt        = HD_VIDEO_PXLFMT_YUV420;
    src_img.line_ofs   = src_img.width;
    src_img.size       = src_img.line_ofs * src_img.height * 3 / 2;

    vendor_ai_net_set(proc_id_, (VENDOR_AI_NET_PARAM_ID)VENDOR_AI_NET_PARAM_IN(0, 0), &src_img);

    // --- Execute ---
    vendor_ai_net_proc(proc_id_);

    // --- Extract outputs ---
    auto* bufs = static_cast<VENDOR_AI_BUF*>(out_bufs_vp_);
    for (UINT32 i = 0; i < out_buf_cnt_; ++i) {
        VENDOR_AI_BUF& b = bufs[i];
        ::memset(&b, 0, sizeof(b));

        vendor_ai_net_get(proc_id_, (VENDOR_AI_NET_PARAM_ID)out_path_list_[i], &b);
        hd_common_mem_flush_cache(reinterpret_cast<void*>(b.va), b.size);

        int elem_count = static_cast<int>(b.width) * b.height *
                         b.channel * b.batch_num;

        Tensor t;
        t.shape = Shape(static_cast<int>(b.batch_num),
                        static_cast<int>(b.channel),
                        static_cast<int>(b.height),
                        static_cast<int>(b.width));
        t.data.resize(elem_count);

        vendor_ai_cpu_util_fixed2float(
            reinterpret_cast<void*>(b.va), b.fmt,
            t.data.data(), b.scale_ratio, elem_count);

        outputs.push_back(std::move(t));
    }
}

REGISTER_ADAPTER(Backend::NTCNN, NtcnnAdapter);

} // namespace edgeinfer
