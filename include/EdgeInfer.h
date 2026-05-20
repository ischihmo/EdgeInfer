//
// EdgeInfer - Public C API (stable ABI)
//

#ifndef EDGEINFER_H
#define EDGEINFER_H

#include <stdint.h>

// ── Visibility ────────────────────────────────────────────
#if defined(_WIN32)
  #define EDGEINFER_C_API __declspec(dllexport)
#else
  #define EDGEINFER_C_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ── Opaque Handle ─────────────────────────────────────────
typedef void* EdgeInferHandle;

// ── Result Types ──────────────────────────────────────────
typedef struct { float x1, y1, x2, y2; float score; int32_t label; } EdgeInferBox;
typedef struct { EdgeInferBox* boxes; int32_t count; } EdgeInferDetectResult;
typedef struct { float* scores; int32_t* labels; int32_t count; } EdgeInferClassifyResult;
typedef struct { float* data; int32_t size; } EdgeInferEmbeddingResult;

// ── Lifecycle ─────────────────────────────────────────────
EDGEINFER_C_API EdgeInferHandle EdgeInferCreate(const char* config_path);
EDGEINFER_C_API EdgeInferHandle EdgeInferCreateFromJson(const char* json_config);
EDGEINFER_C_API void            EdgeInferDestroy(EdgeInferHandle handle);

// ── Single Image ──────────────────────────────────────────
EDGEINFER_C_API int32_t EdgeInferDetect(EdgeInferHandle handle,
    const uint8_t* img_data, int32_t w, int32_t h, int32_t c,
    EdgeInferDetectResult* result);

EDGEINFER_C_API int32_t EdgeInferClassify(EdgeInferHandle handle,
    const uint8_t* img_data, int32_t w, int32_t h, int32_t c,
    EdgeInferClassifyResult* result);

EDGEINFER_C_API int32_t EdgeInferExtract(EdgeInferHandle handle,
    const uint8_t* img_data, int32_t w, int32_t h, int32_t c,
    EdgeInferEmbeddingResult* result);

// ── Batch ─────────────────────────────────────────────────
EDGEINFER_C_API int32_t EdgeInferDetectBatch(EdgeInferHandle handle,
    const uint8_t* const* img_data_array,
    const int32_t* widths, const int32_t* heights,
    int32_t batch_size, int32_t c,
    EdgeInferDetectResult* results);

// ── Cleanup ───────────────────────────────────────────────
EDGEINFER_C_API void EdgeInferFreeDetectResult(EdgeInferDetectResult* result);
EDGEINFER_C_API void EdgeInferFreeClassifyResult(EdgeInferClassifyResult* result);
EDGEINFER_C_API void EdgeInferFreeEmbeddingResult(EdgeInferEmbeddingResult* result);

// ── Version ───────────────────────────────────────────────
EDGEINFER_C_API const char* EdgeInferVersion(void);

#ifdef __cplusplus
}
#endif

#endif // EDGEINFER_H
