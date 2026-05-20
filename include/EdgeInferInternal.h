//
// EdgeInfer - Internal Header Aggregator
// Used for building the shared library, not public API.
//

#ifndef EDGEINFER_INTERNAL_H
#define EDGEINFER_INTERNAL_H

// --- Types ---
#include "types/image.h"
#include "types/types.h"
#include "types/config.h"

// --- Tools ---
#include "tools/json.h"
#include "tools/yaml.h"

// --- Core ---
#include "core/processor.h"
#include "core/engine.h"
#include "core/factory.h"
#include "core/pipeline.h"

// --- Adapters ---
#include "adapters/base_adapter.h"
#include "adapters/ncnn_adapter.h"
#include "adapters/mnn_adapter.h"
#include "adapters/cvdnn_adapter.h"

// --- Processors ---
#include "processors/resize.h"
#include "processors/normalize.h"
#include "processors/letterbox.h"
#include "processors/nms.h"
#include "processors/softmax.h"
#include "processors/yolo_decode.h"
#include "processors/nv12_to_bgr.h"

// --- Models ---
#include "models/base_model.h"
#include "models/detector.h"
#include "models/classifier.h"
#include "models/recognizer.h"

// --- Inline Utilities ---
#include "core/preprocess.hpp"
#include "core/postprocess.hpp"

#endif // EDGEINFER_INTERNAL_H
