cmake_minimum_required(VERSION 3.10)
project(EdgeInfer VERSION 4.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Build switches (set via -D from build.sh) ──────────────
set(EDGEINFER_BUILD_DEMO   OFF CACHE BOOL "Build demo application")
set(EDGEINFER_WITH_NCNN    OFF CACHE BOOL "Enable NCNN backend")
set(EDGEINFER_WITH_MNN     OFF CACHE BOOL "Enable MNN backend")
set(EDGEINFER_WITH_OPENCV  OFF CACHE BOOL "Enable OpenCV support (implies CVDNN)")
set(EDGEINFER_WITH_NTCNN   OFF CACHE BOOL "Enable Novatek NTCNN backend (ARM cross-compile)")
# CVDNN is always enabled together with OpenCV — not a standalone switch
set(EDGEINFER_WITH_CVDNN   ${EDGEINFER_WITH_OPENCV} CACHE BOOL "" FORCE)

# ── Architecture-specific third-party root ──────────────────
set(THIRD_PARTY_ARCH "x86" CACHE STRING "Target arch for third-party libs (x86 | arm-ca53)")
set(THIRD_PARTY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/${THIRD_PARTY_ARCH}")

# ── Project paths ───────────────────────────────────────────
set(NCNN_ROOT       "${THIRD_PARTY_ROOT}/ncnn"
                    CACHE PATH "ncnn install root")
set(MNN_ROOT        "${THIRD_PARTY_ROOT}/mnn"
                    CACHE PATH "mnn install root")
set(OPENCV_ROOT     "${THIRD_PARTY_ROOT}/opencv"
                    CACHE PATH "OpenCV install root")
# ntcnn is always ARM — locked to arm-ca53
set(NTCNN_ROOT      "${CMAKE_CURRENT_SOURCE_DIR}/third_party/arm-ca53/ntcnn"
                    CACHE PATH "Novatek ntcnn SDK root")
set(JSON_INCLUDE    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann/include"
                    CACHE PATH "nlohmann/json include dir")
set(RYML_INCLUDE    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/rapidyaml/include"
                    CACHE PATH "rapidyaml include dir")

# ── Compiler — set from build.sh, fallback to system ────────
if(NOT CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER gcc)
endif()
if(NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER g++)
endif()

find_package(Threads REQUIRED)
find_package(OpenMP)

# ── Internal include directories ────────────────────────────
set(EDGEINFER_PUBLIC_INCLUDE  "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(EDGEINFER_SRC_ROOT        "${CMAKE_CURRENT_SOURCE_DIR}/src")

# ── Core source files ───────────────────────────────────────
set(EDGEINFER_SOURCES
    src/types/image.cpp
    src/tools/json.cpp
    src/tools/yaml.cpp
    src/core/engine.cpp
    src/core/pipeline.cpp
    src/core/processor.cpp
    src/core/config_parser.cpp
    src/adapters/base_adapter.cpp
    src/processors/softmax.cpp
    src/processors/letterbox.cpp
    src/processors/yolo_decode.cpp
    src/api/EdgeInfer_cpp.cpp
    src/api/EdgeInfer_c.cpp
)

# Backend sources
if(EDGEINFER_WITH_NCNN)
    list(APPEND EDGEINFER_SOURCES src/adapters/ncnn_adapter.cpp)
endif()
if(EDGEINFER_WITH_MNN)
    list(APPEND EDGEINFER_SOURCES src/adapters/mnn_adapter.cpp)
endif()
if(EDGEINFER_WITH_CVDNN)
    list(APPEND EDGEINFER_SOURCES src/adapters/cvdnn_adapter.cpp)
endif()
if(EDGEINFER_WITH_NTCNN)
    list(APPEND EDGEINFER_SOURCES src/adapters/ntcnn_adapter.cpp)
endif()

# ── Shared library ──────────────────────────────────────────
add_library(edgeinfer SHARED ${EDGEINFER_SOURCES})

target_compile_definitions(edgeinfer PRIVATE EDGEINFER_BUILDING_SO)

if(EDGEINFER_WITH_OPENCV)
    target_compile_definitions(edgeinfer PRIVATE EDGEINFER_WITH_OPENCV)
endif()

# OpenCV support — compiled into the .so, transparent to callers

target_include_directories(edgeinfer PRIVATE
    ${EDGEINFER_PUBLIC_INCLUDE}
    ${JSON_INCLUDE}
    ${RYML_INCLUDE}
)

target_include_directories(edgeinfer PUBLIC
    ${EDGEINFER_PUBLIC_INCLUDE}
)

if(EDGEINFER_WITH_NCNN)
    if(NCNN_ROOT)
        target_include_directories(edgeinfer PRIVATE ${NCNN_ROOT}/include)
        target_link_directories(edgeinfer PRIVATE ${NCNN_ROOT}/lib)
    endif()
    target_link_libraries(edgeinfer PRIVATE ncnn)
endif()

if(EDGEINFER_WITH_MNN)
    if(MNN_ROOT)
        target_include_directories(edgeinfer PRIVATE ${MNN_ROOT}/include)
        target_link_directories(edgeinfer PRIVATE ${MNN_ROOT}/lib)
    endif()
    target_link_libraries(edgeinfer PRIVATE MNN MNN_Express)
    # Allow cross-linker to resolve MNN shared libs at build time
    set_property(TARGET edgeinfer APPEND_STRING PROPERTY
        LINK_FLAGS " -Wl,-rpath-link,${MNN_ROOT}/lib")
endif()

if(EDGEINFER_WITH_OPENCV AND OPENCV_ROOT)
    target_include_directories(edgeinfer PRIVATE ${OPENCV_ROOT}/include/opencv4)
    target_link_directories(edgeinfer PRIVATE ${OPENCV_ROOT}/lib)
    target_link_libraries(edgeinfer PRIVATE opencv_world)
endif()

if(EDGEINFER_WITH_NTCNN AND NTCNN_ROOT)
    target_include_directories(edgeinfer PRIVATE ${NTCNN_ROOT}/include)
    target_link_directories(edgeinfer PRIVATE ${NTCNN_ROOT}/lib)
    target_link_libraries(edgeinfer PRIVATE
        hdal prebuilt_ai vendor_ai2 vendor_ai2_pub vos
        pthread rt dl m)
    # Allow cross-linker to resolve NTCNN shared libs at build time
    set_property(TARGET edgeinfer APPEND_STRING PROPERTY
        LINK_FLAGS " -Wl,-rpath-link,${NTCNN_ROOT}/lib")
endif()

target_link_libraries(edgeinfer PRIVATE Threads::Threads)

if(OpenMP_FOUND OR OpenMP_CXX_FOUND)
    target_link_libraries(edgeinfer PRIVATE OpenMP::OpenMP_CXX)
endif()

# Hide internal symbols, export only public API
set_target_properties(edgeinfer PROPERTIES
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# ── Demo ────────────────────────────────────────────────────
if(EDGEINFER_BUILD_DEMO)
    add_executable(demo demo/main.cpp)

    target_include_directories(demo PRIVATE
        ${EDGEINFER_PUBLIC_INCLUDE}
    )

    target_link_libraries(demo PRIVATE edgeinfer)

    if(EDGEINFER_WITH_NCNN AND NCNN_ROOT)
        target_link_directories(demo PRIVATE ${NCNN_ROOT}/lib)
    endif()

    if(EDGEINFER_WITH_MNN AND MNN_ROOT)
        target_link_directories(demo PRIVATE ${MNN_ROOT}/lib)
        set_property(TARGET demo APPEND_STRING PROPERTY
            LINK_FLAGS " -Wl,-rpath-link,${MNN_ROOT}/lib")
    endif()

    if(EDGEINFER_WITH_NTCNN AND NTCNN_ROOT)
        target_link_directories(demo PRIVATE ${NTCNN_ROOT}/lib)
        set_property(TARGET demo APPEND_STRING PROPERTY
            LINK_FLAGS " -Wl,-rpath-link,${NTCNN_ROOT}/lib")
    endif()

endif()

# ── Install ─────────────────────────────────────────────────
include(GNUInstallDirs)

install(TARGETS edgeinfer
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# ── Third-party runtime libraries ────────────────────────────
if(EDGEINFER_WITH_MNN AND MNN_ROOT)
    file(GLOB MNN_LIBS "${MNN_ROOT}/lib/libMNN*.so*")
    install(FILES ${MNN_LIBS} DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

if(EDGEINFER_WITH_OPENCV AND OPENCV_ROOT)
    file(GLOB OPENCV_LIBS "${OPENCV_ROOT}/lib/libopencv_world.so*")
    install(FILES ${OPENCV_LIBS} DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

# Public headers
install(FILES
    include/EdgeInfer.h
    include/EdgeInfer.hpp
    include/types/image.h
    include/types/types.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/EdgeInfer
)

if(EDGEINFER_BUILD_DEMO)
    install(TARGETS demo
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    install(FILES
        demo/config.json
        demo/config.yaml
        DESTINATION ${CMAKE_INSTALL_DATADIR}/EdgeInfer
    )
endif()
