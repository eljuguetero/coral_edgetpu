cmake_minimum_required(VERSION 3.7)
project(egdetpu_video_inference)
find_package( OpenCV REQUIRED)

option(WITH_INF_ENGINE "Enable OpenVino Inference Engine support" ON)
if(WITH_INF_ENGINE)
  set(CMAKE_INSTALL_RPATH
    "/opt/intel/openvino/opencv/lib"
    "/opt/intel/openvino/inference_engine/lib/intel64"
    "/opt/intel/openvino/inference_engine/lib/aarch64"
    "/opt/intel/openvino/inference_engine/lib/armv7l"
    "/opt/intel/openvino/inference_engine/external/tbb/lib"
    "/opt/intel/openvino/inference_engine/external/mkltiny_lnx/lib"
    "/opt/intel/openvino/deployment_tools/ngraph/lib"
  )
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  set(CMAKE_EXE_LINKER_FLAGS "-Wl,--disable-new-dtags -Wl,--as-needed")
  add_definitions(-DWITH_INF_ENGINE)
endif(WITH_INF_ENGINE)

option(WITH_GTK "Enable GTK support" OFF)
if(WITH_GTK)
  add_definitions(-DWITH_GTK)
endif(WITH_GTK)

FIND_LIBRARY(LIB_EDGETPU NAMES edgetpu)

find_package(Eigen3)

set(CMAKE_C_FLAGS "-Wall -pthread")
set(CMAKE_C_FLAGS_DEBUG "-g -O0")
set(CMAKE_C_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c++11 -lstdc++ -ldl")
set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})

# Setting Lib path
set(TF_LITE_LIB "tensorflow-lite -ledgetpu -ldl")

include_directories(/opt/tensorflow/tensorflow/lite/tools/make/downloads/eigen)

# Setting all include and link directories.
##########################################################################################################################
include_directories(${CMAKE_SOURCE_DIR}/)
include_directories(${CMAKE_SOURCE_DIR}/src/)
##########################################################################################################################
# project source
include_directories(${CMAKE_SOURCE_DIR}/src/utils)
include_directories(${CMAKE_SOURCE_DIR}/src/common_engine)
include_directories(${CMAKE_SOURCE_DIR}/src/classification_engine)
include_directories(${CMAKE_SOURCE_DIR}/src/image_preprocessing)
include_directories(${CMAKE_SOURCE_DIR}/src/detection_engine)
include_directories(${CMAKE_SOURCE_DIR}/src/ultraface_engine)
include_directories(${CMAKE_SOURCE_DIR}/src/humanpose_engine)

##########################################################################################################################
include_directories(${CMAKE_SOURCE_DIR}/libedgetpu/)
include_directories(${CMAKE_SOURCE_DIR}/include)
##########################################################################################################################
# project includes
include_directories(${CMAKE_SOURCE_DIR}/include/utils)
include_directories(${CMAKE_SOURCE_DIR}/include/common_engine)
include_directories(${CMAKE_SOURCE_DIR}/include/classification_engine)
include_directories(${CMAKE_SOURCE_DIR}/include/image_preprocessing)
include_directories(${CMAKE_SOURCE_DIR}/include/detection_engine)
include_directories(${CMAKE_SOURCE_DIR}/include/ultraface_engine)
include_directories(${CMAKE_SOURCE_DIR}/include/humanpose_engine)

##########################################################################################################################
include_directories(${CMAKE_SOURCE_DIR}/include/thirdparty/cxxopts)
include_directories(/opt/tensorflow/)
include_directories(/opt/tensorflow/tensorflow/lite/tools/make/downloads/absl/)
include_directories(/opt/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include)
include_directories(${CMAKE_BINARY_DIR}/glog_install/include)
##########################################################################################################################
link_directories(${CMAKE_BINARY_DIR})
##########################################################################################################################
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/${ARCH})

add_library(label_utils
  src/utils/label_utils.cc
  include/utils/label_utils.h)

add_library(pose_decoder
        src/humanpose_engine/posenet_decoder_op.cc
        src/humanpose_engine/posenet_decoder.cc
        include/humanpose_engine/posenet_decoder.h
        include/humanpose_engine/posenet_decoder_op.h)
target_link_libraries(pose_decoder ${TF_LITE_LIB})

add_library(engine
        src/common_engine/engine.cc
        include/common_engine/engine.h)
target_link_libraries(engine label_utils pose_decoder ${TF_LITE_LIB})
add_dependencies(engine label_utils pose_decoder)

add_library(classification_engine
        src/classification_engine/classification_engine.cc
        include/classification_engine/classification_engine.h)
target_link_libraries(classification_engine engine pose_decoder ${TF_LITE_LIB} ${OpenCV_LIBS})
add_dependencies(classification_engine engine pose_decoder)

add_library(image_preprocessing
        src/image_preprocessing/img_prep.cc
        include/image_preprocessing/img_prep.h)
target_link_libraries(image_preprocessing ${OpenCV_LIBS})

add_library(detection_engine
        src/detection_engine/detection_engine.cc
        include/detection_engine/detection_engine.h)
target_link_libraries(detection_engine engine pose_decoder ${TF_LITE_LIB} ${OpenCV_LIBS})
add_dependencies(detection_engine engine )

add_library(ultraface_engine
        src/ultraface_engine/ultraface_engine.cc
        include/ultraface_engine/ultraface_engine.h)
target_link_libraries(ultraface_engine engine pose_decoder ${TF_LITE_LIB} ${OpenCV_LIBS})
add_dependencies(ultraface_engine engine )

add_library(humanpose_engine
        src/humanpose_engine/humanpose_engine.cc
        include/humanpose_engine/humanpose_engine.h)
target_link_libraries(humanpose_engine engine pose_decoder ${TF_LITE_LIB} ${OpenCV_LIBS})
add_dependencies(humanpose_engine engine pose_decoder)

add_executable(classification_camera
        src/classification_camera.cc
        /opt/tensorflow/tensorflow/lite/tools/make/downloads/fft2d/fftsg.c
        /opt/tensorflow/tensorflow/lite/tools/optimize/sparsity/format_converter.cc
        )
target_link_libraries(classification_camera image_preprocessing classification_engine engine label_utils ${OpenCV_LIBS} ${TF_LITE_LIB} ${LIB_EDGETPU})
add_dependencies(classification_camera image_preprocessing classification_engine engine label_utils )

add_executable(detection_camera
        src/detection_camera.cc
        /opt/tensorflow/tensorflow/lite/tools/make/downloads/fft2d/fftsg.c
        /opt/tensorflow/tensorflow/lite/tools/optimize/sparsity/format_converter.cc
        )
target_link_libraries(detection_camera image_preprocessing detection_engine engine label_utils ${OpenCV_LIBS} ${TF_LITE_LIB} ${LIB_EDGETPU})
add_dependencies(detection_camera image_preprocessing detection_engine engine label_utils )

add_executable(ultraface_camera
        src/ultraface_camera.cc
        /opt/tensorflow/tensorflow/lite/tools/make/downloads/fft2d/fftsg.c
        /opt/tensorflow/tensorflow/lite/tools/optimize/sparsity/format_converter.cc
        )
target_link_libraries(ultraface_camera image_preprocessing ultraface_engine engine label_utils ${OpenCV_LIBS} ${TF_LITE_LIB} ${LIB_EDGETPU})
add_dependencies(ultraface_camera image_preprocessing ultraface_engine engine label_utils )

add_executable(humanpose_camera
        src/humanpose_camera.cc
        /opt/tensorflow/tensorflow/lite/tools/make/downloads/fft2d/fftsg.c
        /opt/tensorflow/tensorflow/lite/tools/optimize/sparsity/format_converter.cc
        )
target_link_libraries(humanpose_camera image_preprocessing humanpose_engine engine label_utils pose_decoder ${OpenCV_LIBS} ${TF_LITE_LIB} ${LIB_EDGETPU})
add_dependencies(humanpose_camera image_preprocessing humanpose_engine engine label_utils pose_decoder)



