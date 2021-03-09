//
// Created by eashwara on 19.05.20.
//

#ifndef EGDETPU_VIDEO_INFERENCE_ULTRAFACE_ENGINE_H
#define EGDETPU_VIDEO_INFERENCE_ULTRAFACE_ENGINE_H

#include <array>
#include <map>
#include <string>

#include "engine.h"
#include "opencv2/opencv.hpp"

#define num_featuremap 4
#define hard_nms 1
#define blending_nms 2

namespace edge {

typedef struct FaceInfo {
  float x1;
  float y1;
  float x2;
  float y2;
  float score;

  float landmarks[10];
} FaceInfo;

class UltraFaceEngine : public Engine {
 public:
  // Constructor that loads the model and label into the program.
  UltraFaceEngine(
      const std::string& model,
      const std::shared_ptr<edgetpu::EdgeTpuContext>& edgetpu_context,
      const bool edgetpu, float score_threshold_ = 0.7,
      float iou_threshold_ = 0.3, int topk_ = -1)
      : Engine(model, edgetpu_context, edgetpu) {
    std::cout << "Detection Engine loaded successfully" << std::endl;
  }

  std::vector<FaceInfo> DetectWithOutputVector(
    const std::vector<float>& inf_vec, const float& threshold);

 private:
  void generateBBox(std::vector<FaceInfo>& bbox_collection, cv::Mat scores,
                    cv::Mat boxes, float score_threshold, int num_anchors);

  void nms(std::vector<FaceInfo>& input, std::vector<FaceInfo>& output,
           int type = blending_nms);

 private:
  int image_w;
  int image_h;

  float iou_threshold;

  const float mean_vals[3] = {127, 127, 127};
  const float norm_vals[3] = {1.0 / 128, 1.0 / 128, 1.0 / 128};

  const float center_variance = 0.1;
  const float size_variance = 0.2;
  const std::vector<std::vector<float>> min_boxes = {{10.0f, 16.0f, 24.0f},
                                                     {32.0f, 48.0f},
                                                     {64.0f, 96.0f},
                                                     {128.0f, 192.0f, 256.0f}};
  const std::vector<float> strides = {8.0, 16.0, 32.0, 64.0};
  std::vector<std::vector<float>> featuremap_size;
  std::vector<std::vector<float>> shrinkage_size;
  std::vector<int> w_h_list;

  std::vector<std::vector<float>> priors = {};
};
}  // namespace edge
#endif  // EGDETPU_VIDEO_INFERENCE_ULTRAFACE_ENGINE_H
