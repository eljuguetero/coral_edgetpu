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

  void InitAll(const float det_score=0.6, const float nms_iou=0.5);
  std::vector<std::pair<cv::Rect, float>> Decode(const std::vector<std::vector<float> > &outputs, const cv::Size &img_size);


 private:
 void NMS(std::vector<std::pair<cv::Rect, float>> &input, std::vector<std::pair<cv::Rect, float>> &output);
  int iw_, ih_;
  float th_;
  float nms_th_;
  std::vector<std::vector<float>> priors = {};
};
}  // namespace edge
#endif  // EGDETPU_VIDEO_INFERENCE_ULTRAFACE_ENGINE_H
