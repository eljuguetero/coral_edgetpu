#include "ultraface_engine.h"
#include "opencv2/opencv.hpp"

#include <queue>
#include <tuple>

#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))

namespace edge {

std::vector<FaceInfo> UltraFaceEngine::DetectWithOutputVector(
    const std::vector<float>& inf_vec, const float& threshold) {
  const auto* result_raw = inf_vec.data();
  std::vector<std::vector<float>> results(m_output_shape.size());
  int offset = 0;
  for (size_t i = 0; i < m_output_shape.size(); ++i) {
    const size_t size_of_output_tensor_i = m_output_shape[i];
    results[i].resize(size_of_output_tensor_i);
    std::memcpy(results[i].data(), result_raw + offset,
                sizeof(float) * size_of_output_tensor_i);
    offset += size_of_output_tensor_i;
  }
  std::vector<FaceInfo> inf_results;

  return inf_results;
}

void UltraFaceEngine::generateBBox(std::vector<FaceInfo>& bbox_collection,
                                   cv::Mat scores, cv::Mat boxes,
                                   float score_threshold, int num_anchors) {
  float* score_value = (float*)(scores.data);
  float* bbox_value = (float*)(boxes.data);
  for (int i = 0; i < num_anchors; i++) {
    float score = score_value[2 * i + 1];
    if (score_value[2 * i + 1] > score_threshold) {
      FaceInfo rects = {0};
      float x_center =
          bbox_value[i * 4] * center_variance * priors[i][2] + priors[i][0];
      float y_center =
          bbox_value[i * 4 + 1] * center_variance * priors[i][3] + priors[i][1];
      float w = exp(bbox_value[i * 4 + 2] * size_variance) * priors[i][2];
      float h = exp(bbox_value[i * 4 + 3] * size_variance) * priors[i][3];

      rects.x1 = clip(x_center - w / 2.0, 1) * image_w;
      rects.y1 = clip(y_center - h / 2.0, 1) * image_h;
      rects.x2 = clip(x_center + w / 2.0, 1) * image_w;
      rects.y2 = clip(y_center + h / 2.0, 1) * image_h;
      rects.score = clip(score_value[2 * i + 1], 1);
      bbox_collection.push_back(rects);
    }
  }
}

void UltraFaceEngine::nms(std::vector<FaceInfo>& input,
                          std::vector<FaceInfo>& output, int type) {
  std::sort(
      input.begin(), input.end(),
      [](const FaceInfo& a, const FaceInfo& b) { return a.score > b.score; });

  int box_num = input.size();

  std::vector<int> merged(box_num, 0);

  for (int i = 0; i < box_num; i++) {
    if (merged[i]) continue;
    std::vector<FaceInfo> buf;

    buf.push_back(input[i]);
    merged[i] = 1;

    float h0 = input[i].y2 - input[i].y1 + 1;
    float w0 = input[i].x2 - input[i].x1 + 1;

    float area0 = h0 * w0;

    for (int j = i + 1; j < box_num; j++) {
      if (merged[j]) continue;

      float inner_x0 = input[i].x1 > input[j].x1 ? input[i].x1 : input[j].x1;
      float inner_y0 = input[i].y1 > input[j].y1 ? input[i].y1 : input[j].y1;

      float inner_x1 = input[i].x2 < input[j].x2 ? input[i].x2 : input[j].x2;
      float inner_y1 = input[i].y2 < input[j].y2 ? input[i].y2 : input[j].y2;

      float inner_h = inner_y1 - inner_y0 + 1;
      float inner_w = inner_x1 - inner_x0 + 1;

      if (inner_h <= 0 || inner_w <= 0) continue;

      float inner_area = inner_h * inner_w;

      float h1 = input[j].y2 - input[j].y1 + 1;
      float w1 = input[j].x2 - input[j].x1 + 1;

      float area1 = h1 * w1;

      float score;

      score = inner_area / (area0 + area1 - inner_area);

      if (score > iou_threshold) {
        merged[j] = 1;
        buf.push_back(input[j]);
      }
    }
    switch (type) {
      case hard_nms: {
        output.push_back(buf[0]);
        break;
      }
      case blending_nms: {
        float total = 0;
        for (int i = 0; i < buf.size(); i++) {
          total += exp(buf[i].score);
        }
        FaceInfo rects;
        memset(&rects, 0, sizeof(rects));
        for (int i = 0; i < buf.size(); i++) {
          float rate = exp(buf[i].score) / total;
          rects.x1 += buf[i].x1 * rate;
          rects.y1 += buf[i].y1 * rate;
          rects.x2 += buf[i].x2 * rate;
          rects.y2 += buf[i].y2 * rate;
          rects.score += buf[i].score * rate;
        }
        output.push_back(rects);
        break;
      }
      default: {
        printf("wrong type of nms.");
        exit(-1);
      }
    }
  }
}
}  // namespace edge
