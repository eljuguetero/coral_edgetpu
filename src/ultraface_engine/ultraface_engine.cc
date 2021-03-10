#include "ultraface_engine.h"
#include "opencv2/opencv.hpp"

#include <queue>
#include <tuple>


namespace  {
inline float clip(float x, float y=1.0F) {
    return  std::min(std::max(0.0F, x), y);
}
const cv::Scalar kMean = {127, 127, 127};
const float kScale = 1.0 / 128;
const float kCenterVariance = 0.1;
const float kSizeVariance = 0.2;
const std::vector<std::vector<float>> kMinBoxes = {
    {10.0f,  16.0f,  24.0f},
    {32.0f,  48.0f},
    {64.0f,  96.0f},
    {128.0f, 192.0f, 256.0f}};
const std::vector<float> kStrides = {8.0, 16.0, 32.0, 64.0};
}


namespace edge {


void UltraFaceEngine::InitAll(const float det_score, const float nms_iou)
{
    iw_  = 320;
    ih_  = 240;
    th_  = det_score;
    nms_th_  = nms_iou;

    std::vector<std::vector<float>> featuremap_size;
    std::vector<std::vector<float>> shrinkage_size;
    for (auto size : {iw_, ih_}) {
        std::vector<float> fm_item;
        for (float stride : kStrides) {
            fm_item.push_back(ceil(size / stride));
        }
        featuremap_size.push_back(fm_item);
    }
    shrinkage_size.push_back(kStrides);
    shrinkage_size.push_back(kStrides);

    /* generate prior anchors */
    for (int index = 0; index < 4; index++) {
        float scale_w = iw_ / shrinkage_size[0][index];
        float scale_h = ih_ / shrinkage_size[1][index];
        for (int j = 0; j < featuremap_size[1][index]; j++) {
            for (int i = 0; i < featuremap_size[0][index]; i++) {
                float x_center = (i + 0.5) / scale_w;
                float y_center = (j + 0.5) / scale_h;

                for (float k : kMinBoxes[index]) {
                    float w = k / iw_;
                    float h = k / ih_;
                    priors.push_back({clip(x_center), clip(y_center), clip(w), clip(h)});
                }
            }
        }
    }
}

std::vector<std::pair<cv::Rect, float>> UltraFaceEngine::Decode(const std::vector<std::vector<float>> &outputs, const cv::Size &img_size)
{
    std::vector<std::pair<cv::Rect, float>> bboxes_scores, result;
    const float *bboxes_ptr = outputs[0].data();
    const float *scores_ptr = outputs[1].data();

    const int frame_width = img_size.width;
    const int frame_height = img_size.height;

    for (size_t i = 0; i < priors.size(); i++) {
        if (scores_ptr[i * 2 + 1] > th_) {
            cv::Rect box;
            float x_center = bboxes_ptr[i * 4] * kCenterVariance * priors[i][2] + priors[i][0];
            float y_center = bboxes_ptr[i * 4 + 1] * kCenterVariance * priors[i][3] + priors[i][1];
            float w = exp(bboxes_ptr[i * 4 + 2] * kSizeVariance) * priors[i][2];
            float h = exp(bboxes_ptr[i * 4 + 3] * kSizeVariance) * priors[i][3];

            box.x = static_cast<int>( clip(x_center - 0.5*w)*frame_width );
            box.y = static_cast<int>( clip(y_center - 0.5*h)*frame_height );
            box.width = static_cast<int>( clip(w)*frame_width );
            box.height = static_cast<int>( clip(h)*frame_height );

            bboxes_scores.push_back({box, scores_ptr[i * 2 + 1]});
        }
    }
    NMS(bboxes_scores, result);
    return result;
}

void UltraFaceEngine::NMS(std::vector<std::pair<cv::Rect, float>> &input, std::vector<std::pair<cv::Rect, float>> &output) {
    std::sort(input.begin(), input.end(), [](const std::pair<cv::Rect, float> &a, const std::pair<cv::Rect, float> &b) { return a.second > b.second; });
    int box_num = input.size();
    std::vector<int> merged(box_num, 0);

    for (int i = 0; i < box_num; i++) {
        if (merged[i])
            continue;
        std::vector<std::pair<cv::Rect, float>> buf;

        buf.push_back(input[i]);
        merged[i] = 1;
        float h0 = input[i].first.width;
        float w0 = input[i].first.height;

        float area0 = h0 * w0;
        for (int j = i + 1; j < box_num; j++) {
            if (merged[j])
                continue;

            float inner_area = ( input[i].first&input[j].first ).area();
            if (inner_area == 0)
                continue;

            float area1 = input[j].first.area();
            float score = inner_area / (area0 + area1 - inner_area);

            if (score > nms_th_) {
                merged[j] = 1;
                buf.push_back(input[j]);
            }
        }
        output.push_back(buf[0]);
    }
}

}  // namespace edge
