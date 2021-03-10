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

std::vector<cv::Rect> UltraFaceEngine::Decode(const std::vector<std::vector<float>> &outputs, const cv::Size &img_size,  std::vector<float> *out_scores=nullptr)
{
    std::vector<cv::Rect> bboxes_candidates;
    std::vector<float> scores_candidates;
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

            bboxes_candidates.push_back(box);
            scores_candidates.push_back(scores_ptr[i * 2 + 1]);
        }
    }
    std::vector<int> indices;
    cv::dnn::NMSBoxes(bboxes_candidates,scores_candidates,th_,nms_th_,indices);
    std::vector<cv::Rect> faces;
    if (out_scores){
        out_scores->clear();
    }
    for (auto ii : indices){
        faces.push_back(bboxes_candidates[ii]);
        if (out_scores){
            out_scores->push_back(scores_candidates[ii]);
        }
    }
    return faces;
}

}  // namespace edge
