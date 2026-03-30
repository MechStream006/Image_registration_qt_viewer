#ifndef EVENT2_PREPROCESS_H
#define EVENT2_PREPROCESS_H

#include <opencv2/opencv.hpp>

// 🔥 Preprocess single frame (16-bit input → float output)
void preprocessFrameECC(const cv::Mat& input16,
    cv::Mat& output32);

// 🔥 Batch processing
void preprocessSequence(const std::vector<cv::Mat>& inputFrames,
    std::vector<cv::Mat>& outputFrames);

#endif