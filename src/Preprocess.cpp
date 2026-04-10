#include "Preprocess.h"
#include <iostream>

void preprocessFrameECC(const cv::Mat& input16,
    cv::Mat& output32)
{
    // 1. Convert to float
    input16.convertTo(output32, CV_32F);

    // 2. Log transform (Beer-Lambert linearization)
    //    Values after log: ~[0, 11.09]  (ln(65536+1) ≈ 11.09)
    output32 += 1.0f;
    cv::log(output32, output32);

    // 3. Mean shift (per-frame brightness normalization)
    //    Compensates for frame-to-frame brightness drift (AGC, breathing, etc.)
    //    Essential for stable ECC registration
    cv::Scalar mean, stddev;
    cv::meanStdDev(output32, mean, stddev);
    output32 -= static_cast<float>(mean[0]);

    // 4. Normalize to [0, 1] using 3-sigma window
    //    Most image content falls within ±3σ of the mean
    float sigma = static_cast<float>(stddev[0]);
    float lo = -3.0f * sigma;
    float hi =  3.0f * sigma;
    output32 = (output32 - lo) / (hi - lo);

    // 5. Clamp [0, 1]
    cv::threshold(output32, output32, 0.0f, 0.0f, cv::THRESH_TOZERO);
    cv::threshold(output32, output32, 1.0f, 1.0f, cv::THRESH_TRUNC);
}

// 🔥 Batch Processing (Parallel)
void preprocessSequence(const std::vector<cv::Mat>& inputFrames,
    std::vector<cv::Mat>& outputFrames)
{
    outputFrames.resize(inputFrames.size());

#pragma omp parallel for
    for (int i = 0; i < (int)inputFrames.size(); i++)
    {
        preprocessFrameECC(inputFrames[i], outputFrames[i]);
    }

    std::cout << "[Preprocess] Done: "
        << outputFrames.size() << " frames\n";
}