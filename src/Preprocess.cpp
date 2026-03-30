#include "Preprocess.h"
#include <iostream>

void preprocessFrameECC(const cv::Mat& input16,
    cv::Mat& output32)
{
    // 1. Convert to float (preserve precision)
    input16.convertTo(output32, CV_32F);

    // 2. Normalize to [0,1]
    double minVal, maxVal;
    cv::minMaxLoc(output32, &minVal, &maxVal);

    if (maxVal - minVal < 1e-6)
    {
        output32 = cv::Mat::zeros(input16.size(), CV_32F);
        return;
    }

    output32 = (output32 - minVal) / (maxVal - minVal);

    // 3. Log Transform (contrast compression)
    output32 = output32 + 1e-6f;
    cv::log(output32, output32);

    // 4. Normalize again
    cv::normalize(output32, output32, 0.0f, 1.0f, cv::NORM_MINMAX);

    // 5. CLAHE (adaptive contrast)
    cv::Mat temp8U;
    output32.convertTo(temp8U, CV_8U, 255.0);

    static cv::Ptr<cv::CLAHE> clahe =
        cv::createCLAHE(2.0, cv::Size(8, 8));

    cv::Mat claheOut;
    clahe->apply(temp8U, claheOut);

    claheOut.convertTo(output32, CV_32F, 1.0 / 255.0);

    // 6. Gaussian Blur (denoise)
    cv::GaussianBlur(output32, output32,
        cv::Size(5, 5), 1.2);

    // 7. Edge Enhancement (Laplacian)
    cv::Mat lap;
    cv::Laplacian(output32, lap, CV_32F, 3);

    output32 = output32 + 0.2f * lap;

    // 8. Clamp to [0,1]
    cv::threshold(output32, output32, 0.0, 0.0, cv::THRESH_TOZERO);
    cv::threshold(output32, output32, 1.0, 1.0, cv::THRESH_TRUNC);

    // 9. Safety check
    if (!cv::checkRange(output32))
    {
        std::cout << "❌ Invalid values in preprocessing\n";
    }
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