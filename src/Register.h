#ifndef REGISTER_H
#define REGISTER_H

#include <opencv2/opencv.hpp>
#include <vector>

struct ECCResult
{
    cv::Mat warpMatrix;
    int     iterations = 0;
    bool    converged = false;
    double  finalRho = 0.0;
};

// =============================================================
// Register single frame
//   ref    : preprocessed float32 [0,1] — reference (frame 0)
//   src    : preprocessed float32 [0,1] — moving frame i
//   dst    : warped float32 output (same type as src)
//   result : warp matrix + diagnostics
// =============================================================
bool registerFrameECC(
    const cv::Mat& ref,
    const cv::Mat& src,
    cv::Mat& dst,
    ECCResult& result,
    int motionType = cv::MOTION_EUCLIDEAN,
    int pyramidLevels = 3
);

// =============================================================
// Register full sequence
//   Input/output: preprocessed float32 [0,1] frames
//   ECC computed on these frames → warp applied to same frames
//   registeredFrames[0] = processedFrames[0] (identity)
//   registeredFrames[i] = warp(processedFrames[i])
// =============================================================
bool registerSequenceECC(
    const std::vector<cv::Mat>& processedFrames,   // float32 [0,1]
    std::vector<cv::Mat>& registeredFrames,  // float32 [0,1] output
    int motionType = cv::MOTION_EUCLIDEAN,
    int pyramidLevels = 3
);

// =============================================================
// DSA subtraction
//   Input:  preprocessed float32 [0,1] frames
//   Output: CV_16U frames for VTK (vessels dark, background white)
//   maskFrameIndex: 0 = frame[0], -1 = auto-detect brightest
// =============================================================
void computeDSA(
    const std::vector<cv::Mat>& preprocessedFrames,
    std::vector<cv::Mat>& dsaFrames,
    int maskFrameIndex = 0
);

#endif