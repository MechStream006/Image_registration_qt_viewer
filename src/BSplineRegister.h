#ifndef BSPLINE_REGISTER_H
#define BSPLINE_REGISTER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "RegionConfig.h"

// =============================================================
// B-Spline Free-Form Deformation (FFD) registration via ITK
//
// Used as Stage 2 (refinement) after ECC Affine (Stage 1) for
// Cardiac and Abdomen regions.
//
// Algorithm:
//   • Transform:  itk::BSplineTransform<double,2,3> (cubic, 2D)
//   • Metric:     itk::MattesMutualInformationImageToImageMetricv4
//                 — computed only on non-vessel (tissue) pixels
//                 — robust to iodine contrast intensity changes
//   • Optimizer:  itk::LBFGSBOptimizerv4
//                 — bounded L-BFGS; bounds prevent image folding
//   • Multi-res:  image pyramid (shrink factors set per region)
//
// Vessel masking:
//   The ITK metric's fixed-image spatial mask is set to the
//   TISSUE mask (inverted vessel mask from makeVesselMask()).
//   This ensures the MI cost function ignores iodine-filled
//   vessel pixels — the optimizer registers tissue only.
//
// Input frames must be float32 [0,1], already affine-registered
// by Stage 1.  Output is float32 [0,1] in the same domain.
// =============================================================

// Register one frame against the reference using B-Spline FFD.
// fixedMat  = reference frame (affine-registered)
// movingMat = frame to refine   (affine-registered)
// Returns true on success; on failure registeredMat = movingMat.
bool registerFrameBSpline(
    const cv::Mat&        fixedMat,
    const cv::Mat&        movingMat,
    cv::Mat&              registeredMat,
    const RegionParameters& params
);

// Run B-Spline refinement on a full sequence (OpenMP parallel).
// affineFrames = output of Stage 1 (ECC Affine)
bool registerSequenceBSpline(
    const std::vector<cv::Mat>& affineFrames,
    std::vector<cv::Mat>&       registeredFrames,
    int                         referenceFrameIndex,
    const RegionParameters&     params
);

#endif // BSPLINE_REGISTER_H
