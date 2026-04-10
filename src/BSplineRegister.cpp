#include "BSplineRegister.h"
#include "Register.h"   // makeVesselMask

#include <iostream>
#include <itkMultiThreaderBase.h>

// ITK
#include <itkImage.h>
#include <itkBSplineTransform.h>
#include <itkBSplineTransformInitializer.h>
#include <itkImageRegistrationMethodv4.h>
#include <itkMattesMutualInformationImageToImageMetricv4.h>
#include <itkLBFGSBOptimizerv4.h>
#include <itkResampleImageFilter.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkImageMaskSpatialObject.h>

// =============================================================
// Type aliases — keep the boilerplate in one place
// =============================================================
using FloatImage2D   = itk::Image<float,         2>;
using UCharImage2D   = itk::Image<unsigned char, 2>;
using BSplineType    = itk::BSplineTransform<double, 2, 3>;   // cubic, 2-D
using MetricType     = itk::MattesMutualInformationImageToImageMetricv4<
                           FloatImage2D, FloatImage2D>;
using OptimizerType  = itk::LBFGSBOptimizerv4;
using RegType        = itk::ImageRegistrationMethodv4<
                           FloatImage2D, FloatImage2D, BSplineType>;
using ResampleType   = itk::ResampleImageFilter<FloatImage2D, FloatImage2D>;
using InterpType     = itk::LinearInterpolateImageFunction<FloatImage2D, double>;
using MaskObjectType = itk::ImageMaskSpatialObject<2>;
using InitializerType= itk::BSplineTransformInitializer<BSplineType, FloatImage2D>;

// =============================================================
// cv::Mat (CV_32F, continuous, row-major) ↔ ITK image (float,2D)
//
// ITK 2-D image memory: pixel[row][col] → linear index = row*W + col
// OpenCV CV_32F memory: same layout when mat.isContinuous()
// Direct memcpy is safe.
// =============================================================
static FloatImage2D::Pointer matToITK(const cv::Mat& mat)
{
    CV_Assert(mat.type() == CV_32F);

    auto img = FloatImage2D::New();
    FloatImage2D::SizeType  sz  = {{ (unsigned int)mat.cols,
                                     (unsigned int)mat.rows }};
    FloatImage2D::IndexType idx = {{ 0, 0 }};
    img->SetRegions(FloatImage2D::RegionType(idx, sz));
    img->Allocate();
    img->FillBuffer(0.0f);

    if (mat.isContinuous())
    {
        memcpy(img->GetBufferPointer(), mat.ptr<float>(0),
               (size_t)mat.rows * mat.cols * sizeof(float));
    }
    else
    {
        for (int y = 0; y < mat.rows; y++)
            memcpy(img->GetBufferPointer() + y * mat.cols,
                   mat.ptr<float>(y), mat.cols * sizeof(float));
    }
    return img;
}

static cv::Mat itkToMat(FloatImage2D::Pointer img)
{
    auto sz = img->GetLargestPossibleRegion().GetSize();
    cv::Mat mat((int)sz[1], (int)sz[0], CV_32F);
    memcpy(mat.ptr<float>(0), img->GetBufferPointer(),
           sz[1] * sz[0] * sizeof(float));
    return mat;
}

// =============================================================
// Build ITK tissue mask from an OpenCV vessel mask.
// vesselMask8: CV_8U, 255 = vessel pixel, 0 = tissue pixel
// Output:      CV_8U ITK image, 255 = tissue (MI included),
//              0 = vessel (MI excluded)
// =============================================================
static UCharImage2D::Pointer buildTissueMask(const cv::Mat& vesselMask8)
{
    auto img = UCharImage2D::New();
    UCharImage2D::SizeType  sz  = {{ (unsigned int)vesselMask8.cols,
                                     (unsigned int)vesselMask8.rows }};
    UCharImage2D::IndexType idx = {{ 0, 0 }};
    img->SetRegions(UCharImage2D::RegionType(idx, sz));
    img->Allocate();

    unsigned char* dst = img->GetBufferPointer();
    for (int y = 0; y < vesselMask8.rows; y++)
    {
        const unsigned char* src = vesselMask8.ptr<unsigned char>(y);
        for (int x = 0; x < vesselMask8.cols; x++)
            dst[y * vesselMask8.cols + x] = (src[x] == 0) ? 255 : 0;
    }
    return img;
}

// =============================================================
// registerFrameBSpline
// =============================================================
bool registerFrameBSpline(
    const cv::Mat&          fixedMat,
    const cv::Mat&          movingMat,
    cv::Mat&                registeredMat,
    const RegionParameters& params)
{
    CV_Assert(fixedMat.type()  == CV_32F);
    CV_Assert(movingMat.type() == CV_32F);

    // Limit ITK internal threads so one sequential B-Spline call
    // doesn't saturate all cores and blow out memory via ITK's own
    // multi-threaded metric gradient computation.
    itk::MultiThreaderBase::SetGlobalMaximumNumberOfThreads(4);

    // ---- Work at half resolution to keep memory manageable ----
    // 1344×1344 full-res ITK registration: ~2–3 GB per call
    // (pyramid copies + MattesMI gradient buffers + LBFGSB vectors).
    // Half-res (672×672): ~200–400 MB — safe for sequential use.
    // The output is upsampled back to full resolution after warping.
    cv::Mat fixedHalf, movingHalf;
    cv::resize(fixedMat,  fixedHalf,  cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
    cv::resize(movingMat, movingHalf, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);

    // ---- Convert to ITK images --------------------------------
    auto fixedImage  = matToITK(fixedHalf);
    auto movingImage = matToITK(movingHalf);

    // ---- Build vessel/tissue mask for the metric --------------
    // Iodine threshold 30 (8U), dilate 3px (half-res equivalent of 5px).
    cv::Mat fixed8, moving8;
    fixedHalf.convertTo(fixed8,  CV_8U, 255.0);
    movingHalf.convertTo(moving8, CV_8U, 255.0);
    cv::Mat vesselMask = makeVesselMask(fixed8, moving8, 30, 3);

    auto tissueMaskImg = buildTissueMask(vesselMask);
    auto maskObject    = MaskObjectType::New();
    maskObject->SetImage(tissueMaskImg);
    maskObject->Update();

    // ---- B-Spline transform -----------------------------------
    // Mesh size = image size / grid spacing.
    // BSplineTransformInitializer fills origin, direction, physical
    // dimensions from the fixed image automatically.
    auto bsplineTransform = BSplineType::New();

    BSplineType::MeshSizeType meshSize;
    unsigned int nodes = std::max(2u,
        (unsigned int)std::ceil((float)fixedMat.cols / params.bsplineGridSpacing));
    meshSize.Fill(nodes);

    auto initializer = InitializerType::New();
    initializer->SetTransform(bsplineTransform);
    initializer->SetImage(fixedImage);
    initializer->SetTransformDomainMeshSize(meshSize);
    initializer->InitializeTransform();
    bsplineTransform->SetIdentity();   // start from zero deformation

    // ---- Metric — Mattes MI with tissue mask ------------------
    auto metric = MetricType::New();
    metric->SetNumberOfHistogramBins(32);   // 32 bins: sufficient at half-res, lower memory
    metric->SetFixedImageMask(maskObject);

    // ---- Optimizer — LBFGSB with per-parameter bounds ---------
    // Bounding each control-point displacement to ±gridSpacing/2
    // prevents image folding (a medically unsafe artifact).
    auto optimizer = OptimizerType::New();
    optimizer->SetMaximumNumberOfFunctionEvaluations(params.bsplineMaxIter);
    optimizer->SetCostFunctionConvergenceFactor(1e7);  // ~1e-7 relative improvement
    optimizer->SetGradientConvergenceTolerance(1e-4);
    optimizer->SetTrace(false);

    unsigned int nParams = bsplineTransform->GetNumberOfParameters();
    double       bound   = params.bsplineGridSpacing * 0.5;
    OptimizerType::BoundValueType     lower(nParams, -bound);
    OptimizerType::BoundValueType     upper(nParams,  bound);
    OptimizerType::BoundSelectionType boundSel(nParams, 2);  // 2 = both bounds active
    optimizer->SetLowerBound(lower);
    optimizer->SetUpperBound(upper);
    optimizer->SetBoundSelection(boundSel);

    // ---- Registration framework -------------------------------
    auto registration = RegType::New();
    registration->SetFixedImage(fixedImage);
    registration->SetMovingImage(movingImage);
    registration->SetMetric(metric);
    registration->SetOptimizer(optimizer);
    registration->SetInitialTransform(bsplineTransform);
    registration->InPlaceOn();   // optimised transform written back to bsplineTransform

    // Multi-resolution image pyramid
    unsigned int nLevels = (unsigned int)params.bsplineMultiResLevels;
    registration->SetNumberOfLevels(nLevels);

    RegType::ShrinkFactorsArrayType    shrink(nLevels);
    RegType::SmoothingSigmasArrayType  sigma(nLevels);
    for (unsigned int l = 0; l < nLevels; l++)
    {
        unsigned int s = 1u << (nLevels - 1u - l);   // 4,2,1 or 2,1
        shrink[l] = s;
        sigma[l]  = s * 0.5;
    }
    registration->SetShrinkFactorsPerLevel(shrink);
    registration->SetSmoothingSigmasPerLevel(sigma);
    registration->SetSmoothingSigmasAreSpecifiedInPhysicalUnits(false);

    // Random 10% spatial sampling — sufficient at half-res, halves metric memory
    registration->SetMetricSamplingStrategy(RegType::RANDOM);
    registration->SetMetricSamplingPercentage(0.10);

    // ---- Run --------------------------------------------------
    try { registration->Update(); }
    catch (const itk::ExceptionObject& e)
    {
        std::cerr << "[BSpline] ITK exception: " << e.GetDescription() << "\n";
        registeredMat = movingMat.clone();
        return false;
    }

    // ---- Apply optimised transform at half resolution ---------
    auto resampler    = ResampleType::New();
    auto interpolator = InterpType::New();
    resampler->SetInput(movingImage);
    resampler->SetTransform(bsplineTransform);
    resampler->SetInterpolator(interpolator);
    resampler->SetSize(fixedImage->GetLargestPossibleRegion().GetSize());
    resampler->SetOutputOrigin(fixedImage->GetOrigin());
    resampler->SetOutputSpacing(fixedImage->GetSpacing());
    resampler->SetOutputDirection(fixedImage->GetDirection());
    resampler->SetDefaultPixelValue(0.0f);
    resampler->Update();

    // ---- Upsample result back to original full resolution -----
    cv::Mat halfResult = itkToMat(resampler->GetOutput());
    cv::resize(halfResult, registeredMat,
               cv::Size(fixedMat.cols, fixedMat.rows),
               0, 0, cv::INTER_LINEAR);

    // Clamp output to valid float32 range [0,1]
    cv::threshold(registeredMat, registeredMat, 0.0, 0.0, cv::THRESH_TOZERO);
    cv::threshold(registeredMat, registeredMat, 1.0, 1.0, cv::THRESH_TRUNC);

    return true;
}

// =============================================================
// registerSequenceBSpline
// =============================================================
bool registerSequenceBSpline(
    const std::vector<cv::Mat>& affineFrames,
    std::vector<cv::Mat>&       registeredFrames,
    int                         referenceFrameIndex,
    const RegionParameters&     params)
{
    if (affineFrames.empty()) return false;

    int n = (int)affineFrames.size();
    registeredFrames.resize(n);
    registeredFrames[referenceFrameIndex] = affineFrames[referenceFrameIndex].clone();

    int failCount = 0;

    // Sequential — B-Spline is memory-heavy at 1344×1344 (~200-400 MB
    // per call at half-res after the optimisation).  Running N threads
    // simultaneously would multiply that by the thread count and exhaust
    // RAM.  ITK already uses 4 internal threads (SetGlobalMaximumNumberOfThreads)
    // for metric gradient computation, so CPU utilisation stays good.
    for (int i = 0; i < n; i++)
    {
        if (i == referenceFrameIndex) continue;

        bool ok = registerFrameBSpline(
            affineFrames[referenceFrameIndex],
            affineFrames[i],
            registeredFrames[i],
            params);

        if (!ok)
        {
            registeredFrames[i] = affineFrames[i].clone();
            failCount++;
        }
    }

    std::cout << "[BSpline] Done  ref=" << referenceFrameIndex
              << "  region=" << params.name
              << "  grid=" << params.bsplineGridSpacing << "px"
              << "  failures=" << failCount << "\n";
    return true;
}
