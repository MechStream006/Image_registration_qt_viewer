#include "Register.h"
#include "BSplineRegister.h"
#include <iostream>
#include <cmath>
#include <limits>
#include <chrono>

// Simple scoped timer — prints elapsed ms on destruction
struct ScopedTimer {
    const char* label;
    std::chrono::high_resolution_clock::time_point t0;
    ScopedTimer(const char* l) : label(l), t0(std::chrono::high_resolution_clock::now()) {}
    ~ScopedTimer() {
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "[TIMER] " << label << ": " << (int)ms << " ms\n";
    }
};


// =============================================================
// Internal helper — build binary vessel mask
//
// Returns CV_8U mask where pixels are 255 if the frame is darker
// than the reference by more than `thresh` 8U units.
// These are iodine-filled vessel pixels.  Dilated by `dilateR` px
// to also cover vessel edges where gradient leaks would mislead flow.
//
// Key insight: instead of clipping (max), we replace vessel pixels
// in the frame with the reference value before running flow.
// This gives the flow zero signal in vessel regions → it cannot
// track vessels → no floating in the DSA output.
// =============================================================
cv::Mat makeVesselMask(const cv::Mat& ref8, const cv::Mat& frame8,
                        int thresh, int dilateR)
{
    CV_Assert(ref8.type() == CV_8U && frame8.type() == CV_8U);

    // Signed difference: ref - frame (vessels are darker with iodine)
    cv::Mat diff;
    cv::subtract(ref8, frame8, diff);   // saturates at 0 for negative

    // Adaptive threshold: use Otsu on the difference image.
    // If Otsu threshold is lower than the manual hint, prefer Otsu
    // (it adapts to actual iodine contrast). If Otsu finds no clear
    // bimodal split it returns 0 — fall back to manual threshold.
    cv::Mat otsuMask;
    double otsuVal = cv::threshold(diff, otsuMask, 0, 255,
                                   cv::THRESH_BINARY | cv::THRESH_OTSU);
    int effectiveThresh = (otsuVal > 5) ? (int)otsuVal : thresh;

    cv::Mat mask;
    cv::threshold(diff, mask, effectiveThresh, 255, cv::THRESH_BINARY);

    // Generous dilation to catch vessel edges and prevent flow from
    // being pulled by iodine gradients at vessel boundaries.
    if (dilateR > 0)
    {
        cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(2 * dilateR + 1, 2 * dilateR + 1));
        cv::dilate(mask, mask, kernel);
    }
    return mask;
}

// Smooth both channels of a CV_32FC2 flow field.
static void smoothFlow(cv::Mat& flow, double sigma = 7.0)
{
    cv::Mat ch[2];
    cv::split(flow, ch);
    cv::GaussianBlur(ch[0], ch[0], cv::Size(0, 0), sigma);
    cv::GaussianBlur(ch[1], ch[1], cv::Size(0, 0), sigma);
    cv::merge(ch, 2, flow);
}

// =============================================================
// Fit a global transform from dense flow vectors sampled outside
// vessel regions only.
//
// Why this eliminates floating completely:
//   Per-pixel flow warp has independent displacement at every pixel.
//   Even with vessel masking, flow vectors inside vessel regions are
//   non-zero (interpolated from nearby tissue) → vessels shift →
//   DSA floating.
//
//   Constraining to a global transform (affine or homography) means
//   every pixel is moved by the SAME mathematical rule.  Vessel pixels
//   move only as much as the surrounding tissue dictates — no
//   vessel-specific artifact is possible.
//
//   RANSAC rejects outlier flow vectors (vessel edges, motion
//   boundaries) and fits only the majority inliers (smooth tissue).
//
// motionType — one of:
//   cv::MOTION_AFFINE        → 6-DOF full affine (shear, non-uniform scale)
//   cv::MOTION_EUCLIDEAN     → 4-DOF (translation + rotation + uniform scale)
//   cv::MOTION_HOMOGRAPHY    → 8-DOF homography (perspective)
//
// Returns the warp matrix, or empty Mat on failure (caller must handle).
// =============================================================
static cv::Mat fitGlobalFromFlow(
    const cv::Mat& flow,
    const cv::Mat& vesselMask8,
    int motionType = cv::MOTION_AFFINE)
{
    // Adaptive stride so we sample ~2000–4000 grid points regardless of image size
    int minDim = std::min(flow.rows, flow.cols);
    int stride  = std::max(4, minDim / 64);

    std::vector<cv::Point2f> srcPts, dstPts;
    srcPts.reserve(4096);
    dstPts.reserve(4096);

    for (int y = stride / 2; y < flow.rows; y += stride)
    {
        for (int x = stride / 2; x < flow.cols; x += stride)
        {
            if (vesselMask8.at<uchar>(y, x) != 0) continue;  // skip vessel pixels

            cv::Vec2f f = flow.at<cv::Vec2f>(y, x);
            srcPts.push_back(cv::Point2f((float)x,         (float)y));
            dstPts.push_back(cv::Point2f((float)x + f[0],  (float)y + f[1]));
        }
    }

    if ((int)srcPts.size() < 20)
    {
        std::cout << "  [fitGlobal] too few non-vessel points (" << srcPts.size() << ") — skipping\n";
        return cv::Mat();
    }

    cv::Mat inliers;
    cv::Mat warp;
    const double RANSAC_THRESH = 2.5;   // pixels; tighter = more conservative fit

    if (motionType == cv::MOTION_HOMOGRAPHY)
    {
        warp = cv::findHomography(srcPts, dstPts, cv::RANSAC, RANSAC_THRESH, inliers);
    }
    else if (motionType == cv::MOTION_AFFINE)
    {
        warp = cv::estimateAffine2D(srcPts, dstPts, inliers, cv::RANSAC, RANSAC_THRESH);
    }
    else  // EUCLIDEAN / partial affine (translation + rotation + uniform scale)
    {
        warp = cv::estimateAffinePartial2D(srcPts, dstPts, inliers, cv::RANSAC, RANSAC_THRESH);
    }

    int inlierCount = inliers.empty() ? 0 : cv::countNonZero(inliers);
    std::cout << "  [fitGlobal] pts=" << srcPts.size()
              << "  inliers=" << inlierCount << "\n";

    return warp;
}

static void applyFlowWarp(const cv::Mat& src, cv::Mat& dst, const cv::Mat& flow)
{
    cv::Mat ch[2];
    cv::split(flow, ch);   // ch[0]=dx, ch[1]=dy

    cv::Mat mapX(src.size(), CV_32F);
    cv::Mat mapY(src.size(), CV_32F);

    for (int y = 0; y < src.rows; y++)
    {
        const float* dx = ch[0].ptr<float>(y);
        const float* dy = ch[1].ptr<float>(y);
        float* mx = mapX.ptr<float>(y);
        float* my = mapY.ptr<float>(y);
        for (int x = 0; x < src.cols; x++)
        {
            mx[x] = x + dx[x];
            my[x] = y + dy[x];
        }
    }
    cv::remap(src, dst, mapX, mapY, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
}

// =============================================================
// Auto-detect mask frame — minimum std-dev = pre-contrast frame
// =============================================================
int autoDetectMaskFrame(const std::vector<cv::Mat>& frames)
{
    if (frames.empty()) return 0;

    // Only search the first 30% of frames (minimum 3, maximum all).
    // Pre-contrast mask frames are always early in the acquisition —
    // searching later frames risks picking a post-contrast frame where
    // iodine has already arrived (uniform fill can lower std-dev).
    int searchEnd = std::max(3, (int)(frames.size() * 0.30));
    searchEnd = std::min(searchEnd, (int)frames.size());

    int    bestIdx = 0;
    double minStd  = std::numeric_limits<double>::max();

    for (int i = 0; i < searchEnd; i++)
    {
        cv::Scalar mean, stddev;
        cv::meanStdDev(frames[i], mean, stddev);
        if (stddev[0] < minStd) { minStd = stddev[0]; bestIdx = i; }
    }

    std::cout << "[AutoMask] Frame=" << bestIdx << "  std=" << minStd
              << "  (searched first " << searchEnd << " of " << frames.size() << ")\n";
    return bestIdx;
}

// =============================================================
// ECC — single frame (multi-scale pyramid, coarse → fine)
// =============================================================
bool registerFrameECC(
    const cv::Mat& ref,
    const cv::Mat& src,
    cv::Mat&       dst,
    ECCResult&     result,
    int            motionType,
    int            pyramidLevels,
    int            maxIterations,
    double         epsilon,
    const cv::Mat& initialWarp)
{
    std::vector<cv::Mat> refPyr(pyramidLevels);
    std::vector<cv::Mat> srcPyr(pyramidLevels);

    // Small pre-blur before pyramid: smooths sharp high-contrast edges (e.g.
    // skull boundary in Neuro) so they don't dominate the ECC gradient and
    // pull the optimizer away from the tissue alignment signal.
    // sigma=1.5 px — removes 1-2 pixel edge ringing without blurring vessels.
    const double ECC_PREBLUR = 1.5;
    cv::GaussianBlur(ref, refPyr[0], cv::Size(0, 0), ECC_PREBLUR);
    cv::GaussianBlur(src, srcPyr[0], cv::Size(0, 0), ECC_PREBLUR);

    for (int i = 1; i < pyramidLevels; i++)
    {
        cv::pyrDown(refPyr[i-1], refPyr[i]);
        cv::pyrDown(srcPyr[i-1], srcPyr[i]);
    }

    int warpRows = (motionType == cv::MOTION_HOMOGRAPHY) ? 3 : 2;

    // Use caller-supplied initialisation if provided (e.g. Phase Corr estimate
    // for neuro pipeline), otherwise start from identity transform.
    cv::Mat warp;
    if (!initialWarp.empty())
        warp = initialWarp.clone();
    else
        warp = cv::Mat::eye(warpRows, 3, CV_32F);

    cv::TermCriteria criteria(
        cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
        maxIterations, epsilon);

    for (int level = pyramidLevels - 1; level >= 0; level--)
    {
        double scale = std::pow(2.0, level);
        cv::Mat levelWarp = warp.clone();
        levelWarp.at<float>(0, 2) /= (float)scale;
        levelWarp.at<float>(1, 2) /= (float)scale;

        // Build interior mask at this pyramid level.
        // Threshold the reference to find the bright FOV (excludes collimator
        // shadow / black background), then erode by ~20 px (scaled) to pull
        // the mask away from the FOV boundary (skull edge in Neuro, body edge
        // elsewhere).  ECC ignores masked-out pixels, so the sharp boundary
        // gradient cannot dominate the optimizer — only interior tissue drives
        // the alignment.
        const int ERODE_FULLRES_PX = 20;
        int erodeR = std::max(2, (int)(ERODE_FULLRES_PX / scale + 0.5));
        cv::Mat levelMask;
        cv::threshold(refPyr[level], levelMask, 0.05, 1.0, cv::THRESH_BINARY);
        levelMask.convertTo(levelMask, CV_8U, 255.0);
        cv::Mat morphEl = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(2 * erodeR + 1, 2 * erodeR + 1));
        cv::erode(levelMask, levelMask, morphEl);

        // Safety: if mask is too small (< 10% of pixels), skip masking.
        int maskPx = cv::countNonZero(levelMask);
        int totalPx = levelMask.rows * levelMask.cols;
        cv::Mat eccMask = (maskPx > totalPx / 10) ? levelMask : cv::Mat();

        try
        {
            result.finalRho = cv::findTransformECC(
                refPyr[level], srcPyr[level], levelWarp, motionType, criteria, eccMask);
            result.converged = true;
        }
        catch (const cv::Exception& e)
        {
            std::cout << "[ECC] Failed at level " << level << ": " << e.what() << "\n";
            result.converged = false;
            dst = src.clone();
            return false;
        }

        levelWarp.at<float>(0, 2) *= (float)scale;
        levelWarp.at<float>(1, 2) *= (float)scale;
        warp = levelWarp;
    }

    result.warpMatrix = warp;
    cv::Size sz = ref.size();

    if (motionType == cv::MOTION_HOMOGRAPHY)
        cv::warpPerspective(src, dst, warp, sz, cv::INTER_LINEAR + cv::WARP_INVERSE_MAP);
    else
        cv::warpAffine(src, dst, warp, sz, cv::INTER_LINEAR + cv::WARP_INVERSE_MAP);

    return true;
}

// =============================================================
// ECC — full sequence (OpenMP parallel)
// =============================================================
bool registerSequenceECC(
    const std::vector<cv::Mat>& frames,
    std::vector<cv::Mat>&       registeredFrames,
    int    referenceFrameIndex,
    int    motionType,
    int    pyramidLevels,
    int    maxIterations,
    double epsilon,
    std::vector<cv::Mat>*       outWarps)
{
    ScopedTimer _t("ECC Registration");
    if (frames.empty()) return false;

    int n = (int)frames.size();
    registeredFrames.resize(n);
    registeredFrames[referenceFrameIndex] = frames[referenceFrameIndex].clone();

    int warpRows = (motionType == cv::MOTION_HOMOGRAPHY) ? 3 : 2;
    if (outWarps)
    {
        outWarps->resize(n);
        (*outWarps)[referenceFrameIndex] = cv::Mat::eye(warpRows, 3, CV_32F);
    }

    // Vessel masking: replace contrast-agent pixels in each frame with reference
    // values before ECC so the optimizer aligns tissue only, not vessels.
    const int    ECC_IODINE_THRESH = 30;
    const int    ECC_MASK_DILATE   = 5;
    const cv::Mat& ref32 = frames[referenceFrameIndex];
    cv::Mat ref8;
    ref32.convertTo(ref8, CV_8U, 255.0);

    int failCount = 0;

#pragma omp parallel for schedule(dynamic) reduction(+:failCount)
    for (int i = 0; i < n; i++)
    {
        if (i == referenceFrameIndex) continue;

        cv::Mat frame8;
        frames[i].convertTo(frame8, CV_8U, 255.0);
        cv::Mat vesselMask = makeVesselMask(ref8, frame8, ECC_IODINE_THRESH, ECC_MASK_DILATE);
        cv::Mat frameMasked = frames[i].clone();
        ref32.copyTo(frameMasked, vesselMask);

        cv::Mat tempDst;
        ECCResult result;
        bool ok = registerFrameECC(ref32, frameMasked, tempDst, result,
                                   motionType, pyramidLevels, maxIterations, epsilon);
        if (ok)
        {
            if (outWarps) (*outWarps)[i] = result.warpMatrix.clone();

            if (motionType == cv::MOTION_HOMOGRAPHY)
                cv::warpPerspective(frames[i], registeredFrames[i], result.warpMatrix,
                                    frames[i].size(),
                                    cv::INTER_LINEAR + cv::WARP_INVERSE_MAP,
                                    cv::BORDER_REPLICATE);
            else
                cv::warpAffine(frames[i], registeredFrames[i], result.warpMatrix,
                               frames[i].size(),
                               cv::INTER_LINEAR + cv::WARP_INVERSE_MAP,
                               cv::BORDER_REPLICATE);
        }
        else
        {
            if (outWarps) (*outWarps)[i] = cv::Mat::eye(warpRows, 3, CV_32F);
            registeredFrames[i] = frames[i].clone();
            failCount++;
        }
    }

    std::cout << "[ECC] Done  ref=" << referenceFrameIndex
              << "  motion=" << motionType
              << "  levels=" << pyramidLevels
              << "  failures=" << failCount << "\n";
    return true;
}

// =============================================================
// Phase Correlation — translation only, FFT-based
//
// Fastest method. Computes frequency-domain cross-correlation
// to find the dominant X/Y shift in a single operation.
// Only models translation — no rotation or local deformation.
// Best for: table movement, patient shift.
// =============================================================
bool registerSequencePhaseCorr(
    const std::vector<cv::Mat>& frames,
    std::vector<cv::Mat>&       registeredFrames,
    int referenceFrameIndex,
    std::vector<cv::Mat>*  outWarps)
{
    ScopedTimer _t("PhaseCorr Registration");
    if (frames.empty()) return false;

    int n = (int)frames.size();
    registeredFrames.resize(n);
    registeredFrames[referenceFrameIndex] = frames[referenceFrameIndex].clone();

    if (outWarps)
    {
        outWarps->resize(n);
        (*outWarps)[referenceFrameIndex] = cv::Mat::eye(2, 3, CV_32F);
    }

    // Hanning window reduces spectral leakage — improves sub-pixel accuracy
    cv::Mat hann;
    cv::createHanningWindow(hann, frames[0].size(), CV_32F);

    // Pre-blur reference once
    const double PC_PREBLUR_SIGMA = 3.0;
    const float  PC_IODINE_THRESH = 20.0f / 255.0f; // float32 threshold
    const int    PC_MASK_DILATE   = 3;

    cv::Mat ref32 = frames[referenceFrameIndex];
    cv::Mat ref8, ref8Blur;
    ref32.convertTo(ref8, CV_8U, 255.0);
    cv::GaussianBlur(ref32, ref8Blur, cv::Size(0, 0), PC_PREBLUR_SIGMA);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++)
    {
        if (i == referenceFrameIndex) continue;

        // Vessel replacement in float32 domain
        cv::Mat frame8;
        frames[i].convertTo(frame8, CV_8U, 255.0);
        cv::Mat vesselMask = makeVesselMask(ref8, frame8, (int)(PC_IODINE_THRESH * 255), PC_MASK_DILATE);

        cv::Mat frameMasked = frames[i].clone();
        ref32.copyTo(frameMasked, vesselMask);

        cv::Mat frameBlur;
        cv::GaussianBlur(frameMasked, frameBlur, cv::Size(0, 0), PC_PREBLUR_SIGMA);

        cv::Point2d shift = cv::phaseCorrelate(ref8Blur, frameBlur, hann);

        // WARP_INVERSE_MAP: matrix maps dst → src, so translation = +shift
        cv::Mat warp = (cv::Mat_<float>(2, 3)
            << 1.f, 0.f, (float)shift.x,
               0.f, 1.f, (float)shift.y);

        if (outWarps) (*outWarps)[i] = warp.clone();

        cv::warpAffine(frames[i], registeredFrames[i], warp, frames[i].size(),
                       cv::INTER_LINEAR + cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);
    }

    std::cout << "[PhaseCorr] Done  ref=" << referenceFrameIndex << "\n";
    return true;
}

// =============================================================
// DIS → Full Affine (6 DOF)
//
// Computes dense DIS optical flow on vessel-masked frames, then
// fits a full 6-DOF affine transform via RANSAC using only
// non-vessel flow vectors.
//
// Why global transform eliminates floating completely:
//   Per-pixel flow warp has independent displacement at each pixel.
//   Even masked correctly, flow vectors inside vessel regions are
//   interpolated from tissue borders → vessels shift → DSA floating.
//   A global affine applies the SAME mathematical transform to every
//   pixel.  Vessel pixels move only as the tissue majority dictates.
//   No vessel-specific artifact is geometrically possible.
//
// 6-DOF affine handles: translation, rotation, shear, non-uniform
// scale.  Covers breathing, table shift, moderate patient movement.
// =============================================================
bool registerSequenceDIS(
    const std::vector<cv::Mat>& frames,
    std::vector<cv::Mat>&       registeredFrames,
    int referenceFrameIndex,
    std::vector<cv::Mat>*  outWarps)
{
    if (frames.empty()) return false;

    int n = (int)frames.size();
    registeredFrames.resize(n);
    registeredFrames[referenceFrameIndex] = frames[referenceFrameIndex].clone();
    if (outWarps) { outWarps->resize(n); (*outWarps)[referenceFrameIndex] = cv::Mat::eye(2, 3, CV_32F); }

    const int    DIS_IODINE_THRESH = 30;
    const int    DIS_MASK_DILATE   = 5;
    const double DIS_PREBLUR_SIGMA = 4.0;
    const double DIS_FLOW_SIGMA    = 5.0;   // light smooth before RANSAC fit

    cv::Mat ref8, ref8Blur;
    frames[referenceFrameIndex].convertTo(ref8, CV_8U, 255.0);
    cv::GaussianBlur(ref8, ref8Blur, cv::Size(0, 0), DIS_PREBLUR_SIGMA);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++)
    {
        if (i == referenceFrameIndex) continue;

        cv::Mat frame8;
        frames[i].convertTo(frame8, CV_8U, 255.0);
        cv::Mat vesselMask = makeVesselMask(ref8, frame8, DIS_IODINE_THRESH, DIS_MASK_DILATE);

        cv::Mat frameMasked = frame8.clone();
        ref8.copyTo(frameMasked, vesselMask);

        cv::Mat frameMaskedBlur;
        cv::GaussianBlur(frameMasked, frameMaskedBlur, cv::Size(0, 0), DIS_PREBLUR_SIGMA);

        auto dis = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_FAST);
        cv::Mat flow;
        dis->calc(ref8Blur, frameMaskedBlur, flow);
        smoothFlow(flow, DIS_FLOW_SIGMA);

        // Fit 6-DOF affine from non-vessel flow vectors via RANSAC
        cv::Mat warp = fitGlobalFromFlow(flow, vesselMask, cv::MOTION_AFFINE);
        if (warp.empty())
        {
            if (outWarps) (*outWarps)[i] = cv::Mat::eye(2, 3, CV_32F);
            registeredFrames[i] = frames[i].clone();
            continue;
        }

        if (outWarps) (*outWarps)[i] = warp.clone();
        cv::warpAffine(frames[i], registeredFrames[i], warp, frames[i].size(),
                       cv::INTER_LINEAR + cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);
    }

    std::cout << "[DIS→Affine] Done  ref=" << referenceFrameIndex << "\n";
    return true;
}

// =============================================================
// DIS → Partial Affine / Euclidean (4 DOF)
//
// Same DIS flow estimation as above, but fits a more constrained
// 4-DOF transform: translation + rotation + uniform scale only.
// No shear.  More stable when tissue coverage is uneven.
//
// Use this when full affine produces distortion on low-contrast
// frames or when the vessel area is large relative to the image.
// =============================================================
bool registerSequenceFarnebackGPU(
    const std::vector<cv::Mat>& frames,
    std::vector<cv::Mat>&       registeredFrames,
    int referenceFrameIndex,
    std::vector<cv::Mat>*  outWarps)
{
    if (frames.empty()) return false;

    int n = (int)frames.size();
    registeredFrames.resize(n);
    registeredFrames[referenceFrameIndex] = frames[referenceFrameIndex].clone();
    if (outWarps) { outWarps->resize(n); (*outWarps)[referenceFrameIndex] = cv::Mat::eye(2, 3, CV_32F); }

    const int    FB_IODINE_THRESH = 25;   // slightly more sensitive — catches dim vessel edges
    const int    FB_MASK_DILATE   = 6;
    const double FB_PREBLUR_SIGMA = 5.0;
    const double FB_FLOW_SIGMA    = 5.0;

    cv::Mat ref8, ref8Blur;
    frames[referenceFrameIndex].convertTo(ref8, CV_8U, 255.0);
    cv::GaussianBlur(ref8, ref8Blur, cv::Size(0, 0), FB_PREBLUR_SIGMA);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++)
    {
        if (i == referenceFrameIndex) continue;

        cv::Mat frame8;
        frames[i].convertTo(frame8, CV_8U, 255.0);
        cv::Mat vesselMask = makeVesselMask(ref8, frame8, FB_IODINE_THRESH, FB_MASK_DILATE);

        cv::Mat frameMasked = frame8.clone();
        ref8.copyTo(frameMasked, vesselMask);

        cv::Mat frameMaskedBlur;
        cv::GaussianBlur(frameMasked, frameMaskedBlur, cv::Size(0, 0), FB_PREBLUR_SIGMA);

        auto dis = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_FAST);
        cv::Mat flow;
        dis->calc(ref8Blur, frameMaskedBlur, flow);
        smoothFlow(flow, FB_FLOW_SIGMA);

        // Fit 4-DOF partial affine (translation + rotation + uniform scale)
        cv::Mat warp = fitGlobalFromFlow(flow, vesselMask, cv::MOTION_EUCLIDEAN);
        if (warp.empty())
        {
            if (outWarps) (*outWarps)[i] = cv::Mat::eye(2, 3, CV_32F);
            registeredFrames[i] = frames[i].clone();
            continue;
        }

        if (outWarps) (*outWarps)[i] = warp.clone();
        cv::warpAffine(frames[i], registeredFrames[i], warp, frames[i].size(),
                       cv::INTER_LINEAR + cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);
    }

    std::cout << "[DIS→Partial Affine] Done  ref=" << referenceFrameIndex << "\n";
    return true;
}

// =============================================================
// DIS → Homography (8 DOF)
//
// Fits an 8-DOF homography from non-vessel flow vectors.
// Handles perspective distortion — useful when the C-arm tilts
// slightly between the mask frame and contrast frames, causing
// the image plane to change orientation.
//
// More DOF = more expressive but more sensitive to outliers.
// RANSAC handles this; however if tissue coverage is <30% of
// the image, prefer the affine method instead.
// =============================================================
bool registerSequenceNvidiaFlow(
    const std::vector<cv::Mat>& frames,
    std::vector<cv::Mat>&       registeredFrames,
    int referenceFrameIndex,
    std::vector<cv::Mat>*  outWarps)
{
    if (frames.empty()) return false;

    int n = (int)frames.size();
    registeredFrames.resize(n);
    registeredFrames[referenceFrameIndex] = frames[referenceFrameIndex].clone();
    if (outWarps) { outWarps->resize(n); (*outWarps)[referenceFrameIndex] = cv::Mat::eye(3, 3, CV_64F); }

    const int    NV_IODINE_THRESH = 25;
    const int    NV_MASK_DILATE   = 6;
    const double NV_PREBLUR_SIGMA = 5.0;
    const double NV_FLOW_SIGMA    = 5.0;

    cv::Mat ref8, ref8Blur;
    frames[referenceFrameIndex].convertTo(ref8, CV_8U, 255.0);
    cv::GaussianBlur(ref8, ref8Blur, cv::Size(0, 0), NV_PREBLUR_SIGMA);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++)
    {
        if (i == referenceFrameIndex) continue;

        cv::Mat frame8;
        frames[i].convertTo(frame8, CV_8U, 255.0);
        cv::Mat vesselMask = makeVesselMask(ref8, frame8, NV_IODINE_THRESH, NV_MASK_DILATE);

        cv::Mat frameMasked = frame8.clone();
        ref8.copyTo(frameMasked, vesselMask);

        cv::Mat frameMaskedBlur;
        cv::GaussianBlur(frameMasked, frameMaskedBlur, cv::Size(0, 0), NV_PREBLUR_SIGMA);

        auto dis = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_FAST);
        cv::Mat flow;
        dis->calc(ref8Blur, frameMaskedBlur, flow);
        smoothFlow(flow, NV_FLOW_SIGMA);

        // Fit 8-DOF homography from non-vessel flow vectors
        cv::Mat H = fitGlobalFromFlow(flow, vesselMask, cv::MOTION_HOMOGRAPHY);
        if (H.empty())
        {
            if (outWarps) (*outWarps)[i] = cv::Mat::eye(3, 3, CV_64F);
            registeredFrames[i] = frames[i].clone();
            continue;
        }

        if (outWarps) (*outWarps)[i] = H.clone();
        cv::warpPerspective(frames[i], registeredFrames[i], H, frames[i].size(),
                            cv::INTER_LINEAR + cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);
    }

    std::cout << "[DIS→Homography] Done  ref=" << referenceFrameIndex << "\n";
    return true;
}

// =============================================================
// Neuro pipeline: Phase Corr translation init → ECC Euclidean
//
// Phase Corr gives a precise translation estimate in ~5 ms per
// frame.  This pre-initialises the ECC warp so the optimiser
// only needs to solve for the residual rotation (1 DOF instead
// of searching all 3 DOF from scratch).
//
// MOTION_EUCLIDEAN (3 DOF: tx, ty, θ) is physically correct for
// a rigid skull.  Affine's extra DOF (shear, scale) cannot occur
// for bone and introduce optimisation noise.
// =============================================================
// =============================================================
// Phase Corr + ECC — shared engine for all region pipelines.
// motionType: cv::MOTION_EUCLIDEAN (neuro) or cv::MOTION_AFFINE
//             (cardiac / abdomen / peripheral-with-init).
// OpenMP parallel across frames — each thread independent.
// =============================================================
static bool registerSequencePhaseECC(
    const std::vector<cv::Mat>& frames,
    std::vector<cv::Mat>&       registeredFrames,
    int                         referenceFrameIndex,
    int                         motionType,
    int                         eccLevels,
    int                         eccIter,
    double                      eccEps,
    std::vector<cv::Mat>*       outWarps = nullptr)
{
    ScopedTimer _t("PhaseECC Registration");
    if (frames.empty()) return false;

    int n = (int)frames.size();
    registeredFrames.resize(n);
    registeredFrames[referenceFrameIndex] = frames[referenceFrameIndex].clone();

    if (outWarps)
    {
        outWarps->resize(n);
        (*outWarps)[referenceFrameIndex] = cv::Mat::eye(2, 3, CV_32F);
    }

    const int    IODINE_THRESH = 30;
    const int    MASK_DILATE   = 5;
    const double PREBLUR       = 2.0;

    const cv::Mat& ref32 = frames[referenceFrameIndex];
    cv::Mat ref8;
    ref32.convertTo(ref8, CV_8U, 255.0);

    cv::Mat hann;
    cv::createHanningWindow(hann, frames[0].size(), CV_32F);

    cv::Mat refBlur;
    cv::GaussianBlur(ref32, refBlur, cv::Size(0, 0), PREBLUR);

    int failCount = 0;

#pragma omp parallel for schedule(dynamic) reduction(+:failCount)
    for (int i = 0; i < n; i++)
    {
        if (i == referenceFrameIndex) continue;

        cv::Mat frame8;
        frames[i].convertTo(frame8, CV_8U, 255.0);
        cv::Mat vesselMask = makeVesselMask(ref8, frame8, IODINE_THRESH, MASK_DILATE);

        cv::Mat frameMasked = frames[i].clone();
        ref32.copyTo(frameMasked, vesselMask);

        cv::Mat frameMaskedBlur;
        cv::GaussianBlur(frameMasked, frameMaskedBlur, cv::Size(0, 0), PREBLUR);

        cv::Point2d shift = cv::phaseCorrelate(refBlur, frameMaskedBlur, hann);

        cv::Mat initWarp = (cv::Mat_<float>(2, 3)
            << 1.f, 0.f, (float)shift.x,
               0.f, 1.f, (float)shift.y);

        // Estimate warp from vessel-masked frame — tissue only drives ECC.
        // Apply the recovered warp to the original (unmasked) frame.
        cv::Mat tempDst;
        ECCResult result;
        bool ok = registerFrameECC(
            ref32, frameMasked, tempDst, result,
            motionType, eccLevels, eccIter, eccEps, initWarp);

        if (ok)
        {
            if (outWarps) (*outWarps)[i] = result.warpMatrix.clone();
            cv::warpAffine(frames[i], registeredFrames[i], result.warpMatrix,
                           frames[i].size(),
                           cv::INTER_LINEAR + cv::WARP_INVERSE_MAP,
                           cv::BORDER_REPLICATE);
        }
        else
        {
            if (outWarps) (*outWarps)[i] = initWarp.clone();
            cv::warpAffine(frames[i], registeredFrames[i], initWarp,
                           frames[i].size(),
                           cv::INTER_LINEAR + cv::WARP_INVERSE_MAP,
                           cv::BORDER_REPLICATE);
            failCount++;
        }
    }

    std::cout << "[PhaseECC] Done  ref=" << referenceFrameIndex
              << "  motionType=" << motionType
              << "  failures=" << failCount << "\n";
    return true;
}

bool registerSequenceNeuro(
    const std::vector<cv::Mat>& frames,
    std::vector<cv::Mat>&       registeredFrames,
    int referenceFrameIndex,
    std::vector<cv::Mat>*  outWarps)
{
    return registerSequencePhaseECC(frames, registeredFrames, referenceFrameIndex,
                                    cv::MOTION_EUCLIDEAN, 6, 500, 1e-7, outWarps);
}

// Forward declaration
static std::vector<cv::Mat> buildVesselMasksFromDSA(
    const std::vector<cv::Mat>& dsaFrames16, int referenceFrameIndex);

// =============================================================
// Region-aware dispatcher
// =============================================================
bool registerSequenceForRegion(
    const std::vector<cv::Mat>& frames,
    std::vector<cv::Mat>&       registeredFrames,
    int                         referenceFrameIndex,
    BodyRegion                  region,
    std::vector<cv::Mat>*       outWarps,
    std::vector<cv::Mat>*       outDeformFields)
{
    ScopedTimer _t("ForRegion (full pipeline)");
    const RegionParameters& p = getRegionParams(region);
    bool ok = false;

    // All regions now use parameters from RegionConfig.
    // Neuro, Cardiac, Abdomen: Phase Corr init + ECC refinement
    // Peripheral, Auto: ECC only (no phase corr init)
    if (p.usePhaseCorInit)
    {
        ok = registerSequencePhaseECC(frames, registeredFrames,
                                      referenceFrameIndex,
                                      p.eccMotionType,
                                      p.eccPyramidLevels,
                                      p.eccMaxIterations,
                                      p.eccEpsilon, outWarps);
    }
    else
    {
        ok = registerSequenceECC(frames, registeredFrames, referenceFrameIndex,
                                  p.eccMotionType, p.eccPyramidLevels,
                                  p.eccMaxIterations, p.eccEpsilon, outWarps);
    }

    if (!ok) return false;

    // Stage 2: non-rigid refinement
    if (p.useBSpline)
    {
        // MI + B-Spline FFD via ITK (Neuro)
        ScopedTimer _bs("BSpline MI refinement");
        std::vector<cv::Mat> bsplineResult;
        registerSequenceBSpline(registeredFrames, bsplineResult,
                                referenceFrameIndex, p);
        registeredFrames = std::move(bsplineResult);
    }
    else if (p.useDeformable)
    {
        // DIS optical flow (Cardiac, Abdomen, Peripheral)
        std::vector<cv::Mat> deformResult;
        std::vector<cv::Mat> fields;
        registerSequenceDeformable(registeredFrames, deformResult, fields,
                                   referenceFrameIndex, p.deformRegSigma);
        registeredFrames = std::move(deformResult);
        if (outDeformFields) *outDeformFields = std::move(fields);
    }

    return true;
}

// =============================================================
// ORB Feature Matching → Affine via RANSAC
//
// Detects ORB keypoints only in tissue regions (vessel mask
// excluded from frame; reference is pre-contrast so all pixels
// are tissue).  Matches with BFMatcher + Lowe ratio test.
// Fits full 6-DOF affine via RANSAC on matched point pairs.
//
// Completely different paradigm from flow-based methods:
//   - No dense correspondence needed
//   - Robust to illumination/contrast changes between frames
//   - Works well even with large patient motion
//   - Falls back gracefully if too few inliers found
// =============================================================
bool registerSequenceORBAffine(
    const std::vector<cv::Mat>& frames,
    std::vector<cv::Mat>&       registeredFrames,
    int referenceFrameIndex,
    std::vector<cv::Mat>*  outWarps)
{
    ScopedTimer _t("ORB Affine Registration");
    if (frames.empty()) return false;

    int n = (int)frames.size();
    registeredFrames.resize(n);
    registeredFrames[referenceFrameIndex] = frames[referenceFrameIndex].clone();
    if (outWarps) { outWarps->resize(n); (*outWarps)[referenceFrameIndex] = cv::Mat::eye(2, 3, CV_32F); }

    const int   ORB_IODINE_THRESH = 30;
    const int   ORB_MASK_DILATE   = 5;
    const int   ORB_MAX_FEATURES  = 2000;
    const float RATIO_THRESH      = 0.75f;  // Lowe ratio test threshold
    const double RANSAC_THRESH    = 3.0;    // pixels

    cv::Mat ref8;
    frames[referenceFrameIndex].convertTo(ref8, CV_8U, 255.0);

    // Detect reference keypoints once — no vessel mask needed (pre-contrast frame)
    auto refOrb = cv::ORB::create(ORB_MAX_FEATURES);
    std::vector<cv::KeyPoint> refKP;
    cv::Mat refDesc;
    refOrb->detectAndCompute(ref8, cv::noArray(), refKP, refDesc);

    if (refDesc.empty())
    {
        std::cout << "[ORB→Affine] Reference has no keypoints — falling back to identity\n";
        for (int i = 0; i < n; i++)
            if (i != referenceFrameIndex) registeredFrames[i] = frames[i].clone();
        return false;
    }

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++)
    {
        if (i == referenceFrameIndex) continue;

        cv::Mat frame8;
        frames[i].convertTo(frame8, CV_8U, 255.0);
        cv::Mat vesselMask = makeVesselMask(ref8, frame8, ORB_IODINE_THRESH, ORB_MASK_DILATE);

        // Tissue-only mask for frame (ORB ignores vessel regions)
        cv::Mat tissueMask;
        cv::bitwise_not(vesselMask, tissueMask);

        // Per-thread ORB instance for thread safety
        auto frameOrb = cv::ORB::create(ORB_MAX_FEATURES);
        std::vector<cv::KeyPoint> frameKP;
        cv::Mat frameDesc;
        frameOrb->detectAndCompute(frame8, tissueMask, frameKP, frameDesc);

        if (frameDesc.empty() || (int)frameKP.size() < 10)
        {
            if (outWarps) (*outWarps)[i] = cv::Mat::eye(2, 3, CV_32F);
            registeredFrames[i] = frames[i].clone();
            continue;
        }

        // BFMatcher with kNN(2) for ratio test
        cv::BFMatcher matcher(cv::NORM_HAMMING, false);
        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher.knnMatch(refDesc, frameDesc, knnMatches, 2);

        std::vector<cv::Point2f> srcPts, dstPts;
        for (auto& m : knnMatches)
        {
            if (m.size() >= 2 && m[0].distance < RATIO_THRESH * m[1].distance)
            {
                srcPts.push_back(refKP[m[0].queryIdx].pt);
                dstPts.push_back(frameKP[m[0].trainIdx].pt);
            }
        }

        if ((int)srcPts.size() < 10)
        {
            if (outWarps) (*outWarps)[i] = cv::Mat::eye(2, 3, CV_32F);
            registeredFrames[i] = frames[i].clone();
            continue;
        }

        cv::Mat inliers;
        cv::Mat warp = cv::estimateAffine2D(srcPts, dstPts, inliers, cv::RANSAC, RANSAC_THRESH);

        int inlierCount = inliers.empty() ? 0 : cv::countNonZero(inliers);
        std::cout << "  [ORB→Affine] frame=" << i
                  << "  matches=" << srcPts.size()
                  << "  inliers=" << inlierCount << "\n";

        if (warp.empty())
        {
            if (outWarps) (*outWarps)[i] = cv::Mat::eye(2, 3, CV_32F);
            registeredFrames[i] = frames[i].clone();
            continue;
        }

        if (outWarps) (*outWarps)[i] = warp.clone();
        cv::warpAffine(frames[i], registeredFrames[i], warp, frames[i].size(),
                       cv::INTER_LINEAR + cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);
    }

    std::cout << "[ORB→Affine] Done  ref=" << referenceFrameIndex << "\n";
    return true;
}

// =============================================================
// Real Farneback Dense Optical Flow — per-pixel warp
//
// Uses cv::calcOpticalFlowFarneback (polynomial expansion basis).
// Per-pixel warp is applied directly — may produce floating
// artifacts in DSA if vessels move between frames.
// Included for comparison and for non-DSA registration use.
// =============================================================
bool registerSequenceFarnebackFlow(
    const std::vector<cv::Mat>& frames,
    std::vector<cv::Mat>&       registeredFrames,
    int referenceFrameIndex)
{
    if (frames.empty()) return false;

    int n = (int)frames.size();
    registeredFrames.resize(n);
    registeredFrames[referenceFrameIndex] = frames[referenceFrameIndex].clone();

    const int    FB_IODINE_THRESH = 30;
    const int    FB_MASK_DILATE   = 5;
    const double FB_PREBLUR_SIGMA = 2.0;  // Farneback handles its own smoothing via poly_sigma
    // Farneback params: pyr_scale, levels, winsize, iterations, poly_n, poly_sigma
    const double FB_PYR_SCALE  = 0.5;
    const int    FB_LEVELS     = 4;
    const int    FB_WINSIZE    = 15;
    const int    FB_ITERATIONS = 3;
    const int    FB_POLY_N     = 5;
    const double FB_POLY_SIGMA = 1.2;

    cv::Mat ref8;
    frames[referenceFrameIndex].convertTo(ref8, CV_8U, 255.0);
    cv::Mat ref8Blur;
    cv::GaussianBlur(ref8, ref8Blur, cv::Size(0, 0), FB_PREBLUR_SIGMA);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++)
    {
        if (i == referenceFrameIndex) continue;

        cv::Mat frame8;
        frames[i].convertTo(frame8, CV_8U, 255.0);
        cv::Mat vesselMask = makeVesselMask(ref8, frame8, FB_IODINE_THRESH, FB_MASK_DILATE);

        cv::Mat frameMasked = frame8.clone();
        ref8.copyTo(frameMasked, vesselMask);

        cv::Mat frameMaskedBlur;
        cv::GaussianBlur(frameMasked, frameMaskedBlur, cv::Size(0, 0), FB_PREBLUR_SIGMA);

        cv::Mat flow;
        cv::calcOpticalFlowFarneback(
            ref8Blur, frameMaskedBlur, flow,
            FB_PYR_SCALE, FB_LEVELS, FB_WINSIZE,
            FB_ITERATIONS, FB_POLY_N, FB_POLY_SIGMA, 0);

        applyFlowWarp(frames[i], registeredFrames[i], flow);
    }

    std::cout << "[Farneback Flow] Done  ref=" << referenceFrameIndex << "\n";
    return true;
}

// =============================================================
// DIS ULTRAFAST Raw Per-pixel Flow
//
// Original aggressive DIS variant — per-pixel warp applied
// directly without global-transform fitting.
// May produce floating in DSA.  Included for comparison.
// =============================================================
bool registerSequenceDISFlowRaw(
    const std::vector<cv::Mat>& frames,
    std::vector<cv::Mat>&       registeredFrames,
    int referenceFrameIndex)
{
    if (frames.empty()) return false;

    int n = (int)frames.size();
    registeredFrames.resize(n);
    registeredFrames[referenceFrameIndex] = frames[referenceFrameIndex].clone();

    const int    NV_IODINE_THRESH = 20;
    const int    NV_MASK_DILATE   = 8;
    const double NV_PREBLUR_SIGMA = 6.0;
    const double NV_FLOW_SIGMA    = 40.0;  // heavy smooth — best effort to suppress floating

    cv::Mat ref8, ref8Blur;
    frames[referenceFrameIndex].convertTo(ref8, CV_8U, 255.0);
    cv::GaussianBlur(ref8, ref8Blur, cv::Size(0, 0), NV_PREBLUR_SIGMA);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++)
    {
        if (i == referenceFrameIndex) continue;

        cv::Mat frame8;
        frames[i].convertTo(frame8, CV_8U, 255.0);
        cv::Mat vesselMask = makeVesselMask(ref8, frame8, NV_IODINE_THRESH, NV_MASK_DILATE);

        cv::Mat frameMasked = frame8.clone();
        ref8.copyTo(frameMasked, vesselMask);

        cv::Mat frameMaskedBlur;
        cv::GaussianBlur(frameMasked, frameMaskedBlur, cv::Size(0, 0), NV_PREBLUR_SIGMA);

        auto dis = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_ULTRAFAST);
        cv::Mat flow;
        dis->calc(ref8Blur, frameMaskedBlur, flow);
        smoothFlow(flow, NV_FLOW_SIGMA);

        applyFlowWarp(frames[i], registeredFrames[i], flow);
    }

    std::cout << "[DIS Flow Raw] Done  ref=" << referenceFrameIndex << "\n";
    return true;
}

// =============================================================
// Dispatcher
// =============================================================
bool registerSequence(
    const std::vector<cv::Mat>& frames,
    std::vector<cv::Mat>&       registeredFrames,
    int                         referenceFrameIndex,
    RegistrationMethod          method,
    std::vector<cv::Mat>*       outWarps)
{
    switch (method)
    {
    case REG_ECC_CPU:
        return registerSequenceECC(frames, registeredFrames, referenceFrameIndex,
                                   cv::MOTION_AFFINE, 3, 100, 1e-5, outWarps);
    case REG_PHASE_CORR:
        return registerSequencePhaseCorr(frames, registeredFrames, referenceFrameIndex, outWarps);
    case REG_DIS_CPU:
        return registerSequenceDIS(frames, registeredFrames, referenceFrameIndex, outWarps);
    case REG_FARNEBACK_GPU:
        return registerSequenceFarnebackGPU(frames, registeredFrames, referenceFrameIndex, outWarps);
    case REG_NVIDIA_FLOW:
        return registerSequenceNvidiaFlow(frames, registeredFrames, referenceFrameIndex, outWarps);
    case REG_ECC_TRANSLATION:
        return registerSequenceECC(frames, registeredFrames, referenceFrameIndex,
                                   cv::MOTION_TRANSLATION, 4, 100, 1e-5, outWarps);
    case REG_ECC_HOMOGRAPHY:
        return registerSequenceECC(frames, registeredFrames, referenceFrameIndex,
                                   cv::MOTION_HOMOGRAPHY, 3, 100, 1e-5, outWarps);
    case REG_ORB_AFFINE:
        return registerSequenceORBAffine(frames, registeredFrames, referenceFrameIndex, outWarps);
    case REG_FARNEBACK_FLOW:
        return registerSequenceFarnebackFlow(frames, registeredFrames, referenceFrameIndex);
    case REG_DIS_FLOW_RAW:
        return registerSequenceDISFlowRaw(frames, registeredFrames, referenceFrameIndex);
    default:
        return registerSequenceECC(frames, registeredFrames, referenceFrameIndex,
                                   cv::MOTION_AFFINE, 3, 100, 1e-5, outWarps);
    }
}

// =============================================================
// DSA subtraction
//
// Formula:  DSA(i) = clamp( (In(i) − Ref(i)) * gain + 0.25, 0, 1 )
//
// offset = 0.25 (= 16384/65535 ≈ 65536/4 / 65535)
//   • Neutral background pixels land at 0.25 (16383 in uint16)
//   • Vessel pixels (In < Ref after iodine injection) are < 0.25
//   • Gain scales only the vessel deviation — background stays fixed
//     at 0.25 regardless of gain, so no background clipping
//
// VTK display: level=16383 window=32767 shows [0, 32767]
//   • Background  → mid-gray  (16383)
//   • Strong vessels → dark/black (< 16383, approaching 0)
//
// CLAHE clipLimit=2.0 (gentle) — optional, for low-contrast data
// =============================================================
void computeDSA(
    const std::vector<cv::Mat>& frames,
    std::vector<cv::Mat>&       dsaFrames,
    int         maskFrameIndex,
    float       gain,
    bool        useClahe,
    float       bgSuppressSigma,
    DSAMaskMode maskMode)
{
    if (frames.empty()) return;

    const int   n      = (int)frames.size();
    const float OFFSET = 0.25f;

    dsaFrames.resize(n);

    // ------------------------------------------------------------------
    // Build reference image according to mask mode
    // ------------------------------------------------------------------
    cv::Mat ref;

    if (maskMode == MASK_TEMPORAL_MEDIAN)
    {
        // Per-pixel temporal median across ALL frames.
        // Background tissue is present in every frame → lands at the median.
        // Contrast vessels are dark only during bolus (<50% of frames) →
        //   do NOT shift the median → background cancels cleanly.
        // This handles non-rigid tissue motion (brain pulsation, bowel)
        // that rigid/affine registration cannot correct.
        const int rows = frames[0].rows;
        const int cols = frames[0].cols;
        ref.create(rows, cols, CV_32F);

#pragma omp parallel for schedule(static)
        for (int y = 0; y < rows; y++)
        {
            std::vector<float> vals(n);
            const int mid = n / 2;
            for (int x = 0; x < cols; x++)
            {
                for (int k = 0; k < n; k++)
                    vals[k] = frames[k].at<float>(y, x);
                std::nth_element(vals.begin(), vals.begin() + mid, vals.end());
                ref.at<float>(y, x) = vals[mid];
            }
        }
        std::cout << "[DSA] Temporal median reference computed (" << n << " frames)\n";
    }
    else
    {
        // Average all pre-contrast frames [0 .. maskFrameIndex].
        // Multi-frame average reduces quantum noise compared to a single frame.
        ref = cv::Mat::zeros(frames[0].size(), CV_32F);
        const int refCount = maskFrameIndex + 1;
        for (int k = 0; k < refCount; k++)
            ref += frames[k];
        ref /= static_cast<float>(refCount);
        std::cout << "[DSA] Pre-contrast reference: avg of frames 0.."
                  << maskFrameIndex << "\n";
    }

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
    {
        // (In - Ref) * gain  [+ optional high-pass]  + offset
        cv::Mat dsa;
        cv::subtract(frames[i], ref, dsa);   // float32 difference (neg at vessels)
        dsa *= gain;                          // amplify vessel contrast

        // Gaussian high-pass: subtract blurred (low-freq) component.
        // Removes broad soft-tissue haze while preserving thin vessels.
        // Applied before OFFSET so the signal stays centred at 0.
        if (bgSuppressSigma > 0.0f)
        {
            cv::Mat bgLow;
            cv::GaussianBlur(dsa, bgLow, cv::Size(0, 0), bgSuppressSigma);
            dsa -= bgLow;
        }

        dsa += OFFSET;                        // shift neutral to 0.25 (mid-gray)

        // Clamp to valid [0, 1] display range
        cv::threshold(dsa, dsa, 0.0, 0.0, cv::THRESH_TOZERO);
        cv::threshold(dsa, dsa, 1.0, 1.0, cv::THRESH_TRUNC);

        cv::Mat dsa16;
        dsa.convertTo(dsa16, CV_16U, 65535.0);

        if (useClahe)
        {
            // Gentle CLAHE — clipLimit 2.0 avoids over-enhancement
            auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
            clahe->apply(dsa16, dsaFrames[i]);
        }
        else
        {
            dsaFrames[i] = std::move(dsa16);
        }
    }

    std::cout << "[DSA] Done"
              << "  mode=" << (maskMode == MASK_TEMPORAL_MEDIAN ? "temporal_median" : "precontrast")
              << "  gain=" << gain
              << "  clahe=" << (useClahe ? "on" : "off")
              << "  bgSigma=" << bgSuppressSigma << "\n";
}

// =============================================================
// Deformable refinement — Stage 2
//
// Runs DIS optical flow on affine-registered frames with vessel
// masking, then applies heavy Gaussian regularisation so only
// smooth tissue deformation passes through.
//
// The regularisation sigma is the key parameter:
//   sigma = 3.0  — cardiac (fine local motion allowed)
//   sigma = 5.0  — abdomen (smoother, prevents bowel-gas fitting)
//   sigma > 8.0  — nearly rigid (almost no local correction)
//
// Output: deformably warped frames + displacement fields (CV_32FC2).
// Displacement fields are stored so the same warp can be applied
// to log-domain frames for proper DSA subtraction.
// =============================================================
bool registerSequenceDeformable(
    const std::vector<cv::Mat>& affineFrames,
    std::vector<cv::Mat>&       registeredFrames,
    std::vector<cv::Mat>&       deformFields,
    int                         referenceFrameIndex,
    float                       regSigma)
{
    ScopedTimer _t("Deformable (DIS flow)");
    if (affineFrames.empty()) return false;

    int n = (int)affineFrames.size();
    registeredFrames.resize(n);
    deformFields.resize(n);
    registeredFrames[referenceFrameIndex] = affineFrames[referenceFrameIndex].clone();
    deformFields[referenceFrameIndex] = cv::Mat();   // identity — no deformation

    const int THRESH = 30;
    const int DILATE = 9;

    cv::Mat ref8;
    affineFrames[referenceFrameIndex].convertTo(ref8, CV_8U, 255.0);

    int failCount = 0;

#pragma omp parallel for schedule(dynamic) reduction(+:failCount)
    for (int i = 0; i < n; i++)
    {
        if (i == referenceFrameIndex) continue;

        cv::Mat frame8;
        affineFrames[i].convertTo(frame8, CV_8U, 255.0);
        cv::Mat vesselMask = makeVesselMask(ref8, frame8, THRESH, DILATE);

        cv::Mat frameMasked = frame8.clone();
        ref8.copyTo(frameMasked, vesselMask);

        auto dis = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM);
        cv::Mat flow;
        dis->calc(ref8, frameMasked, flow);

        smoothFlow(flow, regSigma);

        deformFields[i] = flow.clone();
        applyFlowWarp(affineFrames[i], registeredFrames[i], flow);
    }

    std::cout << "[Deformable] Done  ref=" << referenceFrameIndex
              << "  sigma=" << regSigma << "\n";
    return true;
}

// =============================================================
// Deformable registration with EXTERNAL vessel masks
//
// Same as above but uses pre-computed vessel masks (e.g. from
// a previous DSA pass) instead of intensity-difference masks.
// This gives much more accurate vessel exclusion → cleaner flow.
// =============================================================
bool registerSequenceDeformableWithMasks(
    const std::vector<cv::Mat>& affineFrames,
    const std::vector<cv::Mat>& vesselMasks,
    std::vector<cv::Mat>&       registeredFrames,
    std::vector<cv::Mat>&       deformFields,
    int                         referenceFrameIndex,
    float                       regSigma)
{
    ScopedTimer _t("Deformable Pass 2 (DSA masks)");
    if (affineFrames.empty()) return false;

    int n = (int)affineFrames.size();
    registeredFrames.resize(n);
    deformFields.resize(n);
    registeredFrames[referenceFrameIndex] = affineFrames[referenceFrameIndex].clone();
    deformFields[referenceFrameIndex] = cv::Mat();

    cv::Mat ref8;
    affineFrames[referenceFrameIndex].convertTo(ref8, CV_8U, 255.0);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++)
    {
        if (i == referenceFrameIndex) continue;

        cv::Mat frame8;
        affineFrames[i].convertTo(frame8, CV_8U, 255.0);

        // Use external mask if available, fallback to intensity diff
        cv::Mat frameMasked = frame8.clone();
        if (i < (int)vesselMasks.size() && !vesselMasks[i].empty())
        {
            // Dilate the DSA-derived mask for safety margin
            cv::Mat dilatedMask;
            cv::Mat kernel = cv::getStructuringElement(
                cv::MORPH_ELLIPSE, cv::Size(11, 11));
            cv::dilate(vesselMasks[i], dilatedMask, kernel);
            ref8.copyTo(frameMasked, dilatedMask);
        }
        else
        {
            cv::Mat mask = makeVesselMask(ref8, frame8, 30, 9);
            ref8.copyTo(frameMasked, mask);
        }

        auto dis = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM);
        cv::Mat flow;
        dis->calc(ref8, frameMasked, flow);

        smoothFlow(flow, regSigma);

        deformFields[i] = flow.clone();
        applyFlowWarp(affineFrames[i], registeredFrames[i], flow);
    }

    std::cout << "[Deformable-P2] Done  ref=" << referenceFrameIndex
              << "  sigma=" << regSigma << "\n";
    return true;
}

// =============================================================
// Build vessel masks from DSA frames
//
// DSA frames show vessels directly as dark pixels on gray background.
// Otsu threshold + dilation → binary vessel mask per frame.
// Much more accurate than intensity-difference masking.
// =============================================================
static std::vector<cv::Mat> buildVesselMasksFromDSA(
    const std::vector<cv::Mat>& dsaFrames16,
    int referenceFrameIndex)
{
    int n = (int)dsaFrames16.size();
    std::vector<cv::Mat> masks(n);

    // DSA convention: gray background (~mid-range), vessels are dark
    // Invert so vessels become bright, then Otsu threshold
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
    {
        if (i == referenceFrameIndex) { masks[i] = cv::Mat(); continue; }

        cv::Mat inv;
        cv::bitwise_not(dsaFrames16[i], inv);  // invert: vessels become bright

        // Convert to 8-bit for Otsu
        cv::Mat inv8;
        inv.convertTo(inv8, CV_8U, 255.0 / 65535.0);

        // Otsu finds the threshold separating background from vessel
        cv::Mat mask;
        cv::threshold(inv8, mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        masks[i] = mask;
    }

    std::cout << "[VesselMask] Built " << n << " masks from DSA\n";
    return masks;
}

// =============================================================
// Build log-domain frames from raw uint16
//
// log(raw + 1) — Beer-Lambert linearisation.
// NO per-frame normalisation: all frames share the same absolute
// scale so subtraction gives physically correct iodine signal.
// =============================================================
void buildLogFrames(
    const std::vector<cv::Mat>& rawFrames16,
    std::vector<cv::Mat>&       logFrames)
{
    ScopedTimer _t("Build log frames");
    int n = (int)rawFrames16.size();
    logFrames.resize(n);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
    {
        cv::Mat f;
        rawFrames16[i].convertTo(f, CV_32F);
        f += 1.0f;
        cv::log(f, f);
        logFrames[i] = f;
    }

    std::cout << "[Log] Built " << n << " log-domain frames\n";
}

// =============================================================
// Apply registration warps to log-domain frames
//
// Stage 1: affine warp matrix (2x3 or 3x3)
// Stage 2: displacement field (CV_32FC2) if present
//
// Two sequential interpolations — quality loss is negligible
// since the deformable residual is small after affine.
// =============================================================
void buildRegisteredLogFrames(
    const std::vector<cv::Mat>& rawFrames16,
    const std::vector<cv::Mat>& warpMatrices,
    const std::vector<cv::Mat>& deformFields,
    std::vector<cv::Mat>&       registeredLogFrames,
    int                         referenceFrameIndex)
{
    ScopedTimer _t("Build registered log frames");
    int n = (int)rawFrames16.size();
    registeredLogFrames.resize(n);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
    {
        // Raw → log domain
        cv::Mat logFrame;
        rawFrames16[i].convertTo(logFrame, CV_32F);
        logFrame += 1.0f;
        cv::log(logFrame, logFrame);

        // Stage 1: affine warp
        if (i != referenceFrameIndex && !warpMatrices.empty() && !warpMatrices[i].empty())
        {
            cv::Mat warped;
            if (warpMatrices[i].rows == 3 && warpMatrices[i].cols == 3)
                cv::warpPerspective(logFrame, warped, warpMatrices[i], logFrame.size(),
                                    cv::INTER_LINEAR + cv::WARP_INVERSE_MAP,
                                    cv::BORDER_REPLICATE);
            else
                cv::warpAffine(logFrame, warped, warpMatrices[i], logFrame.size(),
                               cv::INTER_LINEAR + cv::WARP_INVERSE_MAP,
                               cv::BORDER_REPLICATE);
            logFrame = warped;
        }

        // Stage 2: deformable displacement field
        if (!deformFields.empty() && i < (int)deformFields.size() && !deformFields[i].empty())
        {
            cv::Mat ch[2];
            cv::split(deformFields[i], ch);

            cv::Mat mapX(logFrame.size(), CV_32F);
            cv::Mat mapY(logFrame.size(), CV_32F);
            for (int y = 0; y < logFrame.rows; y++)
            {
                const float* dx = ch[0].ptr<float>(y);
                const float* dy = ch[1].ptr<float>(y);
                float* mx = mapX.ptr<float>(y);
                float* my = mapY.ptr<float>(y);
                for (int x = 0; x < logFrame.cols; x++)
                {
                    mx[x] = (float)x + dx[x];
                    my[x] = (float)y + dy[x];
                }
            }
            cv::Mat deformed;
            cv::remap(logFrame, deformed, mapX, mapY,
                      cv::INTER_LINEAR, cv::BORDER_REPLICATE);
            logFrame = deformed;
        }

        registeredLogFrames[i] = logFrame;
    }

    std::cout << "[LogWarp] Applied warps to " << n << " log frames\n";
}

// =============================================================
// Build log-domain reference mask
//
// MASK_PRECONTRAST:     average of log frames [0 .. maskFrameIndex]
// MASK_TEMPORAL_MEDIAN: per-pixel median across ALL log frames
// =============================================================
cv::Mat buildLogMask(
    const std::vector<cv::Mat>& logFrames,
    int                         maskFrameIndex,
    DSAMaskMode                 mode)
{
    int n = (int)logFrames.size();
    cv::Mat mask;

    if (mode == MASK_TEMPORAL_MEDIAN)
    {
        const int rows = logFrames[0].rows;
        const int cols = logFrames[0].cols;
        mask.create(rows, cols, CV_32F);

#pragma omp parallel for schedule(static)
        for (int y = 0; y < rows; y++)
        {
            std::vector<float> vals(n);
            const int mid = n / 2;
            for (int x = 0; x < cols; x++)
            {
                for (int k = 0; k < n; k++)
                    vals[k] = logFrames[k].at<float>(y, x);
                std::nth_element(vals.begin(), vals.begin() + mid, vals.end());
                mask.at<float>(y, x) = vals[mid];
            }
        }
        std::cout << "[LogMask] Temporal median of " << n << " frames\n";
    }
    else
    {
        mask = cv::Mat::zeros(logFrames[0].size(), CV_32F);
        int refCount = maskFrameIndex + 1;
        for (int k = 0; k < refCount; k++)
            mask += logFrames[k];
        mask /= (float)refCount;
        std::cout << "[LogMask] Pre-contrast avg of frames 0.." << maskFrameIndex << "\n";
    }

    return mask;
}

// =============================================================
// Log-domain DSA subtraction — physics-correct
//
// DSA(i) = (maskLog − frameLog) * scale * gain
//
// In Beer-Lambert:  ln(I₀e^-μt) = ln(I₀) − μt
//   maskLog − frameLog = μ_iodine · t_vessel
//   This is the PURE IODINE SIGNAL, independent of tissue.
//
// The scale factor normalises log-domain differences to a [0,1]
// friendly display range: scale = 1 / (6 * σ_mask).
// This makes gain 1.0x behave similarly to the old DSA.
//
// Background → mid-gray (offset 0.25 in [0,1], = 16383 uint16)
// Vessels → bright (iodine absorbs → maskLog > frameLog → positive)
//
// Sign convention: maskLog − frameLog is POSITIVE at vessels
// (iodine attenuates more → lower raw value → lower log value
//  in frame than in mask).  We display vessels as DARK, so we
// negate: DSA = −(mask − frame) * scale * gain + offset
//            = (frame − mask) * scale * gain + offset
// This keeps the same display convention as the old DSA.
// =============================================================
// -----------------------------------------------------------------
void computeDSALogDomain(
    const std::vector<cv::Mat>& logFrames,
    const cv::Mat&              maskLog,
    std::vector<cv::Mat>&       dsaFrames,
    float                       gain,
    bool                        useClahe,
    float                       bgSuppressSigma)
{
    ScopedTimer _t("DSA Log-domain subtraction");
    if (logFrames.empty()) return;

    const int   n      = (int)logFrames.size();
    const float OFFSET = 0.25f;

    dsaFrames.resize(n);

    // --- Spatially-varying intensity correction ---
    // Fast large-scale blur: pyrDown 4x (16x fewer pixels), blur σ=4,
    // then pyrUp back. Approximates σ≈64 Gaussian at ~20x less cost.
    auto fastLargeBlur = [](const cv::Mat& src) -> cv::Mat {
        cv::Mat d1, d2, d3, d4;
        cv::pyrDown(src, d1);
        cv::pyrDown(d1, d2);
        cv::pyrDown(d2, d3);
        cv::pyrDown(d3, d4);
        cv::GaussianBlur(d4, d4, cv::Size(0, 0), 4.0);
        cv::Mat u3, u2, u1, u0;
        cv::pyrUp(d4, u3, d3.size());
        cv::pyrUp(u3, u2, d2.size());
        cv::pyrUp(u2, u1, d1.size());
        cv::pyrUp(u1, u0, src.size());
        return u0;
    };

    cv::Mat maskLocal = fastLargeBlur(maskLog);

    std::vector<cv::Mat> normLog(n);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
    {
        cv::Mat frameLocal = fastLargeBlur(logFrames[i]);
        cv::Mat correction = maskLocal - frameLocal;
        normLog[i] = logFrames[i] + correction;
    }

    // Auto-scale
    cv::Scalar maskMean, maskStd;
    cv::meanStdDev(maskLog, maskMean, maskStd);
    float dsaScale = 1.0f / std::max(0.01f, 6.0f * (float)maskStd[0]);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
    {
        cv::Mat dsa;
        cv::subtract(normLog[i], maskLog, dsa);
        dsa *= dsaScale * gain;

        // Optional Gaussian high-pass: remove broad tissue haze
        if (bgSuppressSigma > 0.0f)
        {
            cv::Mat bgLow;
            cv::GaussianBlur(dsa, bgLow, cv::Size(0, 0), bgSuppressSigma);
            dsa -= bgLow;
        }

        dsa += OFFSET;

        // Clamp [0, 1]
        cv::threshold(dsa, dsa, 0.0, 0.0, cv::THRESH_TOZERO);
        cv::threshold(dsa, dsa, 1.0, 1.0, cv::THRESH_TRUNC);

        cv::Mat dsa16;
        dsa.convertTo(dsa16, CV_16U, 65535.0);

        if (useClahe)
        {
            auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
            clahe->apply(dsa16, dsaFrames[i]);
        }
        else
        {
            dsaFrames[i] = std::move(dsa16);
        }
    }

    std::cout << "[DSA-Log] Done  scale=" << dsaScale
              << "  gain=" << gain
              << "  clahe=" << (useClahe ? "on" : "off")
              << "  bgSigma=" << bgSuppressSigma << "\n";
}

// =============================================================
// TEMPORAL SMOOTHING — warp matrices (Gaussian-weighted)
//
// Each frame's warp is replaced by a Gaussian-weighted average of
// all warps within ±windowRadius frames.  sigma = windowRadius/2
// gives heavy centre weight with smooth tails.  The reference
// frame keeps its identity warp untouched.
// =============================================================
void temporalSmoothWarps(
    std::vector<cv::Mat>& warps,
    int                   refIdx,
    int                   windowRadius)
{
    ScopedTimer _t("Temporal smooth warps");
    if (warps.empty() || windowRadius <= 0) return;

    const int N = (int)warps.size();
    const double sigma = std::max(1.0, windowRadius / 2.0);
    const double twoSigma2 = 2.0 * sigma * sigma;

    std::vector<cv::Mat> smoothed(N);

    for (int i = 0; i < N; i++)
    {
        if (i == refIdx || warps[i].empty())
        {
            smoothed[i] = warps[i].clone();
            continue;
        }

        int lo = std::max(0, i - windowRadius);
        int hi = std::min(N - 1, i + windowRadius);

        cv::Mat acc = cv::Mat::zeros(warps[i].size(), CV_64F);
        double wSum = 0.0;

        for (int j = lo; j <= hi; j++)
        {
            if (warps[j].empty()) continue;
            double d = (double)(j - i);
            double w = std::exp(-(d * d) / twoSigma2);
            cv::Mat w64;
            warps[j].convertTo(w64, CV_64F);
            acc += w * w64;
            wSum += w;
        }

        if (wSum > 0.0)
            acc /= wSum;

        acc.convertTo(smoothed[i], warps[i].type());
    }

    warps = std::move(smoothed);
    std::cout << "[TemporalSmooth] Warps: Gaussian radius=" << windowRadius
              << " sigma=" << sigma << "\n";
}

// =============================================================
// TEMPORAL SMOOTHING — displacement fields (Gaussian-weighted)
// =============================================================
void temporalSmoothFields(
    std::vector<cv::Mat>& fields,
    int                   refIdx,
    int                   windowRadius)
{
    ScopedTimer _t("Temporal smooth fields");
    if (fields.empty() || windowRadius <= 0) return;

    const int N = (int)fields.size();
    const double sigma = std::max(1.0, windowRadius / 2.0);
    const double twoSigma2 = 2.0 * sigma * sigma;

    std::vector<cv::Mat> smoothed(N);

    for (int i = 0; i < N; i++)
    {
        if (i == refIdx || fields[i].empty())
        {
            smoothed[i] = fields[i].empty() ? fields[i] : fields[i].clone();
            continue;
        }

        int lo = std::max(0, i - windowRadius);
        int hi = std::min(N - 1, i + windowRadius);

        cv::Mat acc = cv::Mat::zeros(fields[i].size(), CV_32FC2);
        double wSum = 0.0;

        for (int j = lo; j <= hi; j++)
        {
            if (j < N && !fields[j].empty())
            {
                double d = (double)(j - i);
                double w = std::exp(-(d * d) / twoSigma2);
                cv::Mat scaled;
                fields[j].convertTo(scaled, CV_32FC2, w);
                acc += scaled;
                wSum += w;
            }
        }

        if (wSum > 0.0)
            acc /= (float)wSum;

        smoothed[i] = acc;
    }

    fields = std::move(smoothed);
    std::cout << "[TemporalSmooth] Fields: Gaussian radius=" << windowRadius
              << " sigma=" << sigma << "\n";
}
