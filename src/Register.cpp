#include "Register.h"
#include <iostream>

#define NOMINMAX
#include <windows.h>

// =============================================================
// Console output on Windows (Qt swallows stdout otherwise)
// =============================================================
#ifdef _WIN32
#include <windows.h>
static bool s_consoleReady = false;
static void ensureConsole()
{
    if (s_consoleReady) return;
    AllocConsole();
    freopen_s((FILE**)stdout, "CONOUT$", "w", stdout);
    freopen_s((FILE**)stderr, "CONOUT$", "w", stderr);
    s_consoleReady = true;
}
#else
static void ensureConsole() {}
#endif

// =============================================================
// Build Gaussian pyramid
// =============================================================
static void buildPyramid(const cv::Mat& src,
    std::vector<cv::Mat>& pyr,
    int levels)
{
    pyr.resize(levels);
    pyr[0] = src.clone();
    for (int i = 1; i < levels; i++)
        cv::pyrDown(pyr[i - 1], pyr[i]);
}

// Scale translation components when going coarse→fine
static void scaleWarpUp(cv::Mat& warp)
{
    warp.at<float>(0, 2) *= 2.0f;
    warp.at<float>(1, 2) *= 2.0f;
}

// =============================================================
// Register single frame (preprocessed float32 in, float32 out)
// =============================================================
bool registerFrameECC(
    const cv::Mat& ref,
    const cv::Mat& src,
    cv::Mat& dst,
    ECCResult& result,
    int motionType,
    int pyramidLevels)
{
    ensureConsole();

    std::vector<cv::Mat> refPyr, srcPyr;
    buildPyramid(ref, refPyr, pyramidLevels);
    buildPyramid(src, srcPyr, pyramidLevels);

    cv::Mat warp = cv::Mat::eye(2, 3, CV_32F);
    result.converged = true;
    result.finalRho = 1.0;

    for (int lvl = pyramidLevels - 1; lvl >= 0; lvl--)
    {
        cv::TermCriteria tc(
            cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
            50, 1e-4);

        // Mask out near-black border pixels (image boundary artifacts)
        cv::Mat eccMask;
        cv::threshold(refPyr[lvl], eccMask, 0.02f, 1.0f, cv::THRESH_BINARY);
        eccMask.convertTo(eccMask, CV_8U, 255);

        try
        {
            double rho = cv::findTransformECC(
                refPyr[lvl], srcPyr[lvl],
                warp, motionType, tc, eccMask);

            result.finalRho = rho;

            printf("  [ECC lvl=%d] rho=%.4f  tx=%.2f  ty=%.2f\n",
                lvl, rho,
                warp.at<float>(0, 2), warp.at<float>(1, 2));
            fflush(stdout);

            if (rho < 0.5)
            {
                printf("  [ECC] LOW RHO — falling back to identity\n");
                fflush(stdout);
                warp = cv::Mat::eye(2, 3, CV_32F);
                result.converged = false;
                break;
            }
        }
        catch (const cv::Exception& e)
        {
            printf("  [ECC lvl=%d] EXCEPTION: %s\n", lvl, e.what());
            fflush(stdout);
            warp = cv::Mat::eye(2, 3, CV_32F);
            result.converged = false;
            break;
        }

        if (lvl > 0) scaleWarpUp(warp);
    }

    result.warpMatrix = warp.clone();

    // Apply warp to preprocessed float32 frame
    cv::warpAffine(
        src, dst,
        warp,
        src.size(),
        cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
        cv::BORDER_REPLICATE);

    return result.converged;
}

// =============================================================
// Register full sequence — all preprocessed float32
// =============================================================
bool registerSequenceECC(
    const std::vector<cv::Mat>& processedFrames,
    std::vector<cv::Mat>& registeredFrames,
    int motionType,
    int pyramidLevels)
{
    ensureConsole();

    if (processedFrames.empty())
    {
        printf("[Register] ERROR: empty input\n");
        fflush(stdout);
        return false;
    }

    int N = (int)processedFrames.size();
    registeredFrames.resize(N);

    // Frame 0 = reference, no warp needed
    registeredFrames[0] = processedFrames[0].clone();

    printf("[Register] Starting: %d frames, motionType=%d, pyramid=%d\n",
        N, motionType, pyramidLevels);
    fflush(stdout);

    int failures = 0;

    for (int i = 1; i < N; i++)
    {
        printf("[Register] Frame %d/%d\n", i, N - 1);
        fflush(stdout);

        ECCResult res;

        bool ok = registerFrameECC(
            processedFrames[0],    // reference: always frame 0
            processedFrames[i],    // moving: frame i
            registeredFrames[i],   // output: warped float32
            res,
            motionType,
            pyramidLevels);

        if (!ok)
        {
            ++failures;
            // Fallback: use unwarped preprocessed frame
            registeredFrames[i] = processedFrames[i].clone();
            printf("[Register] Frame %d FALLBACK (rho=%.3f)\n",
                i, res.finalRho);
            fflush(stdout);
        }
        else
        {
            printf("[Register] Frame %d OK  rho=%.3f  tx=%.2f  ty=%.2f\n",
                i, res.finalRho,
                res.warpMatrix.at<float>(0, 2),
                res.warpMatrix.at<float>(1, 2));
            fflush(stdout);
        }
    }

    printf("[Register] Done: %d OK, %d fallbacks\n",
        N - 1 - failures, failures);
    fflush(stdout);

    return failures < N / 2;
}

// =============================================================
// DSA — subtract preprocessed frames
//
// Both input frames are float32 [0,1].
// pre-contrast mask is BRIGHT (high value near 1.0)
// post-contrast frame is DARKER at vessels (lower value)
// diff = mask - frame[i] → POSITIVE at vessels
// After invert → vessels DARK on WHITE background
void computeDSA(
    const std::vector<cv::Mat>& frames,   // float32 [0,1]
    std::vector<cv::Mat>& dsaFrames,
    int maskFrameIndex)
{
    if (frames.empty()) return;

    int N = frames.size();
    dsaFrames.resize(N);

    const cv::Mat& mask = frames[0];

    const float scaleFactor = 20000.0f;   // base amplification
    const float alpha = 5.0f;             // 🔥 contrast boost (3–8 range)

    for (int i = 0; i < N; i++)
    {
        // Step 1: subtraction
        cv::Mat diff = mask - frames[i];

        // Step 2: remove negatives
        cv::threshold(diff, diff, 0.0f, 0.0f, cv::THRESH_TOZERO);

        // Step 3: amplify
        diff = diff * scaleFactor;

        // Step 4: normalize to [0,1]
        diff = diff / 65535.0f;

        // 🔥 Step 5: NON-LINEAR BOOST (KEY)
        // log-like enhancement → boosts weak vessels
        diff = diff / (diff + (1.0f / alpha));

        // Step 6: back to 16-bit
        diff = diff * 65535.0f;

        cv::Mat out16;
        diff.convertTo(out16, CV_16U);

        // Step 7: invert (vessels dark)
        dsaFrames[i] = cv::Scalar(50000) - out16;
    }

    std::cout << "[DSA NONLINEAR BOOST] Done\n";
}