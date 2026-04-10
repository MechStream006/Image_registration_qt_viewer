#ifndef REGISTER_H
#define REGISTER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "RegionConfig.h"

struct ECCResult
{
    cv::Mat warpMatrix;
    int     iterations = 0;
    bool    converged  = false;
    double  finalRho   = 0.0;
};

// =============================================================
// Registration method selector
// =============================================================
enum RegistrationMethod
{
    // --- Stable: global transform only, no floating possible ---
    REG_ECC_CPU          = 0,  // ECC Affine 6DOF pyramid — accurate, slow
    REG_PHASE_CORR       = 1,  // Phase Correlation — translation 2DOF, fastest
    REG_DIS_CPU          = 2,  // DIS flow → fit Affine 6DOF via RANSAC
    REG_FARNEBACK_GPU    = 3,  // DIS flow → fit Partial Affine 4DOF via RANSAC
    REG_NVIDIA_FLOW      = 4,  // DIS flow → fit Homography 8DOF via RANSAC
    REG_ECC_TRANSLATION  = 5,  // ECC Translation 2DOF — fastest ECC variant
    REG_ECC_HOMOGRAPHY   = 6,  // ECC Homography 8DOF — handles C-arm tilt
    REG_ORB_AFFINE       = 7,  // ORB feature matching → Affine via RANSAC

    // --- Per-pixel flow: may have floating artifacts in DSA ---
    REG_FARNEBACK_FLOW   = 8,  // Real Farneback dense flow, warp applied directly
    REG_DIS_FLOW_RAW     = 9,  // DIS ULTRAFAST raw flow (aggressive, max floating risk)
};

// =============================================================
// Shared vessel mask helper — used by Register.cpp and
// BSplineRegister.cpp.  Returns CV_8U mask: 255 = vessel pixel
// (frame darker than ref by > thresh), 0 = tissue pixel.
// Mask is dilated by dilateR pixels to cover vessel edges.
// =============================================================
cv::Mat makeVesselMask(const cv::Mat& ref8, const cv::Mat& frame8,
                        int thresh, int dilateR);

// =============================================================
// Auto-detect pre-contrast mask frame (min std-dev)
// =============================================================
int autoDetectMaskFrame(const std::vector<cv::Mat>& preprocessedFrames);

// =============================================================
// ECC — single frame
//
// New parameters vs original:
//   maxIterations — per pyramid level (default 100; use 200 for neuro)
//   epsilon       — convergence threshold (default 1e-5; use 1e-6 for neuro)
//   initialWarp   — optional starting warp (e.g. from Phase Corr).
//                   If empty, identity is used.
// =============================================================
bool registerFrameECC(
    const cv::Mat& ref,
    const cv::Mat& src,
    cv::Mat&       dst,
    ECCResult&     result,
    int            motionType    = cv::MOTION_AFFINE,
    int            pyramidLevels = 3,
    int            maxIterations = 100,
    double         epsilon       = 1e-5,
    const cv::Mat& initialWarp   = cv::Mat()
);

// =============================================================
// ECC — full sequence (OpenMP parallel)
// outWarps: if non-null, receives the 2x3 (or 3x3) warp matrix
//           per frame.  Identity for the reference frame.
// =============================================================
bool registerSequenceECC(
    const std::vector<cv::Mat>& frames,
    std::vector<cv::Mat>&       registeredFrames,
    int    referenceFrameIndex = 0,
    int    motionType          = cv::MOTION_AFFINE,
    int    pyramidLevels       = 3,
    int    maxIterations       = 100,
    double epsilon             = 1e-5,
    std::vector<cv::Mat>*       outWarps = nullptr
);

// =============================================================
// Neuro-specific pipeline: Phase Corr init → ECC Euclidean
//
// Phase Correlation provides a fast, accurate translation
// estimate which initialises the ECC warp.  ECC EUCLIDEAN (3
// DOF: tx, ty, θ) then refines only the rotation component.
//   • 5 pyramid levels  — handles large head motion
//   • 200 iterations    — tighter convergence at fine scale
//   • 1e-6 epsilon      — sub-pixel accuracy
// B-Spline is NOT applied (skull is a rigid body).
// =============================================================
bool registerSequenceNeuro(
    const std::vector<cv::Mat>& frames,
    std::vector<cv::Mat>&       registeredFrames,
    int referenceFrameIndex = 0,
    std::vector<cv::Mat>*  outWarps = nullptr
);

// =============================================================
// Region-aware dispatcher
//
// Selects the correct pipeline for the given BodyRegion:
//   Neuro      — registerSequenceNeuro (Phase Corr + ECC rigid)
//   Cardiac    — ECC Affine → B-Spline FFD (fine grid)
//   Abdomen    — ECC Affine → B-Spline FFD (coarse grid)
//   Peripheral — ECC Affine (standalone)
//   Auto       — falls back to registerSequence(method)
// =============================================================
bool registerSequenceForRegion(
    const std::vector<cv::Mat>& frames,
    std::vector<cv::Mat>&       registeredFrames,
    int                         referenceFrameIndex,
    BodyRegion                  region,
    std::vector<cv::Mat>*       outWarps        = nullptr,
    std::vector<cv::Mat>*       outDeformFields = nullptr
);

// =============================================================
// Phase Correlation (translation only)
// =============================================================
bool registerSequencePhaseCorr(
    const std::vector<cv::Mat>& frames,
    std::vector<cv::Mat>&       registeredFrames,
    int referenceFrameIndex = 0,
    std::vector<cv::Mat>*  outWarps = nullptr
);

// =============================================================
// DIS Dense Optical Flow (non-rigid, CPU)
// =============================================================
bool registerSequenceDIS(
    const std::vector<cv::Mat>& frames,
    std::vector<cv::Mat>&       registeredFrames,
    int referenceFrameIndex = 0,
    std::vector<cv::Mat>*  outWarps = nullptr
);

// =============================================================
// Farneback Dense Optical Flow (non-rigid, GPU)
// Falls back to DIS CPU if no CUDA
// =============================================================
bool registerSequenceFarnebackGPU(
    const std::vector<cv::Mat>& frames,
    std::vector<cv::Mat>&       registeredFrames,
    int referenceFrameIndex = 0,
    std::vector<cv::Mat>*  outWarps = nullptr
);

// =============================================================
// NVIDIA Hardware Optical Flow (non-rigid, RTX+)
// Falls back to Farneback GPU → DIS CPU if unsupported
// =============================================================
bool registerSequenceNvidiaFlow(
    const std::vector<cv::Mat>& frames,
    std::vector<cv::Mat>&       registeredFrames,
    int referenceFrameIndex = 0,
    std::vector<cv::Mat>*  outWarps = nullptr
);

// =============================================================
// ORB feature matching → Affine via RANSAC
// =============================================================
bool registerSequenceORBAffine(
    const std::vector<cv::Mat>& frames,
    std::vector<cv::Mat>&       registeredFrames,
    int referenceFrameIndex = 0,
    std::vector<cv::Mat>*  outWarps = nullptr
);

// =============================================================
// Real Farneback dense optical flow (per-pixel warp)
// Note: may produce floating artifacts in DSA
// =============================================================
bool registerSequenceFarnebackFlow(
    const std::vector<cv::Mat>& frames,
    std::vector<cv::Mat>&       registeredFrames,
    int referenceFrameIndex = 0
);

// =============================================================
// DIS ULTRAFAST raw per-pixel flow (original aggressive variant)
// Note: may produce floating artifacts in DSA
// =============================================================
bool registerSequenceDISFlowRaw(
    const std::vector<cv::Mat>& frames,
    std::vector<cv::Mat>&       registeredFrames,
    int referenceFrameIndex = 0
);

// =============================================================
// Dispatcher — call any method by enum
// =============================================================
bool registerSequence(
    const std::vector<cv::Mat>& frames,
    std::vector<cv::Mat>&       registeredFrames,
    int                         referenceFrameIndex,
    RegistrationMethod          method,
    std::vector<cv::Mat>*       outWarps = nullptr
);

// =============================================================
// DSA mask mode
//
// MASK_PRECONTRAST   — average of frames [0..maskFrameIndex]
// MASK_TEMPORAL_MEDIAN — per-pixel median across ALL frames.
// =============================================================
enum DSAMaskMode
{
    MASK_PRECONTRAST      = 0,
    MASK_TEMPORAL_MEDIAN  = 1
};

// =============================================================
// Deformable refinement (Stage 2)
//
// Runs DIS optical flow on affine-registered frames, then
// applies heavy Gaussian regularisation (sigma = regSigma) to
// the flow field.  Regularisation prevents vessel tracking —
// only smooth tissue deformation passes through.
//
// Input: affine-registered float32 [0,1] frames from Stage 1.
// Output: deformably corrected frames + displacement fields.
// =============================================================
bool registerSequenceDeformable(
    const std::vector<cv::Mat>& affineFrames,
    std::vector<cv::Mat>&       registeredFrames,
    std::vector<cv::Mat>&       deformFields,
    int                         referenceFrameIndex,
    float                       regSigma = 3.0f
);

// =============================================================
// Multi-pass deformable with external vessel masks
// =============================================================
bool registerSequenceDeformableWithMasks(
    const std::vector<cv::Mat>& affineFrames,
    const std::vector<cv::Mat>& vesselMasks,
    std::vector<cv::Mat>&       registeredFrames,
    std::vector<cv::Mat>&       deformFields,
    int                         referenceFrameIndex,
    float                       regSigma = 3.0f
);

// =============================================================
// Log-domain DSA pipeline
//
// Physics-correct DSA: subtraction is done in log(raw) domain
// where tissue attenuation cancels perfectly (Beer-Lambert law).
//
// buildLogFrames:  raw uint16 → log(raw+1) float32
//   No per-frame normalisation — all frames share the same
//   absolute scale so subtraction is physically meaningful.
//
// buildRegisteredLogFrames:  applies registration warps to log
//   frames.  Handles both affine matrices and deformation fields.
//
// buildLogMask:  builds the reference mask in log domain
//   (pre-contrast average or temporal median).
//
// computeDSALogDomain:  log(mask) - log(frame) = iodine signal.
//   Auto-scales to display range using mask std-dev.
// =============================================================
void buildLogFrames(
    const std::vector<cv::Mat>& rawFrames16,
    std::vector<cv::Mat>&       logFrames
);

void buildRegisteredLogFrames(
    const std::vector<cv::Mat>& rawFrames16,
    const std::vector<cv::Mat>& warpMatrices,
    const std::vector<cv::Mat>& deformFields,
    std::vector<cv::Mat>&       registeredLogFrames,
    int                         referenceFrameIndex
);

cv::Mat buildLogMask(
    const std::vector<cv::Mat>& logFrames,
    int                         maskFrameIndex,
    DSAMaskMode                 mode
);

// =============================================================
// Temporal smoothing of registration warps
//
// Averages warp matrices / displacement fields over a time window
// (±windowRadius frames).  Eliminates frame-to-frame jitter in
// the registration — vessel positions become stable in DSA.
//
// Without this, each frame's warp is computed independently and
// small intensity variations (iodine flow, noise) cause the warp
// to wobble slightly → vessels appear to "float" in DSA.
// =============================================================
void temporalSmoothWarps(
    std::vector<cv::Mat>& warps,
    int                   refIdx,
    int                   windowRadius = 3
);

void temporalSmoothFields(
    std::vector<cv::Mat>& fields,
    int                   refIdx,
    int                   windowRadius = 3
);

void computeDSALogDomain(
    const std::vector<cv::Mat>& logFrames,
    const cv::Mat&              maskLog,
    std::vector<cv::Mat>&       dsaFrames,
    float                       gain            = 1.0f,
    bool                        useClahe        = false,
    float                       bgSuppressSigma = 0.0f
);

// =============================================================
// DSA subtraction (legacy — kept for fallback)
//
// Formula:  DSA(i) = clamp( (In(i) − Ref) * gain + 0.25, 0, 1 )
//
//   Ref is built according to maskMode (see above).
//   gain         — user-adjustable contrast amplifier (0.5 – 4.0)
//   useClahe     — optional local contrast enhancement (clipLimit 2.0)
//   bgSuppressSigma — >0 → Gaussian high-pass on the difference
//
// Input frames must be float32 [0,1] in log domain (preprocessed).
// Output is CV_16U [0,65535] ready for VTK display.
// =============================================================
void computeDSA(
    const std::vector<cv::Mat>& preprocessedFrames,
    std::vector<cv::Mat>&       dsaFrames,
    int         maskFrameIndex  = 0,
    float       gain            = 1.0f,
    bool        useClahe        = false,
    float       bgSuppressSigma = 0.0f,
    DSAMaskMode maskMode        = MASK_PRECONTRAST
);

#endif
