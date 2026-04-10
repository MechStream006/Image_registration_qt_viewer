#ifndef REGION_CONFIG_H
#define REGION_CONFIG_H

#include <string>

// =============================================================
// Body region selector — drives the registration pipeline
// choice and all associated parameters automatically.
//
// Each region has fundamentally different motion characteristics:
//   Neuro      — rigid skull, tiny motion (< 2mm), no B-Spline
//   Cardiac    — non-rigid heart beat + breathing, fine B-Spline
//   Abdomen    — respiratory deformation, coarse B-Spline
//   Peripheral — limb bones, mostly global affine
//
// Auto-detect reads DICOM tag 0018|0015 (BodyPartExamined).
// User can override via the UI combo box at any time.
// =============================================================
enum BodyRegion
{
    REGION_AUTO       = 0,   // detect from DICOM; fallback to manual method
    REGION_NEURO      = 1,   // cerebral / carotid angiography
    REGION_CARDIAC    = 2,   // coronary angiography
    REGION_ABDOMEN    = 3,   // aorta, renal, mesenteric, celiac
    REGION_PERIPHERAL = 4    // iliac, femoral, tibial, upper limb
};

// =============================================================
// All tunable parameters for a given region, in one struct.
// Add new parameters here rather than scattering magic numbers.
// =============================================================
struct RegionParameters
{
    BodyRegion  region;
    const char* name;

    // --- Stage 1: ECC ---
    int    eccMotionType;       // cv::MOTION_EUCLIDEAN  or  cv::MOTION_AFFINE
    int    eccPyramidLevels;    // 5 for neuro (rigid, handles large motion)
                                // 3 for others
    int    eccMaxIterations;    // per pyramid level; 200 for neuro, 100 others
    double eccEpsilon;          // convergence threshold; 1e-6 neuro, 1e-5 others
    bool   usePhaseCorInit;     // Phase Corr translation pre-init (neuro only)

    // --- Stage 2: Deformable refinement ---
    // Uses DIS optical flow with heavy Gaussian regularisation.
    // Regularisation sigma controls smoothness — prevents the flow
    // field from tracking vessels (which would cause floating in DSA).
    bool   useDeformable;       // false = Stage 1 only
    float  deformRegSigma;      // Gaussian regularisation sigma (pixels)

    // --- Stage 2 alt: B-Spline FFD via ITK (MI metric) ---
    // When useBSpline=true, replaces DIS optical flow with ITK
    // B-Spline registration using Mattes Mutual Information.
    bool   useBSpline;          // true = use MI+B-Spline instead of DIS
    int    bsplineGridSpacing;  // control point spacing in pixels (e.g. 64)
    int    bsplineMaxIter;      // LBFGSB max function evaluations
    int    bsplineMultiResLevels; // image pyramid levels (e.g. 3)
};

// =============================================================
// Returns pre-configured parameters for a region.
// REGION_AUTO returns conservative affine defaults.
// =============================================================
const RegionParameters& getRegionParams(BodyRegion region);

// =============================================================
// Reads DICOM tag 0018|0015 (BodyPartExamined) and maps it to
// one of our BodyRegion values.  Returns REGION_AUTO if the tag
// is missing, empty, or not recognised.
// =============================================================
BodyRegion detectRegionFromDICOM(const std::string& dicomPath);

// Short display name (e.g. "Neuro", "Cardiac")
const char* regionDisplayName(BodyRegion region);

#endif // REGION_CONFIG_H
