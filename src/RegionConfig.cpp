#include "RegionConfig.h"

#include <itkGDCMImageIO.h>
#include <itkMetaDataObject.h>

#include <opencv2/opencv.hpp>   // cv::MOTION_* constants
#include <algorithm>
#include <cctype>
#include <iostream>

// =============================================================
// Static parameter table — one entry per BodyRegion enum value.
// Keep in the same order as the enum (0–4).
// =============================================================
static const RegionParameters REGION_TABLE[] =
{
    // ---- REGION_AUTO (index 0) --------------------------------
    {
        REGION_AUTO, "Auto",
        /* eccMotionType   */ cv::MOTION_AFFINE,
        /* eccPyramidLevels*/ 4,
        /* eccMaxIter      */ 300,
        /* eccEpsilon      */ 1e-6,
        /* usePhaseCorInit */ false,
        /* useDeformable   */ false,
        /* deformRegSigma  */ 0.0f,
        /* useBSpline      */ false,
        /* bsplineGridSpacing */ 0,
        /* bsplineMaxIter     */ 0,
        /* bsplineMultiResLvl */ 0
    },

    // ---- REGION_NEURO (index 1) --------------------------------
    // Stage 1: Phase Corr + ECC Euclidean (rigid skull alignment)
    // Stage 2: MI + B-Spline FFD via ITK (replaces DIS optical flow)
    //   - Mutual Information metric ignores iodine intensity changes
    //   - B-Spline with 64px grid = smooth non-rigid correction
    //   - 3-level multi-res pyramid for coarse-to-fine
    //   - DIS deformable kept OFF (B-Spline replaces it)
    {
        REGION_NEURO, "Neuro",
        /* eccMotionType   */ cv::MOTION_EUCLIDEAN,
        /* eccPyramidLevels*/ 5,
        /* eccMaxIter      */ 300,
        /* eccEpsilon      */ 1e-6,
        /* usePhaseCorInit */ true,
        /* useDeformable   */ false,
        /* deformRegSigma  */ 0.0f,
        /* useBSpline      */ true,
        /* bsplineGridSpacing */ 64,
        /* bsplineMaxIter     */ 200,
        /* bsplineMultiResLvl */ 3
    },

    // ---- REGION_CARDIAC (index 2) -----------------------------
    // Stage 1: Phase Corr + ECC Affine
    // Stage 2: DIS optical flow σ=7.0 (B-Spline OFF for now)
    {
        REGION_CARDIAC, "Cardiac",
        /* eccMotionType   */ cv::MOTION_AFFINE,
        /* eccPyramidLevels*/ 5,
        /* eccMaxIter      */ 300,
        /* eccEpsilon      */ 1e-6,
        /* usePhaseCorInit */ true,
        /* useDeformable   */ true,
        /* deformRegSigma  */ 7.0f,
        /* useBSpline      */ false,
        /* bsplineGridSpacing */ 0,
        /* bsplineMaxIter     */ 0,
        /* bsplineMultiResLvl */ 0
    },

    // ---- REGION_ABDOMEN (index 3) -----------------------------
    // Stage 1: Phase Corr + ECC Affine
    // Stage 2: DIS optical flow σ=6.0 (B-Spline OFF for now)
    {
        REGION_ABDOMEN, "Abdomen",
        /* eccMotionType   */ cv::MOTION_AFFINE,
        /* eccPyramidLevels*/ 4,
        /* eccMaxIter      */ 300,
        /* eccEpsilon      */ 1e-6,
        /* usePhaseCorInit */ true,
        /* useDeformable   */ true,
        /* deformRegSigma  */ 6.0f,
        /* useBSpline      */ false,
        /* bsplineGridSpacing */ 0,
        /* bsplineMaxIter     */ 0,
        /* bsplineMultiResLvl */ 0
    },

    // ---- REGION_PERIPHERAL (index 4) --------------------------
    // Same as Neuro but with DIS deformable (B-Spline OFF for now)
    {
        REGION_PERIPHERAL, "Peripheral",
        /* eccMotionType   */ cv::MOTION_EUCLIDEAN,
        /* eccPyramidLevels*/ 5,
        /* eccMaxIter      */ 300,
        /* eccEpsilon      */ 1e-6,
        /* usePhaseCorInit */ true,
        /* useDeformable   */ true,
        /* deformRegSigma  */ 5.0f,
        /* useBSpline      */ false,
        /* bsplineGridSpacing */ 0,
        /* bsplineMaxIter     */ 0,
        /* bsplineMultiResLvl */ 0
    }
};

// =============================================================
const RegionParameters& getRegionParams(BodyRegion region)
{
    int idx = static_cast<int>(region);
    if (idx < 0 || idx > 4) idx = 0;
    return REGION_TABLE[idx];
}

const char* regionDisplayName(BodyRegion region)
{
    return getRegionParams(region).name;
}

// =============================================================
// detectRegionFromDICOM
// Reads DICOM tag 0018|0015 (BodyPartExamined), normalises the
// string, and maps it to a BodyRegion.
// =============================================================
BodyRegion detectRegionFromDICOM(const std::string& dicomPath)
{
    auto dicomIO = itk::GDCMImageIO::New();
    dicomIO->SetFileName(dicomPath);

    try { dicomIO->ReadImageInformation(); }
    catch (const itk::ExceptionObject& e)
    {
        std::cerr << "[Region] DICOM read error: " << e.GetDescription() << "\n";
        return REGION_AUTO;
    }

    const itk::MetaDataDictionary& dict = dicomIO->GetMetaDataDictionary();

    std::string bodyPart;
    if (!itk::ExposeMetaData<std::string>(dict, "0018|0015", bodyPart))
    {
        std::cout << "[Region] Tag 0018|0015 not present — using Auto\n";
        return REGION_AUTO;
    }

    // Normalise: upper-case, strip DICOM padding spaces
    std::transform(bodyPart.begin(), bodyPart.end(),
                   bodyPart.begin(), [](unsigned char c){ return std::toupper(c); });
    auto first = bodyPart.find_first_not_of(' ');
    auto last  = bodyPart.find_last_not_of(' ');
    if (first == std::string::npos) { std::cout << "[Region] Empty tag — using Auto\n"; return REGION_AUTO; }
    bodyPart = bodyPart.substr(first, last - first + 1);

    std::cout << "[Region] BodyPartExamined = \"" << bodyPart << "\"\n";

    // --- Neuro ---
    if (bodyPart == "BRAIN"    || bodyPart == "HEAD"       ||
        bodyPart == "NECK"     || bodyPart == "CAROTID"    ||
        bodyPart == "CEREBRAL" || bodyPart == "SKULL"      ||
        bodyPart == "INTRACRANIAL")
        return REGION_NEURO;

    // --- Cardiac ---
    if (bodyPart == "HEART"    || bodyPart == "CORONARY"   ||
        bodyPart == "CHEST"    || bodyPart == "THORAX"     ||
        bodyPart == "CARDIAC"  || bodyPart == "CORO"       ||
        bodyPart == "PULMONARY")
        return REGION_CARDIAC;

    // --- Abdomen ---
    if (bodyPart == "ABDOMEN"     || bodyPart == "ABDOMINAL"  ||
        bodyPart == "LIVER"       || bodyPart == "KIDNEY"     ||
        bodyPart == "RENAL"       || bodyPart == "AORTA"      ||
        bodyPart == "MESENTERIC"  || bodyPart == "CELIAC"     ||
        bodyPart == "PELVIS"      || bodyPart == "PELVIC"     ||
        bodyPart == "PORTAL"      || bodyPart == "SPLENIC")
        return REGION_ABDOMEN;

    // --- Peripheral ---
    if (bodyPart == "LOWER EXTREMITY" || bodyPart == "UPPER EXTREMITY" ||
        bodyPart == "FEMORAL"   || bodyPart == "LEG"        ||
        bodyPart == "ARM"       || bodyPart == "HAND"       ||
        bodyPart == "FOOT"      || bodyPart == "ILIAC"      ||
        bodyPart == "TIBIAL"    || bodyPart == "POPLITEAL"  ||
        bodyPart == "PERIPHERAL"|| bodyPart == "RUNOFF"     ||
        bodyPart == "FOREARM"   || bodyPart == "WRIST")
        return REGION_PERIPHERAL;

    std::cout << "[Region] Unrecognised value \"" << bodyPart
              << "\" — using Auto (Affine)\n";
    return REGION_AUTO;
}
