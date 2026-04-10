# DSA Image Registration Viewer

A real-time Digital Subtraction Angiography (DSA) viewer with multi-stage image registration for catheterization lab quality improvement. Built with C++, Qt, VTK, ITK, and OpenCV.

## Overview

DSA removes background tissue from X-ray angiograms by subtracting a pre-contrast mask frame from contrast-filled frames, revealing only iodine-filled blood vessels. Patient motion between frames causes misalignment, producing ghost artifacts in the subtracted images.

This application implements a **region-specific multi-stage registration pipeline** that corrects patient motion and produces clean DSA output with minimal background artifacts.

## Features

- **DICOM loader** with automatic metadata extraction
- **Multiple display modes**: Original, Preprocessed, DSA (unregistered), DSA (ECC registered), DSA (DIS flow), DSA (Farneback), DSA (ORB Affine)
- **Region-specific registration** with auto-detection from DICOM BodyPartExamined tag
- **Real-time playback** with frame-by-frame navigation
- **DSA controls**: adjustable gain, CLAHE enhancement, background suppression (Gaussian high-pass)
- **Mask modes**: Pre-contrast frame, Temporal median
- **Auto mask frame detection** from pre-contrast frames

## Registration Pipeline

### Architecture

The pipeline uses a **dual-track approach**: normalized frames for registration, unnormalized log-domain frames for DSA subtraction. Registration outputs warp matrices that are applied to raw log frames, preserving physically correct Beer-Lambert attenuation for tissue cancellation.

### Stage One — Global Alignment

| Region | Motion Model | Init | DOF | Description |
|---|---|---|---|---|
| **Neuro** | ECC Euclidean | Phase Correlation | Three | Rigid skull: translate + rotate |
| **Cardiac** | ECC Affine | Phase Correlation | Six | Non-rigid chest: + scale, shear |
| **Abdomen** | ECC Affine | Phase Correlation | Six | Respiratory motion: + scale, shear |
| **Peripheral** | ECC Euclidean | Phase Correlation | Three | Rigid limbs: translate + rotate |

- **Phase Correlation**: FFT-based coarse translation estimate for initialization
- **ECC (Enhanced Correlation Coefficient)**: Multi-scale pyramid refinement (coarse-to-fine, hundreds of iterations per level)

### Stage Two — Non-Rigid Refinement

| Region | Method | Key Parameter |
|---|---|---|
| **Neuro** | ITK B-Spline FFD + Mattes Mutual Information | Multi-level pyramid, bounded LBFGSB optimizer |
| **Cardiac** | DIS Optical Flow + Gaussian regularization | High sigma (smooth correction) |
| **Abdomen** | DIS Optical Flow + Gaussian regularization | Medium-high sigma |
| **Peripheral** | DIS Optical Flow + Gaussian regularization | Medium sigma |

**B-Spline FFD (Neuro):**
- Uses Mutual Information metric, which is robust to iodine-induced intensity changes
- Vessel pixels excluded from MI computation via tissue masking
- LBFGSB optimizer with bounded control points to prevent image folding
- Runs at half-resolution for memory efficiency

**DIS Optical Flow (Cardiac/Abdomen/Peripheral):**
- Dense Inverse Search at full resolution (PRESET_MEDIUM)
- Heavy Gaussian regularization prevents vessel tracking (floating artifacts)

### Post-Processing

- **Temporal warp smoothing**: Gaussian-weighted averaging of warp parameters over adaptive window, eliminates frame-to-frame vessel jitter
- **Spatially-varying intensity correction**: Fast pyramid blur corrects per-pixel brightness drift
- **Log-domain subtraction**: Physically correct Beer-Lambert tissue cancellation
- **Background suppression**: Gaussian high-pass filter removes low-frequency tissue haze

## Tech Stack

| Component | Purpose |
|---|---|
| C++ (modern standard) | Core language |
| Qt | GUI framework |
| VTK | Image display and rendering |
| ITK | B-Spline registration, DICOM I/O |
| OpenCV | ECC, optical flow, image processing |
| OpenMP | Parallel preprocessing |
| MSVC (Visual Studio) | Compiler |

## Project Structure

```
qt_viewer/
  CMakeLists.txt
  src/
    main.cpp                # Application entry point
    MainWindow.cpp/.h       # Qt UI: controls, layout, signal wiring
    VTKViewer.cpp/.h        # VTK-based DICOM display, DSA pipeline orchestration
    Register.cpp/.h         # ECC, Phase Corr, DIS, ORB, DSA computation
    BSplineRegister.cpp/.h  # ITK B-Spline FFD with Mutual Information
    RegionConfig.cpp/.h     # Per-region parameter tables, DICOM region detection
    Preprocess.cpp/.h       # Frame normalization, log-domain conversion
```

## Building

### Prerequisites

- Windows
- Visual Studio (MSVC toolchain)
- CMake
- Qt (msvc build)
- OpenCV (built from source)
- VTK (built from source)
- ITK (built from source)

### Build Steps

```bash
cd qt_viewer
mkdir build && cd build
cmake ..
cmake --build . --config Release --parallel
```

### Optional: CUDA Acceleration

OpenCV requires a compatible CUDA toolkit for GPU-accelerated operations. Verify version compatibility before building.

```bash
# Rebuild OpenCV with CUDA
cmake -DWITH_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR="<path-to-cuda>" ...
```

## Usage

- Launch the application
- Click **Open** to load a DICOM directory (single angiography run)
- The region is auto-detected from DICOM metadata (or select manually from the dropdown)
- Registration runs automatically on load
- Use the left/right panel dropdowns to compare display modes
- Adjust **Gain**, **BG Suppress**, **CLAHE**, and **Mask** mode as needed
- Use **Play** or the slider for frame-by-frame review

## Region Auto-Detection

The application reads DICOM tag BodyPartExamined and maps it to the appropriate registration pipeline:

- **Neuro**: HEAD, BRAIN, SKULL, NECK, CAROTID, CEREBRAL
- **Cardiac**: HEART, CORONARY, CHEST, THORAX, CARDIAC, PULMONARY
- **Abdomen**: ABDOMEN, AORTA, RENAL, LIVER, PELVIS, MESENTERIC, CELIAC, HEPATIC
- **Peripheral**: LEG, ARM, KNEE, FEMORAL, ILIAC, TIBIAL, POPLITEAL, FOOT, HAND, BRACHIAL

## Algorithm Details

### Why Log-Domain Subtraction?

X-ray attenuation follows the Beer-Lambert law. Taking the logarithm linearizes the exponential attenuation relationship, making subtraction physically meaningful — it isolates the iodine signal with proper tissue cancellation, unlike simple intensity subtraction which leaves residual tissue texture.

### Vessel Masking for Registration

Iodine-filled vessels corrupt intensity-based registration by introducing bright regions that don't exist in the reference frame. The pipeline:

- Detects vessel pixels using Otsu auto-threshold on the frame-reference difference
- Dilates the mask to cover vessel boundaries
- Excludes masked pixels from the registration cost function (ECC or MI)

### Temporal Warp Smoothing

Each frame's registration is computed independently against the mask frame. Small noise variations cause the warp to wobble slightly between adjacent frames, producing vessel "floating" in the DSA loop. Gaussian-weighted temporal averaging with adaptive radius smooths these fluctuations while preserving genuine motion correction.

## License

Proprietary - Internal use only.

