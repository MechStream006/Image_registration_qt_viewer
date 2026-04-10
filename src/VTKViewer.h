#pragma once

#include <QVTKOpenGLNativeWidget.h>
#include <vtkSmartPointer.h>
#include <QTimer>
#include <QScrollBar>
#include <QString>

#include <opencv2/opencv.hpp>
#include <vector>

#include "Register.h"       // RegistrationMethod enum
#include "RegionConfig.h"   // BodyRegion enum

enum DisplayMode
{
    RAW        = 0,
    PROCESSED  = 1,
    REGISTERED = 2,
    DSA_RAW    = 3,
    DSA_ECC    = 4
};

class vtkImageViewer2;
class vtkRenderer;
class vtkImageData;

class VTKViewer : public QVTKOpenGLNativeWidget
{
    Q_OBJECT

public:
    explicit VTKViewer(QWidget* parent = nullptr);

    void loadDICOM(const QString& path);
    void setRegistrationMethod(RegistrationMethod method);

    void startPlayback();
    void stopPlayback();
    void setSlice(int slice);
    void setPlaybackSpeed(int ms);
    void updateScrollbars();

    int getMinSlice() const;
    int getMaxSlice() const;

    vtkRenderer* getRenderer();
    void setDisplayMode(DisplayMode mode);
    void resetView();

    int getMaskFrameIndex() const;

    // DSA parameters — can be changed live without re-registering
    void setDSAGain(float gain);
    void setDSAClahe(bool enabled);
    void setBgSuppression(float sigma);       // 0=off, >0=Gaussian high-pass sigma (px)
    void setDSAMaskMode(DSAMaskMode mode);    // pre-contrast avg vs temporal median
    // Region selection
    // selectedRegion — user's explicit choice (or REGION_AUTO)
    // detectedRegion — read from DICOM tag on loadDICOM
    // Active region  — selectedRegion if not AUTO, else detectedRegion
    void       setBodyRegion(BodyRegion region);
    BodyRegion getDetectedRegion() const { return detectedRegion; }

signals:
    void sliceChanged(int slice);

protected:
    void resizeEvent(QResizeEvent* event) override;

private:
    void runRegistration(RegistrationMethod method);
    void recomputeDSAVolumes();   // fast recompute on gain/clahe change, no re-registration
    vtkSmartPointer<vtkImageData> makeVolume(const std::vector<cv::Mat>& frames);

    vtkSmartPointer<vtkImageViewer2> viewer;

    vtkSmartPointer<vtkImageData> rawVolume;
    vtkSmartPointer<vtkImageData> processedVolume;
    vtkSmartPointer<vtkImageData> registeredVolume;
    vtkSmartPointer<vtkImageData> dsaRawVolume;
    vtkSmartPointer<vtkImageData> dsaECCVolume;

    QTimer*     timer;
    QScrollBar* hScroll;
    QScrollBar* vScroll;

    int currentSlice = 0;
    int minSlice     = 0;
    int maxSlice     = 0;

    int imgWidth  = 0;
    int imgHeight = 0;
    int imgDepth  = 0;

    double defaultWindow = 65535;
    double defaultLevel  = 32767;

    int maskFrameIndex = 0;

    std::vector<cv::Mat> cachedRawFrames16;       // raw uint16 — source of truth
    std::vector<cv::Mat> cachedProcessedFloat;   // normalised float32 — for registration
    std::vector<cv::Mat> cachedRegisteredFloat;  // warped normalised — for REGISTERED display

    // Log-domain DSA data (physics-correct subtraction)
    std::vector<cv::Mat> cachedUnregisteredLog;  // log(raw+1), no warp — for unregistered DSA
    std::vector<cv::Mat> cachedRegisteredLog;    // log(raw+1) + warps — for registered DSA
    std::vector<cv::Mat> cachedWarpMatrices;     // affine 2x3 per frame from registration
    std::vector<cv::Mat> cachedDeformFields;     // CV_32FC2 displacement per frame (may be empty)

    RegistrationMethod   currentRegMethod = REG_ECC_CPU;
    DisplayMode          currentMode      = RAW;
    bool                 viewerInitialized = false;

    float       dsaGain     = 1.0f;
    bool        dsaClahe    = false;
    float       dsaBgSigma  = 40.0f;             // Gaussian high-pass sigma; removes low-freq haze
    DSAMaskMode dsaMaskMode = MASK_PRECONTRAST;

    BodyRegion selectedRegion = REGION_AUTO;  // explicit UI selection
    BodyRegion detectedRegion = REGION_AUTO;  // from DICOM tag

    BodyRegion activeRegion() const
    {
        return (selectedRegion != REGION_AUTO) ? selectedRegion : detectedRegion;
    }
};
