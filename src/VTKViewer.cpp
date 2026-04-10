#include "VTKViewer.h"
#include "Preprocess.h"
#include "Register.h"

#include <iostream>
#include <chrono>

struct ScopedTimerV {
    const char* label;
    std::chrono::high_resolution_clock::time_point t0;
    ScopedTimerV(const char* l) : label(l), t0(std::chrono::high_resolution_clock::now()) {}
    ~ScopedTimerV() {
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "[TIMER] " << label << ": " << (int)ms << " ms\n";
    }
};
#include <QResizeEvent>
#include <QScrollBar>

#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkImageViewer2.h>
#include <vtkImageData.h>
#include <vtkCamera.h>
#include <vtkImageFlip.h>

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkGDCMImageIO.h>

#include <opencv2/opencv.hpp>

// =============================================================
// CONSTRUCTOR — VTK/scrollbar/timer setup, then load default
// =============================================================
VTKViewer::VTKViewer(QWidget* parent)
    : QVTKOpenGLNativeWidget(parent)
{
    viewer = vtkSmartPointer<vtkImageViewer2>::New();

    hScroll = new QScrollBar(Qt::Horizontal, this);
    vScroll = new QScrollBar(Qt::Vertical,   this);
    hScroll->hide();
    vScroll->hide();

    connect(hScroll, &QScrollBar::valueChanged, this, [this](int value)
        {
            auto cam = viewer->GetRenderer()->GetActiveCamera();
            double pos[3]; cam->GetPosition(pos);
            pos[0] = value; cam->SetPosition(pos);
            viewer->Render();
        });

    connect(vScroll, &QScrollBar::valueChanged, this, [this](int value)
        {
            auto cam = viewer->GetRenderer()->GetActiveCamera();
            double pos[3]; cam->GetPosition(pos);
            pos[1] = value; cam->SetPosition(pos);
            viewer->Render();
        });

    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, [this]()
        {
            currentSlice++;
            if (currentSlice > maxSlice) currentSlice = minSlice;
            viewer->SetSlice(currentSlice);
            viewer->Render();
            emit sliceChanged(currentSlice);
        });

    loadDICOM("D:/Image_registration/qt_viewer/test.dcm");
}

// =============================================================
// LOAD DICOM — full preprocessing pipeline
//   Safe to call multiple times (new file or reload).
//   Caches processedFloat for re-registration on method change.
// =============================================================
void VTKViewer::loadDICOM(const QString& path)
{
    ScopedTimerV _tTotal("TOTAL loadDICOM");
    if (timer->isActive()) timer->stop();

    auto _tRead = std::chrono::high_resolution_clock::now();
    using ImageType = itk::Image<unsigned short, 3>;
    auto dicomIO = itk::GDCMImageIO::New();
    auto reader  = itk::ImageFileReader<ImageType>::New();
    reader->SetImageIO(dicomIO);
    reader->SetFileName(path.toStdString());
    reader->Update();
    std::cout << "[TIMER] DICOM read: "
              << (int)std::chrono::duration<double,std::milli>(
                     std::chrono::high_resolution_clock::now() - _tRead).count()
              << " ms\n";

    auto itkImage = reader->GetOutput();
    auto sz       = itkImage->GetLargestPossibleRegion().GetSize();

    imgWidth  = sz[0];
    imgHeight = sz[1];
    imgDepth  = sz[2];

    unsigned short* itkPtr = itkImage->GetBufferPointer();

    // -----------------------------------------------------------
    // Extract raw CV_16U frames — kept for log-domain DSA
    // -----------------------------------------------------------
    cachedRawFrames16.clear();
    cachedRawFrames16.reserve(imgDepth);
    for (int z = 0; z < imgDepth; z++)
    {
        cv::Mat frame(imgHeight, imgWidth, CV_16U);
        for (int y = 0; y < imgHeight; y++)
            memcpy(frame.ptr(y),
                   itkPtr + z * imgWidth * imgHeight + y * imgWidth,
                   imgWidth * sizeof(unsigned short));
        cachedRawFrames16.push_back(frame);
    }

    // -----------------------------------------------------------
    // Preprocess → float32 [0,1] (per-frame normalised — for registration)
    // -----------------------------------------------------------
    {   ScopedTimerV _tp("Preprocessing");
        preprocessSequence(cachedRawFrames16, cachedProcessedFloat);
    }

    // -----------------------------------------------------------
    // Build log-domain frames (unnormalised — for DSA subtraction)
    // -----------------------------------------------------------
    buildLogFrames(cachedRawFrames16, cachedUnregisteredLog);

    std::vector<cv::Mat> processed16;
    processed16.reserve(imgDepth);
    for (auto& f : cachedProcessedFloat)
    {
        cv::Mat tmp; f.convertTo(tmp, CV_16U, 65535.0);
        processed16.push_back(tmp);
    }

    // -----------------------------------------------------------
    // Auto-detect mask frame
    // -----------------------------------------------------------
    maskFrameIndex = autoDetectMaskFrame(cachedProcessedFloat);

    // -----------------------------------------------------------
    // Auto-detect body region from DICOM tag (Option C):
    // if the user has explicitly selected a region, honour it;
    // otherwise read the DICOM BodyPartExamined tag.
    // -----------------------------------------------------------
    if (selectedRegion == REGION_AUTO)
        detectedRegion = detectRegionFromDICOM(path.toStdString());
    else
        detectedRegion = selectedRegion;   // explicit override

    // -----------------------------------------------------------
    // Fill fixed volumes (don't change with registration method)
    // -----------------------------------------------------------
    rawVolume       = makeVolume(cachedRawFrames16);
    processedVolume = makeVolume(processed16);

    // DSA unregistered — log-domain subtraction (physics-correct)
    cv::Mat unregMask = buildLogMask(cachedUnregisteredLog, maskFrameIndex, MASK_PRECONTRAST);
    std::vector<cv::Mat> dsaUnregFrames;
    computeDSALogDomain(cachedUnregisteredLog, unregMask, dsaUnregFrames, dsaGain, dsaClahe, dsaBgSigma);
    dsaRawVolume = makeVolume(dsaUnregFrames);

    // -----------------------------------------------------------
    // Registration + ECC DSA (depends on current method)
    // -----------------------------------------------------------
    runRegistration(currentRegMethod);

    // -----------------------------------------------------------
    // Wire VTK viewer (first load only)
    // -----------------------------------------------------------
    if (!viewerInitialized)
    {
        auto flip = vtkSmartPointer<vtkImageFlip>::New();
        flip->SetInputData(rawVolume);
        flip->SetFilteredAxis(1);
        flip->Update();

        viewer->SetInputConnection(flip->GetOutputPort());
        viewer->SetRenderWindow(this->renderWindow());
        viewer->SetupInteractor(this->interactor());

        auto renderer = viewer->GetRenderer();
        renderer->GetActiveCamera()->ParallelProjectionOn();
        renderer->ResetCamera();

        viewerInitialized = true;
    }
    else
    {
        setDisplayMode(currentMode);
    }

    viewer->SetColorWindow(65535);
    viewer->SetColorLevel(32767);

    minSlice     = 0;
    maxSlice     = imgDepth - 1;
    currentSlice = 0;
    viewer->SetSlice(currentSlice);
    viewer->Render();
}

// =============================================================
// RUN REGISTRATION — re-registers using the chosen method
//   Only updates registeredVolume + dsaECCVolume.
//   rawVolume / processedVolume / dsaRawVolume are unchanged.
// =============================================================
void VTKViewer::runRegistration(RegistrationMethod method)
{
    ScopedTimerV _tReg("TOTAL runRegistration");
    // --- Stage 1 (+2): Registration on normalised frames ---
    // Outputs: cachedRegisteredFloat (for REGISTERED display)
    //          cachedWarpMatrices    (affine per frame)
    //          cachedDeformFields    (displacement fields, may be empty)
    cachedWarpMatrices.clear();
    cachedDeformFields.clear();

    BodyRegion ar = activeRegion();
    if (ar != REGION_AUTO)
        registerSequenceForRegion(cachedProcessedFloat, cachedRegisteredFloat,
                                   maskFrameIndex, ar,
                                   &cachedWarpMatrices, &cachedDeformFields);
    else
        registerSequence(cachedProcessedFloat, cachedRegisteredFloat,
                          maskFrameIndex, method, &cachedWarpMatrices);

    // --- Temporal smoothing of warps to eliminate frame-to-frame jitter ---
    // Adaptive radius: half the sequence length → nearly global averaging
    // so every frame converges to a very similar geometric transform.
    int smoothRadius = std::max(3, imgDepth / 2);
    if (!cachedWarpMatrices.empty())
        temporalSmoothWarps(cachedWarpMatrices, maskFrameIndex, smoothRadius);
    if (!cachedDeformFields.empty())
        temporalSmoothFields(cachedDeformFields, maskFrameIndex, smoothRadius);

    std::vector<cv::Mat> registered16;
    registered16.reserve(imgDepth);
    for (auto& f : cachedRegisteredFloat)
    {
        cv::Mat tmp; f.convertTo(tmp, CV_16U, 65535.0);
        registered16.push_back(tmp);
    }
    registeredVolume = makeVolume(registered16);

    // --- Build log-domain registered frames for DSA ---
    // Apply the SAME warps to raw log frames (no per-frame normalisation)
    // so that subtraction gives physically correct iodine signal.
    if (!cachedWarpMatrices.empty())
    {
        buildRegisteredLogFrames(cachedRawFrames16, cachedWarpMatrices,
                                  cachedDeformFields, cachedRegisteredLog,
                                  maskFrameIndex);
    }
    else
    {
        // Fallback: registration method didn't output warps.
        // Use the old normalised DSA path.
        cachedRegisteredLog.clear();
    }

    // --- DSA on registered log frames ---
    std::vector<cv::Mat> dsaRegFrames;
    if (!cachedRegisteredLog.empty())
    {
        cv::Mat regMask = buildLogMask(cachedRegisteredLog, maskFrameIndex, dsaMaskMode);
        computeDSALogDomain(cachedRegisteredLog, regMask, dsaRegFrames,
                             dsaGain, dsaClahe, dsaBgSigma);
    }
    else
    {
        // Fallback to old method
        computeDSA(cachedRegisteredFloat, dsaRegFrames, maskFrameIndex,
                    dsaGain, dsaClahe, dsaBgSigma, dsaMaskMode);
    }

    dsaECCVolume = makeVolume(dsaRegFrames);
}

// =============================================================
// SET REGISTRATION METHOD — re-registers in place
// =============================================================
void VTKViewer::setRegistrationMethod(RegistrationMethod method)
{
    currentRegMethod = method;
    if (!cachedProcessedFloat.empty())
        runRegistration(method);
    // Caller must call setDisplayMode() after this to refresh the view
}

// =============================================================
// MAKE VOLUME — fill vtkImageData from CV_16U frame vector
// =============================================================
vtkSmartPointer<vtkImageData> VTKViewer::makeVolume(const std::vector<cv::Mat>& frames)
{
    auto vol = vtkSmartPointer<vtkImageData>::New();
    vol->SetDimensions(imgWidth, imgHeight, imgDepth);
    vol->SetExtent(0, imgWidth-1, 0, imgHeight-1, 0, imgDepth-1);
    vol->AllocateScalars(VTK_UNSIGNED_SHORT, 1);

    unsigned short* ptr = static_cast<unsigned short*>(vol->GetScalarPointer());
    for (int z = 0; z < imgDepth; z++)
        for (int y = 0; y < imgHeight; y++)
            memcpy(ptr + z * imgWidth * imgHeight + y * imgWidth,
                   frames[z].ptr(y), imgWidth * sizeof(unsigned short));
    return vol;
}

// =============================================================
// MODE SWITCH
// =============================================================
void VTKViewer::setDisplayMode(DisplayMode mode)
{
    currentMode = mode;

    vtkSmartPointer<vtkImageData> src;
    switch (mode)
    {
    case RAW:        src = rawVolume;        break;
    case PROCESSED:  src = processedVolume;  break;
    case REGISTERED: src = registeredVolume; break;
    case DSA_RAW:    src = dsaRawVolume;     break;
    case DSA_ECC:    src = dsaECCVolume;     break;
    default:         src = rawVolume;        break;
    }

    auto flip = vtkSmartPointer<vtkImageFlip>::New();
    flip->SetInputData(src);
    flip->SetFilteredAxis(1);
    flip->Update();

    viewer->SetInputConnection(flip->GetOutputPort());
    viewer->SetSlice(currentSlice);

    if (mode == DSA_RAW || mode == DSA_ECC)
    {
        // Neutral background lands at offset=0.25 → 16383 in uint16
        // regardless of gain (gain only moves vessel pixels, not background).
        // Window [0, 32767]: background at mid-gray, vessels appear dark.
        viewer->SetColorLevel(16383);
        viewer->SetColorWindow(32767);
    }
    else
    {
        viewer->SetColorWindow(defaultWindow);
        viewer->SetColorLevel(defaultLevel);
    }

    viewer->Render();
}

// =============================================================
// RESIZE
// =============================================================
void VTKViewer::resizeEvent(QResizeEvent* event)
{
    QVTKOpenGLNativeWidget::resizeEvent(event);
    if (viewer)
    {
        viewer->GetRenderWindow()->SetSize(this->width(), this->height());
        viewer->Render();
    }
    hScroll->setGeometry(0, height()-20, width()-20, 20);
    vScroll->setGeometry(width()-20, 0, 20, height()-20);
}

// =============================================================
// RESET VIEW
// =============================================================
void VTKViewer::resetView()
{
    auto renderer = viewer->GetRenderer();
    auto camera   = renderer->GetActiveCamera();
    renderer->ResetCamera();
    renderer->ResetCameraClippingRange();

    double* bounds = viewer->GetInput()->GetBounds();
    camera->SetParallelScale(std::max(bounds[1]-bounds[0], bounds[3]-bounds[2]) / 2.0);

    viewer->SetColorWindow(defaultWindow);
    viewer->SetColorLevel(defaultLevel);
    currentSlice = minSlice;
    viewer->SetSlice(currentSlice);
    hScroll->hide();
    vScroll->hide();
    viewer->Render();
}

// =============================================================
// SCROLLBARS
// =============================================================
void VTKViewer::updateScrollbars()
{
    double scale = viewer->GetRenderer()->GetActiveCamera()->GetParallelScale();
    if (scale < 500) { hScroll->show(); vScroll->show(); }
    else             { hScroll->hide(); vScroll->hide(); }
}

// =============================================================
// PLAYBACK
// =============================================================
void VTKViewer::startPlayback()          { timer->start(80); }
void VTKViewer::stopPlayback()           { timer->stop(); }
void VTKViewer::setPlaybackSpeed(int ms) { timer->setInterval(ms); }

// =============================================================
// SLICE
// =============================================================
void VTKViewer::setSlice(int slice)
{
    currentSlice = slice;
    viewer->SetSlice(slice);
    viewer->Render();
}

int VTKViewer::getMinSlice() const { return minSlice; }
int VTKViewer::getMaxSlice() const { return maxSlice; }

// =============================================================
// RENDERER / MASK
// =============================================================
vtkRenderer* VTKViewer::getRenderer()    { return viewer->GetRenderer(); }
int VTKViewer::getMaskFrameIndex() const { return maskFrameIndex; }

// =============================================================
// BODY REGION — changing region re-runs the full registration
// with the appropriate pipeline (neuro / cardiac / abdomen /
// peripheral).  Safe to call before or after loadDICOM.
// =============================================================
void VTKViewer::setBodyRegion(BodyRegion region)
{
    selectedRegion = region;
    if (region != REGION_AUTO)
        detectedRegion = region;   // explicit override beats DICOM detection

    if (cachedProcessedFloat.empty()) return;   // no data loaded yet

    runRegistration(currentRegMethod);
    setDisplayMode(currentMode);
}

// =============================================================
// DSA GAIN — update live without re-running registration
// =============================================================
void VTKViewer::setDSAGain(float gain)
{
    dsaGain = gain;
    recomputeDSAVolumes();
    if (currentMode == DSA_RAW || currentMode == DSA_ECC)
        setDisplayMode(currentMode);
}

// =============================================================
// DSA CLAHE toggle — update live without re-running registration
// =============================================================
void VTKViewer::setDSAClahe(bool enabled)
{
    dsaClahe = enabled;
    recomputeDSAVolumes();
    if (currentMode == DSA_RAW || currentMode == DSA_ECC)
        setDisplayMode(currentMode);
}

// =============================================================
// BG SUPPRESSION — Gaussian high-pass sigma, live update
// =============================================================
void VTKViewer::setBgSuppression(float sigma)
{
    dsaBgSigma = sigma;
    recomputeDSAVolumes();
    if (currentMode == DSA_RAW || currentMode == DSA_ECC)
        setDisplayMode(currentMode);
}

// =============================================================
// DSA MASK MODE — pre-contrast average vs temporal median
// =============================================================
void VTKViewer::setDSAMaskMode(DSAMaskMode mode)
{
    dsaMaskMode = mode;
    recomputeDSAVolumes();
    if (currentMode == DSA_RAW || currentMode == DSA_ECC)
        setDisplayMode(currentMode);
}

// =============================================================
// Recompute both DSA volumes from cached float frames.
// Fast — no registration, just arithmetic.  Called on
// every gain or CLAHE toggle change.
// =============================================================
void VTKViewer::recomputeDSAVolumes()
{
    if (cachedRawFrames16.empty()) return;

    // --- Unregistered DSA (log-domain) ---
    if (!cachedUnregisteredLog.empty())
    {
        cv::Mat unregMask = buildLogMask(cachedUnregisteredLog, maskFrameIndex, MASK_PRECONTRAST);
        std::vector<cv::Mat> dsaUnreg;
        computeDSALogDomain(cachedUnregisteredLog, unregMask, dsaUnreg,
                             dsaGain, dsaClahe, dsaBgSigma);
        dsaRawVolume = makeVolume(dsaUnreg);
    }

    // --- Registered DSA (log-domain if warps available, else old method) ---
    if (!cachedRegisteredLog.empty())
    {
        cv::Mat regMask = buildLogMask(cachedRegisteredLog, maskFrameIndex, dsaMaskMode);
        std::vector<cv::Mat> dsaReg;
        computeDSALogDomain(cachedRegisteredLog, regMask, dsaReg,
                             dsaGain, dsaClahe, dsaBgSigma);
        dsaECCVolume = makeVolume(dsaReg);
    }
    else if (!cachedRegisteredFloat.empty())
    {
        std::vector<cv::Mat> dsaReg;
        computeDSA(cachedRegisteredFloat, dsaReg, maskFrameIndex,
                    dsaGain, dsaClahe, dsaBgSigma, dsaMaskMode);
        dsaECCVolume = makeVolume(dsaReg);
    }
}
