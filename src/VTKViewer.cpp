#include "VTKViewer.h"
#include "Preprocess.h"
#include "Register.h"

#include <iostream>
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

VTKViewer::VTKViewer(QWidget* parent)
    : QVTKOpenGLNativeWidget(parent)
{
    using ImageType = itk::Image<unsigned short, 3>;

    auto dicomIO = itk::GDCMImageIO::New();
    auto reader = itk::ImageFileReader<ImageType>::New();
    reader->SetImageIO(dicomIO);
    reader->SetFileName("D:/Image_registration/qt_viewer/test.dcm");
    reader->Update();

    auto itkImage = reader->GetOutput();
    auto region = itkImage->GetLargestPossibleRegion();
    auto size = region.GetSize();

    int width = size[0];
    int height = size[1];
    int depth = size[2];

    unsigned short* itkPtr = itkImage->GetBufferPointer();

    // =============================================================
    // STEP 1: Extract raw CV_16U frames from ITK buffer
    // =============================================================
    std::vector<cv::Mat> rawFrames;
    rawFrames.reserve(depth);

    for (int z = 0; z < depth; z++)
    {
        cv::Mat frame(height, width, CV_16U);
        for (int y = 0; y < height; y++)
            memcpy(frame.ptr(y),
                itkPtr + z * width * height + y * width,
                width * sizeof(unsigned short));
        rawFrames.push_back(frame);
    }

    // =============================================================
    // STEP 2: Preprocess raw → float32 [0,1]
    //   Used for: display, registration, and DSA subtraction
    // =============================================================
    std::vector<cv::Mat> processedFloat;
    preprocessSequence(rawFrames, processedFloat);

    // float32 → CV_16U for PREPROCESSED viewer mode
    std::vector<cv::Mat> processed16;
    processed16.reserve(depth);
    for (auto& f : processedFloat)
    {
        cv::Mat temp16;
        f.convertTo(temp16, CV_16U, 65535.0);
        processed16.push_back(temp16);
    }

    // =============================================================
    // STEP 3: Registration
    //   ECC computed ON preprocessed float32 frames
    //   Warp also APPLIED TO preprocessed float32 frames
    //   registeredFloat is float32 [0,1] — display-ready after →16U
    //
    //   Why not raw? Raw is dark 16-bit — ECC struggles on it AND
    //   display would need window/level tuning. Preprocessed is
    //   already normalized [0,1] — works directly for both.
    // =============================================================
    std::vector<cv::Mat> registeredFloat;

    registerSequenceECC(
        processedFloat,        // float32 [0,1]: ECC input AND warp target
        registeredFloat,       // float32 [0,1]: warped output
        cv::MOTION_EUCLIDEAN,
        3);

    // float32 → CV_16U for REGISTERED viewer mode
    std::vector<cv::Mat> registered16;
    registered16.reserve(depth);
    for (auto& f : registeredFloat)
    {
        cv::Mat temp16;
        f.convertTo(temp16, CV_16U, 65535.0);
        registered16.push_back(temp16);
    }

    // =============================================================
    // STEP 4: DSA
    //   Both DSA modes operate on float32 [0,1] preprocessed frames
    //   DSA unregistered: uses processedFloat (motion artifacts visible)
    //   DSA registered:   uses registeredFloat (motion corrected)
    //   Result is CV_16U: vessels dark, background white
    // =============================================================
    std::vector<cv::Mat> dsaUnregFrames;
    computeDSA(processedFloat, dsaUnregFrames, 0);

    std::vector<cv::Mat> dsaRegFrames;
    computeDSA(registeredFloat, dsaRegFrames, 0);

    // =============================================================
    // STEP 5: Fill all VTK volumes
    // =============================================================
    auto fillVTKVolume = [&](const std::vector<cv::Mat>& frameVec)
        -> vtkSmartPointer<vtkImageData>
        {
            auto vol = vtkSmartPointer<vtkImageData>::New();
            vol->SetDimensions(width, height, depth);
            vol->SetExtent(0, width - 1, 0, height - 1, 0, depth - 1);
            vol->AllocateScalars(VTK_UNSIGNED_SHORT, 1);

            unsigned short* ptr =
                static_cast<unsigned short*>(vol->GetScalarPointer());

            for (int z = 0; z < depth; z++)
                for (int y = 0; y < height; y++)
                    memcpy(ptr + z * width * height + y * width,
                        frameVec[z].ptr(y),
                        width * sizeof(unsigned short));
            return vol;
        };

    rawVolume = fillVTKVolume(rawFrames);
    processedVolume = fillVTKVolume(processed16);
    registeredVolume = fillVTKVolume(registered16);
    dsaRawVolume = fillVTKVolume(dsaUnregFrames);
    dsaRegisteredVolume = fillVTKVolume(dsaRegFrames);

    // =============================================================
    // STEP 6: VTK viewer setup -- start in RAW mode
    // =============================================================
    viewer = vtkSmartPointer<vtkImageViewer2>::New();

    auto flip = vtkSmartPointer<vtkImageFlip>::New();
    flip->SetInputData(rawVolume);
    flip->SetFilteredAxis(1);
    flip->Update();

    viewer->SetInputConnection(flip->GetOutputPort());
    viewer->SetRenderWindow(this->renderWindow());
    viewer->SetupInteractor(this->interactor());

    viewer->SetColorWindow(65535);
    viewer->SetColorLevel(32767);

    minSlice = 0;
    maxSlice = depth - 1;
    currentSlice = 0;

    viewer->SetSlice(currentSlice);

    auto renderer = viewer->GetRenderer();
    auto camera = renderer->GetActiveCamera();
    camera->ParallelProjectionOn();
    renderer->ResetCamera();

    // =============================================================
    // SCROLLBARS
    // =============================================================
    hScroll = new QScrollBar(Qt::Horizontal, this);
    vScroll = new QScrollBar(Qt::Vertical, this);

    hScroll->hide();
    vScroll->hide();

    connect(hScroll, &QScrollBar::valueChanged, this, [this](int value)
        {
            auto cam = viewer->GetRenderer()->GetActiveCamera();
            double pos[3];
            cam->GetPosition(pos);
            pos[0] = value;
            cam->SetPosition(pos);
            viewer->Render();
        });

    connect(vScroll, &QScrollBar::valueChanged, this, [this](int value)
        {
            auto cam = viewer->GetRenderer()->GetActiveCamera();
            double pos[3];
            cam->GetPosition(pos);
            pos[1] = value;
            cam->SetPosition(pos);
            viewer->Render();
        });

    // =============================================================
    // PLAYBACK TIMER
    // =============================================================
    timer = new QTimer(this);

    connect(timer, &QTimer::timeout, this, [this]()
        {
            currentSlice++;
            if (currentSlice > maxSlice)
                currentSlice = minSlice;

            viewer->SetSlice(currentSlice);
            viewer->Render();
            emit sliceChanged(currentSlice);
        });

    viewer->Render();
}

// =============================================================
// MODE SWITCH
// =============================================================
void VTKViewer::setDisplayMode(DisplayMode mode)
{
    vtkSmartPointer<vtkImageData> src;

    switch (mode)
    {
    case RAW:            src = rawVolume;            break;
    case PROCESSED:      src = processedVolume;      break;
    case REGISTERED:     src = registeredVolume;     break;
    case DSA_RAW:        src = dsaRawVolume;         break;
    case DSA_REGISTERED: src = dsaRegisteredVolume;  break;
    default:             src = rawVolume;            break;
    }

    auto flip = vtkSmartPointer<vtkImageFlip>::New();
    flip->SetInputData(src);
    flip->SetFilteredAxis(1);
    flip->Update();

    viewer->SetInputConnection(flip->GetOutputPort());
    viewer->SetSlice(currentSlice);

    // Full 16-bit range for all modes -- CLAHE inside computeDSA
    // already handled local contrast for DSA modes
    viewer->SetColorWindow(65535);
    viewer->SetColorLevel(32767);

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

    hScroll->setGeometry(0, height() - 20, width() - 20, 20);
    vScroll->setGeometry(width() - 20, 0, 20, height() - 20);
}

// =============================================================
// RESET VIEW
// =============================================================
void VTKViewer::resetView()
{
    auto renderer = viewer->GetRenderer();
    auto camera = renderer->GetActiveCamera();

    renderer->ResetCamera();
    renderer->ResetCameraClippingRange();

    double* bounds = viewer->GetInput()->GetBounds();
    double  w = bounds[1] - bounds[0];
    double  h = bounds[3] - bounds[2];
    double  maxDim = std::max(w, h);

    camera->SetParallelScale(maxDim / 2.0);

    viewer->SetColorWindow(defaultWindow);
    viewer->SetColorLevel(defaultLevel);

    currentSlice = minSlice;
    viewer->SetSlice(currentSlice);

    hScroll->hide();
    vScroll->hide();

    viewer->Render();
}

// =============================================================
// SCROLLBAR VISIBILITY
// =============================================================
void VTKViewer::updateScrollbars()
{
    auto   cam = viewer->GetRenderer()->GetActiveCamera();
    double scale = cam->GetParallelScale();

    if (scale < 500)
    {
        hScroll->show();
        vScroll->show();
    }
    else
    {
        hScroll->hide();
        vScroll->hide();
    }
}

// =============================================================
// PLAYBACK
// =============================================================
void VTKViewer::startPlayback() { timer->start(80); }
void VTKViewer::stopPlayback() { timer->stop(); }

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
// RENDERER ACCESS
// =============================================================
vtkRenderer* VTKViewer::getRenderer()
{
    return viewer->GetRenderer();
}