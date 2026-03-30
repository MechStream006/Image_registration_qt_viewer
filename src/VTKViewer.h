#pragma once

#include <QVTKOpenGLNativeWidget.h>
#include <vtkSmartPointer.h>
#include <QTimer>
#include <QScrollBar>

#include <opencv2/opencv.hpp>

// 🔥 DISPLAY MODE
enum DisplayMode
{
    RAW = 0,
    PROCESSED = 1,
    REGISTERED = 2,
    DSA_RAW = 3,
    DSA_REGISTERED = 4
};

class vtkImageViewer2;
class vtkRenderer;
class vtkImageData;

class VTKViewer : public QVTKOpenGLNativeWidget
{
    Q_OBJECT

public:
    explicit VTKViewer(QWidget* parent = nullptr);

    void startPlayback();
    void stopPlayback();
    void setSlice(int slice);
    void setPlaybackSpeed(int ms);
    void updateScrollbars();

    int getMinSlice() const;
    int getMaxSlice() const;

    vtkRenderer* getRenderer();

    // 🔥 NEW
    void setDisplayMode(DisplayMode mode);

    void resetView();

signals:
    void sliceChanged(int slice);

protected:
    void resizeEvent(QResizeEvent* event) override;

private:
    vtkSmartPointer<vtkImageViewer2> viewer;

    vtkSmartPointer<vtkImageData> rawVolume;
    vtkSmartPointer<vtkImageData> processedVolume;
    vtkSmartPointer<vtkImageData> registeredVolume;
    vtkSmartPointer<vtkImageData> dsaRawVolume;
    vtkSmartPointer<vtkImageData> dsaRegisteredVolume;

    QTimer* timer;

    int currentSlice;
    int minSlice;
    int maxSlice;

    double defaultWindow = 65535;
    double defaultLevel = 32767;

    QScrollBar* hScroll;
    QScrollBar* vScroll;

    bool isZoomed = false;
};