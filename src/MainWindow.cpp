#include "MainWindow.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QSlider>
#include <QWidget>
#include <QComboBox>
#include <QFileDialog>

#include "VTKViewer.h"

// VTK
#include <vtkRenderer.h>
#include <vtkCamera.h>
#include <vtkRenderWindow.h>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    QWidget* central = new QWidget(this);
    setCentralWidget(central);

    QVBoxLayout* mainLayout = new QVBoxLayout(central);

    // =========================
    // 🔥 TOOLBAR
    // =========================
    QHBoxLayout* toolbar = new QHBoxLayout();

    openBtn = new QPushButton("Open");
    zoomInBtn = new QPushButton("+");
    zoomOutBtn = new QPushButton("-");
    resetBtn = new QPushButton("Reset");

    toolbar->addWidget(openBtn);
    toolbar->addWidget(zoomInBtn);
    toolbar->addWidget(zoomOutBtn);
    toolbar->addWidget(resetBtn);

    // 🔥 SPEED
    speedBox = new QComboBox();
    speedBox->addItem("0.25x", 200);
    speedBox->addItem("0.5x", 150);
    speedBox->addItem("0.8x", 100);
    speedBox->addItem("1x", 80);
    speedBox->addItem("1.25x", 60);
    speedBox->addItem("1.5x", 50);
    speedBox->addItem("2x", 30);
    speedBox->setCurrentIndex(3);

    toolbar->addWidget(speedBox);

    // 🔥 MODE SELECTORS
    leftModeBox = new QComboBox();
    leftModeBox->addItem("RAW");
    leftModeBox->addItem("PREPROCESSED");
    leftModeBox->addItem("REGISTERED");
    leftModeBox->addItem("DSA (unregistered)");
    leftModeBox->addItem("DSA (registered)");

    rightModeBox = new QComboBox();
    rightModeBox->addItem("RAW");
    rightModeBox->addItem("PREPROCESSED");
    rightModeBox->addItem("REGISTERED");
    rightModeBox->addItem("DSA (unregistered)");
	rightModeBox->addItem("DSA (registered)");

    toolbar->addWidget(leftModeBox);
    toolbar->addWidget(rightModeBox);

    mainLayout->addLayout(toolbar);

    // =========================
    // 🔥 VIEWERS
    // =========================
    QHBoxLayout* viewerLayout = new QHBoxLayout();

    viewer1 = new VTKViewer(this);
    viewer2 = new VTKViewer(this);

    viewerLayout->addWidget(viewer1);
    viewerLayout->addWidget(viewer2);

    viewerLayout->setStretch(0, 1);
    viewerLayout->setStretch(1, 1);

    mainLayout->addLayout(viewerLayout, 1);

    // =========================
    // 🔥 CONTROLS
    // =========================
    QHBoxLayout* controls = new QHBoxLayout();

    playBtn = new QPushButton("Play");
    controls->addWidget(playBtn);

    slider = new QSlider(Qt::Horizontal);
    controls->addWidget(slider);

    mainLayout->addLayout(controls);

    // =========================
    // 🔥 SLIDER INIT
    // =========================
    slider->setMinimum(viewer1->getMinSlice());
    slider->setMaximum(viewer1->getMaxSlice());

    // =========================
    // 🔥 DEFAULT MODES
    // =========================
    leftModeBox->setCurrentIndex(0);   // RAW
    rightModeBox->setCurrentIndex(1);  // PROCESSED

    viewer1->setDisplayMode(RAW);
    viewer2->setDisplayMode(PROCESSED);

    // =========================
    // 🔥 PLAY / PAUSE
    // =========================
    connect(playBtn, &QPushButton::clicked, this, [this]() {

        static bool playing = false;

        if (!playing)
        {
            viewer1->startPlayback();
            viewer2->startPlayback();
            playBtn->setText("Pause");
        }
        else
        {
            viewer1->stopPlayback();
            viewer2->stopPlayback();
            playBtn->setText("Play");
        }

        playing = !playing;
        });

    // =========================
    // 🔥 SLIDER SYNC
    // =========================
    connect(viewer1, &VTKViewer::sliceChanged, slider, &QSlider::setValue);

    connect(slider, &QSlider::valueChanged, this, [this](int value) {
        viewer1->setSlice(value);
        viewer2->setSlice(value);
        });

    // =========================
    // 🔥 ZOOM
    // =========================
    connect(zoomInBtn, &QPushButton::clicked, this, [this]() {

        auto cam1 = viewer1->getRenderer()->GetActiveCamera();
        auto cam2 = viewer2->getRenderer()->GetActiveCamera();

        cam1->Zoom(1.2);
        cam2->Zoom(1.2);

        viewer1->renderWindow()->Render();
        viewer2->renderWindow()->Render();

        viewer1->updateScrollbars();
        viewer2->updateScrollbars();
        });

    connect(zoomOutBtn, &QPushButton::clicked, this, [this]() {

        auto cam1 = viewer1->getRenderer()->GetActiveCamera();
        auto cam2 = viewer2->getRenderer()->GetActiveCamera();

        cam1->Zoom(0.8);
        cam2->Zoom(0.8);

        viewer1->renderWindow()->Render();
        viewer2->renderWindow()->Render();

        });

    // =========================
    // 🔥 SPEED
    // =========================
    connect(speedBox, &QComboBox::currentIndexChanged, this, [this](int) {

        int interval = speedBox->currentData().toInt();

        viewer1->setPlaybackSpeed(interval);
        viewer2->setPlaybackSpeed(interval);
        });

    // RESET
    connect(resetBtn, &QPushButton::clicked, this, [this]() {

        viewer1->resetView();
        viewer2->resetView();
        });

    // =========================
    // 🔥 MODE SWITCH
    // =========================
    connect(leftModeBox, &QComboBox::currentIndexChanged,
        this, [this](int index)
        {
            viewer1->setDisplayMode(static_cast<DisplayMode>(index));
        });

    connect(rightModeBox, &QComboBox::currentIndexChanged,
        this, [this](int index)
        {
            viewer2->setDisplayMode(static_cast<DisplayMode>(index));
        });
}