#include "MainWindow.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QSlider>
#include <QWidget>
#include <QComboBox>
#include <QFileDialog>
#include <QLabel>

#include "VTKViewer.h"
#include "RegionConfig.h"

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
    leftModeBox->addItem("DSA (ECC registered)");

    rightModeBox = new QComboBox();
    rightModeBox->addItem("RAW");
    rightModeBox->addItem("PREPROCESSED");
    rightModeBox->addItem("REGISTERED");
    rightModeBox->addItem("DSA (unregistered)");
    rightModeBox->addItem("DSA (ECC registered)");

    // REGION selector (Option C: auto-detect + user override)
    regionBox = new QComboBox();
    regionBox->addItem("Region: Auto (DICOM)",  REGION_AUTO);
    regionBox->addItem("Neuro",                 REGION_NEURO);
    regionBox->addItem("Cardiac",               REGION_CARDIAC);
    regionBox->addItem("Abdomen",               REGION_ABDOMEN);
    regionBox->addItem("Peripheral",            REGION_PERIPHERAL);
    regionBox->setToolTip(
        "Auto: reads DICOM BodyPartExamined tag and selects the\n"
        "appropriate registration pipeline automatically.\n"
        "Override: forces a specific pipeline regardless of DICOM.");
    regionBox->setStyleSheet(
        "QComboBox { background:#1a1a2e; color:#88aaff; "
        "border:1px solid #334; padding:2px 4px; }");

    toolbar->addWidget(leftModeBox);
    toolbar->addWidget(rightModeBox);
    toolbar->addWidget(regionBox);

    mainLayout->addLayout(toolbar);

    // =========================
    // 🔥 DSA CONTROLS ROW
    // =========================
    QHBoxLayout* dsaBar = new QHBoxLayout();

    // Label
    QLabel* dsaBarLabel = new QLabel("DSA:");
    dsaBarLabel->setStyleSheet("color:#aaa; font-weight:bold;");
    dsaBar->addWidget(dsaBarLabel);

    // Gain slider — range 10–80 maps to gain 0.5x–4.0x (step 0.05)
    QLabel* gainMinLabel = new QLabel("Gain 0.5x");
    gainMinLabel->setStyleSheet("color:#888; font-size:10px;");
    dsaBar->addWidget(gainMinLabel);

    dsaGainSlider = new QSlider(Qt::Horizontal);
    dsaGainSlider->setMinimum(10);   // 10/20 = 0.5x
    dsaGainSlider->setMaximum(80);   // 80/20 = 4.0x
    dsaGainSlider->setValue(20);     // 20/20 = 1.0x  (default)
    dsaGainSlider->setFixedWidth(180);
    dsaGainSlider->setToolTip("DSA contrast gain (0.5x – 4.0x)");
    dsaBar->addWidget(dsaGainSlider);

    QLabel* gainMaxLabel = new QLabel("4.0x");
    gainMaxLabel->setStyleSheet("color:#888; font-size:10px;");
    dsaBar->addWidget(gainMaxLabel);

    dsaGainLabel = new QLabel("Gain: 1.0x");
    dsaGainLabel->setFixedWidth(72);
    dsaGainLabel->setStyleSheet("color:#00cc44; font-family:monospace; font-size:11px;");
    dsaBar->addWidget(dsaGainLabel);

    dsaBar->addSpacing(16);

    // CLAHE toggle
    claheBtn = new QPushButton("CLAHE");
    claheBtn->setCheckable(true);
    claheBtn->setChecked(false);
    claheBtn->setFixedWidth(60);
    claheBtn->setToolTip("Local contrast enhancement (CLAHE) — toggle on/off");
    claheBtn->setStyleSheet(
        "QPushButton { background:#333; color:#aaa; border:1px solid #555; border-radius:3px; }"
        "QPushButton:checked { background:#005522; color:#00ff88; border:1px solid #00cc44; }");
    dsaBar->addWidget(claheBtn);

    dsaBar->addSpacing(16);

    // BG Suppress — Gaussian high-pass to remove soft-tissue haze
    QLabel* bgLabel = new QLabel("BG Suppress:");
    bgLabel->setStyleSheet("color:#aaa;");
    dsaBar->addWidget(bgLabel);

    bgSuppressBox = new QComboBox();
    bgSuppressBox->addItem("Off",    0.0f);
    bgSuppressBox->addItem("Low (σ=20)",  20.0f);
    bgSuppressBox->addItem("Med (σ=40)",  40.0f);
    bgSuppressBox->addItem("High (σ=60)", 60.0f);
    bgSuppressBox->setCurrentIndex(2);   // Med (σ=40) — removes low-freq background haze
    bgSuppressBox->setToolTip(
        "Gaussian high-pass filter on the DSA result.\n"
        "Removes broad soft-tissue haze while preserving thin vessels.\n"
        "Low/Med/High = sigma 20/40/60 px.");
    dsaBar->addWidget(bgSuppressBox);

    dsaBar->addSpacing(16);

    // Mask mode — pre-contrast average vs temporal median
    QLabel* maskModeLabel = new QLabel("Mask:");
    maskModeLabel->setStyleSheet("color:#aaa;");
    dsaBar->addWidget(maskModeLabel);

    maskModeBox = new QComboBox();
    maskModeBox->addItem("Pre-contrast",     MASK_PRECONTRAST);
    maskModeBox->addItem("Temporal Median",  MASK_TEMPORAL_MEDIAN);
    maskModeBox->setCurrentIndex(0);
    maskModeBox->setToolTip(
        "Pre-contrast: average of frames before contrast arrives.\n"
        "  Good when patient is still.\n\n"
        "Temporal Median: per-pixel median across ALL frames.\n"
        "  Handles non-rigid tissue (brain pulsation, bowel motion).\n"
        "  Best for cases where registration leaves soft tissue ghosts.");
    dsaBar->addWidget(maskModeBox);

    dsaBar->addStretch();
    mainLayout->addLayout(dsaBar);

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
    // STATUS BAR
    // =========================
    statusLabel = new QLabel(this);
    statusLabel->setAlignment(Qt::AlignCenter);
    statusLabel->setStyleSheet(
        "QLabel { background:#1a1a1a; color:#00cc44; "
        "padding:4px; font-family:monospace; font-size:11px; }");
    mainLayout->addWidget(statusLabel);

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

        // Update status bar with current frame number
        BodyRegion selected = static_cast<BodyRegion>(regionBox->currentData().toInt());
        BodyRegion detected = viewer1->getDetectedRegion();
        QString regionStr = (selected == REGION_AUTO)
            ? QString("Auto → %1").arg(regionDisplayName(detected))
            : regionDisplayName(selected);
        float gain = dsaGainSlider->value() / 20.0f;
        int total = viewer1->getMaxSlice() + 1;
        statusLabel->setText(
            QString("Frame: %1/%2  |  Mask: %3  |  Region: %4  |  Gain: %5x")
            .arg(value + 1)
            .arg(total)
            .arg(viewer1->getMaskFrameIndex())
            .arg(regionStr)
            .arg(gain, 0, 'f', 2));
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
    // OPEN DICOM
    // =========================
    connect(openBtn, &QPushButton::clicked, this, [this]() {

        QString path = QFileDialog::getOpenFileName(
            this, "Open DICOM", "", "DICOM Files (*.dcm);;All Files (*)");

        if (path.isEmpty()) return;

        // Stop playback before loading
        viewer1->stopPlayback();
        viewer2->stopPlayback();
        playBtn->setText("Play");

        viewer1->loadDICOM(path);
        viewer2->loadDICOM(path);

        // Update slider range for new sequence length
        slider->setMinimum(viewer1->getMinSlice());
        slider->setMaximum(viewer1->getMaxSlice());
        slider->setValue(0);

        // Reset modes to defaults
        leftModeBox->setCurrentIndex(0);
        rightModeBox->setCurrentIndex(1);

        // Show auto-detected region (most useful feedback for the user)
        BodyRegion detected = viewer1->getDetectedRegion();
        BodyRegion selected = static_cast<BodyRegion>(regionBox->currentData().toInt());
        QString regionStr = (selected == REGION_AUTO)
            ? QString("Auto → %1").arg(regionDisplayName(detected))
            : regionDisplayName(selected);

        float gain = dsaGainSlider->value() / 20.0f;
        int total = viewer1->getMaxSlice() + 1;
        statusLabel->setText(
            QString("Frame: 1/%1  |  Mask: %2  |  Region: %3  |  Gain: %4x")
            .arg(total)
            .arg(viewer1->getMaskFrameIndex())
            .arg(regionStr)
            .arg(gain, 0, 'f', 2));
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

    // =========================
    // REGION SELECTOR
    // =========================
    connect(regionBox, &QComboBox::currentIndexChanged,
        this, [this](int)
        {
            auto region = static_cast<BodyRegion>(regionBox->currentData().toInt());

            viewer1->setBodyRegion(region);
            viewer1->setDisplayMode(static_cast<DisplayMode>(leftModeBox->currentIndex()));

            viewer2->setBodyRegion(region);
            viewer2->setDisplayMode(static_cast<DisplayMode>(rightModeBox->currentIndex()));

            // Show detected region in status bar (relevant for Auto mode)
            BodyRegion detected = viewer1->getDetectedRegion();
            QString detectedStr = (region == REGION_AUTO)
                ? QString("Auto → %1").arg(regionDisplayName(detected))
                : regionDisplayName(region);

            float gain = dsaGainSlider->value() / 20.0f;
            int total = viewer1->getMaxSlice() + 1;
            int curFrame = slider->value() + 1;
            statusLabel->setText(
                QString("Frame: %1/%2  |  Mask: %3  |  Region: %4  |  Gain: %5x")
                .arg(curFrame)
                .arg(total)
                .arg(viewer1->getMaskFrameIndex())
                .arg(detectedStr)
                .arg(gain, 0, 'f', 2));
        });

    // =========================
    // DSA GAIN SLIDER
    // =========================
    connect(dsaGainSlider, &QSlider::valueChanged, this, [this](int value) {
        float gain = value / 20.0f;   // 10→0.5, 20→1.0, 40→2.0, 80→4.0
        dsaGainLabel->setText(QString("Gain: %1x").arg(gain, 0, 'f', 2));
        viewer1->setDSAGain(gain);
        viewer2->setDSAGain(gain);
    });

    // =========================
    // CLAHE TOGGLE
    // =========================
    connect(claheBtn, &QPushButton::toggled, this, [this](bool checked) {
        viewer1->setDSAClahe(checked);
        viewer2->setDSAClahe(checked);
    });

    // =========================
    // BG SUPPRESS
    // =========================
    connect(bgSuppressBox, &QComboBox::currentIndexChanged, this, [this](int) {
        float sigma = bgSuppressBox->currentData().toFloat();
        viewer1->setBgSuppression(sigma);
        viewer2->setBgSuppression(sigma);
    });

    // =========================
    // MASK MODE
    // =========================
    connect(maskModeBox, &QComboBox::currentIndexChanged, this, [this](int) {
        auto mode = static_cast<DSAMaskMode>(maskModeBox->currentData().toInt());
        viewer1->setDSAMaskMode(mode);
        viewer2->setDSAMaskMode(mode);
    });

    // =========================
    // STATUS — shown after viewers finish loading
    // =========================
    statusLabel->setText(
        QString("Mask Frame: %1  |  Frames: %2  |  DSA Gain: 1.00x")
        .arg(viewer1->getMaskFrameIndex())
        .arg(viewer1->getMaxSlice() + 1));
}