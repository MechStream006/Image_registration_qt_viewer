#pragma once

#include <QMainWindow>

class VTKViewer;
class QPushButton;
class QSlider;
class QComboBox;
class QLabel;

class MainWindow : public QMainWindow
{
public:
    MainWindow(QWidget* parent = nullptr);

private:
    VTKViewer* viewer1;
    VTKViewer* viewer2;

    // Playback
    QPushButton* playBtn;
    QSlider*     slider;

    // Toolbar — navigation
    QPushButton* openBtn;
    QPushButton* zoomInBtn;
    QPushButton* zoomOutBtn;
    QPushButton* resetBtn;
    QComboBox*   speedBox;

    // Toolbar — mode / region
    QComboBox*   leftModeBox;
    QComboBox*   rightModeBox;
    QComboBox*   regionBox;

    // Toolbar — DSA controls
    QSlider*     dsaGainSlider;
    QLabel*      dsaGainLabel;
    QPushButton* claheBtn;
    QComboBox*   bgSuppressBox;
    QComboBox*   maskModeBox;

    QLabel* statusLabel;
};