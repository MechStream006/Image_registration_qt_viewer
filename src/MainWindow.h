#pragma once

#include <QMainWindow>

class VTKViewer;
class QPushButton;
class QSlider;
class QComboBox;

class MainWindow : public QMainWindow
{
public:
    MainWindow(QWidget* parent = nullptr);

private:
    // 🔥 VIEWERS
    VTKViewer* viewer1;
    VTKViewer* viewer2;

    // 🔥 PLAYBACK
    QPushButton* playBtn;
    QSlider* slider;

    // 🔥 TOOLBAR
    QPushButton* openBtn;
    QPushButton* zoomInBtn;
    QPushButton* zoomOutBtn;
    QPushButton* resetBtn;

    QComboBox* speedBox;

    // 🔥 NEW (MODE CONTROL)
    QComboBox* leftModeBox;
    QComboBox* rightModeBox;
};