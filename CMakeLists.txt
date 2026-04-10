cmake_minimum_required(VERSION 3.16)
project(QtTest)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# Qt
set(CMAKE_PREFIX_PATH "C:/Qt/6.8.2/msvc2022_64")
find_package(Qt6 REQUIRED COMPONENTS Widgets)

# OpenCV
set(OpenCV_DIR "D:/Image_registration/dev/opencv-build/install/x64/vc17/lib")
find_package(OpenCV REQUIRED)

# OpenMP (for preprocessing)
find_package(OpenMP REQUIRED)

# VTK
set(VTK_DIR "D:/Image_registration/dev/VTK-install/lib/cmake/vtk-9.6")
find_package(VTK REQUIRED)

# ITK
set(ITK_DIR "D:/Image_registration/dev/ITK-install/lib/cmake/ITK-6.0")
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

# Executable
add_executable(QtTest
    src/main.cpp
    src/MainWindow.cpp
    src/MainWindow.h
    src/VTKViewer.cpp
    src/VTKViewer.h
    src/Preprocess.cpp
    src/Preprocess.h
    src/Register.cpp
    src/Register.h
    src/BSplineRegister.cpp
    src/BSplineRegister.h
    src/RegionConfig.cpp
    src/RegionConfig.h
)

# Link
target_link_libraries(QtTest PRIVATE
    Qt6::Widgets
    ${VTK_LIBRARIES}
    ${ITK_LIBRARIES}
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
)

target_include_directories(QtTest PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

vtk_module_autoinit(
    TARGETS QtTest
    MODULES ${VTK_LIBRARIES}
)

# ---- Qt Deployment (AUTO COPY DLLs) ----

if (WIN32)
    find_program(WINDEPLOYQT_EXECUTABLE windeployqt HINTS "${Qt6_DIR}/../../../bin")

    add_custom_command(TARGET QtTest POST_BUILD
        COMMAND "${WINDEPLOYQT_EXECUTABLE}"
                --no-compiler-runtime
                --no-translations
                $<TARGET_FILE:QtTest>
        COMMENT "Running windeployqt to copy Qt dependencies..."
    )
endif()