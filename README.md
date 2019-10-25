## RGBD SLAM

A simple slam implementation for rgbd camera.
I used gaoxiang12 tutorial to build this simple SLAM.

# Build

To build do these steps:

- `mkdir build`
- `cd build`
- `cmake ..`
- `make`

If it fails, check the CMakeLists.txt and remove the line:

- `SET(OpenCV_DIR /installation/OpenCV-/lib/cmake/opencv4)`

Compiled file is saved on:
- `/rgbd-slam/bin/`
