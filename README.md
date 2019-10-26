## RGBD SLAM

A simple slam implementation for rgbd camera.
I used gaoxiang12 tutorial to build this simple SLAM.

### Build

To build do these steps:

- `mkdir build`
- `cd build`
- `cmake ..`
- `make`

If it fails with errors related to OpenCV, check the CMakeLists.txt and remove the line:

- `SET(OpenCV_DIR /installation/OpenCV-/lib/cmake/opencv4)`

because I use OpenCV 4.1...

Compiled file is saved on:
- `/rgbd-slam/bin/`
