#include <fstream>
#include <vector>

// OpenCV library
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

// Point Cloud Library
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

// Definition of PointCloud and PointT types
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

using namespace std;
using namespace cv;

struct CAMERA_INTRINSIC_PARAMETERS
{
	double cx, cy, fx, fy, scale;
};

//Convert RGB image/map to point cloud
PointCloud::Ptr image2PointCloud(Mat& rgb, Mat& depth, CAMERA_INTRINSIC_PARAMETERS & Camera);

//Get 3D coordinates, using depth image and projection.
Point3f point2dTo3d(Point3f& point, CAMERA_INTRINSIC_PARAMETERS& Camera);
