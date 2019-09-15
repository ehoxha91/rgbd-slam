
#include <iostream>
#include <string>
using namespace std;


// OpenCV library
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
// PCL library
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

// Define the PC type
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud; 

// camera internal reference
const double camera_factor = 1000; //1000 - represent 1m
const double camera_cx = 325.5;
const double camera_cy = 253.5;
const double camera_fx = 518.0;
const double camera_fy = 519.0;


int main( int argc, char** argv )
{
    // Read ./data/rgb.png and ./data/depth.png
    // then convert it to point cloud

    //
    Mat rgb, depth;
    rgb = imread( "rgb.png" );
    // rgb image is a color image of 8UC3 - (8-bit unsigend char, 3 channels) 
    // depth is a single channel of 16UC1
    // note: flags are set to -1, indicating that the original data 
    // is read without any modification
    depth = imread( "depth.png", -1 );

    // point cloud variables
    // smart pointers, creat an empty point cloud
    // This type of pointer is automatically released when
    // is used up.
    PointCloud::Ptr cloud ( new PointCloud );

    // Traversing the depth map
    for (int m = 0; m < depth.rows; m++)
        for (int n=0; n < depth.cols; n++)
        {
            // Get value at (m,n) in the depth map
            ushort d = depth.ptr<ushort>(m)[n];

            // There may be no value, if so, skip the point
            if (d == 0)
                continue;
            
	    // Otherwise, add a point to the point cloud:
            PointT p;

            // Calculate the 3D coordinates of this point.
            p.z = double(d) / camera_factor;
            p.x = (n - camera_cx) * p.z / camera_fx;
            p.y = (m - camera_cy) * p.z / camera_fy;
            
            // Get color of the point cloud from rgb image
            p.b = rgb.ptr<uchar>(m)[n*3];
            p.g = rgb.ptr<uchar>(m)[n*3+1];
            p.r = rgb.ptr<uchar>(m)[n*3+2];

            // Add point "p" to the point cloud
            cloud->points.push_back( p );
	    cout <<"depth = " + d<<endl;
        }
    // Set and save point cloud
    cloud->height = 1;
    cloud->width = cloud->points.size();
    cout<<"point cloud size = "<<cloud->points.size()<<endl;
    cloud->is_dense = false;
    pcl::io::savePCDFile( "./pointcloud.pcd", *cloud );
    
    // Clear the data and exit
    cloud->points.clear();
    cout<<"Point cloud saved."<<endl;
    return 0;
}
