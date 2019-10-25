#include<iostream>
#include "slamBase.h"

#include <opencv2/core/eigen.hpp>

#include <pcl-1.8/pcl/common/transforms.h>
#include <pcl-1.8/pcl/visualization/cloud_viewer.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    //ParameterReader pd;

    Frame frame1, frame2; //Two frames

    frame1.rgb = imread("rgb1.png");  
    frame1.depth = imread("depth1.png",-1);
    
    frame2.rgb = imread("rgb2.png");
    frame2.depth = imread("depth2.png",-1);

    cout << "Extracting features..."<<endl;    

    // //Compute keypoints and descriptors
    computeKeyPointsAndDesp(frame1);
    computeKeyPointsAndDesp(frame2);

    //Read camera parameters
    CamIParam camera;
    camera.cx=325.5;
    camera.cy=253.5;
    camera.fx=518.0;
    camera.fy=519.0;
    camera.scale=1000.0;

    //Solve PnP optimization:
    cout<<"\nSolving PnP...\n"<<endl;
    PnP_Result result = estimateMotion( frame1, frame2, camera );

    //Print the result vectors:
    cout<<result.rvec<<endl<<result.tvec<<endl;

    //Conver the rotation vector to rotation matrix:

    Mat R;
    Rodrigues(result.rvec, R); //Convert using Rodrigues formula;
    Eigen::Matrix3d r;
    cv2eigen(R, r);

    //Convert the translation vector and rotation matrix into a transformation matrix

    Eigen::Isometry3d T = Eigen::Isometry3d::Identity(); //Homogenous transformation matrix
    
    Eigen::AngleAxisd angle(r);
    cout << "Homogenous Transformation Matrix \n ################\n";

    Eigen::Translation<double,3> trans(result.tvec.at<double>(0,0), result.tvec.at<double>(0,1), result.tvec.at<double>(0,2)); 
    T = angle;
    T(0,3) = result.tvec.at<double>(0,0);
    T(1,3) = result.tvec.at<double>(0,1);
    T(2,3) = result.tvec.at<double>(0,2);
    cout<<"T = "<<endl;
    for (size_t i = 0; i < 4; i++)
    {
        cout << "[";
        for (size_t j = 0; j < 4; j++)
        {
            /* code */
            cout <<" " << T(i,j);
        }
        cout << " ]\n";
        
    }
    

    //Converting images to point cloud:
    cout<<"Converting images to cloud..."<<endl;
    PointCloud::Ptr cloud1 = image2PointCloud( frame1.rgb, frame1.depth, camera );
    PointCloud::Ptr cloud2 = image2PointCloud( frame2.rgb, frame2.depth, camera );

    //Combine clouds:
    cout <<"Combining clouds..."<<endl;
    PointCloud::Ptr output(new PointCloud());
    pcl::transformPointCloud(*cloud1, *output, T.matrix() );
    *output += *cloud2;
    
    pcl::io::savePCDFile("combinedPointCloud.pcd", *output);
    cout<<"Final result saved."<<endl;

    pcl::visualization::CloudViewer viewer( "viewer" );
    viewer.showCloud( output );
    while( !viewer.wasStopped() )
    {
        
    }

    return 0;
}