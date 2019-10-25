#include <iostream>
#include <fstream>
#include <sstream>
#include "slamBase.h"

using namespace cv;
using namespace std;

bool vizualize = true;  //set false if we don't want to show the point cloud
int min_inliers = 5;
int min_good_match=10;
double max_norm=0.3;
int good_match_threshold=10;

Frame readFrame(int index);
double normofTransform(Mat rvec, Mat tvec );

int main(int argc, char** argv)
{
    int startIndex = 1;
    int endIndex = 2; 

    cout << "Initialization...\n";
    //Camera settings
    CamIParam camera;
    camera.cx=325.5;
    camera.cy=253.5;
    camera.fx=518.0;
    camera.fy=519.0;
    camera.scale=1000.0; 
    /* Freiburgh dataset 1*/
    /*
    camera.cx=318.6;
    camera.cy=255.3;
    camera.fx=517.3;
    camera.fy=516.5;
    camera.scale=5000.0; */
   
    int currIndex = startIndex; //we start from here
    Frame lastFrame = readFrame(currIndex);
    computeKeyPointsAndDesp(lastFrame); //detect keypoints and calculate descriptors
    
    PointCloud::Ptr pcloud = image2PointCloud(lastFrame.rgb, lastFrame.depth, camera);
    pcl::visualization::CloudViewer viewer("Viewer");

    for (currIndex=startIndex+1; currIndex <= endIndex; currIndex++)
    {
        cout << "Frame "<<currIndex<<endl;
        Frame currFrame = readFrame(currIndex);
        computeKeyPointsAndDesp(currFrame);
        PnP_Result result = estimateMotion(lastFrame, currFrame, camera);
        if(result.inliers < min_inliers)  //If we don't have enough inliers, drop the frame
            continue;

        //Calculate whether the range of motion is too large
        double norm = normofTransform(result.rvec, result.tvec);
        cout<<"Norm = " << norm<<endl;

        if(norm>=max_norm)
            continue;

        Eigen::Isometry3d T = cvMat2Eigen(result.rvec, result.tvec); //Compose Homogenous Transform matrix;
        cout <<"T = "<<T.matrix()<<endl;
        pcloud =  joinPointCloud(pcloud, currFrame, T, camera);
        if(vizualize == true)
            viewer.showCloud(pcloud);
        
        lastFrame = currFrame;
    }
    waitKey(10000);
    pcl::io::savePCDFile( "resultPCL.pcd", *pcloud );

    return 0;
}

Frame readFrame(int index)
{
    Frame frame;
    string rgbd_filename = "r"+to_string(index)+".png";
    string depth_filename = to_string(index)+".png";

    frame.rgb = imread(rgbd_filename);
    frame.depth = imread(depth_filename,-1);
    return frame;
}

double normofTransform(Mat rvec, Mat tvec )
{
    return fabs(min(norm(rvec), 2*M_PI-norm(rvec)))+ fabs(norm(tvec)); //fabs - absolute value  // norm - gjatesia e.g. norm(3,4) = (9+16)
}

