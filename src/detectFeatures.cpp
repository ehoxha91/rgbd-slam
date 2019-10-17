#include <iostream>
#include "slamBase.h"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>


using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    cv::Mat rgb1 = imread("rgb1.png");
    cv::Mat rgb2 = imread("rgb2.png");
    cv::Mat depth1 = imread("depth1.png",-1);
    cv::Mat depth2 = imread("depth2.png",-1);
    cout <<"VERSION " <<CV_VERSION<<endl;

    // Feature extractor and description
    Ptr<FeatureDetector> detector;
    Ptr<DescriptorExtractor> descriptor;
    
    // We use ORB extractor and descriptor
    //detector = xfeatures2d::SIFT::create();
    //descriptor = xfeatures2d::SIFT::create();
    detector = ORB::create();
    descriptor = ORB::create();

    vector<KeyPoint> kp1, kp2;

    //Feature detection using ORB
    detector->detect(rgb1, kp1);
    detector->detect(rgb2, kp2);

    cout <<"Number of KeyPoints of two images: "<< kp1.size()<<", "<<kp2.size()<<endl;

    Mat imgShow;
    drawKeypoints(rgb1, kp1, imgShow,Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("keypoints1", imgShow);
    imwrite("keypoits.png", imgShow); // Save the picture with keypoints
    waitKey(0);

    //Feature descriptor
    Mat desp1, desp2;
    descriptor->compute(rgb1, kp1, desp1);
    descriptor->compute(rgb2, kp2, desp2);

    //Feature matching
    vector<DMatch> matches;
    BFMatcher matcher;
    matcher.match(desp1, desp2, matches);
    cout << "Found " << matches.size()<<" matches"<<endl;

    //Display matching features
    Mat imgMatches;
    drawMatches(rgb1, kp1, rgb2, kp2, matches, imgMatches);
    imshow("Matches",imgMatches);
    imwrite("matches.png", imgMatches);
    waitKey(0);

    //Filter outliers and keep only matches that fullfill the criteria
    vector<DMatch> goodMatches;
    double minDist = 9999;
    for (size_t i = 0; i < matches.size(); i++)
    {
        if(matches[i].distance <minDist)
        {
            minDist = matches[i].distance;
        }
    }
    cout <<"Minimum distane: " <<minDist<<endl;
    for (size_t i = 0; i < matches.size(); i++)
    {
        if(matches[i].distance <10*minDist)
        {
            goodMatches.push_back(matches[i]);
        }
    }
    
    //Display good matches
    cout<<"Good matches: " << goodMatches.size() <<endl;
    drawMatches(rgb1, kp1, rgb2, kp2, goodMatches, imgMatches);
    imshow("Good matches", imgMatches);
    imwrite("good_matches.png", imgMatches);
    waitKey(0);
    
    //Calculate camera motion
    //Key function: solvePnPRansac()
    //Prepare necessary parameters for calling this function

    //3D point from 1st frame
    vector<Point3f> pts_obj;

    //Image point from 2nd frame
    vector<Point2f> pts_img;

    //Camera intrisic parameters:
    CAMERA_INTRINSIC_PARAMETERS C;  //(struct)
    C.cx = 325.5;
    C.cy = 253.5;
    C.fx = 518.0;
    C.fy = 519.0;
    C.scale = 1000.0;

      for (size_t i=0; i<goodMatches.size(); i++)
    {
        //qurey is the first, train is the second
        cv::Point2f p = kp1[goodMatches[i].queryIdx].pt;

        //Get depth d from the depth image; x-column, y-row
        ushort d = depth1.ptr<ushort>( int(p.y) )[ int(p.x) ];
        if (d == 0)
            continue;
        pts_img.push_back( cv::Point2f( kp2[goodMatches[i].trainIdx].pt ) );

        //Get depth d from the depth image; x-column, y-row
        cv::Point3f pt ( p.x, p.y, d );
        cv::Point3f pd = point2dTo3d( pt, C );
        pts_obj.push_back( pd );
    }

    double camera_matrix_data[3][3] = {
        {C.fx, 0, C.cx},
        {0, C.fy, C.cy},
        {0, 0, 1}
    };

    // Camera Matrix
    cv::Mat cameraMatrix( 3, 3, CV_64F, camera_matrix_data );
    cv::Mat rvec, tvec, inliers;
    
    // Solve PnP
    cv::solvePnPRansac( pts_obj, pts_img, cameraMatrix, Mat(), rvec, tvec, false , 1000 , 8.0 , 0.99 , inliers );

    cout<<"inliers: "<<inliers.rows<<endl;
    cout<<"R="<<rvec<<endl;
    cout<<"t="<<tvec<<endl;

    //Draw inliers
    vector<DMatch> matchesShow;
    for (size_t i = 0; i < inliers.rows; i++)
    {
        matchesShow.push_back(goodMatches[inliers.ptr<int>(i)[0]]);
    }

    drawMatches(rgb1, kp1, rgb2, kp2, matchesShow, imgMatches);
    imshow("Inlier matches", imgMatches);
    imwrite("inliers.png", imgMatches);
    waitKey(0);
    

    return 0;
}