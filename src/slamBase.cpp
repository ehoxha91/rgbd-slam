#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "slamBase.h"

using namespace std;
using namespace cv;

//Convert RGB image/map to point cloud
PointCloud::Ptr image2PointCloud( Mat& rgb, Mat& depth, CamIParam& camera )
{
	PointCloud::Ptr cloud (new PointCloud);
	for (int m = 0; m < depth.rows; m++)
	{
		for (int n = 0; n < depth.cols; n++)
		{
			//get the value of depth from depth image:
			ushort d = depth.ptr<ushort> (m)[n];
			//IF d == 0, we don't have enough info for point cloud, skip it.
			if(d == 0) continue;

			//ELSE:
			PointT p;

			//Use depth to generate 3D coordinates
			p.z = double (d)/camera.scale;
			p.x = (n - camera.cx) * p.z / camera.fx;
			p.y = (m - camera.cy) * p.z / camera.fy;

			//Read the colors:
			p.b = rgb.ptr<uchar>(m)[n*3];
			p.g = rgb.ptr<uchar>(m)[n*3+1];
			p.r = rgb.ptr<uchar>(m)[n*3+2];

			//push the point to the list:
			cloud->points.push_back(p);
		}
	}

	cloud->height = 1;
	cloud->width = cloud->points.size();
	cloud->is_dense = false;

	return cloud;
}

//Get 3D coordinates, using depth image and projection.
cv::Point3f point2dTo3d(Point3f& point, CamIParam& camera)
{
	Point3f p; //3D point where we will save the new point;
	p.z = double (point.z)/camera.scale;
	p.x = (point.x - camera.cx) * p.z / camera.fx;
	p.y = (point.y - camera.cy) * p.z / camera.fy;
	return p;
}

//Extract KeyPoints and feature descriptors
void computeKeyPointsAndDesp(Frame& frame)
{
	cv::Ptr<FeatureDetector> _detector;
	cv::Ptr<DescriptorExtractor> _descriptor;
	
	_detector = ORB::create();
	_descriptor = ORB::create();

	//Detect features
	_detector->detect(frame.rgb, frame.kp);
	//Calculate descriptors
	_descriptor->compute(frame.rgb, frame.kp, frame.desp);
}

//Estimate motion between two frames
PnP_Result estimateMotion(Frame& frame1, Frame& frame2, CamIParam& camera)
{
	vector<DMatch> matches;
	BFMatcher matcher;
	matcher.match(frame1.desp, frame2.desp, matches);

	cout << "Total matches found: "<<matches.size()<<"."<<endl;
	vector<DMatch> goodMatches;
	double minDist =9999;
	double good_match_threshold = 10.0;//atof(pd.getData("good_match_threhold").c_str());
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
        if(matches[i].distance <good_match_threshold*minDist)
        {
            goodMatches.push_back(matches[i]);
        }
    }
	cout<<"Good matches: "<<goodMatches.size()<<"."<<endl;

	vector<Point3f> pts_obj; //3D points from 1st frame
	vector<Point2f> pts_img; //Image points from 2nd image
	
	for (size_t i = 0; i < goodMatches.size(); i++)
	{
		Point2f p = frame1.kp[goodMatches[i].queryIdx].pt;
		ushort d = frame1.depth.ptr<ushort>(int(p.y))[int(p.x)];
		if(d == 0) continue;

		pts_img.push_back(Point2f(frame2.kp[goodMatches[i].trainIdx].pt));
		
		//Get 3D coordinates of the point;
		Point3f pt ( p.x, p.y, d );
        Point3f pd = point2dTo3d( pt, camera );
		pts_obj.push_back( pd );
	}

	double camera_matrix_data[3][3] = {
				{camera.fx, 0, camera.cx},
				{0, camera.fy, camera.cy},
				{0, 0, 1}};

	//Solve PnP
	cout <<"Solving PnP"<<endl;

	Mat cameraMatrix(3,3,CV_64F, camera_matrix_data);
	Mat rvec, tvec, inliers;
	cv::solvePnPRansac( pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 8.0, 0.99, inliers);
	PnP_Result result;
	result.rvec = rvec; 
	result.tvec = tvec; 
	result.inliers = inliers.rows;

	return result;
	
}

