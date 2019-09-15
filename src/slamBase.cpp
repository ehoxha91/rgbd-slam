#include "slamBase.h"

//Convert RGB image/map to point cloud
PointCloud::Ptr image2PointCloud( Mat& rgb, Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera )
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
cv::Point3f point2dTo3d(Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera)
{
	Point3f p; //3D point where we will save the new point;
	p.z = double (point.z)/camera.scale;
	p.x = (point.x - camera.cx) * p.z / camera.fx;
	p.y = (point.y - camera.cy) * p.z / camera.fy;
	return p;
}


