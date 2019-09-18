#include <fstream>
#include <vector>
#include <map>
using namespace std;

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

// Point Cloud Library
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

// Definition of PointCloud and PointT types
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;


typedef struct CAMERA_INTRINSIC_PARAMETERS
{
	double cx, cy, fx, fy, scale;
}CamIParam;

typedef struct FRAME
{
	cv::Mat rgb, depth; 	 //Color frame and depth frame
	cv::Mat desp;		     //Feature descriptor of the frame
	vector<cv::KeyPoint> kp; //KeyPoint
}Frame;

typedef struct RESULT_OF_PNP
{
	cv::Mat rvec, tvec; //Rotation and translation result vectors
	int inliers;	//Number of inliers
}PnP_Result;


//Extract KeyPoints and feature descriptors
void computeKeyPointsAndDesp(Frame& frame);

//Estimate motion between two frames
PnP_Result estimateMotion(Frame& frame1, Frame& frame2, CamIParam& camera);

//Convert RGB image/map to point cloud
PointCloud::Ptr image2PointCloud(cv::Mat& rgb, cv::Mat& depth, CamIParam& Camera);

//Get 3D coordinates, using depth image and projection.
cv::Point3f point2dTo3d(cv::Point3f& point, CamIParam& Camera);


//Class to read parameter file
class ParameterReader
{
	public:
		ParameterReader(string filename =" ./parameters.txt")
		{
			ifstream fin(filename.c_str());
			if(!fin)
			{
				std::cerr<<"Couldn't read parameter file!\n";
				return;
			}
			while (!fin.eof())
			{
				string str;
				getline(fin, str);
				if(str[0]=='#')
					continue;
				
				int pos = str.find("=");
				if(pos ==-1)
					continue;

				string key = str.substr(0, pos);
				string value = str.substr(pos+1, str.length());

				data[key] = value;
				if(!fin.good())
					break;
			}
		}
		
		string getData(string key)
		{
			map<string, string>::iterator iter = data.find(key);
			if(iter == data.end())
			{
				cerr<<"Parameter name "<<key<<" not found!\n";
				return string("NOT_FOUND");
			}
			return iter->second;
		}

		map<string, string> data;
};