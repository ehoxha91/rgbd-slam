#include <iostream>
#include <fstream>
#include <vector>
#include <map>
//

//Eigen
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

// Point Cloud Library
#include <pcl-1.8/pcl/io/pcd_io.h>
#include <pcl-1.8/pcl/filters/passthrough.h>
#include <pcl-1.8/pcl/point_types.h>
#include <pcl-1.8/pcl/common/transforms.h>
#include <pcl-1.8/pcl/visualization/cloud_viewer.h>
#include <pcl-1.8/pcl/filters/voxel_grid.h>
#include <pcl-1.8/pcl/filters/passthrough.h>

using namespace std;

enum CHECK_RESULT {NOT_MATCHED=0, TOO_FAR_AWAY, TOO_CLOSE, KEYFRAME}; 

//Definition of PointCloud and PointT types
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
	int frameID;
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

//After we estimate Translation and Rotation vector convert that information to T homogenous matrix
Eigen::Isometry3d cvMat2Eigen(cv::Mat& rvec, cv::Mat& tvec);

//Join Point clouds together
PointCloud::Ptr joinPointCloud(PointCloud::Ptr original, Frame& newFrame, Eigen::Isometry3d T, CamIParam& camera);

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


#define RESET "\033[0m"
#define BLACK "\033[30m" /* Black */
#define RED "\033[31m" /* Red */
#define GREEN "\033[32m" /* Green */
#define YELLOW "\033[33m" /* Yellow */
#define BLUE "\033[34m" /* Blue */
#define MAGENTA "\033[35m" /* Magenta */
#define CYAN "\033[36m" /* Cyan */
#define WHITE "\033[37m" /* White */
#define BOLDBLACK "\033[1m\033[30m" /* Bold Black */
#define BOLDRED "\033[1m\033[31m" /* Bold Red */
#define BOLDGREEN "\033[1m\033[32m" /* Bold Green */
#define BOLDYELLOW "\033[1m\033[33m" /* Bold Yellow */
#define BOLDBLUE "\033[1m\033[34m" /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m" /* Bold Magenta */
#define BOLDCYAN "\033[1m\033[36m" /* Bold Cyan */
#define BOLDWHITE "\033[1m\033[37m" /* Bold White */