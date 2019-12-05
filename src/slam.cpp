#include <iostream>
#include <fstream>
#include <sstream>
#include "slamBase.h"

using namespace cv;
using namespace std;


#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/slam3d/vertex_se3.h>

#include "OptimizerE.h"

#pragma region Parameters
//Camera settings for freiburgh data
CamIParam camera;
bool vizualize = true;  //set false if we don't want to show the point cloud
int min_inliers = 5;
int min_good_match=10;
double max_norm=0.3;
int good_match_threshold=10;
double keyframe_threshold = 0.1;
double max_norm_lp=2.0;
bool check_loop_closure = true;
int nearby_loops=5;   //How many loops to search in the neighbourhood
int random_loops=5;   //How many loops to search in randomly

#pragma endregion


CHECK_RESULT checkKeyFrame(Frame &frame1, Frame &frame2, bool is_loops, PointCloud::Ptr &output);
Frame readFrame(int index);
double normofTransform(Mat rvec, Mat tvec );
void nothing();
//Loop Closure:
//1. Check for loops in neighbourhood, close range search!
void checkNearbyLoops( vector<Frame>& frames, Frame& currentFrame, PointCloud::Ptr &output);
//2. Check random keyframes for loop closure!
void checkRandomLoops( vector<Frame>& frames, Frame& currentFrame, PointCloud::Ptr &output);

OptimizerE optimizeE;       //Optimizer class 

int main(int argc, char** argv)
{
    int startIndex = 0;
    int endIndex = 13;
   /*
    camera.cx=325.5;
    camera.cy=253.5;
    camera.fx=518.0;
    camera.fy=519.0;
    camera.scale=1000.0; 
    camera.cx=318.6;
    camera.cy=255.3;
    camera.fx=517.3;
    camera.fy=516.5;
    camera.scale=5000.0;
    */

    camera.fx = 517.3;
    camera.fy = 516.5;
    camera.cx = 318.6;
    camera.cy = 255.3;
    camera.scale= 5000.0;
    vector<Frame> keyframes; //We save keyframes on this vector

    cout << GREEN"Initialization...\n";

    int currIndex = startIndex; //we start from here
    Frame currFrame = readFrame(currIndex);
    computeKeyPointsAndDesp(currFrame); //detect keypoints and calculate descriptors

    PointCloud::Ptr pcloud = image2PointCloud(currFrame.rgb, currFrame.depth, camera);
    //pcl::visualization::CloudViewer viewer("Viewer");

    //Initialize g2o optimization
    optimizeE.globalOptimizer.setVerbose(false);
    
    //Add the first vertex to the optimizer:
    optimizeE.AddVertex(startIndex);

    keyframes.push_back(currFrame);

    for (currIndex = startIndex+1; currIndex <= endIndex; currIndex++)
    {
        cout << "Frame "<<currIndex<<endl;
        currFrame = readFrame(currIndex);
        computeKeyPointsAndDesp(currFrame);
        CHECK_RESULT result = checkKeyFrame( keyframes.back(), currFrame, false, pcloud);

        switch (result)
        {
            case NOT_MATCHED:
                cout<<RED"Not enough inliers."<<endl;
                break;

            case TOO_FAR_AWAY:
                cout<<RED"Too far away, may be an error."<<endl;
                break;

            case TOO_CLOSE:
                cout<<RESET"Too close, not a keyframe"<<endl;
                break;

            case KEYFRAME:
                cout<<GREEN"This is a new keyframe"<<endl;

                if(check_loop_closure)
                {   //Now time to check for loopclosing
                    checkNearbyLoops(keyframes,currFrame, pcloud);
                    checkRandomLoops(keyframes,currFrame, pcloud);
                }
                keyframes.push_back(currFrame);

                break;
            default:
                break;
        }
    }
    //We start optimization of all nodes:
    optimizeE.Optimize(100);

    pcl::io::savePCDFile("NonOptimizedPointCloud.pcd", *pcloud);
    cout<<"Final result saved."<<endl;

    pcl::visualization::CloudViewer viewer( "viewer" );
    viewer.showCloud( pcloud );
    while( !viewer.wasStopped() )
    {

    }

    //Using optimized graph to generate the point cloud!
    cout<<YELLOW"Saving the point cloud map..."<<endl;
    PointCloud::Ptr output (new PointCloud());
    PointCloud::Ptr tmp (new PointCloud());

    //Grid filter, to adjust the resolution, faster!
    pcl::VoxelGrid<PointT> voxel;
    pcl::PassThrough<PointT> pass;
    pass.setFilterFieldName("z");  //Set a limit for the depth
    pass.setFilterLimits(0.0,0.4); //filter info that are deeper than 4m

    voxel.setLeafSize(0.01f,0.01f,0.01f);
    int breaktest = 1;
    for (size_t i = 0; i < keyframes.size(); i++)
    {
        // cout << "Inside the for loop\n";
        // nothing();
        // g2o::VertexSE3 *v = dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex(keyframes[i].frameID));
        // breaktest =2;
        // cout <<"T_opt = [" <<v->estimate().matrix() <<"]"<<endl;

        //Take a frame from g2o
        //g2o::VertexSE3 *vertex = dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex(keyframes[i].frameID));
        //cout<<vertex->estimate().matrix()<<endl;
        //Eigen::Isometry3d pose = vertex->estimate();
        //PointCloud::Ptr newCloud = image2PointCloud( keyframes[i].rgb, keyframes[i].depth, camera ); //Generate point cloud
        //voxel.setInputCloud(newCloud);
        //voxel.filter(*tmp);
        //pass.setInputCloud(tmp);
        //pass.filter(*newCloud);
        //pcl::transformPointCloud(*newCloud, *tmp, pose.matrix());
        //*output += *tmp;
        //tmp->clear();
        //newCloud->clear();
    }
    //voxel.setInputCloud(output);
    //voxel.filter(*tmp);
 

    waitKey(10000);
    //pcl::io::savePCDFile("resultPCL.pcd", *output );
    return 0;
}

Frame readFrame(int index)
{
    Frame frame;
    string rgbd_filename =  "r" + to_string(index)+".png";
    string depth_filename = "d" + to_string(index)+".png";

    frame.rgb = imread(rgbd_filename);
    frame.depth = imread(depth_filename,-1);
    frame.frameID = index;
    return frame;
}

double normofTransform(Mat rvec, Mat tvec )
{
    return fabs(min(norm(rvec), 2*M_PI-norm(rvec)))+ fabs(norm(tvec)); //fabs - absolute value  // norm - gjatesia e.g. norm(3,4) = (9+16)
}

CHECK_RESULT checkKeyFrame(Frame &frame1, Frame &frame2, bool is_loops, PointCloud::Ptr &output)
{
    PnP_Result result = estimateMotion(frame1, frame2, camera);

    if(result.inliers < min_inliers)  //If we don't have enough inliers, drop the frame
        return NOT_MATCHED;

    //Calculate range of motion!
    double norm = normofTransform(result.rvec, result.tvec);
    cout<<"norm = "<<norm<<endl;

    if(is_loops)
    {
        if ( norm >= max_norm )
            return TOO_FAR_AWAY; //If range of motion is too large, we may have error
    }
    else
    {
        if(norm>=max_norm_lp)   //If no loop closure!
            return TOO_FAR_AWAY;
    }

    if(norm <= keyframe_threshold)
        return TOO_CLOSE; //Too close with the previous frame!

    //If no loops add this vertex to the graph for optimizing
    if(is_loops == false)
    {
        optimizeE.AddVertex(frame2.frameID);
    }
    //Create the link between two edges:
    Eigen::Matrix<double, 6, 6> infoMatrix = Eigen::Matrix<double, 6, 6>::Identity();
    for (size_t i = 0; i < 6; i++)
    {
        infoMatrix(i,i) = 100;
    }
    Eigen::Isometry3d T = cvMat2Eigen(result.rvec, result.tvec); //get Homogenous transformation matrxi from Vectors
    optimizeE.AddEdge(frame1.frameID, frame2.frameID, infoMatrix, T);

    cout<<"Frame ID"<< frame2.frameID<<"\nT " <<T.matrix()<<endl;
    
    //To generate non-optimized point cloud map!
    PointCloud::Ptr cloud2 = image2PointCloud( frame2.rgb, frame2.depth, camera );
    cout <<"Combining clouds..."<<endl;
    pcl::transformPointCloud(*cloud2, *output, T.matrix() );

    return KEYFRAME;    //We have a keyframe
}

void checkNearbyLoops( vector<Frame>& frames, Frame& currentFrame, PointCloud::Ptr &output)
{
    if(frames.size() <= nearby_loops)
    {
        //We don't even have as much as we need to check, so let's check everything
        for (size_t i = 0; i < frames.size(); i++)
        {
            checkKeyFrame(frames[i], currentFrame, true, output);
        }
    }
    else
    {
        //Check the frames in neighbourhood, number of frames to be checked "nearby_loops"
        for (size_t i = frames.size() - nearby_loops; i < frames.size(); i++)
        {
            //If number of frames is 200, then last 5 are our "nearby frames", 200, 199, ..., 196
            checkKeyFrame(frames[i], currentFrame, true, output);
        }

    }


}

void checkRandomLoops( vector<Frame>& frames, Frame& currentFrame,  PointCloud::Ptr &output)
{
      if ( frames.size() <= random_loops )
    {
        // no enough keyframes, check everyone
        for (size_t i=0; i<frames.size(); i++)
        {
            checkKeyFrame( frames[i], currentFrame, true, output );
        }
    }
    else
    {
        // randomly check loops
        for (int i=0; i<random_loops; i++)
        {
            int index = rand()%frames.size();
            checkKeyFrame( frames[i], currentFrame, true, output );
        }
    }
}
