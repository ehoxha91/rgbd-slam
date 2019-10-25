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


CHECK_RESULT checkKeyFrame(Frame &frame1, Frame &frame2, g2o::SparseOptimizer &optimizer, bool is_loops=false);
Frame readFrame(int index);
double normofTransform(Mat rvec, Mat tvec );

//Loop Closure:
//1. Check for loops in neighbourhood, close range search!
void checkNearbyLoops( vector<Frame>& frames, Frame& currentFrame, g2o::SparseOptimizer& optimizer);
//2. Check random keyframes for loop closure!
void checkRandomLoops( vector<Frame>& frames, Frame& currentFrame, g2o::SparseOptimizer& optimizer);

int main(int argc, char** argv)
{
    int startIndex = 1;
    int endIndex = 6; 
    camera.cx=325.5;
    camera.cy=253.5;
    camera.fx=518.0;
    camera.fy=519.0;
    camera.scale=1000.0;

    vector<Frame> keyframes; //We save keyframes on this vector

    cout << GREEN"Initialization...\n";

    int currIndex = startIndex; //we start from here
    Frame currFrame = readFrame(currIndex);
    computeKeyPointsAndDesp(currFrame); //detect keypoints and calculate descriptors

    //PointCloud::Ptr pcloud = image2PointCloud(currFrame.rgb, currFrame.depth, camera);
    //pcl::visualization::CloudViewer viewer("Viewer");

    //Initialize g2o
    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver (new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>());
    std::unique_ptr<g2o::BlockSolver_6_3> blockSolver (new g2o::BlockSolver_6_3(std::move(linearSolver)));
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));

    g2o::SparseOptimizer globalOptimizer;
    globalOptimizer.setAlgorithm(solver);

    //Disable outputing debug information
    //globalOptimizer.setVerbose(false);

    //Add the first vertex to the optimizer:
    g2o::VertexSE3* v = new g2o::VertexSE3();
    v->setId(currIndex);
    v->setEstimate(Eigen::Isometry3d::Identity()); // Estimate as the unit matrix, first step
    v->setFixed(true);  //First vertex/frame/pose is the world coordinate, I matrix; Is fixed, no optimization on this one;
    globalOptimizer.addVertex(v);

    keyframes.push_back(currFrame);

    for (currIndex = startIndex+1; currIndex <= endIndex; currIndex++)
    {
        cout << "Frame "<<currIndex<<endl;
        currFrame = readFrame(currIndex);
        computeKeyPointsAndDesp(currFrame);
        CHECK_RESULT result = checkKeyFrame( keyframes.back(), currFrame, globalOptimizer, false);

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
                    checkNearbyLoops(keyframes,currFrame, globalOptimizer);
                    checkRandomLoops(keyframes,currFrame, globalOptimizer);
                }
                keyframes.push_back(currFrame);

                break;
            default:
                break;
        }
    }

    //We start optimization of all nodes:
    cout <<RESET"\n Optimizing the graph with: "<<globalOptimizer.vertices().size()<<" nodes\n";
    globalOptimizer.save("nonoptimizedGraph.g2o");
    globalOptimizer.initializeOptimization();
    globalOptimizer.optimize(100); //Try 100 iteration for optimization
    globalOptimizer.save("optimizedGraph.g2o");
    cout <<GREEN"Optimization done"<<endl;
    globalOptimizer.clear();

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

    for (size_t i = 0; i < keyframes.size(); i++)
    {
        //Take a frame from g2o
        cout<<YELLOW"Saving 1"<<endl;
        g2o::VertexSE3 *vertex = dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex(keyframes[i].frameID));
        //Optimized pose for the frame:
        cout<<YELLOW"Saving 2"<<endl;

        Eigen::Isometry3d pose = vertex->estimate();
        cout<<"Pose="<<endl<<pose.matrix()<<endl;

        cout<<YELLOW"Saving 3"<<endl;
        PointCloud::Ptr newCloud = image2PointCloud( keyframes[i].rgb, keyframes[i].depth, camera ); //Generate point cloud
        //Filter the pointcloud
        cout<<YELLOW"Saving 4"<<endl;
        voxel.setInputCloud(newCloud);
        cout<<YELLOW"Saving 5"<<endl;
        voxel.filter(*tmp);
        cout<<YELLOW"Saving 6"<<endl;
        pass.setInputCloud(tmp);
        cout<<YELLOW"Saving 7"<<endl;
        pass.filter(*newCloud);
        cout<<YELLOW"Saving 8"<<endl;
        pcl::transformPointCloud(*newCloud, *tmp, pose.matrix());
        *output += *tmp;
        cout<<YELLOW"Saving 9"<<endl;
        tmp->clear();
        cout<<YELLOW"Saving 10"<<endl;
        newCloud->clear();
        cout<<YELLOW"Saving 11"<<endl;
    }
    voxel.setInputCloud(output);
    voxel.filter(*tmp);
    

    waitKey(10000);
    pcl::io::savePCDFile("resultPCL.pcd", *output );
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

CHECK_RESULT checkKeyFrame(Frame &frame1, Frame &frame2, g2o::SparseOptimizer &optimizer, bool is_loops)
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
        g2o::VertexSE3 *vertex = new g2o::VertexSE3();
        vertex->setId(frame2.frameID);
        vertex->setEstimate(Eigen::Isometry3d::Identity());
        optimizer.addVertex(vertex);
    }
    //Create the link between two edges:
    g2o::EdgeSE3 *edge = new g2o::EdgeSE3();
    //Connect two vertices with this edge:
    edge->setVertex(0, optimizer.vertex(frame1.frameID));
    edge->setVertex(1, optimizer.vertex(frame2.frameID));
    edge->setRobustKernel(new g2o::RobustKernelHuber());

    // Information matrix
    // The information matrix is the inverse of the covariance matrix, 
    // indicating our pre-estimation of the accurracy of the edges
    // Pose is 6D vector, we need the information and covariance matrix to be 6x6. 
    // We assume that position and angle estimates are independent.
    // This means that Covariance matrix is a diagonal matrix diagonal(0.01, 0.01, 0.01, 0.01, 0.01, 0.01)
    // Information matrix is the inverse of this matrix and is a diagonal(100,100,100,100,100,100);
    // The larger the value of the element in the diagonal, the more accurate is the estimation
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();

    for (size_t i = 0; i < 6; i++)
    {
        information(i,i) = 100;
    }
    edge->setInformation(information);  //We link the information matrix with the edge
    Eigen::Isometry3d T = cvMat2Eigen(result.rvec, result.tvec); //get Homogenous transformation matrxi from Vectors
    edge->setMeasurement(T.inverse());  //As estimate we consider the T matrix from PnPRansac, Inverse(from frame 1 to 2)
    optimizer.addEdge(edge);            //In the end add this edge to the optimizer

    return KEYFRAME;    //We have a keyframe 
}

void checkNearbyLoops( vector<Frame>& frames, Frame& currentFrame, g2o::SparseOptimizer& optimizer)
{
    if(frames.size() <= nearby_loops)
    {
        //We don't even have as much as we need to check, so let's check everything
        for (size_t i = 0; i < frames.size(); i++)
        {
            checkKeyFrame(frames[i], currentFrame, optimizer, true);
        }
    }
    else
    {
        //Check the frames in neighbourhood, number of frames to be checked "nearby_loops"
        for (size_t i = frames.size() - nearby_loops; i < frames.size(); i++)
        {
            //If number of frames is 200, then last 5 are our "nearby frames", 200, 199, ..., 196
            checkKeyFrame(frames[i], currentFrame, optimizer, true);
        }
        
    }
    

}

void checkRandomLoops( vector<Frame>& frames, Frame& currentFrame, g2o::SparseOptimizer& optimizer)
{
      if ( frames.size() <= random_loops )
    {
        // no enough keyframes, check everyone
        for (size_t i=0; i<frames.size(); i++)
        {
            checkKeyFrame( frames[i], currentFrame, optimizer, true );
        }
    }
    else
    {
        // randomly check loops
        for (int i=0; i<random_loops; i++)
        {
            int index = rand()%frames.size();
            checkKeyFrame( frames[i], currentFrame, optimizer, true );
        }
    }
}
