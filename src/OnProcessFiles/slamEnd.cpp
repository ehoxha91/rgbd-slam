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
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

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
    int endIndex = 8; 

    cout << "Initialization...\n";
    //Camera settings
    CamIParam camera;
    camera.cx=325.5;
    camera.cy=253.5;
    camera.fx=518.0;
    camera.fy=519.0;
    camera.scale=1000.0; 

    int currIndex = startIndex; //we start from here
    Frame lastFrame = readFrame(currIndex);
    computeKeyPointsAndDesp(lastFrame); //detect keypoints and calculate descriptors

    PointCloud::Ptr pcloud = image2PointCloud(lastFrame.rgb, lastFrame.depth, camera);
    pcl::visualization::CloudViewer viewer("Viewer");

    //Initialize g2o
    //Chose an optimization method
    /*
    typedef g2o::BlockSolver_6_3 SlamBlockSolver;
    typedef g2o::LinearSolverEigen<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;

    //initialize the solver
    SlamLinearSolver* linearSolver = new SlamLinearSolver();
    linearSolver->setBlockOrdering(false);
    SlamBlockSolver* blockSolver = new SlamBlockSolver(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(blockSolver);
    */
    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver (new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>());
    std::unique_ptr<g2o::BlockSolver_6_3> blockSolver (new g2o::BlockSolver_6_3(std::move(linearSolver)));
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));

    g2o::SparseOptimizer globalOptimizer;
    globalOptimizer.setAlgorithm(solver);
    //Disable outputing debug information
    globalOptimizer.setVerbose(false);

    //Add the first vertex to the optimizer:
    g2o::VertexSE3* v = new g2o::VertexSE3();
    v->setId(currIndex);
    v->setEstimate(Eigen::Isometry3d::Identity()); // Estimate as the unit matrix, first step
    v->setFixed(true);  //First vertex/frame/pose is the world coordinate, I matrix; Is fixed, no optimization on this one;
    globalOptimizer.addVertex(v);

    int lastIndex = currIndex; //Id of the previous frame;

    for (currIndex = startIndex+1; currIndex <= endIndex; currIndex++)
    {
        cout << "Frame "<<currIndex<<endl;
        Frame currFrame = readFrame(currIndex);
        computeKeyPointsAndDesp(currFrame);
        PnP_Result result = estimateMotion(lastFrame, currFrame, camera);
        
        if(result.inliers < min_inliers)  //If we don't have enough inliers, drop the frame
            continue;
        double norm = normofTransform(result.rvec, result.tvec);
        cout<<"norm = "<<norm<<endl;
        if ( norm >= max_norm )
            continue;

        Eigen::Isometry3d T = cvMat2Eigen(result.rvec, result.tvec);
        cout <<"T = "<<T.matrix()<<endl;
        if(vizualize == true)
        {
            pcloud =  joinPointCloud(pcloud, currFrame, T, camera);
            viewer.showCloud(pcloud);
        }

        //Add link current and previous vertex
        //1. Add the new vertex to the graph
        //2. Connect new vertex with the previous vertex with an edge.
        
        //1.
        g2o::VertexSE3 *vertex = new g2o::VertexSE3();
        vertex->setId(currIndex);
        vertex->setEstimate(Eigen::Isometry3d::Identity());
        globalOptimizer.addVertex(vertex);
        //2.
        g2o::EdgeSE3 *edge = new g2o::EdgeSE3();
        edge->vertices()[0] = globalOptimizer.vertex(lastIndex); //First vertex of the edge
        edge->vertices()[1] = globalOptimizer.vertex(currIndex); //The other vertex that is connected with this edge.

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
        edge->setInformation(information); //We link the information matrix with the edge
        edge->setMeasurement(T);           //As estimate we consider the T matrix from PnPRansac

        //Add the edge to the graph now!
        globalOptimizer.addEdge(edge);

        lastFrame = currFrame;  //Make the current frame be last for next iteration
        lastIndex = currIndex;  //Also make current index be last for the next iteration
    }

    //We start optimization of all nodes:
    cout << "\n Optimizing the graph with: "<<globalOptimizer.vertices().size()<<" nodes\n";
    globalOptimizer.save("nonoptimizedGraph.g2o");
    globalOptimizer.initializeOptimization();
    globalOptimizer.optimize(100); //Try 100 iteration for optimization
    globalOptimizer.save("optimizedGraph.g2o");
    cout << "Optimization done"<<endl;
    globalOptimizer.clear();

    waitKey(10000);
    pcl::io::savePCDFile( "resultPCL.pcd", *pcloud );
    return 0;
}

Frame readFrame(int index)
{
    Frame frame;
    string rgbd_filename =  "r" + to_string(index)+".png";
    string depth_filename = "d" + to_string(index)+".png";

    frame.rgb = imread(rgbd_filename);
    frame.depth = imread(depth_filename,-1);
    return frame;
}

double normofTransform(Mat rvec, Mat tvec )
{
    return fabs(min(norm(rvec), 2*M_PI-norm(rvec)))+ fabs(norm(tvec)); //fabs - absolute value  // norm - gjatesia e.g. norm(3,4) = (9+16)
}