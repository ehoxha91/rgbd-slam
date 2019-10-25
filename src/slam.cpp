#include <iostream>
#include <fstream>
#include <sstream>
#include "slamBase.h"

using namespace cv;
using namespace std;

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


CHECK_RESULT checkKeyFrame(Frame &frame1, Frame &frame2, bool is_loops=false);
Frame readFrame(int index);
double normofTransform(Mat rvec, Mat tvec );
//Loop Closure:
void checkNearbyLoops( vector<Frame>& frames, Frame& currentFrame);
void checkRandomLoops( vector<Frame>& frames, Frame& currentFrame);

OptimizerE optimizeE;       //Optimizer class 

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
        CHECK_RESULT result = checkKeyFrame( keyframes.back(), currFrame, false);

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
                    checkNearbyLoops(keyframes,currFrame);
                    checkRandomLoops(keyframes,currFrame);
                }
                keyframes.push_back(currFrame);

                break;
            default:
                break;
        }
    }
    //We start optimization of all nodes:
    optimizeE.Optimize(100);

    for (size_t i = 0; i < keyframes.size(); i++)
    {
        //Take a frame from g2o
    }
    
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

CHECK_RESULT checkKeyFrame(Frame &frame1, Frame &frame2, bool is_loops)
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

    return KEYFRAME;    //We have a keyframe 
}

void checkNearbyLoops( vector<Frame>& frames, Frame& currentFrame)
{
    if(frames.size() <= nearby_loops)
    {
        //We don't even have as much as we need to check, so let's check everything
        for (size_t i = 0; i < frames.size(); i++)
        {
            checkKeyFrame(frames[i], currentFrame, true);
        }
    }
    else
    {
        //Check the frames in neighbourhood, number of frames to be checked "nearby_loops"
        for (size_t i = frames.size() - nearby_loops; i < frames.size(); i++)
        {
            //If number of frames is 200, then last 5 are our "nearby frames", 200, 199, ..., 196
            checkKeyFrame(frames[i], currentFrame, true);
        }
        
    }
    

}

void checkRandomLoops( vector<Frame>& frames, Frame& currentFrame)
{
      if ( frames.size() <= random_loops )
    {
        // no enough keyframes, check everyone
        for (size_t i=0; i<frames.size(); i++)
        {
            checkKeyFrame( frames[i], currentFrame, true );
        }
    }
    else
    {
        // randomly check loops
        for (int i=0; i<random_loops; i++)
        {
            int index = rand()%frames.size();
            checkKeyFrame( frames[i], currentFrame, true );
        }
    }
}
