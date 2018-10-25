#ifndef __LIMS2_VISION_GLOBAL_
#define __LIMS2_VISION_GLOBAL_

#include <string>
#include <ros/ros.h>
#include <opencv2/highgui.hpp>
#include <boost/thread/mutex.hpp>

extern const std::string OUT_FOLDER;
extern const int   IMG_WIDTH;
extern const int   IMG_HEIGHT;
extern const double KArr[9];
extern const float STABLE_LIMIT;
extern const float Z_TRACK_LIMIT;
extern const cv::Rect IMG_RECT;
extern const float HUMAN_DETECTION_THR;

bool eqStamp(const ros::Time & lt, const ros::Time & rt);
bool isValid(const cv::Rect & rect);
bool isValid(const ros::Time & t);
std::string makeString(const cv::Rect & rect);
std::string makeString(const cv::Point & pnt);
std::string makeString(const cv::Point2d & pnt);
std::string makeString(const cv::Point3d & pnt);
std::string makeString(const ros::Time & stamp);
std::string makeString(const ros::Duration & stamp);

int COORD_CLIP( int v, int l, int u );
int COORD_CLIP_L( int v, int l );
int COORD_CLIP_U( int v, int u );
cv::Point COORD_CLIP(const cv::Point & p, const cv::Point & l, const cv::Point & u);
cv::Point COORD_CLIP_L(const cv::Point & p, const cv::Point & l);
cv::Point COORD_CLIP_U(const cv::Point & p, const cv::Point & u);

cv::Point center(const cv::Rect & rect);
float dist(const cv::Point & p1, const cv::Point & p2);

cv::Rect  extendRectHor(const cv::Rect & rect, const int & inc);
cv::Rect  extendRectVer(const cv::Rect & rect, const int & inc);
cv::Rect  extendRect(const cv::Rect & rect, const cv::Size & inc);

void crossLine( cv::Mat img, const cv::Point & pt, const cv::Scalar & color );

cv::Mat linspace(float startP, float endP, int interval);

void maxLocStereo(cv::Mat * imgs, cv::Point * Locs);
void minLocStereo(cv::Mat * imgs, cv::Point * Locs);

void removeIsolPixel(const cv::Mat & in, cv::Mat & out);

/////////////////////////////////
//
// StereoROI
//
struct StereoROI
{
    cv::Rect        _ROI[2];
    ros::Time       _stamp;
    boost::mutex    _mutex;    

    bool isValid();
    void getOverlapRatio(const StereoROI & hroi, float * overlapRatio);
    StereoROI & operator=(const StereoROI & rhs);    
};

#endif  // __LIMS2_VISION_GLOBAL_
