#ifndef __LIMS2_VISION_GLOBAL_
#define __LIMS2_VISION_GLOBAL_

#include <string>
#include <ros/ros.h>
#include <opencv2/highgui.hpp>

extern const std::string OUT_FOLDER;
extern const int   IMG_WIDTH;
extern const int   IMG_HEIGHT;

bool eqStamp(const ros::Time & lt, const ros::Time & rt);
bool isValid(const cv::Rect & rect);
std::string makeString(const cv::Rect & rect);
std::string makeString(const cv::Point & pnt);
std::string makeString(const ros::Time & stamp);

int COORD_CLIP( int v, int l, int u );
int COORD_CLIP_L( int v, int l );
int COORD_CLIP_U( int v, int u );
cv::Point COORD_CLIP(const cv::Point & p, const cv::Point & l, const cv::Point & u);
cv::Point COORD_CLIP_L(const cv::Point & p, const cv::Point & l);
cv::Point COORD_CLIP_U(const cv::Point & p, const cv::Point & u);

#endif  // __LIMS2_VISION_GLOBAL_
