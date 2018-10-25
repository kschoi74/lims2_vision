#include <lims2_vision/lims2_vision_global.h>
#include <ros/ros.h>
//#include <ros/console.h>
#include <opencv2/imgproc.hpp>

const std::string OUT_FOLDER("/home/kschoi/Downloads/");
const int   IMG_WIDTH = 1280;
const int   IMG_HEIGHT = 720;
const double KArr[] = { 696.692, 0, 670.839,   0, 696.692, 352.867,   0, 0, 1 };
const float STABLE_LIMIT = 0.8f;
const float Z_TRACK_LIMIT = 1.4f;
const cv::Rect IMG_RECT( 0, 0, IMG_WIDTH, IMG_HEIGHT );
const float HUMAN_DETECTION_THR = 0.85;

bool eqStamp(const ros::Time & lt, const ros::Time & rt)
{
    return ( lt.sec == rt.sec && lt.nsec == rt.nsec ) ? true : false;
}

bool isValid(const cv::Rect & rect)
{
    return rect.x != -1;
}

bool isValid(const ros::Time & t)
{
    return t.sec != 0 || t.nsec != 0;
}

std::string makeString(const cv::Rect & rect)
{
    std::string res("(");
    res += std::to_string(rect.x) + ", " + std::to_string(rect.y) + ", "
         + std::to_string(rect.width) + ", " + std::to_string(rect.height) + ")";
    return res;
}

std::string makeString(const cv::Point & pnt)
{
    std::string res("(");
    res += std::to_string(pnt.x) + ", " + std::to_string(pnt.y) + ")";
    return res;
}

std::string makeString(const cv::Point2d & pnt)
{
    std::string res("(");
    res += std::to_string(pnt.x) + ", " + std::to_string(pnt.y) + ")";
    return res;
}

std::string makeString(const cv::Point3d & pnt)
{
    std::string res("(");
    res += std::to_string(pnt.x) + ", " + std::to_string(pnt.y) + ", " 
         + std::to_string(pnt.z) + ")";
    return res;
}

std::string makeString(const ros::Time & stamp)
{
    return std::to_string(stamp.sec) + "." + std::to_string(stamp.nsec);
}

std::string makeString(const ros::Duration & stamp)
{
    return std::to_string(stamp.sec) + "." + std::to_string(stamp.nsec);
}

int COORD_CLIP( int v, int l, int u )
{
    return cv::min(cv::max( v, l ), u);
}

int COORD_CLIP_L( int v, int l )
{
    return cv::max( v, l );
}

int COORD_CLIP_U( int v, int u )
{
    return cv::min( v, u );
}

cv::Point COORD_CLIP(const cv::Point & p, const cv::Point & l, const cv::Point & u)
{
    return COORD_CLIP_U(COORD_CLIP_L(p, l), u);
}

cv::Point COORD_CLIP_L(const cv::Point & p, const cv::Point & l)
{
    return cv::Point( COORD_CLIP_L(p.x, l.x), COORD_CLIP_L(p.y, l.y) );
}

cv::Point COORD_CLIP_U(const cv::Point & p, const cv::Point & u)
{
    return cv::Point( COORD_CLIP_U(p.x, u.x), COORD_CLIP_U(p.y, u.y) );
}

cv::Point center(const cv::Rect & rect)
{
    return (rect.tl() + rect.br()) / 2;
}

float dist(const cv::Point & p1, const cv::Point & p2)
{
    cv::Point p = p1 - p2;
    return sqrt(p.x * p.x + p.y * p.y);
}

cv::Rect  extendRectHor(const cv::Rect & rect, const int & inc)
{
    ROS_ASSERT( inc > 0 );
    
    cv::Rect res(rect.x - inc / 2, rect.y, rect.width + inc, rect.height);
    res &= IMG_RECT;

    return res;
}

cv::Rect  extendRectVer(const cv::Rect & rect, const int & inc)
{
    ROS_ASSERT( inc > 0 );
    
    cv::Rect res(rect.x, rect.y - inc / 2, rect.width, rect.height + inc);
    res &= IMG_RECT;

    return res;
}

cv::Rect  extendRect(const cv::Rect & rect, const cv::Size & inc)
{
    ROS_ASSERT( inc.area() > 0 );
    
    cv::Rect res = rect + inc;
    res -= cv::Point(inc/2);
    res &= IMG_RECT;    

    return res;
}

void crossLine( cv::Mat img, const cv::Point & pt, const cv::Scalar & color )
{
    cv::Point ph(5,0);
    cv::Point pv(0,5);
    cv::Point ph1(1,0);
    cv::Point pv1(0,1);
    cv::line( img, pt - ph, pt - ph1, color );
    cv::line( img, pt + ph1, pt + ph, color );
    cv::line( img, pt - pv, pt - pv1, color );
    cv::line( img, pt + pv1, pt + pv, color );
}

cv::Mat linspace(float startP, float endP, int interval)
{
    float spacing = interval - 1;
    cv::Mat y(spacing, 1, CV_32F);
    for (int i = 0 ; i < y.rows; ++i)
        y.at<float>(i) = startP + i*(endP - startP) / spacing;

    return y;
}

void maxLocStereo(cv::Mat * imgs, cv::Point * Locs)
{
    cv::Mat sumMax(imgs[0].rows, 1, CV_32F);
    cv::Mat maxIndex;

    double maxVal[2];
    cv::Point maxLoc;
    for (int r = 0; r < imgs[0].rows; r++)
    {
        minMaxLoc(imgs[0].row(r), NULL, &maxVal[0], NULL, &Locs[0]);
        minMaxLoc(imgs[1].row(r), NULL, &maxVal[1], NULL, &Locs[1]);
        sumMax.at<float>(r, 0) = maxVal[0] + maxVal[1];
        maxIndex.push_back(cv::Vec2i(Locs[0].x, Locs[1].x));
    }
    minMaxLoc(sumMax, NULL, NULL, NULL, &maxLoc);
    cv::Vec2i xx = maxIndex.at<cv::Vec2i>(maxLoc.y, 0);
    Locs[0].x = xx(0);
    Locs[1].x = xx(1);
    Locs[0].y = Locs[1].y = maxLoc.y;
}

void minLocStereo(cv::Mat * imgs, cv::Point * Locs)
{
    cv::Mat sumMin(imgs[0].rows, 1, CV_32F);
    cv::Mat minIndex;

    double minVal[2];
    cv::Point minLoc;
    for (int r = 0; r < imgs[0].rows; r++)
    {
        minMaxLoc(imgs[0].row(r), &minVal[0], NULL, &Locs[0]);
        minMaxLoc(imgs[1].row(r), &minVal[1], NULL, &Locs[1]);
        sumMin.at<float>(r, 0) = minVal[0] + minVal[1];
        minIndex.push_back(cv::Vec2i(Locs[0].x, Locs[1].x));
    }
    minMaxLoc(sumMin, NULL, NULL, &minLoc);
    cv::Vec2i xx = minIndex.at<cv::Vec2i>(minLoc.y, 0);
    Locs[0].x = xx(0);
    Locs[1].x = xx(1);
    Locs[0].y = Locs[1].y = minLoc.y;
}

void removeIsolPixel(const cv::Mat & in, cv::Mat & out)
{
    ROS_ASSERT(in.type() == CV_8U);
    out = cv::Mat::zeros(in.size(), in.type());
    int x, y;
    int rows = in.rows;
    int cols = in.cols;
    for ( y= 1 ; y < rows - 1; y++)
    {
        const uchar * p0 = in.ptr<uchar>(y-1);
        const uchar * p1 = in.ptr<uchar>(y);
        const uchar * p2 = in.ptr<uchar>(y+1);
        uchar * pd = out.ptr<uchar>(y) + 1;

        for (x=1; x < cols-1; x++, p0++, p1++, p2++, pd++)
        {
            if (p1[1] == 0) continue;
            if (p0[0] + p0[1] + p0[2] + p1[0] + p1[2] + p2[0] + p2[1] + p2[2])
                *pd = 255;
        }
    }
}

/////////////////////////////////
//
// StereoROI
//
bool StereoROI::isValid()
{
     return ::isValid(_ROI[0]) && ::isValid(_ROI[1]);
}

void StereoROI::getOverlapRatio(const StereoROI & hroi, float * overlapRatio)
{
    ROS_ASSERT( overlapRatio != NULL );
    overlapRatio[0] = float((hroi._ROI[0] & _ROI[0]).area()) / hroi._ROI[0].area();
    overlapRatio[1] = float((hroi._ROI[1] & _ROI[1]).area()) / hroi._ROI[1].area();
}

StereoROI & StereoROI::operator=(const StereoROI & rhs)
{
    boost::mutex::scoped_lock lock(_mutex);
    _ROI[0] = rhs._ROI[0];
    _ROI[1] = rhs._ROI[1];
    _stamp = rhs._stamp;
}