#include <lims2_vision/lims2_vision_global.h>

const std::string OUT_FOLDER("/home/kschoi/Downloads/");
const int   IMG_WIDTH = 1280;
const int   IMG_HEIGHT = 720;

bool eqStamp(const ros::Time & lt, const ros::Time & rt)
{
    return ( lt.sec == rt.sec && lt.nsec == rt.nsec ) ? true : false;
}

bool isValid(const cv::Rect & rect)
{
    return rect.x != -1;
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

std::string makeString(const ros::Time & stamp)
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