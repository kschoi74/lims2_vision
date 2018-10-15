#include <lims2_vision/CoRects.h>
#include <ros/ros.h>

using namespace lims2_vision;
using namespace cv;

void CoRects::setOriginCoRects(const std::list<cv::Size> & sizes)
{
	_center = Point(0, 0);
    int nRects = sizes.size();
    
    _rects.resize(nRects);
    _validRects.resize(nRects);

    int i = 0;
    for ( auto & sz : sizes ) {
        Rect r = Rect( -Point(sz/2), sz);
        _rects[i] = r;
        _validRects[i] = r & _frameRect;        
        i++;
    }
}

void CoRects::setCoRects(const std::list<cv::Size> & sizes, const cv::Point & pt)
{
	setOriginCoRects(sizes);
	*this += pt;
}

const cv::Rect & CoRects::getRect(const int n) const
{
    ROS_ASSERT( 0 <= n && n < _validRects.size() );
    return _validRects[n];
}

const cv::Rect CoRects::get0BaseRect(const int n) const
{
    ROS_ASSERT( 0 <= n && n < _validRects.size() );
    return _validRects[n] - _validRects[0].tl();
}

cv::Point CoRects::getOffset(const int n) const
{
    ROS_ASSERT( 0 <= n && n < _validRects.size() );
    return _validRects[n].tl() - _validRects[0].tl();
}