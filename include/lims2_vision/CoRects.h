#ifndef _CORECTS_
#define _CORECTS_

#include <opencv2/imgproc.hpp>
#include <vector>
#include <list>

namespace lims2_vision {

class CoRects {
public:
    std::vector<cv::Rect> _rects;
	std::vector<cv::Rect> _validRects;
    cv::Rect            _frameRect;
	cv::Point			_center;

public:
    CoRects() {}
    CoRects(const cv::Rect & fR) : _frameRect(fR) {}
    
    void setOriginCoRects(const std::list<cv::Size> & sizes);
    void setCoRects(const std::list<cv::Size> & sizes, const cv::Point & pt);    
    const cv::Rect & getRect(const int n) const;   
    const cv::Rect get0BaseRect(const int n) const;
    cv::Point getOffset(const int n) const;
};

static inline
CoRects& operator += ( CoRects& a, const cv::Point& b )
{
	a._center += b;
	int n = a._rects.size();
	for (int i = 0; i < a._rects.size(); i++)
	{
		a._rects[i].x += b.x;
		a._rects[i].y += b.y;
		a._validRects[i] = a._rects[i] & a._frameRect;
	}
	    
    return a;
}

static inline
CoRects& operator -= ( CoRects& a, const cv::Point& b )
{
	a += -b;
	return a;
}

// static inline
// Rect_<_Tp>& operator += ( Rect_<_Tp>& a, const Size_<_Tp>& b )
// {
//     a.width += b.width;
//     a.height += b.height;
//     return a;
// }

// template<typename _Tp> static inline
// Rect_<_Tp>& operator -= ( Rect_<_Tp>& a, const Size_<_Tp>& b )
// {
//     a.width -= b.width;
//     a.height -= b.height;
//     return a;
// }

// template<typename _Tp> static inline
// Rect_<_Tp>& operator &= ( Rect_<_Tp>& a, const Rect_<_Tp>& b )
// {
//     _Tp x1 = std::max(a.x, b.x);
//     _Tp y1 = std::max(a.y, b.y);
//     a.width = std::min(a.x + a.width, b.x + b.width) - x1;
//     a.height = std::min(a.y + a.height, b.y + b.height) - y1;
//     a.x = x1;
//     a.y = y1;
//     if( a.width <= 0 || a.height <= 0 )
//         a = Rect();
//     return a;
// }

// template<typename _Tp> static inline
// Rect_<_Tp>& operator |= ( Rect_<_Tp>& a, const Rect_<_Tp>& b )
// {
//     if (a.empty()) {
//         a = b;
//     }
//     else if (!b.empty()) {
//         _Tp x1 = std::min(a.x, b.x);
//         _Tp y1 = std::min(a.y, b.y);
//         a.width = std::max(a.x + a.width, b.x + b.width) - x1;
//         a.height = std::max(a.y + a.height, b.y + b.height) - y1;
//         a.x = x1;
//         a.y = y1;
//     }
//     return a;
// }


}

#endif 