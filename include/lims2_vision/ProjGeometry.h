#ifndef _PROJ_GEOMETRY_
#define _PROJ_GEOMETRY_

#include <opencv2/core.hpp>

namespace lims2_vision {

class ProjGeometry 
{
protected:
    cv::Matx33d _K;     // camera instrinsic matrix
    cv::Matx33d _Ki;
    cv::Size    _frmDim;

    const double BASELINE_LEN = 0.12;

public:
    void setCamParams(const double * Karr, const cv::Size & frmDim);
    cv::Point3d triangulate(const cv::Point2d & left, const cv::Point2d & right, const bool SideBySide = false);
    double depthFromDisparity(double disp);    
    void projPoint(cv::Point2d & p, cv::Point3d pos, const int & v);
};

}

#endif // _PROJ_GEOMETRY_