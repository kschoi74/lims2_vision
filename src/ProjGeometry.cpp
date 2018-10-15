#include <lims2_vision/ProjGeometry.h>
#include <ros/ros.h>

using namespace lims2_vision;
using namespace cv;
using namespace ros;


void ProjGeometry::setCamParams(const double * Karr, const Size & frmDim)
{
    for (int i = 0, k = 0; i < 3; i++)
        for (int j = 0; j < 3; j++, k++)
        _K(i,j) = Karr[k];

    _Ki = _K.inv();    
    _frmDim = frmDim;    
}

Point3d ProjGeometry::triangulate(const Point2d & left, const Point2d & right, const bool SideBySide)
{
    double rx = right.x;
    if (SideBySide == true)
        rx -= _frmDim.width;

    Point3d p = _Ki * Vec3d(left.x, left.y, 1);
    // baseline / disparity
    double z = depthFromDisparity(left.x - rx);
    return p * z;
}

double ProjGeometry::depthFromDisparity(double disp)
{ 
    //ROS_ASSERT(disp != 0.0); 
    if ( disp == 0.0 )
        disp = 0.1;
    return _K(0, 0) * BASELINE_LEN / disp; 
}

void ProjGeometry::projPoint(cv::Point2d & p, cv::Point3d pos, const int & v)
{
    if ( v == 1 )
        pos.x -= BASELINE_LEN;

    Point3d pt = _K * pos;
    p.x = pt.x / pt.z;
    p.y = pt.y / pt.z;    
}