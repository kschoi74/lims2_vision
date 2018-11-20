#ifndef _LIMS2_CONTROLLER_
#define _LIMS2_CONTROLLER_

#include <lims2_common_pkg/HandPosData.h>
#include <lims2_vision/lims2_vision_global.h>
#include <ros/ros.h>
#include <opencv2/opencv.hpp>

namespace lims2_vision
{
    class Lims2Controller
    {
    protected:
        ros::NodeHandle                 _nh;
        ros::Publisher                  _handPosPub;
        ros::Subscriber                 _handPosAckSub;
        lims2_common_pkg::HandPosData   _msg;
        cv::Point3d                     _pos[2];    // left, right arms

    public:
        Lims2Controller(ros::NodeHandle & nh);
        void manipulate(STATE state, void* addInfo = NULL);
        void publish();
        void pose0(bool check = true);
        void pose(cv::Point3d * pP);
        void HandPosAckCB(const lims2_common_pkg::HandPosData::ConstPtr& msg);
    };
}

#endif  // _LIMS2_CONTROLLER_