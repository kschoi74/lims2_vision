#include <lims2_vision/Lims2Controller.hpp>
#include <opencv2/opencv.hpp>

#ifndef PI
    #define PI       (3.14159265358979323836264338327950)
#endif

#ifndef D2R
    #define D2R		(PI / 180.)
    #define R2D		(180. / PI)
#endif

using namespace lims2_vision;

Lims2Controller::Lims2Controller(ros::NodeHandle & nh)
{
    _nh = nh;
    _handPosPub = _nh.advertise<lims2_common_pkg::HandPosData>("hand_pos_msg", 1);
    _handPosAckSub = nh.subscribe("hand_pos_ackmsg", 1, &Lims2Controller::HandPosAckCB, this);
}

void Lims2Controller::manipulate(STATE state, void* addInfo)
{
    switch(state) {
    case _STATE_LOST:
        pose0(false);
        break;

    case _STATE_HT:
    case _STATE_GT:
    case _STATE_AIM:
    case _STATE_BT:
        pose0();
        break;

    case _STATE_CLEANUP:  //YJK 20180519
        ROS_ASSERT( addInfo );
        pose((cv::Point3d*)addInfo);
        break;
    }
}

void Lims2Controller::publish()
{
    _handPosPub.publish(_msg);
}

void Lims2Controller::pose0(bool check)
{
    const static double INIT_X = 0.4;
    const static double INIT_Y = -0.3;
    const static double INIT_Z = 0.8;

    // RIGHT ARM
    _msg.mode = 1;
    _msg.data[0] = INIT_X;
    _msg.data[1] = INIT_Y;
    _msg.data[2] = INIT_Z;
    _msg.data[3] = 1.;
    _msg.data[4] = 0.;
    _msg.data[5] = 0.;
    _msg.data[6] = 0.;

    if ( check ) {
        if ( _pos[1].x != INIT_X || _pos[1].y != INIT_Y || _pos[1].z != INIT_Z )
            publish();

        // LEFT ARM
        if ( _pos[0].x != INIT_X || _pos[0].y != -INIT_Y || _pos[0].z != INIT_Z )
        {
            _msg.mode = 0;
            _msg.data[1] = -INIT_Y;
            publish();
        }
    }
    else {
        publish();
        
        // LEFT ARM
        _msg.mode = 0;
        _msg.data[1] = -INIT_Y;
        publish();
    }
}

void Lims2Controller::pose(cv::Point3d * pP)
{
    _msg.data[0] = pP->z;               // LIMS2 X : forward
    _msg.data[1] = -pP->x + 0.06;       //       Y : left of LIMS2
    _msg.data[2] = -pP->y + 1.15;       //       Z : up
    _msg.data[3] = 1.;
    _msg.data[4] = 0.;
    _msg.data[5] = 0.;
    _msg.data[6] = 0.;

    _msg.mode = (_msg.data[1] < 0.0) ? 1 /*Right ARM*/ : 0 /*Left ARM*/;
    publish();
}

void Lims2Controller::HandPosAckCB(const lims2_common_pkg::HandPosData::ConstPtr& msg)
{
    ROS_INFO_STREAM("HandPosAckCB: " << msg->data[0] << ", " << msg->data[1] << ", " << msg->data[2]);
    _pos[msg->data[1] < 0] = cv::Point3d(msg->data[0], msg->data[1], msg->data[2]);    
}