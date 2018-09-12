
#include <lims2_vision/BulletTrackROS.h>
#include <lims2_vision/bbox.h>
#include <vector>

using namespace lims2_vision;

BulletTrackROS::BulletTrackROS(ros::NodeHandle& n, ros::NodeHandle& pnh):_pnh(pnh), _it(n) 
{
    boost::mutex::scoped_lock lock(_connect_mutex);

    ROS_INFO("BulletTrackROS constructor!");

    ros::NodeHandle nh;
    image_transport::TransportHints hints("raw", ros::TransportHints(), _pnh);
    _sub[0] = _it.subscribeCamera("image_l", 1, &BulletTrackROS::imgCb_l, this, hints);
    _sub[1] = _it.subscribeCamera("image_r", 1, &BulletTrackROS::imgCb_r, this, hints);
    _subHBBox = nh.subscribe("hbbox", 4, &BulletTrackROS::hbboxCb, this);

    _prvT = ros::Time::now();
}

BulletTrackROS::~BulletTrackROS() {
    _sub[0].shutdown();
    _sub[1].shutdown();
}

void BulletTrackROS::hbboxCb(const lims2_vision::bbox::ConstPtr& msg)
{
    ROS_INFO_STREAM("hbboxCB: [" << msg->p << ": " << msg->l << ", " << msg->t << ", " 
                                 << msg->r << ", " << msg->b << "]");

    _bTrack.locateHuman(msg);
}

void BulletTrackROS::imgCb_l(const sensor_msgs::ImageConstPtr& img_msg, 
                             const sensor_msgs::CameraInfoConstPtr& info_msg) 
{
    const ros::Duration T_THR(1,0);
    ros::Time start_t = ros::Time::now();
    ros::Duration tdiff = start_t - _prvT;
    
    ROS_INFO_STREAM("BTROS imgCB_l: " << tdiff.nsec / 1000000 << "ms, Dim: " << img_msg->width << ", " << img_msg->height);

    if ( _bTrack.setImage(0, img_msg) )
    {   
        ROS_ASSERT(0);  // expect paired after getting the right image
    }

    _prvT = start_t;
}

void BulletTrackROS::imgCb_r(const sensor_msgs::ImageConstPtr& img_msg, 
                             const sensor_msgs::CameraInfoConstPtr& info_msg) 
{
    const ros::Duration T_THR(1,0);
    ros::Time start_t = ros::Time::now();
    ros::Duration tdiff = start_t - _prvT;
    
    ROS_INFO_STREAM("BTROS imgCB_r: " << tdiff.nsec / 1000000 << "ms");

    if ( _bTrack.setImage(1, img_msg) )
    {
        _bTrack.locateGun();
    }
}