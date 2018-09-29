
#include <lims2_vision/BulletTrackROS.h>
#include <lims2_vision/bbox.h>
#include <vector>
#include <string>
#include <boost/format.hpp>

using namespace lims2_vision;
using namespace cv;

BulletTrackROS::BulletTrackROS(ros::NodeHandle& n, ros::NodeHandle& pnh):_pnh(pnh), _it(n) 
{
    boost::mutex::scoped_lock lock(_connect_mutex);

    ROS_INFO("BulletTrackROS constructor!");

    ros::NodeHandle nh;
    image_transport::TransportHints hints("raw", ros::TransportHints(), _pnh);
    _sub[0] = _it.subscribeCamera("image_l", 1, &BulletTrackROS::imgCb_l, this, hints);
    _sub[1] = _it.subscribeCamera("image_r", 1, &BulletTrackROS::imgCb_r, this, hints);
    _subHBBox = nh.subscribe("gbbox", 4, &BulletTrackROS::gbboxCb, this);

    _prvT = ros::Time::now();
}

BulletTrackROS::~BulletTrackROS() {
    _sub[0].shutdown();
    _sub[1].shutdown();
}

void BulletTrackROS::gbboxCb(const bbox::ConstPtr& msg)
{
#ifdef _DEBUG_BT_    
    ROS_INFO_STREAM("gbboxCB: [" << msg->p << ": " << msg->x << ", " << msg->y << ", " 
                                 << msg->width << ", " << msg->height << "] : " << msg->stamp.sec << "." << msg->stamp.nsec);
#endif                                 

    int camPos = msg->p & 0x01;
    
    boost::format fmter("gbboxCB: [%1%: %2%, %3%, %4%, %5%] : %6%.%7%");
    fmter % msg->p % msg->x % msg->y % msg->width % msg->height % msg->stamp.sec % msg->stamp.nsec;
    _bTrack.updateMsg( fmter.str(), 2+camPos );

    _bTrack.setGunROI( camPos, Rect(msg->x, msg->y, msg->width, msg->height) );

    if ( _bTrack.isTrackingReady() )
        _bTrack.findBullet_sa();    
}

void BulletTrackROS::imgCb_l(const sensor_msgs::ImageConstPtr& img_msg, 
                             const sensor_msgs::CameraInfoConstPtr& info_msg) 
{
#ifdef _DEBUG_BT_    
    ROS_INFO_STREAM("imgCb_l: [" << img_msg->header.stamp.sec << "." << img_msg->header.stamp.nsec);
#endif
    const int CAMNUM = 0;
    _bTrack.updateMsg( std::string("imgCb_l: [") + makeString(img_msg->header.stamp) + "]", CAMNUM );
    _bTrack.setImage(CAMNUM, img_msg);    
}

void BulletTrackROS::imgCb_r(const sensor_msgs::ImageConstPtr& img_msg, 
                             const sensor_msgs::CameraInfoConstPtr& info_msg) 
{
#ifdef _DEBUG_BT_    
    ROS_INFO_STREAM("imgCb_r: [" << img_msg->header.stamp.sec << "." << img_msg->header.stamp.nsec);
#endif
    const int CAMNUM = 1;
    _bTrack.updateMsg( std::string("imgCb_r: [") + makeString(img_msg->header.stamp) + "]", CAMNUM );
    _bTrack.setImage(CAMNUM, img_msg);    
}