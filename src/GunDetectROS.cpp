
#include <lims2_vision/lims2_vision_global.h>
#include <lims2_vision/GunDetectROS.h>
#include <lims2_vision/GunDetect.h>
#include <cv_bridge/cv_bridge.h>

using namespace lims2_vision;

GunDetectROS::GunDetectROS(ros::NodeHandle& n, ros::NodeHandle& pnh):_pnh(pnh), _it(n) {
    boost::mutex::scoped_lock lock(_connect_mutex);

    _pnh.getParam("position", _camPos);

    ROS_INFO_STREAM("GunDetectROS constructor!,  CAM# : " << _camPos);
    _pub = n.advertise<lims2_vision::bbox>("gbbox", 2, boost::bind(&GunDetectROS::connectCb, this, _1), 
                                                       boost::bind(&GunDetectROS::disconnectCb, this, _1));

    ros::Time::init();
    _prvT = ros::Time::now();

    _humanROIs.clear();
}

GunDetectROS::~GunDetectROS() {
    _sub.shutdown();
}

void GunDetectROS::imgCb(const sensor_msgs::ImageConstPtr& img_msg, 
                         const sensor_msgs::CameraInfoConstPtr& info_msg) 
{
    const ros::Duration T_THR(1,0);
    ros::Time start_t = ros::Time::now();
    ros::Duration tdiff = start_t - _prvT;
    
    if (tdiff >= T_THR) {
        // human detection using tensorflow
        try {
            int nHuman = _hdt.detectHuman(img_msg, _humanROIs);
        }
        catch (std::runtime_error& e) {
            ROS_ERROR_THROTTLE(1.0, "Could not detect human: %s", e.what());
        }
        _prvT = start_t;

        _hdt.buildHumanTracks( _humanROIs );

        // if there is no person in the scene, ROI is resetted. (-1 -1 -1 -1)
        Rect humanROI = _hdt.getBestHumanROI();        
        bbox hbbox;
        hbbox.stamp = img_msg->header.stamp;
        hbbox.p = _camPos;

        hbbox.x      = humanROI.x;
        hbbox.y      = humanROI.y;
        hbbox.width  = humanROI.width;
        hbbox.height = humanROI.height;        
        _pub.publish(hbbox);
    }       
}

// if anyone connects to me, I subscribe camera
void GunDetectROS::connectCb(const ros::SingleSubscriberPublisher& pub) {
  boost::mutex::scoped_lock lock(_connect_mutex);
  if (!_sub && _pub.getNumSubscribers() > 0) {
    ROS_DEBUG("Connecting to image topic.");
    image_transport::TransportHints hints("raw", ros::TransportHints(), _pnh);
    _sub = _it.subscribeCamera("image", 2, &GunDetectROS::imgCb, this, hints);
  }
}

void GunDetectROS::disconnectCb(const ros::SingleSubscriberPublisher& pub) {
  boost::mutex::scoped_lock lock(_connect_mutex);
  if (_pub.getNumSubscribers() == 0) {
    ROS_DEBUG("Unsubscribing from image topic.");
    _sub.shutdown();
  }
}
