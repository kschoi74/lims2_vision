
#include <lims2_vision/HumanDetectROS.h>
#include <lims2_vision/bbox.h>
#include <vector>

using namespace lims2_vision;

HumanDetectROS::HumanDetectROS(ros::NodeHandle& n, ros::NodeHandle& pnh):_pnh(pnh), _it(n) {
    boost::mutex::scoped_lock lock(_connect_mutex);

    _pnh.getParam("position", _camPos);

    ROS_INFO_STREAM("HumanDetectROS constructor!,  CAM# : " << _camPos);
    _pub = n.advertise<lims2_vision::bbox>("hbbox", 2, boost::bind(&HumanDetectROS::connectCb, this, _1), 
                                                       boost::bind(&HumanDetectROS::disconnectCb, this, _1));

    _prvT = ros::Time::now();
}

HumanDetectROS::~HumanDetectROS() {
    _sub.shutdown();
}

void HumanDetectROS::imgCb(const sensor_msgs::ImageConstPtr& img_msg, 
                           const sensor_msgs::CameraInfoConstPtr& info_msg) 
{
    const ros::Duration T_THR(1,0);
    ros::Time start_t = ros::Time::now();
    ros::Duration tdiff = start_t - _prvT;
    
    if (tdiff < T_THR)  return;

    try {
        std::vector<bbox> bbox_msgs = _hdt.getBoundingBox(img_msg);        

        for (int i = 0 ; i < bbox_msgs.size() ; i++) {
            bbox_msgs[i].p = _camPos;
            _pub.publish(bbox_msgs[i]);
        }

        ros::Time t = ros::Time::now();
        
        ros::Duration tdiff = t - start_t;
        ROS_INFO_STREAM("bbox[" << bbox_msgs[0].p << "] : [" << bbox_msgs[0].l << ", " << bbox_msgs[0].t << 
                              ", " << bbox_msgs[0].r << ", " << bbox_msgs[0].b << 
                              "],   Duration: " << tdiff.nsec / 1000000 << "ms" );
    }
    catch (std::runtime_error& e) {
        ROS_ERROR_THROTTLE(1.0, "Could not detect human: %s", e.what());
    }
    _prvT = start_t;
}

// if anyone connects to me, I subscribe camera
void HumanDetectROS::connectCb(const ros::SingleSubscriberPublisher& pub) {
  boost::mutex::scoped_lock lock(_connect_mutex);
  if (!_sub && _pub.getNumSubscribers() > 0) {
    ROS_DEBUG("Connecting to image topic.");
    image_transport::TransportHints hints("raw", ros::TransportHints(), _pnh);
    _sub = _it.subscribeCamera("image", 2, &HumanDetectROS::imgCb, this, hints);
  }
}

void HumanDetectROS::disconnectCb(const ros::SingleSubscriberPublisher& pub) {
  boost::mutex::scoped_lock lock(_connect_mutex);
  if (_pub.getNumSubscribers() == 0) {
    ROS_DEBUG("Unsubscribing from image topic.");
    _sub.shutdown();
  }
}
