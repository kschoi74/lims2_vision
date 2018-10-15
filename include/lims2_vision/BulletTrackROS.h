#ifndef _BULLETTRACK_ROS_
#define _BULLETTRACK_ROS_

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <boost/thread/mutex.hpp>
#include <lims2_vision/bbox.h>
#include <lims2_vision/BulletTrack.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <list>

namespace lims2_vision
{
    class BulletTrackROS
    {
        public:
        BulletTrackROS(ros::NodeHandle& n, ros::NodeHandle& pnh);
        ~BulletTrackROS();

        private:
        void imgCb_l(const sensor_msgs::ImageConstPtr& img_msg, const sensor_msgs::CameraInfoConstPtr& info_msg);        
        void imgCb_r(const sensor_msgs::ImageConstPtr& img_msg, const sensor_msgs::CameraInfoConstPtr& info_msg);        
        void hbboxCb(const bbox::ConstPtr& msg);

        ros::NodeHandle _pnh; ///< Private nodehandle used to generate the transport hints in the connectCb.
        image_transport::ImageTransport _it; ///< Subscribes to synchronized Image CameraInfo pairs.
        image_transport::CameraSubscriber _sub[2]; ///< Subscriber for image_transport
        ros::Subscriber _subHBBox;

        BulletTrack _bTrack;

        boost::mutex _connect_mutex; ///< Prevents the connectCb and disconnectCb from being called until everything is initialized.
        ros::Time _prvT;        
        
        int _camPos;    // 0 : left, 1 : right
    };
}

#endif  // BULLETTRACK_ROS