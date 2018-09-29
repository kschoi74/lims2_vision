#ifndef GUNDETECT_ROS
#define GUNDETECT_ROS

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <boost/thread/mutex.hpp>
#include <lims2_vision/HumanDetect.h>
#include <lims2_vision/bbox.h>
#include <vector>
#include <set>
#include <list>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

namespace lims2_vision
{
    class GunInfo
    {
        cv::Rect    _bBox;
        float       _prob;
    };

    class GunDetectROS
    {
        public:
        GunDetectROS(ros::NodeHandle& n, ros::NodeHandle& pnh);
        ~GunDetectROS();

        private:
        void imgCb(const sensor_msgs::ImageConstPtr& img_msg, const sensor_msgs::CameraInfoConstPtr& info_msg);        
        void connectCb(const ros::SingleSubscriberPublisher& pub);
        void disconnectCb(const ros::SingleSubscriberPublisher& pub);

        ros::NodeHandle _pnh; ///< Private nodehandle used to generate the transport hints in the connectCb.
        image_transport::ImageTransport _it; ///< Subscribes to synchronized Image CameraInfo pairs.
        image_transport::CameraSubscriber _sub; ///< Subscriber for image_transport
        ros::Publisher _pub; ///< Publisher for output bounding box messages

        lims2_vision::HumanDetect _hdt;

        boost::mutex _connect_mutex; ///< Prevents the connectCb and disconnectCb from being called until everything is initialized.
        ros::Time _prvT;
        
        int _camPos;    // 0 : left, 1 : right
        std::vector<HumanInfo> _humanROIs;
        
    };
}

#endif  // GUNDETECT_ROS