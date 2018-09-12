#ifndef BULLETTRACK
#define BULLETTRACK

#include <ros/ros.h>
#include <ros/package.h>
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <lims2_vision/bbox.h>
#include <boost/thread/mutex.hpp>
#include <set>

bool eqStamp(const ros::Time & lt, const ros::Time & rt);

namespace lims2_vision
{
    class BulletTrack
    {
    protected:
        cv::Rect _HumanROI[2];
        cv::Rect _GunROI[2];
        boost::mutex _HumanROI_mutex[2]; 

        cv_bridge::CvImagePtr   _img_ptrs[2][2];        
        int                     _imgStoreIdx[2];
        bool                    _imgPairing[2];
    public:
        BulletTrack();
        ~BulletTrack();

        bool locateHuman(const lims2_vision::bbox::ConstPtr& msg);
        bool setImage(const int v, const sensor_msgs::ImageConstPtr& img_msg);
        void locateGun();
    };
}

#endif