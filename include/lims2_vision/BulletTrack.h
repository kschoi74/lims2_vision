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
#include <lims2_vision/lims2_vision_global.h>
#include <boost/thread/mutex.hpp>
#include <list>

namespace lims2_vision
{
    static double m = 0.00395; // [kg]
    static double g = 9.80665; // [m/s^2]
    static double Cd = 0.82; // Long Cylinder
    static double Rho = 1.2041; // T = 20 / [kg/m^3]
    static double r = 0.020; // [m]
    static double A = 3.141592 * r * r; // [m^2]    

    class BulletTrack
    {
        enum STATE { _STATE_LOST, _STATE_GT, _STATE_AIM, _STATE_BT };

        bool isImageReady() const { return _isImageReady; }
        bool isAiming() const { return _gunAIM[0] && _gunAIM[1]; }
        bool checkAim(const cv::Rect & gunROI) const { int ga = gunROI.area(); return 0 < ga && ga <= 160 && isValid(gunROI); }
        const ros::Time & getFirstStamp(const int v) const;
        const ros::Time & getLastStamp(const int v) const;
        const ros::Time & getPairStamp(const int i) const;
        bool updatePairStamp();
        void updateImagePtrs();
        void changeState(const STATE state);
        

    protected:        
        bool            _isImageReady;
        bool            _gunAIM[2];
        cv::Rect        _gunROI[2];        
        boost::mutex    _gun_mutex[2];
        boost::mutex    _humanROI_mutex[2]; 

        std::list<cv_bridge::CvImageConstPtr> _img_ptrs[2];
        std::list<ros::Time>    _pairStamps;
        int                     _lostCount[2];                
        STATE                   _state;
        
        cv::Mat                 _bulletKernel;
        std::string             _lastmsg[4];
        cv::Point               _bulletPos2[2];
        cv::Point3d             _bulletPos3[3];
        int                     _positionCount;
        cv::Mat1d               _ITRS;
        double                  _baseline;

    public:
        BulletTrack();
        ~BulletTrack();

        bool locateHuman(const bbox::ConstPtr& msg);
        bool setImage(const int v, const sensor_msgs::ImageConstPtr& img_msg);
        void setGunROI(int v, const cv::Rect & gunROI);
        bool isTrackingReady();   

        ///
        void findBullet_sa();
        bool getBulletEnds(const cv::Mat & bulletImg, cv::Point * endPnts);
        cv::Point3d calcPos3(const cv::Point & p1, const cv::Point & p2);
        ///

        void draw(const int v);
        std::string getStateString();
        void updateMsg(const std::string & str, int i);
    };
}

#endif