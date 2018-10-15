#ifndef _BULLETTRACK_
#define _BULLETTRACK_

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
#include <lims2_vision/ProjGeometry.h>
#include <lims2_vision/CoRects.h>
#include <boost/thread/mutex.hpp>
#include <list>

#define CMV_L    0
#define CMV_R    1
#define CMV_NUM  2
namespace lims2_vision
{
    static double m = 0.00395; // [kg]
    static double g = 9.80665; // [m/s^2]
    static double Cd = 0.82; // Long Cylinder
    static double Rho = 1.2041; // T = 20 / [kg/m^3]
    static double r = 0.020; // [m]
    static double A = 3.141592 * r * r; // [m^2]    

    struct StereoImagePtrs
    {
        cv_bridge::CvImageConstPtr _ptrs[2];
        ros::Time                  _stamp;
        StereoImagePtrs() { _ptrs[0] = _ptrs[1] = NULL; _stamp.sec = _stamp.nsec = 0; }
        bool isPairing() { return eqStamp( _ptrs[0]->header.stamp, _ptrs[1]->header.stamp ); }
    };

    class BulletTrack
    {
        enum STATE { _STATE_LOST, _STATE_HT, _STATE_GT, _STATE_BT };
        enum FORDER { CUR, PRV, PR2 };

        StereoImagePtrs         _simgPtr;

        bool isImageReady() const { return _isImageReady; }        
        const ros::Time & getFirstStamp(const int v) const;
        const ros::Time & getLastStamp(const int v) const;
        const ros::Time & getLastStamp() const;
        bool updateImagePtrs();
        void changeState(const STATE state);
        int getBulletROIImages();
        void predictBulletPosition();

    protected:       
        ProjGeometry    _PG; 
        bool            _isHumanStable;     // by the overlap ratio     
        bool            _isGunStable;       // by movement 
        double          _humanDepth;
        cv::Rect        _humanROI[2];
        boost::mutex    _humanROI_mutex[2]; 

        bool            _isImageReady;
        
        cv::Rect        _gunROI[2];        
        boost::mutex    _gun_mutex;

        cv::Mat         _bulletROIImg[3][2];
        cv::Mat         _difBulletROIImg[2][2];        
        cv::Mat         _gunMask[2];

        cv::Point2d     _predBulletPos2[2];
        cv::Rect        _bulletSR[2];
        CoRects         _BSR[2];
        
        ros::Time       _ROIStamp[3];

        std::vector<cv::Point3d> _traj;
        std::vector<ros::Time>   _stamp;
        cv::Mat                  _predTraj;

        cv::Mat         _eventImgs[3][2];

        boost::mutex    _stereo_mutex;

        std::list<cv_bridge::CvImageConstPtr> _img_ptrs[2]; // initial buffer
        std::list<StereoImagePtrs>            _simg_ptrs;   // stereo images are stored

        cv::Mat         _trackImgs[2];
        
        int                     _gunLostCount;                
        STATE                   _state;
        
        std::string             _lastmsg[4];
        cv::Point               _bulletPos2[2];
        cv::Point3d             _bulletPos3[3];
        int                     _positionCount;

    public:
        BulletTrack();
        ~BulletTrack();

        bool locateHuman(const bbox::ConstPtr& msg);
        bool setImage(const int v, const sensor_msgs::ImageConstPtr& img_msg);        
        bool setHumanROI(int v, const cv::Rect & humanROI);
        
        // Two stereo images are ready & Human is stable, then detect gun
        bool condDetectGun() const { return _isImageReady && _isHumanStable; }        

        // If guns are detected in the stereo image, try to find bullet
        bool condDetectBullet() const { return _isGunStable && _state == _STATE_GT; }
        bool condTrackBullet() const { return _state == _STATE_BT && _simg_ptrs.size() == 3; }

        bool detectGun(); 
        void detectBullet();
        bool trackBullet();
        cv::Point3d predictTrajEnd(const cv::Point3d & p1, const cv::Point3d & p2);
        cv::Point3d predictTrajEnd();
        bool predictTrajectory();

        const cv::Mat & getImage(const FORDER & order, const int & v) {
            ROS_ASSERT( 0 <= order && order <= 2 );  // 0: current t, 1: t-1, 2: t-2
            ROS_ASSERT( 0 <= v && v < 2 ); // 0: left, 1: right
            if (order == CUR)
                return _simg_ptrs.back()._ptrs[v]->image;
            else if (order == PRV)
            {
                std::list<StereoImagePtrs>::iterator it = _simg_ptrs.begin();
                std::advance(it,1);
                return it->_ptrs[v]->image;
            }
            else
                return _simg_ptrs.front()._ptrs[v]->image;
            
        }
        const ros::Time & getStamp(const FORDER & order) {
            ROS_ASSERT( 0 <= order && order <= 2 );  // 0: current t, 1: t-1, 2: t-2            
            if (order == CUR)
                return _simg_ptrs.back()._stamp;
            else if (order == PRV)
            {
                std::list<StereoImagePtrs>::iterator it = _simg_ptrs.begin();
                std::advance(it,1);
                return it->_stamp;
            }
            else
                return _simg_ptrs.front()._stamp;
        }

        void makeEventImage( int n );
        void saveEventImages();

        ///
        bool getBulletEnds(const cv::Mat & bulletImg, cv::Point * endPnts);        
        ///

        void draw(const int v, bool stable);
        void drawStatus(bool stable);
        std::string getStateString();
        void updateMsg(const std::string & str, int i);
    };
}

#endif