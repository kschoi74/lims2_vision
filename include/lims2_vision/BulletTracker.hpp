#ifndef _BULLETTRACKER_
#define _BULLETTRACKER_

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <lims2_vision/lims2_vision_global.h>
#include <lims2_vision/MiniZEDWrapper.hpp>
#include <lims2_vision/ProjGeometry.h>
#include <lims2_vision/CoRects.h>
#include <boost/thread/mutex.hpp>
#include <list>

#define CMV_L    0
#define CMV_R    1
#define CMV_NUM  2

void preProcRegion(cv::Mat & blob);
void getBulletSal(cv::Mat & bSal, const cv::Mat & img, const cv::Mat & difimg, const cv::Mat & mask );

template<typename T>
void sumChannels(cv::Mat & dst, const cv::Mat & clrImg);

namespace lims2_vision
{
    class BulletTracker
    {
        enum STATE { _STATE_LOST, _STATE_HT, _STATE_GT, _STATE_AIM, _STATE_BT, _STATE_CLEANUP };        

        void changeState(const STATE state);
        int  getBulletROIImages();
        void predictBulletPosition();

    protected:       
        // threading
        std::thread     _trackThread;
        bool            _bRun;

        // state-related
        STATE           _state;
        bool            _isHumanStable;     // by the overlap ratio     
        bool            _isGunStable;       // by movement 
        int             _gunLostCount;  
        int             _gunStableCount;
        int             _trackStartCount;              
        int             _cleanupCount;
        
        // image queue
        StereoImgCQ &   _sImgs;

        // image-related
        ros::Time       _ROIStamp[3];
        ros::Time       _sImgStamp;
        int             _sImgIdx[3];

        // human
        StereoROI &     _humanROIsDetected;
        StereoROI       _humanROIs;

        // gun
        cv::Rect        _gunROI[2];                        
        cv::Mat         _gunMask[2];

        // bullet
        CoRects         _BSR[2];
        cv::Mat         _bulletROIImg[3][2];
        cv::Mat         _difBulletROIImg[2][2];        
        
        cv::Point2d     _predBulletPos2[2];        
        
        // geometry calculation
        ProjGeometry    _PG; 

        // results
        std::vector<cv::Point3d> _traj;
        std::vector<ros::Time>   _stamp;
        cv::Mat                  _predTraj;

        // check
        StereoImgCQ     _eventImgs;        
        cv::Mat         _trackImgs[2];

        std::string     _lastmsg[4];

    public:
        BulletTracker(StereoImgCQ & simgs, StereoROI & hRegion);
        ~BulletTracker();
        void shutdown();

        // conditions
        bool condCheckHumanROIs() const { return !::eqStamp(_humanROIsDetected._stamp, _humanROIs._stamp); }
        
        // Two stereo images are ready & Human is stable, then detect gun
        bool condDetectGun() const {  return _isHumanStable; }        

        // If guns are detected in the stereo image, try to find bullet
        bool condDetectBullet() const { return _state == _STATE_AIM; }
        bool condTrackBullet() const;
        bool condPredictTrajectory() const { 
            if ( _traj.empty() )    return false;
            return _traj.back().z < Z_TRACK_LIMIT;
        }

        // human ROI update
        bool checkHumanROIs();

        // 
        void process();
        bool detectGun(); 
        void detectBullet();
        bool trackBullet();        
        
        // results
        bool predictTrajectory(Point3d & predP);
        void predictTrajectoryNCleanup();


        const ros::Time & getLastStamp() const { return _sImgs.getLastStamp(); }

        void makeEventImage( int n );
        void saveEventImages();
        void saveTrackHistImages(int n);

        
        void drawStatus(bool prnStable = true);
        void draw(const int v, bool prnStable);
        void prnStatus();        
        void updateMsg(const std::string & str, int i);
        std::string getStateString();        
    };
}

#endif