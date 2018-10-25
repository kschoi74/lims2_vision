#ifndef _MINIZEDWRAPPER_
#define _MINIZEDWRAPPER_

#include <sl/Camera.hpp>

#include <ros/ros.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <lims2_vision/lims2_vision_global.h>
#include <vector>

using namespace std;
using namespace cv;
using namespace ros;

namespace lims2_vision {

    struct StereoImg {
        Mat _imgs[2];
        Time _stamp;

        Mat &        getImage(int v)            {   return _imgs[v];    }
        void         setImage(Mat & img, int v) {   _imgs[v] = img;     }
        const Time & getStamp() const           {   return _stamp;      }
        void         setStamp(Time & t)         {   _stamp = t;         }
    };

    struct StereoImgCQ {
        vector<StereoImg>   _sImgs;
        int                 _size;
        int                 _idx;   // 0 <= _idx < _size

        StereoImgCQ() : _size(0), _idx(-1) {}

        void resize( int n, const cv::Size & sz ) {
            ROS_ASSERT( 0 < n && n < 100 );
            _size = n;
            _sImgs.resize(n);
            for ( int i = 0 ; i < n ; i++ ) {
                _sImgs[i].getImage(0).create( sz, CV_8UC3 );
                _sImgs[i].getImage(1).create( sz, CV_8UC3 );
            }
            _idx = -1;  // the index of the last image stored
        }

        int getCurrentIndex()   { return _idx; }
        int getIndex(const ros::Time & t) {
            int i, j;
            for ( i = 0, j = _idx ; i < _size ; i++ ) {
                if ( eqStamp(t, _sImgs[j]._stamp) ) break;
                j = (j - 1 + _size) % _size;
            }
            ROS_ASSERT( i != _size );
            return j;
        }

        // for stereo image access
        StereoImg & operator[](int n) {
            ROS_ASSERT(0 <= n && n < _size);
            return _sImgs[n];
        }
        
        StereoImg & getLastStereoImg( int n = 0 ) { // n:0, the last, n:1 the second last (previous)
            int idx = (_idx - n + _size) % _size;
            return _sImgs[idx];
        }

        // the same as the above function except getting the time stamp of the stereo image
        StereoImg & getLastStereoImg( ros::Time & t, int n = 0 ) { // n:0, the last, n:1 the second last
            int idx = (_idx - n + _size) % _size;
            t = _sImgs[idx]._stamp;
            return _sImgs[idx];
        }

        Mat & getLastImage( int v ) {
            ROS_ASSERT( 0 <= v && v < 2 ); // 0: left, 1: right
            return _sImgs[_idx].getImage(v);
        }

        // get an empty image slot for storing 
        Mat & getNextImageSlot( int v ) {   
            int idx = (_idx + 1) % _size;
            return _sImgs[idx].getImage(v);
        }

        // for stamping
        const Time & getLastStamp() {  return _sImgs[_idx].getStamp(); }

        void setNextStamp(Time & t) {
            _idx = (_idx + 1) % _size;
            _sImgs[_idx].setStamp(t);
        }

        void write(string & pathfn) {
            if (_size == 0) { ROS_INFO("Stereo Queue is empty"); return; }
            if ( !isValid(_sImgs[_idx].getStamp()) ) { ROS_INFO("Stereo Queue is not valid"); return; }

            int i, idx, from = -1, to = (_idx+1)%_size; 
            for ( i = 0, idx = (_idx-1+_size)%_size ; i < _size ; i++, idx=(idx-1+_size)%_size ) {
                if ( !isValid(_sImgs[idx].getStamp()) ) {
                    from = (idx+1)%_size;
                    break;
                }                
            }
            if ( from == -1 )   from = to;
            idx = from; i = 0;

            ROS_INFO_STREAM("write: size: " << _idx << "/" << _size << "    from, to: " << from << ", " << to );

            Mat simg( IMG_HEIGHT, IMG_WIDTH*2, CV_8UC3 );
            Mat limg = simg(Rect(0,0,IMG_WIDTH,IMG_HEIGHT));
            Mat rimg = simg(Rect(IMG_WIDTH,0, IMG_WIDTH,IMG_HEIGHT));
            
            do {
                ROS_INFO_STREAM(i << ",  " << idx);
                _sImgs[idx].getImage(0).copyTo(limg);
                _sImgs[idx].getImage(1).copyTo(rimg);
                putText(simg, makeString(_sImgs[idx].getStamp()), 
                        Point(5,150), FONT_HERSHEY_PLAIN, 1.0, Scalar(255,0,255));
                string fn = pathfn + to_string(i) + ".png";
                imwrite(fn, simg);
                idx = (idx+1)%_size;
                i++;
            } while ( idx != to );
            ROS_INFO_STREAM("write end: " << i << ",  " << idx);
        }
    };

    class MiniZEDWrapper {
        // SDK version
        int verMajor;
        int verMinor;
        int verSubMinor;

        ros::NodeHandle nhNs;
        std::thread devicePollThread;
        bool    _bRun;
        bool mStopNode;

        // Launch file parameters
        int resolution;
        int frameRate;        
        int gpuId;
        int zedId;        
        std::string svoFilepath;        
        bool verbose;

        // zed object
        sl::InitParameters param;
        sl::Camera zed;
        unsigned int serial_number;
        int userCamModel; // Camera model set by ROS Param
        sl::MODEL realCamModel; // Camera model requested to SDK

        // flags
        double matResizeFactor;
        int exposure;
        int gain;
        bool autoExposure;
        bool triggerAutoExposure;                

        // Frame and Mat
        int camWidth;
        int camHeight;
        int matWidth;
        int matHeight;        

        StereoImgCQ _simgs;        

        // Mutex
        std::mutex dataMutex;

    public:
        MiniZEDWrapper(ros::NodeHandle& pnh);
        ~MiniZEDWrapper() {
            ROS_INFO("MiniZEDWrapper descructor");
            shutdown();
        }
        void shutdown() {
            if (devicePollThread.joinable()) {
                _bRun = false;
                devicePollThread.join();
            }
        }
        void onInit();
        void device_poll();
        StereoImgCQ & getStereoImageQueue() { return _simgs; }
    };
}

#endif // _MINIZEDWRAPPER_