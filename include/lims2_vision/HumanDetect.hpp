#ifndef _HUMANDETECT_
#define _HUMANDETECT_

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include <vector>
#include <list>
#include <ros/ros.h>
#include <ros/package.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <lims2_vision/lims2_vision_global.h>
#include <lims2_vision/MiniZEDWrapper.hpp>

namespace lims2_vision
{
    const int      HUMAN_LOST_LIMIT(1);

    class HumanInfo;
    class HumanInfoHistory;
    typedef std::vector<HumanInfo>::iterator        HIVIter;
    typedef std::list<HumanInfoHistory>::iterator   HIHLIter;

    class HumanInfo
    {
        float       _prob;  // probability of human        
        cv::Rect    _bBox;
        cv::Point2f _center;        

    public:
        HumanInfo(const float prob, const cv::Rect & bbox);

        float dist(const HumanInfo & human) const;
        float dist(const cv::Point2f & center) const;
        const cv::Point2f & center() const { return _center; }
        const cv::Rect & getROI() const { return _bBox; }
        float getProb() const { return _prob; }
    };    

    class HumanInfoHistory
    {
        std::list<HumanInfo>    _hiHist;
        float                   _prob;
        float                   _avgWidth;
        int                     _lostCount;
    
    public:
        HumanInfoHistory();
        bool update(const HumanInfo & hi);
        const HumanInfo & getLastHumanInfo() const;
        const cv::Point2f & getLastPosition() const;
        const std::list<HumanInfo> & getHumanInfos() const { return _hiHist; }
        bool findClosest(HIVIter & hiiter, std::vector<HumanInfo> & humanInfos);
        float getProb() const { return _prob; }
        float getWidth() const { return _avgWidth; }
        int   getLostCount() const { return _lostCount; }
        void incLostCount() { _lostCount++; }        
    };

    class HumanTracks
    {
        std::list<HumanInfoHistory>  _humanTracks;

    public:
        bool empty() const                          { return _humanTracks.empty(); }
        bool insert(const HumanInfoHistory & hih)   { _humanTracks.push_back(hih); return true; }
        
        bool linkClosestWithin(const HumanInfo & hi, const float radius);
        void update(std::vector<HumanInfo> & humanInfos);        
        void linkClosest(std::vector<HumanInfo> & humanInfos);        
        int purgeLostHumanInfoHistory();        
        cv::Rect getBestHumanROI() const;        
        void draw(cv::Mat & img, int camPos, bool write = false );
        std::string getStatus();
    };

    class HumanDetector
    {
        private:
        //constant tensor names for tensorflow object detection api
        const std::string IMAGE_TENSOR       = "image_tensor:0";
        const std::string DETECTION_BOXES    = "detection_boxes:0";
        const std::string DETECTION_SCORES   = "detection_scores:0";
        const std::string DETECTION_CLASSES  = "detection_classes:0";
        const std::string NUM_DETECTIONS     = "num_detections:0";
        std::thread _hDetectThread;
        bool        _bRun;

    protected:
        float                   _threshold; 
        tensorflow::Session*    _session; 
        tensorflow::GraphDef    _graph_def;
        tensorflow::Tensor      _input_tensor;
        
        HumanTracks             _humanTracks[2];
        std::vector<HumanInfo>  _humanROIs[2];        
        StereoImgCQ &           _sImgs;
        StereoROI &             _hROIs;                

    public:
        HumanDetector(StereoImgCQ & sImgs, StereoROI & hregion);
        ~HumanDetector();
        void shutdown();

        void detect();
        int detectHuman(const cv::Mat & img, std::vector<HumanInfo> & humanInfos);
        cv::Rect getBestHumanROI(int v) const;

        void buildHumanTracks(int v, std::vector<HumanInfo> & humanInfos);        
    };
}

#endif // HUMANDETECT_