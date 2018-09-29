#include <lims2_vision/lims2_vision_global.h>
#include <lims2_vision/HumanDetect.h>
#include <opencv2/imgproc.hpp>

using namespace lims2_vision;
using namespace tensorflow;
using namespace cv;
using namespace std;

//////////////////////////////////
//
// HumanInfo
//
HumanInfo::HumanInfo(const float prob, const cv::Rect & bbox)
{
    _prob = prob;
    _bBox = bbox;
    _center = Point2f(_bBox.x + _bBox.width / 2.0f, _bBox.y + _bBox.height / 2.0f);
}

float HumanInfo::dist(const HumanInfo & human) const
{
    Point2f cdiff = center() - human.center();
    return sqrt(cdiff.x * cdiff.x + cdiff.y * cdiff.y);
}

float HumanInfo::dist(const cv::Point2f & center) const
{
    Point2f cdiff = _center - center;
    return sqrt(cdiff.x * cdiff.x + cdiff.y * cdiff.y);
}

//////////////////////////////////
//
// HumanInfoHistory
//
HumanInfoHistory::HumanInfoHistory() : _prob(0.0f), _avgWidth(0.0f), _lostCount(0)
{    
}

bool HumanInfoHistory::update(const HumanInfo & hi)
{
    _hiHist.push_back(hi);
    while (_hiHist.size() > 5)
        _hiHist.pop_front();

    float prob = 0.0f;
    float avgWidth = 0.0f;
    for ( const auto & hi : _hiHist ) {
        prob += hi.getProb();
        avgWidth += hi.getROI().width;
    }

    _prob = prob / _hiHist.size();
    _avgWidth = avgWidth / _hiHist.size();

    _prob += 1.0f - fabs( (FHD_SIZE.width / 2.0f - hi.center().x) / 1000.0f );
    _prob /= 2.0f;

    _lostCount = 0;

    return true;
}

const HumanInfo & HumanInfoHistory::getLastHumanInfo() const
{
    ROS_ASSERT( _hiHist.size() > 0 );
    return _hiHist.back();
}

const Point2f & HumanInfoHistory::getLastPosition() const
{
    ROS_ASSERT( _hiHist.size() > 0 );
    return getLastHumanInfo().center();    
}

bool HumanInfoHistory::findClosest(HIVIter & hiiter, vector<HumanInfo> & humanInfos)
{
    if ( humanInfos.size() == 0 ) return false;

    float minDist = FHD_SIZE.width + FHD_SIZE.height;
    const Point2f & center = getLastPosition();
    HIVIter it = humanInfos.begin();
    hiiter = it;

    for (int i = 0 ; i < humanInfos.size() ; i++, it++ )
    {
        float curDist = humanInfos[i].dist(center);
        if ( minDist > curDist )
        {            
            minDist = curDist;
            hiiter = it;
        }
    }

    if ( abs(center.x - hiiter->center().x) < _avgWidth ) {
        return true;
    }
    else {
#ifdef _DEBUG_HT_        
        ROS_INFO_STREAM("Human was found, but distant. HIH: (" << center.x << "," << center.y <<")  closest HI: ("
         << hiiter->center().x << "," << hiiter->center().y << ")");         
#endif         
        return false;
    }
}

//////////////////////////////////
//
// HumanTracks
//
// TODO: radius check. 
// if none is linked, return false.
bool HumanTracks::linkClosestWithin(const HumanInfo & hi, const float radius)
{
    ROS_ASSERT( _humanTracks.size() > 0 );

    list<HumanInfoHistory>::iterator it = _humanTracks.begin();
    list<HumanInfoHistory>::iterator minit = it;
    float minDist = it->getLastHumanInfo().dist(hi);
    it++;

    for ( ; it != _humanTracks.end() ; it++ )
    {
        float curDist = it->getLastHumanInfo().dist(hi);
        if ( minDist > curDist )
        {
            minDist = curDist;
            minit = it;
        }
    }

    minit->update(hi);

    return true;
}

void HumanTracks::update(vector<HumanInfo> & humanInfos)
{
    // link HI to a plausible HIH if possible (the closest within human width)
    // the linked HI is removed from humanInfos
    // the HIH that isn't updated increases the lost count.
#ifdef _DEBUG_HT_    
    ROS_INFO("HT::update"); 
#endif
    linkClosest(humanInfos);
    
    // remove HIHs whose lost count is large
    int nDeleted = purgeLostHumanInfoHistory();

#ifdef _DEBUG_HT_        
    if ( nDeleted ) ROS_INFO_STREAM("  HIH purged: " << nDeleted << "  -> " << _humanTracks.size() );
#endif

    // remaining humanInfos create new HIHs
    for ( const auto & hi : humanInfos)
    {
        HumanInfoHistory hiHist;
        hiHist.update(hi);
        insert(hiHist);
    } 

#ifdef _DEBUG_HT_            
    ROS_INFO_STREAM("    after creating HIHs " << _humanTracks.size());
#endif
}

struct em {
float        d;
    HIHLIter hih;
    int      hii;
};

bool compare_em( const em& first, const em& second) {
    return first.d < second.d;
}

void HumanTracks::linkClosest(vector<HumanInfo> & humanInfos)
{
#ifdef _DEBUG_HT_    
    ROS_INFO_STREAM("HT::linkClosest,  # of [HIHs, HIs] - " << _humanTracks.size() << ", " << humanInfos.size());
#endif    

    int nHIs = humanInfos.size();
    if ( nHIs == 0 ) {
        ROS_INFO("HT::link Human is not detected, increase all the lost counts");
        for ( auto & hih : _humanTracks ) 
            hih.incLostCount();
        return;
    }

    if ( _humanTracks.size() == 0 ) {
        ROS_INFO("HT::link There is no HumanInfoHistory, so skip linking");
        return;
    }
    
    list<em> linkCandidate;
    vector<bool> HIdelete(nHIs, false);
    int i;
    //float w;
    for ( HIHLIter hih = _humanTracks.begin() ; hih != _humanTracks.end() ; hih++ )
    {
        const HumanInfo & hi1 = hih->getLastHumanInfo();
        const float hi1x = hi1.center().x;
        const float avgWidth = hih->getWidth();
        //w = avgWidth;
        for ( i = 0 ; i < nHIs ; i++ )
        {
            float d = humanInfos[i].dist( hi1 );
            if ( abs(hi1x - humanInfos[i].center().x) > avgWidth )    continue;

            em e = { d, hih, i };
            linkCandidate.push_back(e);
        }
        hih->incLostCount();    // initially, increase all. Then, reset if it is updated
    }

    linkCandidate.sort(compare_em);    

    while ( ! linkCandidate.empty() )
    {
        em e = linkCandidate.front();
        (e.hih)->update( humanInfos[e.hii] );    // reset lost count within update()        
        linkCandidate.pop_front();        
        HIdelete[e.hii] = true;        

        bool needDelete = false;
        do {
            if ( linkCandidate.empty() )    break;

            needDelete = false;
            list<em>::iterator it;
            for ( it = linkCandidate.begin() ; it != linkCandidate.end() ; it++ )
            {
                if ( it->hih == e.hih || it->hii == e.hii ) {
                    needDelete = true;
                    linkCandidate.erase(it);
                    break;
                }
            }
        } while ( needDelete );
    }

    int nDelete = 0;
    int lastIdx = HIdelete.size() - 1;
    for ( i = 0 ; i < HIdelete.size() ; i++ )
    {
        if ( HIdelete[i] == true ) {
            humanInfos[i] = humanInfos[lastIdx];
            lastIdx--;
            nDelete++;
        }
    }

    if ( nDelete ) 
        humanInfos.erase( humanInfos.end() - nDelete, humanInfos.end() );
        
    ROS_ASSERT( nHIs-nDelete == humanInfos.size() );
#ifdef _DEBUG_HT_    
    ROS_INFO_STREAM("    HIs resized to " << humanInfos.size());
#endif
}

int HumanTracks::purgeLostHumanInfoHistory()
{
#ifdef _DEBUG_HT_    
    ROS_INFO_STREAM("HT::purge, # of HIHs - " << _humanTracks.size() << ", lostCount: " 
                    << _humanTracks.front().getLostCount());
#endif
    int nDelete = 0;
    
    bool needDelete = false;
    do {
        needDelete = false;
        if ( ! _humanTracks.empty() ) {
            for ( HIHLIter it = _humanTracks.begin() ; it != _humanTracks.end() ; it++ ) {
                if ( it->getLostCount() >= HUMAN_LOST_LIMIT ) {
                    needDelete = true;
                    _humanTracks.erase(it);                    
                    nDelete++;
                    break;
                }
            }
        }
    } while ( needDelete );
    
    return nDelete;
}

Rect HumanTracks::getBestHumanROI() const
{
    float maxProb = 0.0f;
    Rect hROI(-1, -1, -1, -1);

    for ( auto & hih : _humanTracks )
    {
        float curProb = hih.getProb();
        if ( maxProb < curProb )
        {
            hROI = hih.getLastHumanInfo().getROI();
            maxProb = curProb;
        }
    }

    return hROI;
}

string HumanTracks::getStatus()
{
    string str("# of tracks: ");
    str += to_string(_humanTracks.size());

    int i = 0;
    for ( auto & hih : _humanTracks )
    {
        str += " -- [" + to_string(i) + "]: ";
        for ( const auto & hi : hih.getHumanInfos() )    
        {
            const Rect & r = hi.getROI();
            str += "(" + to_string(r.x) + "," + to_string(r.y) + "," 
                   + to_string(r.width) + "," + to_string(r.height) + ") ";
        }
        i++;
    }
    return str;
}

void HumanTracks::draw(Mat & img, int camPos, bool write)
{
    // FOR DEBUGGING
    //ROS_INFO( getStatus().c_str() );

    const int INC = 40;
    for ( auto & hih : _humanTracks )
    {
        const HumanInfo & hil = hih.getLastHumanInfo();
        Point2f ctr = hil.center();

        putText( img, to_string(hih.getProb()),
            Point(ctr.x, ctr.y),
            FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 255, 0));

        const list<HumanInfo> & humanInfos = hih.getHumanInfos();
        int g = 255 - (humanInfos.size()-1) * INC;
        for ( const auto & hi : humanInfos )    
        {
            rectangle( img, hi.getROI(), Scalar(0, g, 0), 2 );
            g += INC;
        }
    }

    static int imgnum = 0;    
    if ( write == true ) {
        std::string path1;
        path1 = OUT_FOLDER + to_string(camPos);
        path1 = path1 + "-";
        path1 = path1 + to_string(imgnum);
        path1 = path1 + ".png";
        imwrite( path1, img );

        imgnum++;
    }
}


//////////////////////////////////
//
// HumanDetect
//
HumanDetect::HumanDetect() : _input_tensor(DT_UINT8, TensorShape({1, FHD_SIZE.height, FHD_SIZE.width, 3})), _threshold(0.85)
{
    // Initialize a tensorflow session
    SessionOptions options = SessionOptions();
    options.config.mutable_gpu_options()->set_allow_growth(true);
    Status status = NewSession(options, &_session);
    if (!status.ok()) {
        ROS_INFO_STREAM( "TF Session failed " << status.ToString() << "\n" );
        return;
    }

    status = ReadBinaryProto(Env::Default(), ros::package::getPath("lims2_vision") + 
                "/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb", &_graph_def);
    if (!status.ok()) {
        ROS_INFO_STREAM( "TF Graph read failed " << status.ToString() << "\n" );
        return;
    }

    status = _session->Create(_graph_def);
    if (!status.ok()) {
        ROS_INFO_STREAM( "TF Session Create failed " << status.ToString() << "\n" );
        return;
    }

    // The initial run can require much time.
    std::vector<Tensor> outputs;
    auto run_status = _session->Run({{IMAGE_TENSOR, _input_tensor}},
                                    {DETECTION_BOXES, DETECTION_SCORES, DETECTION_CLASSES, NUM_DETECTIONS}, {}, &outputs);

    if (!run_status.ok()) {
        ROS_INFO_STREAM( "TF Initial Run - Failed to run interference model: " << run_status.ToString() );        
    }
}

HumanDetect::~HumanDetect() { }

int HumanDetect::detectHuman(const sensor_msgs::ImageConstPtr& img_msg, vector<HumanInfo> & humanInfos)
{
    auto image = img_msg->data; // std::vector<uint8>
    const auto rows = img_msg->height;
    const auto cols = img_msg->width;

    ROS_ASSERT( rows == FHD_SIZE.height && cols == FHD_SIZE.width );
    
    auto image_data = _input_tensor.shaped<uint8_t, 3>({rows, cols, 3});
    int i = 0;
    for (auto y = 0 ; y < rows; y++)
        for (auto x = 0; x < cols; ++x)
            for (auto c = 0; c < 3; ++c, ++i)
                image_data(y,x,c) = image[i];    
    
    vector<Tensor> outputs;
    auto run_status = _session->Run({{IMAGE_TENSOR, _input_tensor}},
                                    {DETECTION_BOXES, DETECTION_SCORES, DETECTION_CLASSES, NUM_DETECTIONS}, {}, &outputs);

    if (!run_status.ok()) {
        ROS_INFO_STREAM( "Failed to run interference model: " << run_status.ToString() );        
    }

    const auto boxes_tensor    = outputs[0].shaped<float, 2>({100, 4}); //shape={1, 100, 4}
    const auto scores_tensor   = outputs[1].shaped<float, 1>({100});    //shape={1, 100}
    const auto classes_tensor  = outputs[2].shaped<float, 1>({100});    //shape={1, 100}
    const auto n_detect_tensor = outputs[3].shaped<float, 1>({1});      //shape={1}
    
    humanInfos.clear();

    //retrieve and format valid results
    for(int i = 0; i < n_detect_tensor(0); ++i) {
        const float score = scores_tensor(i);
        const int label_index = classes_tensor(i);
        if (score < _threshold || label_index != 1) {
            continue;
        }
    
        Rect bbox;
        bbox.x = boxes_tensor(i, 1) * cols;
        bbox.y = boxes_tensor(i, 0) * rows;
        bbox.width = boxes_tensor(i, 3) * cols - bbox.x;
        bbox.height = boxes_tensor(i, 2) * rows - bbox.y;
        
        HumanInfo hi( score, bbox );
        humanInfos.push_back(hi);
    }    

    return humanInfos.size();
}

void HumanDetect::buildHumanTracks(vector<HumanInfo> & humanInfos)
{
    _humanTracks.update( humanInfos );    
}

Rect HumanDetect::getBestHumanROI() const
{
    return _humanTracks.getBestHumanROI();
}

void HumanDetect::drawHumanTracks(cv::Mat & img, int camPos )
{
    _humanTracks.draw(img, camPos);
}

