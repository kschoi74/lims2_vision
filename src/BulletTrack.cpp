#include <lims2_vision/BulletTrack.h>
#include <lims2_vision/GunDetect.h>
#include <opencv2/highgui.hpp>

bool eqStamp(const ros::Time & lt, const ros::Time & rt)
{
    if ( lt.sec == rt.sec && lt.nsec == rt.nsec )
        return true;
    else
        return false;
}

using namespace lims2_vision;
using namespace cv;

BulletTrack::BulletTrack()    
{
    ROS_INFO( "BulletTrack Constructor!!" );    
    Rect initRect(-1,-1,-1,-1);
    _HumanROI[0] = initRect;
    _HumanROI[1] = initRect;
    _GunROI[0] = initRect;
    _GunROI[1] = initRect;

    _imgStoreIdx[0] = _imgStoreIdx[1] = 0;
    _img_ptrs[0][0] = _img_ptrs[0][1] = _img_ptrs[1][0] = _img_ptrs[1][1] = NULL;
    _imgPairing[0] = _imgPairing[1] = false;    
}

BulletTrack::~BulletTrack() {
}

bool BulletTrack::locateHuman(const lims2_vision::bbox::ConstPtr& msg)
{
    ROS_ASSERT( 0 <= msg->p && msg->p < 2 );
    boost::mutex::scoped_lock lock(_HumanROI_mutex[msg->p]);

    _HumanROI[msg->p] = Rect( msg->l, msg->t, msg->r - msg->l + 1, msg->b - msg->t + 1);
    Rect & cROI = _HumanROI[msg->p];
    ROS_INFO( "HROI[%d] [%d, %d, %d, %d]", msg->p, cROI.x, cROI.y, cROI.width, cROI.height );

    return true;
}

bool BulletTrack::setImage(const int view, const sensor_msgs::ImageConstPtr& img_msg)
{
    static int lcount = 0;
    static int rcount = 0;
    ROS_ASSERT( img_msg->width == 1280);
    ROS_ASSERT( img_msg->height == 720);

    bool paired = false;
    ////////
    int & imgStoreIdx = _imgStoreIdx[view];
    try {
        _img_ptrs[view][imgStoreIdx] = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exeception: %s", e.what());
        return false;
    }
    ////////

    int otherview = !view;
    auto curStamp = _img_ptrs[view][imgStoreIdx]->header.stamp;

    if ( _img_ptrs[otherview][imgStoreIdx] != NULL ) {
        paired = eqStamp(_img_ptrs[otherview][imgStoreIdx]->header.stamp, curStamp) ? true : false;
        _imgPairing[imgStoreIdx] = paired;
    }

    if ( view == 0 )    lcount++;
    else                rcount++;

    if ( lcount > 5 && rcount > 5 )
    {
        ROS_INFO("P: %p, %p, %p, %p", _img_ptrs[0][0]->image.data, _img_ptrs[0][1]->image.data, 
                                      _img_ptrs[1][0]->image.data, _img_ptrs[1][1]->image.data );
        ROS_INFO("S %d %d: %d, %d, %d, %d", _imgPairing[0], _imgPairing[1], _img_ptrs[0][0]->header.stamp.nsec, _img_ptrs[0][1]->header.stamp.nsec,
                                      _img_ptrs[1][0]->header.stamp.nsec, _img_ptrs[1][1]->header.stamp.nsec );                                      
    }

    ////////
    imgStoreIdx = !imgStoreIdx;
    ////////
    return paired;
}

void BulletTrack::locateGun()
{
    if ( _HumanROI[0].x == -1 || _HumanROI[1].x == -1 )
        return;
        
    int curImgIdx = !_imgStoreIdx[0];
    Mat & img_l = _img_ptrs[0][curImgIdx]->image;
    Mat & img_r = _img_ptrs[1][curImgIdx]->image;
    Rect left = Get_Gun(img_l, _HumanROI[0]);
    Rect right = Get_Gun(img_r, _HumanROI[1]);

    ROS_INFO("GUN L: %d %d %d %d", left.x, left.y, left.width, left.height);
    ROS_INFO("GUN R: %d %d %d %d", right.x, right.y, right.width, right.height);
}