#include <lims2_vision/BulletTrack.h>
#include <lims2_vision/GunDetect.h>
#include <opencv2/highgui.hpp>

using namespace lims2_vision;
using namespace cv;
using namespace std;
using namespace ros;

BulletTrack::BulletTrack()    
{
    ROS_INFO( "BulletTrack Constructor!!" );    
    Rect initRect(-1,-1,-1,-1);
    _gunROI[0] = initRect;
    _gunROI[1] = initRect;
    _gunAIM[0] = false;
    _gunAIM[1] = false;
    _isImageReady = false;

    _img_ptrs[0].clear();
    _img_ptrs[1].clear();    
    _state = _STATE_LOST;
    _lostCount[0] = _lostCount[1] = 0;
    _bulletKernel = Mat(5, 5, CV_8UC1, 1);
    _bulletKernel.at<uchar>(0,0) = 0;
    _bulletKernel.at<uchar>(0,4) = 0;
    _bulletKernel.at<uchar>(4,0) = 0;
    _bulletKernel.at<uchar>(4,4) = 0;
    _positionCount = 0;

    _ITRS.create(3, 3);
    _ITRS = 0;
    _ITRS.at<double>(0, 0) = 696.692;
    _ITRS.at<double>(0, 2) = 670.839;
    _ITRS.at<double>(1, 1) = 696.692;
    _ITRS.at<double>(1, 2) = 352.867;
    _ITRS.at<double>(2, 2) = 1;

    _ITRS = _ITRS.inv();

    _baseline = 120.0;
}

BulletTrack::~BulletTrack() {
}

// private functions
const ros::Time & BulletTrack::getFirstStamp(const int v) const
{
    ROS_ASSERT( 0 <= v && v < 2 );
    ROS_ASSERT( _img_ptrs[v].size() );

    return _img_ptrs[v].front()->header.stamp; 
}

const ros::Time & BulletTrack::getLastStamp(const int v) const
{
    ROS_ASSERT( 0 <= v && v < 2 );
    ROS_ASSERT( _img_ptrs[v].size() );

    return _img_ptrs[v].back()->header.stamp; 
}

const ros::Time & BulletTrack::getPairStamp(const int i) const
{
    ROS_ASSERT( 0 <= i && i < 2 );
    ROS_ASSERT( 1 == _pairStamps.size() || _pairStamps.size() == 2 ); 
    
    return i ? _pairStamps.back() : _pairStamps.front();
}

bool BulletTrack::updatePairStamp()
{
    if ( _img_ptrs[0].size() == 0 || _img_ptrs[1].size() == 0 ) return false;
    
    bool res = false;
    const ros::Time & t = getLastStamp(0);
    if ( eqStamp( t, getLastStamp(1) ) )
    {
        _pairStamps.push_back(t);
        res = true;
    }

    if ( _pairStamps.size() == 2 )
        _isImageReady = true;

    while( _pairStamps.size() > 2 )
        _pairStamps.pop_front();
    
    return res;
}

void BulletTrack::updateImagePtrs()
{
    const auto & prvStamp = getPairStamp(0);   
    while ( getFirstStamp(0) != prvStamp ) _img_ptrs[0].pop_front();
    while ( getFirstStamp(1) != prvStamp ) _img_ptrs[1].pop_front();
}

// public functions
bool BulletTrack::setImage(const int view, const sensor_msgs::ImageConstPtr& img_msg)
{
    ROS_ASSERT( img_msg->width == IMG_WIDTH );
    ROS_ASSERT( img_msg->height == IMG_HEIGHT );

    try {
        _img_ptrs[view].push_back( cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::BGR8) );
    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exeception: %s", e.what());
        return false;
    }
    
    bool res = false;
    if ( updatePairStamp() == true )
    {
        res = true;
        updateImagePtrs();
    }
    
#ifdef _DEBUG_BT_    
    string str("View ");
    str += to_string(view) + ": ";

    for ( const auto & cvimg : _img_ptrs[view] )
    {
        str += to_string(cvimg->header.stamp.sec) + "." + to_string(cvimg->header.stamp.nsec) + " ";        
    }
    ROS_INFO( str.c_str() );

    str = "Pair: ";
    for ( const auto & stamp : _pairStamps )
        str += to_string(stamp.sec) + "." + to_string(stamp.nsec) + " ";        
    
    ROS_INFO( str.c_str() );    
#endif  // _DEBUG_BT_

    return res;
}

void BulletTrack::setGunROI(int v, const Rect & gunROI)
{
    ROS_ASSERT( 0 <= v && v < 2 );
    
    boost::mutex::scoped_lock lock(_gun_mutex[v]);

    bool gunDetected[2] = { isValid(_gunROI[0]), isValid(_gunROI[1]) };

    ROS_INFO_STREAM("BT::gunROI  " << v << "[" << gunDetected[0] << "," << gunDetected[1] << "," 
                    << isValid(gunROI) << "],  lostCount: [" << _lostCount[0] << "," << _lostCount[1]
                    << "],  gunAIM: [" << _gunAIM[0] << "," << _gunAIM[1] << "]  " << checkAim(gunROI) 
                    << " " << getStateString() );

    if ( isValid(gunROI) )        
    {
        _gunROI[v] = gunROI;
        _lostCount[v] = 0;
        _gunAIM[v] = checkAim(_gunROI[v]);

        if ( _gunAIM[0] && _gunAIM[1] && _state < _STATE_AIM )
            changeState( _STATE_AIM );
        else if ( gunDetected[!v] )
            changeState( _STATE_GT );
    }
    else
    {
        _lostCount[v]++;
        if ( _lostCount[v] > 150 ) // 1 second
        {
            _gunROI[v] = gunROI;    // -1, -1, -1, -1
            changeState( _STATE_LOST );
        }
    }
}

bool BulletTrack::isTrackingReady()
{    
    boost::mutex::scoped_lock lock0(_gun_mutex[0]);
    boost::mutex::scoped_lock lock1(_gun_mutex[1]);
    return _gunAIM[0] == true && _gunAIM[1] == true;
}

void BulletTrack::findBullet_sa()
{
    boost::mutex::scoped_lock lock0(_gun_mutex[0]);
    boost::mutex::scoped_lock lock1(_gun_mutex[1]);

    int ROI_range = 127;
    Point endPnts[2];
    static const Point ROIRange( ROI_range, ROI_range );
    static const Point ROI_L_LIMIT( 0, 0 );
    static const Point ROI_U_LIMIT( IMG_WIDTH, IMG_HEIGHT );
    static const Point ROI_U_LIMIT1( IMG_WIDTH-1, IMG_HEIGHT-1 );

    bool found[2] = {false, false};

    static int imgnum = 0;            
    string path1;   

    ROS_ASSERT( eqStamp(_pairStamps.front(), _img_ptrs[0].front()->header.stamp) ) ;
    ROS_ASSERT( eqStamp(_pairStamps.front(), _img_ptrs[1].front()->header.stamp) ) ;
    ROS_ASSERT( eqStamp(_pairStamps.back(), _img_ptrs[0].back()->header.stamp) ) ;
    ROS_ASSERT( eqStamp(_pairStamps.back(), _img_ptrs[1].back()->header.stamp) ) ;

    if ( _state < _STATE_BT )
    {
        // set the gun center as the initial bullet position
        _bulletPos2[0] = (_gunROI[0].tl() + _gunROI[0].br()) / 2;
        _bulletPos2[1] = (_gunROI[1].tl() + _gunROI[1].br()) / 2;
    }    

    Rect ROI0 = Rect( COORD_CLIP_L( _bulletPos2[0] - ROIRange, ROI_L_LIMIT ), 
                      COORD_CLIP_U( _bulletPos2[0] + ROIRange, ROI_U_LIMIT ) );
    
    Mat left_ROI = abs(_img_ptrs[0].front()->image(ROI0) - _img_ptrs[0].back()->image(ROI0));

    found[0] = getBulletEnds(left_ROI, endPnts);
//////
    Mat leftImg = _img_ptrs[0].back()->image.clone();
    left_ROI.copyTo(leftImg(ROI0));
    putText( leftImg, makeString(ROI0) + string(" ") + makeString(_bulletPos2[0]) + makeString(_img_ptrs[0].back()->header.stamp), 
                Point(30, 30), 
                FONT_HERSHEY_PLAIN, 2.0, Scalar(0, 255, 0));
    rectangle( leftImg, ROI0, Scalar(0,255,0) );
    rectangle( leftImg, _gunROI[0], Scalar(0, 255, 255));
    if ( found[0] )    
    {
        rectangle( leftImg, Rect(_bulletPos2[0] - ROIRange + endPnts[0], _bulletPos2[0] - ROIRange + endPnts[1]),
                    Scalar(255, 255, 0) );
    }
    path1 = OUT_FOLDER + string("fB0-") + std::to_string(imgnum) + ".png";
    imwrite( path1, leftImg );
///////

    if ( found[0] )
    {
        if ( _bulletPos2[0].y + endPnts[0].y - (ROI_range + 1) < _gunROI[0].y )
            _bulletPos2[0] = _bulletPos2[0] + endPnts[1] - ROIRange;
        else
            _bulletPos2[0] = _bulletPos2[0] + endPnts[0] - ROIRange;
        
        _bulletPos2[0] = COORD_CLIP(_bulletPos2[0], ROI_L_LIMIT, ROI_U_LIMIT1);        
    }    

    ////////
    Rect ROI1 = Rect( COORD_CLIP_L( _bulletPos2[1] - ROIRange, ROI_L_LIMIT ), 
                      COORD_CLIP_U( _bulletPos2[1] + ROIRange, ROI_U_LIMIT ) );
    
    Mat right_ROI = abs(_img_ptrs[1].front()->image(ROI1) - _img_ptrs[1].back()->image(ROI1));

    found[1] = getBulletEnds(right_ROI, endPnts);
    //////
    Mat rightImg = _img_ptrs[1].back()->image.clone();
    right_ROI.copyTo(rightImg(ROI1));
    putText( rightImg, makeString(ROI1) + string(" ") + makeString(_bulletPos2[1]) + makeString(_img_ptrs[1].back()->header.stamp), Point(30, 30), 
                FONT_HERSHEY_PLAIN, 2.0, Scalar(0, 255, 0));
    rectangle( rightImg, ROI1, Scalar(0,255,0) );
    rectangle( rightImg, _gunROI[1], Scalar(0, 255, 255));
    if ( found[1] )    
    {
        rectangle( rightImg, Rect(_bulletPos2[1] - ROIRange + endPnts[0], _bulletPos2[1] - ROIRange + endPnts[1]),
                    Scalar(255, 255, 0) );
    }
    path1 = OUT_FOLDER + string("fB1-") + std::to_string(imgnum) + ".png";
    imwrite( path1, rightImg );
    imgnum++;
///////
    if ( found[1] )
    {
        if ( _bulletPos2[1].y + endPnts[1].y - (ROI_range + 1) < _gunROI[1].y )
            _bulletPos2[1] = _bulletPos2[1] + endPnts[1] - ROIRange;
        else
            _bulletPos2[1] = _bulletPos2[1] + endPnts[0] - ROIRange;
        
        _bulletPos2[1] = COORD_CLIP(_bulletPos2[1], ROI_L_LIMIT, ROI_U_LIMIT1);        
    }    

    if ( found[0] && found[1] )
    {
        changeState( _STATE_BT );

        _positionCount = _positionCount < 1 + 2 ? _positionCount + 1 : _positionCount;
        _bulletPos3[_positionCount - 1] = calcPos3(Point(_bulletPos2[0].x, 720 - _bulletPos2[0].y ), 
                                               Point(_bulletPos2[1].x, 720 - _bulletPos2[1].y ) );
        if ( _bulletPos3[_positionCount - 1].z >= 3.75 || _bulletPos3[_positionCount - 1].z <= 0)                                               
        {
            _positionCount = 0;
        }
    }
    else
        _positionCount = 0;
    
    if ( _positionCount == 3 )
    {
        // pred_Mat = get_TrajPred( _bulletPos3[0], _bulletPos3[_positionCount - 1]);

        // for ( int i = 0 ; i < 2 ; i++ )
        //     _bulletPos3[i] = _bulletPos3[i + 1];

        // if ( pred_Mat.z >= -1.0 && pred_Mat.z <= 0)
        // {

        // }
    }
}

bool BulletTrack::getBulletEnds(const Mat & bulletImg, Point * endPnts)
{
    ROS_ASSERT( endPnts != NULL );

    int sum =0, count = 0;
    int y = 0, x = 0;
    int w = bulletImg.cols;
    int h = bulletImg.rows;
    bool res = false;

    vector<Mat> chs;
    split(bulletImg, chs);

    threshold( chs[2], chs[2], 10, 1, THRESH_BINARY );
    filter2D( chs[2], chs[0], chs[0].depth(), _bulletKernel );    // 25 - 4 = 21
    threshold( chs[0], chs[0], 20, 255, THRESH_BINARY);         // detected if the circular neighbor is over 20

    Mat colvec;
    reduce( chs[0], colvec, 1, REDUCE_MAX);
    
    for ( y = h - 1 ; y >= 0 ; y-- ) {
        if ( colvec.at<uchar>(y,0) != 0 ) {
            res = true;
            break;
        }
    }

    if ( res == false ) {
        endPnts[0] = endPnts[1] = Point(-1, -1);
        return res;
    }

    endPnts[0].y = y;    
    for ( sum = 0, count = 0, x = 0 ; x < w ; x++ ) {
        if ( chs[0].at<uchar>(endPnts[0].y, x) != 0 )
        {
            sum += x;
            count++;
        }
    }
    endPnts[0].x = sum / count;

    //////
    for ( y = 0 ; y < h ; y++ ) {
        if ( colvec.at<uchar>(y,0) != 0 )
            break;
    }

    if ( y == endPnts[0].y ) {
        endPnts[1] = endPnts[0];
        return res;
    }

    endPnts[1].y = y;
    for ( sum = 0, count = 0, x = 0 ; x < w ; x++ ) {
        if ( chs[0].at<uchar>(endPnts[1].y, x) != 0 )
        {
            sum += x;
            count++;
        }
    }
    endPnts[1].x = sum / count;
    
    return res;
}

Point3d BulletTrack::calcPos3(const Point & p1, const Point & p2)
{
    double y_av = (p1.y + p2.y) / 2.0;

	Mat1d L_Point(3, 3);
    Mat1d R_Point(3, 3);
	L_Point = 1;

	L_Point.at<double>(0, 0) = p1.x;
	L_Point.at<double>(0, 1) = y_av;
	L_Point.at<double>(1, 0) = p1.x;
	L_Point.at<double>(1, 1) = y_av;
	L_Point.at<double>(2, 0) = p1.x;
	L_Point.at<double>(2, 1) = y_av;
	
	R_Point = 1;

	R_Point.at<double>(0, 0) = p2.x;
	R_Point.at<double>(0, 1) = y_av;
	R_Point.at<double>(1, 0) = p2.x;
	R_Point.at<double>(1, 1) = y_av;
	R_Point.at<double>(2, 0) = p2.x;
	R_Point.at<double>(2, 1) = y_av;

	Mat1d mul_result(3, 3);
	Mat1d result(3, 1);

	multiply(_ITRS, L_Point, mul_result);

	reduce(mul_result, result, 1, REDUCE_SUM);

	double z = ( (((696.692 + 699.02) / 2.) * _baseline) / (p1.x - p2.x) ) / 1000;

	result = result * z;

	return Point3d(result.at<double>(0, 0), result.at<double>(1, 0), result.at<double>(2, 0));
}

void BulletTrack::draw(const int v)
{
    Mat img = _img_ptrs[v].back()->image.clone();

    string st = makeString( _img_ptrs[v].back()->header.stamp );

    rectangle( img, _gunROI[v], Scalar(0, 0, 255), 2 );
    putText( img, getStateString() + "[" + st + "]", Point(30, 30),
            FONT_HERSHEY_PLAIN, 2.0, Scalar(0, 255, 0));
    
    static int imgnum = 0;            
    string path1;
    path1 = OUT_FOLDER + std::to_string(v);
    path1 = path1 + "-" + std::to_string(imgnum) + getStateString();
    path1 = path1 + ".png";
    imwrite( path1, img );

    imgnum++;    
}

void BulletTrack::updateMsg(const std::string & str, int i)
{
    ROS_ASSERT( 0 <= i && i < 4 );
    _lastmsg[i] = str;
}

void BulletTrack::changeState(const STATE state)
{
    if ( _state == state )  return;

    _state = state;

    switch( state )
    {
    case _STATE_LOST:
        _lostCount[0] = _lostCount[1] = 0;
        _gunAIM[0] = _gunAIM[1] = false;
        break;
    case _STATE_GT:
        _lostCount[0] = _lostCount[1] = 0;
        break;
    case _STATE_AIM:        
        break;
    case _STATE_BT:
        break;   
    default:
        ROS_ASSERT(0);                     
    }

    ROS_INFO( getStateString().c_str() );
    ROS_INFO( _lastmsg[0].c_str() );
    ROS_INFO( _lastmsg[1].c_str() );
    ROS_INFO( _lastmsg[2].c_str() );
    ROS_INFO( _lastmsg[3].c_str() );
    draw(0);
    draw(1);
}

string BulletTrack::getStateString()
{
    string statestr;
    switch( _state )
    {
    case _STATE_LOST:
        statestr = "== STATE LOST ==";        
        break;
    case _STATE_GT:
        statestr = "== STATE GUN TRACKING ==";        
        break;
    case _STATE_AIM:
        statestr = "== STATE GUN AIMING ==";
        break;
    case _STATE_BT:
        statestr = "== STATE BULLET TRACKING ==";
        break;                
    default:
        ROS_ASSERT(0);
    }

    return statestr;
}
