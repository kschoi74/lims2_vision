#include <lims2_vision/BulletTracker.hpp>
#include <lims2_vision/GunDetect.h>
#include <lims2_vision/CoRects.h>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>


using namespace lims2_vision;
using namespace cv;
using namespace cv::ximgproc;
using namespace std;
using namespace ros;

void preProcRegion(Mat & blob)
{
    //static Mat SE3 = getStructuringElement(MORPH_RECT, Size(3,3));
    static Mat SE7 = getStructuringElement(MORPH_RECT, Size(7,7));
    Mat temp;
    //erode( blob, blob, SE3 );
    removeIsolPixel( blob, temp );
    dilate( temp, blob, SE7 );
}

void getBulletSal( Mat & bSal, const Mat & img, const Mat & difimg, const Mat & mask )
{
    Mat lab; 
    Mat a, b;
    Mat difimg32s, intimg;

    cvtColor( img, lab, COLOR_BGR2Lab);
    extractChannel( lab, a, 1 ); a.convertTo(a, CV_32S);
    extractChannel( lab, b, 2 ); b.convertTo(b, CV_32S);

    difimg.convertTo(difimg32s, CV_32S);

    add( a, b, intimg, mask );
    add( intimg, difimg, bSal, mask, CV_32S );
}

template<typename T>
void sumChannels(cv::Mat & dst, const cv::Mat & clrImg)
{
    typedef Vec<T, 3> VT;

    int nChs = clrImg.channels();
    assert(nChs >= 3);

    int w = clrImg.cols;
    int h = clrImg.rows;
    dst.create(h, w, CV_32F);

    MatConstIterator_<VT> it1 = clrImg.begin<VT>(), it1_end = clrImg.end<VT>();    
    MatIterator_<float> dst_it = dst.begin<float>();

    for (; it1 != it1_end; ++it1, ++dst_it) {
        VT pix1 = *it1;        
        *dst_it = pix1[0] + pix1[1] + pix1[2];
    }    
}

////////////////////////////////////
//
// BulletTracker
//
BulletTracker::BulletTracker(StereoImgCQ & simgs, StereoROI & hRegion, Lims2Controller & controller)    
: _sImgs(simgs),
  _humanROIsDetected(hRegion),
  _controller(controller)
{
    ROS_INFO( "//////////////////////////////////////////" );    
    ROS_INFO( "//" );    
    ROS_INFO( "//    BulletTracker Constructor !!" );    
    ROS_INFO( "//" );    
    ROS_INFO( "//////////////////////////////////////////" );    

    Rect initRect(-1,-1,-1,-1);

    _isHumanStable = false;
    _isGunStable = false;
    
    _gunROI[0] = initRect;
    _gunROI[1] = initRect;    

    _state = _STATE_LOST;
    
    _BSR[0] = CoRects(IMG_RECT);
    _BSR[1] = CoRects(IMG_RECT);

    _eventImgs.resize( 30, IMG_RECT.size() );
    _trackImgs[0].create( IMG_RECT.size(), CV_8UC3 );
    _trackImgs[1].create( IMG_RECT.size(), CV_8UC3 );

    _gunLostCount = 0;
    _gunStableCount= 0;    
    _cleanupCount = 0;

    _predTraj = Mat(1, 3, CV_32F);

    _humanROIs._ROI[0] = initRect;
    _humanROIs._ROI[1] = initRect;
    _humanROIs._stamp = Time(0,0);

    _PG.setCamParams( KArr, IMG_RECT.size());
    _trackThread = std::thread(&BulletTracker::process, this);
}

BulletTracker::~BulletTracker() {
    ROS_INFO("BulletTracker descructor");
    shutdown();    
}

void BulletTracker::shutdown() {
    ROS_INFO("BLT::shutdown");
    if (_trackThread.joinable()) {
        ROS_INFO("BLT::shutdown joining");
        string fn = OUT_FOLDER + "event";
        _eventImgs.write( fn );

        _bRun = false;
        _trackThread.join();
    }
}

void BulletTracker::process()
{
    ros::Rate loop_rate(60);
    _bRun = true;
    
    while ( ros::ok() && _bRun ) {
        
        if ( condCheckHumanROIs() ) {   
            checkHumanROIs();
        }        

        if ( _cleanupCount == 0 ) {
            detectGun();
            detectBullet();
            trackBullet();
            if ( _trackStartCount > 0 ) _trackStartCount--;

            if ( condPredictTrajectory() ) 
            { 
                predictTrajectoryNCleanup();
            }
        }
        else {
            ROS_ASSERT( _state == _STATE_CLEANUP );
            _cleanupCount--;
            if ( _cleanupCount == 0 ) {
                if ( _humanROIs.isValid() )
                    changeState( _STATE_HT );
                else
                    changeState( _STATE_LOST );
            }
        }
        
        static int rateWarnCount = 0;
        if (!loop_rate.sleep()) {
            rateWarnCount++;

            if (rateWarnCount == 10) {
                ROS_DEBUG_THROTTLE( 1.0, "Tracking thread takes much time");
                ROS_DEBUG_STREAM_THROTTLE(
                    1.0, "Expected cycle time: " << loop_rate.expectedCycleTime()
                    << " - Real cycle time: "    << loop_rate.cycleTime());                
            }
        } else {
            rateWarnCount = 0;
        }        
    }
    ROS_DEBUG("BulletTracker thread finished");
}

// public functions
bool BulletTracker::checkHumanROIs()
{
    //ROS_INFO_STREAM("checkHumanROIs : STATE: " << _state);
    static float overlapRatio[2] = { -1.0f, -1.0f };
    _isHumanStable = false;
    
    StereoROI hROIs;
    hROIs = _humanROIsDetected;
    _humanROIs = hROIs;
    
    if ( hROIs.isValid() ) {
        _humanROIs.getOverlapRatio( hROIs, overlapRatio );
        
        // stability check
        double d = _PG.depthFromDisparity( double(center(hROIs._ROI[0]).x - center(hROIs._ROI[1]).x) );
        if ( overlapRatio[0] > STABLE_LIMIT && overlapRatio[1] > STABLE_LIMIT &&  _state >= _STATE_HT ) // human motion is stable
        {
            _isHumanStable = true;
        }                

        if ( _state < _STATE_HT ) {
            _sImgStamp = hROIs._stamp;
            changeState( _STATE_HT );

            //ROS_INFO_STREAM("checkHumanROIs End: STATE: " << _state);
            return _isHumanStable;    
        }
    }
    else {
        changeState( _STATE_LOST );        
    }       

    //ROS_INFO_STREAM("checkHumanROIs End: STATE: " << _state);
    return _isHumanStable;
}

bool BulletTracker::detectGun()
{
    if ( ! condDetectGun() ) return false;

    ROS_ASSERT(_state >= _STATE_HT); 

    Rect ROI[2];
    Point prvGunCtr[2] = { center(_gunROI[0]), center(_gunROI[1]) };
    
    string str = "DG: STATE: " + to_string(_state) + ", Stable: " + to_string(_isHumanStable) + ", " + to_string(_isGunStable);    
    
    StereoImg & curSImg = _sImgs.getLastStereoImg(_sImgStamp);    // _sImgStamp is the time of the latest stereo image

    bool gunDetected = false;
    if ( ! (isValid(_gunROI[0]) && isValid(_gunROI[1])) ) {
        ROS_ASSERT( _humanROIs.isValid() );
        ROI[0] = extendRectHor(_humanROIs._ROI[0], 150);
        ROI[1] = extendRectHor(_humanROIs._ROI[1], 150);
    }
    else {
        ROS_ASSERT( isValid(_gunROI[0]) && isValid(_gunROI[1]) );
        int ext[2] = { max( _gunROI[0].width, _gunROI[0].height ), max( _gunROI[1].width, _gunROI[1].height ) };
        ROI[0] = extendRect(_gunROI[0], Size(ext[0],ext[0]));
        ROI[1] = extendRect(_gunROI[1], Size(ext[1],ext[1]));
    }

    // The state cannot be _LOST since _isHumanStable is true, but can be _HT, _GT, _AIM, _BT
    // if _BT, Tracker should change its state after the bullet disappears. consider _HT, _GT, _AIM
    gunDetected = Get_Gun( curSImg.getImage(0), curSImg.getImage(1), ROI[0], ROI[1], _gunROI[0], _gunROI[1] );
    float d0 = dist(prvGunCtr[0], center(_gunROI[0]));
    float d1 = dist(prvGunCtr[1], center(_gunROI[1]));
    _isGunStable = (gunDetected == true) && (d0 + d1 < 3.0f) 
                && _gunROI[0].width < _gunROI[0].height*1.3 
                && _gunROI[1].width < _gunROI[1].height*1.3;

    if ( _isGunStable )
        _gunStableCount++;
    else
        _gunStableCount = 0;

    if ( gunDetected == true ) {
        str += " detected : ";

        if ( _state == _STATE_HT ) {
            changeState( _STATE_GT );
        }    
        else { // should be _state > _STATE_HT, i.e. one of _STATE_GT, _STATE_AIM, _STATE_BT
            _gunLostCount = 0;

            if ( _state == _STATE_GT ) {
                if ( _gunStableCount > 40 )  // MIN_GUN_STABLE_COUNT: 40
                    changeState( _STATE_AIM );
            }
            else if ( _state == _STATE_AIM ) {
                if ( ! _isGunStable )
                    changeState( _STATE_GT );
            }
        }
    }
    else  {
        str += " not detected : ";
        // _gunROIs are invalid
        // _isGunStable is false.
        // if ( _state == _STATE_HT ) do nothing;
        if ( _state == _STATE_GT || _state == _STATE_AIM )
        {
            _gunLostCount++;
            str += "  gunLost # : " + to_string(_gunLostCount);                

            if ( _state == _STATE_AIM )
                changeState( _STATE_GT );

            if ( _gunLostCount > 30 ) 
                changeState( _STATE_HT );
        }            
    }

    // ROS_INFO_STREAM(str);

    // ROS_INFO_STREAM("  : STATE: " << _state << ", Stable: " << to_string(_isHumanStable) << 
    //                 ", " << to_string(_isGunStable) << "  Count: " << to_string(_gunStableCount) << 
    //                 ", " << to_string(_gunLostCount));

    if ( _isGunStable )
    {
        ROS_ASSERT( _state >= _STATE_GT );
        ROS_ASSERT( isValid(_gunROI[0]));
        ROS_ASSERT( isValid(_gunROI[1]));
        //ROS_INFO("GUN Mask generated");
        _gunMask[0] = curSImg.getImage(0)(_gunROI[0]);
        _gunMask[1] = curSImg.getImage(1)(_gunROI[1]);
        return true;
    }
    else
        return false;
}

bool BulletTracker::condTrackBullet() const { 
    bool newFrame = true;
    if ( _stamp.size() > 0 ) {
        const Time curT = getLastStamp();
        if ( eqStamp(_stamp.back(), curT) )
            newFrame = false;
    }
    return newFrame == true && _trackStartCount == 0 && _state == _STATE_BT; 
}

void BulletTracker::predictBulletPosition()
{
    if ( _state == _STATE_AIM )
    {
        ROS_ASSERT( isValid(_gunROI[0]) && isValid(_gunROI[1]) );
        _predBulletPos2[0] = center( _gunROI[0] );
        _predBulletPos2[1] = center( _gunROI[1] );
    }
    else 
    {
        ROS_ASSERT( _state == _STATE_BT );

        int nTraj = _traj.size();
        ROS_ASSERT( nTraj >= 2 );

        Point3d predBulletPos;
        Duration d0, d1;
        if ( nTraj > 2 )
        {
            d0 = getLastStamp() - _stamp[nTraj-1];
            d1 = _stamp[nTraj-1] - _stamp[nTraj-2];

            predBulletPos = _traj[nTraj-1] + (_traj[nTraj-1] - _traj[nTraj-2]) * (double(d0.nsec) / double(d1.nsec));
        }
        else    // nTraj == 2
        {
            predBulletPos = _traj.back();
        }
        _PG.projPoint( _predBulletPos2[0], predBulletPos, 0 );
        _PG.projPoint( _predBulletPos2[1], predBulletPos, 1 );

        ROS_INFO_STREAM( "predBPos: " << makeString(predBulletPos) << makeString(_predBulletPos2[0]) << makeString(_predBulletPos2[1]) 
                                << ", d0: " << makeString(d0) << ", d1: " << makeString(d1));
        string posStr; 
        for (int i = 0 ; i < nTraj ; i++ )
        {
            Point2d pt[2];
            _PG.projPoint( pt[0], _traj[i], 0 );
            _PG.projPoint( pt[1], _traj[i], 1 );
            posStr = "        : " + makeString( _traj[i]) + " " + makeString(Point(pt[0])) + " " + makeString(Point(pt[1])) + ": " + makeString( _stamp[i] );
            ROS_INFO_STREAM(posStr);
        }

        const double & z1 = _traj[nTraj-1].z;
        const double & z2 = _traj[nTraj-2].z;
        if ( 0.0 < z1 && z1 < 2.0 )
        {
            double z = 0.5;
            Duration d( 0, (z - z1) / (z1 - z2) * double(d1.nsec) );
            Time s = _stamp[nTraj-1];
            s += d;

            Point3d preTargetPos;
            d0 = s - _stamp[nTraj-1];
            preTargetPos = _traj[nTraj-1] + (_traj[nTraj-1] - _traj[nTraj-2]) * (double(d0.nsec) / double(d1.nsec));
            ROS_INFO_STREAM( "predTargetPos: " << makeString(preTargetPos));
        }
    }
}

int BulletTracker::getBulletROIImages()
{
    predictBulletPosition();

    list<Size> CoRectSizes;
    static int count = 0;

    if ( _state == _STATE_AIM )
    {
        const int WR = 7;
        const int HR = 5;
        const float WRf = 3;
        const float HRf = 2;                
        
        CoRectSizes.push_back( Size(_gunROI[0].width*WRf, _gunROI[0].height*HRf) );
        CoRectSizes.push_back( Size(_gunROI[0].width, _gunROI[0].height) );
        _BSR[0].setCoRects( CoRectSizes, _predBulletPos2[0] );

        CoRectSizes.clear();
        CoRectSizes.push_back( Size(_gunROI[1].width*WRf, _gunROI[1].height*HRf) );
        CoRectSizes.push_back( Size(_gunROI[1].width, _gunROI[1].height) );
        _BSR[1].setCoRects( CoRectSizes, _predBulletPos2[1] );
    }
    else
    {
        ROS_ASSERT( _state == _STATE_BT );
        
        CoRectSizes.push_back( Size( 200, 200 ) );
        _BSR[0].setCoRects( CoRectSizes, _predBulletPos2[0] );
        _BSR[1].setCoRects( CoRectSizes, _predBulletPos2[1] );
    }

    ROS_INFO_STREAM( "getBulletROIImages: area: " << makeString(_BSR[0].getRect(0)) << makeString(_BSR[1].getRect(0)) );
    if (!( _BSR[0].getRect(0).area() > 0 && _BSR[1].getRect(0).area() > 0 ))
    {   // reason: in trackBullet(), small values of frame difference result in no blob. (threshold is related)
        // then, the bullet position becomes zero in the x axis of the ROI region.
        // It makes the estimated bullet position go to the ceil.
        saveEventImages();
        // TODO:
        if ( _state == _STATE_BT ) {
            changeState( _STATE_LOST );
            return count;
        }
    }

    _sImgIdx[0] = _sImgs.getCurrentIndex();
    _sImgIdx[1] = (_sImgIdx[0]-1+_sImgs._size)%_sImgs._size;
    _sImgIdx[2] = (_sImgIdx[1]-1+_sImgs._size)%_sImgs._size;

    // current
    StereoImg & sImg0 = _sImgs[ _sImgIdx[0] ];
    _bulletROIImg[0][CMV_L] = sImg0.getImage(CMV_L)(_BSR[0].getRect(0)).clone();
    _bulletROIImg[0][CMV_R] = sImg0.getImage(CMV_R)(_BSR[1].getRect(0)).clone();
    // previous
    StereoImg & sImg1 = _sImgs[ _sImgIdx[1] ];
    _bulletROIImg[1][CMV_L] = sImg1.getImage(CMV_L)(_BSR[0].getRect(0)).clone();
    _bulletROIImg[1][CMV_R] = sImg1.getImage(CMV_R)(_BSR[1].getRect(0)).clone();
    // previous 2
    StereoImg & sImg2 = _sImgs[ _sImgIdx[2] ];
    _bulletROIImg[2][CMV_L] = sImg2.getImage(CMV_L)(_BSR[0].getRect(0)).clone();
    _bulletROIImg[2][CMV_R] = sImg2.getImage(CMV_R)(_BSR[1].getRect(0)).clone();
    
    _difBulletROIImg[0][CMV_L] = abs( _bulletROIImg[0][CMV_L] - _bulletROIImg[1][CMV_L] );
    _difBulletROIImg[0][CMV_R] = abs( _bulletROIImg[0][CMV_R] - _bulletROIImg[1][CMV_R] );  
    _difBulletROIImg[1][CMV_L] = abs( _bulletROIImg[1][CMV_L] - _bulletROIImg[2][CMV_L] );
    _difBulletROIImg[1][CMV_R] = abs( _bulletROIImg[1][CMV_R] - _bulletROIImg[2][CMV_R] );  

    _ROIStamp[0] = sImg0._stamp;
    _ROIStamp[1] = sImg1._stamp;
    _ROIStamp[2] = sImg2._stamp;
    ROS_ASSERT( !eqStamp(_ROIStamp[0], _ROIStamp[1]) );
    ROS_ASSERT( !eqStamp(_ROIStamp[1], _ROIStamp[2]) );

    count++;

    ROS_INFO("getBulletROIImages: end");

    return count - 1;
}

void BulletTracker::detectBullet()
{
    if ( ! condDetectBullet() ) return;
    ROS_ASSERT( _isGunStable == true );

    int i;
    cv::Mat sch[CMV_NUM];
    cv::Mat blob[2];
    cv::Mat labels[CMV_NUM];
    cv::Mat stats[CMV_NUM];
    cv::Mat centroids[CMV_NUM];

    int count;
    count = getBulletROIImages();    

    for ( i = CMV_L ; i < CMV_NUM ; i++) {
        int maxS;
        sumChannels<unsigned char>(sch[i], _difBulletROIImg[0][i]);
        
        threshold(sch[i], blob[i], 50, 255, THRESH_BINARY);

        blob[i].convertTo(blob[i], CV_8U);
        morphologyEx(blob[i], blob[i], MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(5,5)));
        connectedComponentsWithStats(blob[i], labels[i], stats[i], centroids[i]);
        if (stats[i].rows == 1)
            return;        
    }    

    // Bullet Detection Condition
    // The size of the largest blob is larger than gun_area / 10;
    int rowIdx[CMV_NUM];
    int maxArea;
    for (i = CMV_L; i < CMV_NUM; i++) {
        maxArea = 0;
        int maxIdx;
        for (int r = 1; r < stats[i].rows; r++)
        {
            if (stats[i].at<int>(r, 4) > maxArea)   // 4: AREA
            {
                maxArea = stats[i].at<int>(r, 4);
                maxIdx = r;
            }
        }
        if (maxArea < _gunROI[i].area() / 10 || maxArea > _gunROI[i].area() / 2)
            return;
        rowIdx[i] = maxIdx;
        
        blob[i].convertTo(blob[i], CV_8UC3);
    }

    // estimate the motion in each frame
    cv::Mat blur[CMV_NUM];
    GaussianBlur(sch[CMV_L], blur[CMV_L], Size(5, 5), 1.0, 1.0);
    GaussianBlur(sch[CMV_R], blur[CMV_R], Size(5, 5), 1.0, 1.0);

    Point Locs[CMV_NUM];
    maxLocStereo(blur, Locs);

    Locs[CMV_L] += _BSR[0].getRect(0).tl(); 
    Locs[CMV_R] += _BSR[1].getRect(0).tl(); 

    Point3d pos;
    pos = _PG.triangulate(Point2d(center(_gunROI[0])), Point2d(center(_gunROI[1])));
    _traj.push_back(pos);
    _stamp.push_back(_ROIStamp[1]);
    
    string str1 = "Gun Pos: " + makeString(pos);

    pos = _PG.triangulate(Point2d(Locs[CMV_L]), Point2d(Locs[CMV_R]));    
    _traj.push_back(pos);
    _stamp.push_back(_ROIStamp[0]);

    ROS_INFO_STREAM( str1 << "   Initial Bllet Pos: " << makeString(pos) << ", Gun: " << 
                     makeString(_gunROI[0]) << makeString(_gunROI[1]) << makeString(_ROIStamp[1]) <<
                     ", Bullet: " << makeString(Locs[0]) << makeString(Locs[1]) << makeString(_ROIStamp[0]));
    
    _trackImgs[0] = _sImgs[ _sImgIdx[0] ].getImage(0).clone();
    _trackImgs[1] = _sImgs[ _sImgIdx[0] ].getImage(1).clone();

    changeState( _STATE_BT );
}

bool BulletTracker::trackBullet()
{
    // string str = "trackBullet: " + to_string(condTrackBullet()) + "    F#: " + to_string(_sImgs.getCurrentIndex()) + "  " 
    //             + makeString(_sImgs.getLastStamp());
    //ROS_INFO_STREAM( str );

    if ( ! condTrackBullet() )  return false;

    Mat sch[2][2], blob[2][2];
    Mat blur[2][CMV_NUM];
    int i;
    int count;
    count = getBulletROIImages();    

    for ( i = CMV_L ; i < CMV_NUM ; i++) {
        int maxS;
        sumChannels<unsigned char>(sch[0][i], _difBulletROIImg[0][i]);
        sumChannels<unsigned char>(sch[1][i], _difBulletROIImg[1][i]);
        
        GaussianBlur(sch[0][i], blur[0][i], Size(3, 3), 1.0, 1.0);
        GaussianBlur(sch[1][i], blur[1][i], Size(3, 3), 1.0, 1.0);

        threshold(blur[0][i], blob[0][i], 30, 255, THRESH_BINARY);
        threshold(blur[1][i], blob[1][i], 30, 255, THRESH_BINARY);        

        blob[0][i].convertTo(blob[0][i], CV_8U);
        blob[1][i].convertTo(blob[1][i], CV_8U);
    }    

    Mat Blob[3][2];
    Mat bSal[3][2];
    Point Locs[3][CMV_NUM];

    Blob[0][0] = blob[0][0] - blob[1][0]; preProcRegion(Blob[0][0]);
    Blob[0][1] = blob[0][1] - blob[1][1]; preProcRegion(Blob[0][1]);
    
    getBulletSal( bSal[0][0], _bulletROIImg[0][0], blur[0][0], Blob[0][0] );
    getBulletSal( bSal[0][1], _bulletROIImg[0][1], blur[0][1], Blob[0][1] );
    
    maxLocStereo(bSal[0], Locs[0]);
    ///
    Blob[1][0] = blob[0][0] & blob[1][0]; preProcRegion(Blob[1][0]);
    Blob[1][1] = blob[0][1] & blob[1][1]; preProcRegion(Blob[1][1]);

    getBulletSal( bSal[1][0], _bulletROIImg[1][0], (blur[0][0] + blur[1][0])/2, Blob[1][0] );
    getBulletSal( bSal[1][1], _bulletROIImg[1][1], (blur[0][1] + blur[1][0])/2, Blob[1][1] );

    maxLocStereo( bSal[1], Locs[1]);
    ///
    Blob[2][0] = blob[1][0] - blob[0][0]; preProcRegion(Blob[2][0]);
    Blob[2][1] = blob[1][1] - blob[0][1]; preProcRegion(Blob[2][1]);

    getBulletSal( bSal[2][0], _bulletROIImg[2][0], blur[1][0], Blob[2][0] );
    getBulletSal( bSal[2][1], _bulletROIImg[2][1], blur[1][1], Blob[2][1] );

    maxLocStereo( bSal[2], Locs[2]);
    
    ///
    Point3d bulPos3;
    ROS_INFO_STREAM( "  Locs: " << makeString(Locs[2][CMV_L]) << ", " << makeString(Locs[2][CMV_R]) << ", " << makeString(_ROIStamp[2]) << "   " << // prev2
                                   makeString(Locs[1][CMV_L]) << ", " << makeString(Locs[1][CMV_R]) << ", " << makeString(_ROIStamp[1])  << "   " << // prev
                                   makeString(Locs[0][CMV_L]) << ", " << makeString(Locs[0][CMV_R]) << ", " << makeString(_ROIStamp[0])  );          // cur
    if ( Locs[0][CMV_L].x == 0 || Locs[0][CMV_R].x == 0 )
    {
        string fn = OUT_FOLDER + "bulletnd_ROI_00.png";
        imwrite(fn, _bulletROIImg[0][0]);
        fn = OUT_FOLDER + "bulletnd_ROI_01.png";
        imwrite(fn, _bulletROIImg[0][1]);
        fn = OUT_FOLDER + "bulletnd_ROI_10.png";
        imwrite(fn, _bulletROIImg[1][0]);
        fn = OUT_FOLDER + "bulletnd_ROI_11.png";
        imwrite(fn, _bulletROIImg[1][1]);
        fn = OUT_FOLDER + "bulletnd_ROI_20.png";
        imwrite(fn, _bulletROIImg[2][0]);
        fn = OUT_FOLDER + "bulletnd_ROI_21.png";
        imwrite(fn, _bulletROIImg[2][1]);
        fn = OUT_FOLDER + "bulletnd_blur_00.png";
        imwrite(fn, blur[0][0]);
        fn = OUT_FOLDER + "bulletnd_blur_01.png";
        imwrite(fn, blur[0][1]);
        fn = OUT_FOLDER + "bulletnd_blur_10.png";
        imwrite(fn, blur[1][0]);
        fn = OUT_FOLDER + "bulletnd_blur_11.png";
        imwrite(fn, blur[1][1]);
        fn = OUT_FOLDER + "bulletnd_blob_00.png";
        imwrite(fn, blob[0][0]);
        fn = OUT_FOLDER + "bulletnd_blob_01.png";
        imwrite(fn, blob[0][1]);
        fn = OUT_FOLDER + "bulletnd_blob_10.png";
        imwrite(fn, blob[1][0]);
        fn = OUT_FOLDER + "bulletnd_blob_11.png";
        imwrite(fn, blob[1][1]);
        fn = OUT_FOLDER + "bulletnd_Blob_00.png";
        imwrite(fn, Blob[0][0]);
        fn = OUT_FOLDER + "bulletnd_Blob_01.png";
        imwrite(fn, Blob[0][1]);
        fn = OUT_FOLDER + "bulletnd_bSal_00.png";
        imwrite(fn, bSal[0][0]);
        fn = OUT_FOLDER + "bulletnd_bSal_01.png";
        imwrite(fn, bSal[0][1]);
        return false;
    }

    Locs[0][CMV_L] += _BSR[0].getRect(0).tl(); 
    Locs[0][CMV_R] += _BSR[1].getRect(0).tl(); 

    if ( Locs[1][CMV_L].x != 0 && Locs[1][CMV_R].x != 0 )
    {
        Locs[1][CMV_L] += _BSR[0].getRect(0).tl(); 
        Locs[1][CMV_R] += _BSR[1].getRect(0).tl(); 

        bulPos3 = _PG.triangulate( Locs[1][CMV_L], Locs[1][CMV_R] );
        if ( eqStamp(_ROIStamp[1], _stamp.back()) ) {
            _traj.back() = bulPos3;
        }
        else {
            _traj.push_back( bulPos3 );
            _stamp.push_back( _ROIStamp[1] );
        }
    }
    
    Locs[2][CMV_L] += _BSR[0].getRect(0).tl(); 
    Locs[2][CMV_R] += _BSR[1].getRect(0).tl(); 
    ROS_INFO_STREAM( "  Locs: " << makeString(Locs[2][CMV_L]) << ", " << makeString(Locs[2][CMV_R]) << ", " << makeString(_ROIStamp[2]) << "   " << // prev2
                                   makeString(Locs[1][CMV_L]) << ", " << makeString(Locs[1][CMV_R]) << ", " << makeString(_ROIStamp[1])  << "   " << // prev
                                   makeString(Locs[0][CMV_L]) << ", " << makeString(Locs[0][CMV_R]) << ", " << makeString(_ROIStamp[0])  );          // cur

    bulPos3 = _PG.triangulate( Locs[0][0], Locs[0][1]);
    _traj.push_back( bulPos3 );
    _stamp.push_back( _ROIStamp[0] );

    _bulletROIImg[0][0].copyTo(_trackImgs[0](_BSR[0].getRect(0)), Blob[0][0]);
    _bulletROIImg[0][1].copyTo(_trackImgs[1](_BSR[1].getRect(0)), Blob[0][1]);

    ROS_INFO_STREAM("trackBullet End " << makeString(_traj.back()) << ": " << makeString(_ROIStamp[0]));
    return true;
}

void BulletTracker::predictTrajectoryNCleanup()
{
    static int funcCount = 0;

    if ( predictTrajectory(_predEndPoint) ) {
        float z_start = (_traj.front().z + _traj.back().z) / 2.0f;
        for ( int i = 0 ; i < _predTraj.rows; i++ )
        {
            Point3d pt( _predTraj.at<float>(i,0), _predTraj.at<float>(i,1), _predTraj.at<float>(i,2));
            Point2d pt2;
            if ( pt.z > z_start ) continue;
            if ( pt.z < 0.2 ) break;

            _PG.projPoint( pt2, pt, 0 );
            crossLine( _trackImgs[0], Point(pt2), Scalar(255,0,0));

            _PG.projPoint( pt2, pt, 1 );
            crossLine( _trackImgs[1], Point(pt2), Scalar(255,0,0));
            
            ROS_INFO_STREAM( makeString( pt ));
        }
        saveTrackHistImages(funcCount++);

        changeState( _STATE_CLEANUP );
    }    
}

void BulletTracker::saveTrackHistImages(int n)
{
    Time t = Time::now();
    Time t0 = _stamp.back();
    double dt = (double(t.nsec) - t0.nsec) / 1000000.0; // in msec
    double curZ = _traj.back().z - dt / 100.0;

    string fn = OUT_FOLDER + "TrackHist_" + to_string(n);
    string msg = "Last stamp: " + makeString(t0) + ", position: " + makeString( _traj.back());
    string msg1 = "Cur Time: " + makeString(t) + ", est. distance: "  + to_string(curZ);
    string predmsg = "predicted position: ";
    Point3d pt;
    int i;
    for ( i = 0 ; i < _predTraj.rows; i++ ) {
        if ( _predTraj.at<float>(i,2) < 0.5 ) {
            pt = Point3d( _predTraj.at<float>(i,0), _predTraj.at<float>(i,1), _predTraj.at<float>(i,2));
            break;
        }
    }
    predmsg += makeString(pt) + " after " + to_string((curZ - pt.z) * 100.0) + " msec";

    putText( _trackImgs[0], msg, Point(5,20), FONT_HERSHEY_PLAIN, 1.5, Scalar(0,0,255));
    putText( _trackImgs[0], msg1, Point(5,45), FONT_HERSHEY_PLAIN, 1.5, Scalar(0,0,255));
    putText( _trackImgs[0], predmsg, Point(5,70), FONT_HERSHEY_PLAIN, 1.5, Scalar(0,0,255));
    putText( _trackImgs[1], msg, Point(5,20), FONT_HERSHEY_PLAIN, 1.5, Scalar(0,0,255));
    putText( _trackImgs[1], msg1, Point(5,45), FONT_HERSHEY_PLAIN, 1.5, Scalar(0,0,255));
    putText( _trackImgs[1], predmsg, Point(5,70), FONT_HERSHEY_PLAIN, 1.5, Scalar(0,0,255));
    imwrite( fn + "_0.png", _trackImgs[0]);
    imwrite( fn + "_1.png", _trackImgs[1]);
}

bool BulletTracker::predictTrajectory(Point3d & predP)
{
    ROS_INFO_STREAM("predictTrajectory - Trajectory #: " << _traj.size());

    int i, j;
    int from = -1, to = _traj.size() - 1;
    
    for ( i = 0 ; i <= to ; i++ ) {
        const Point3d & p = _traj[i];
        if ( 0.0 < p.z && p.z < 2.3 ) {
            from = i;
            break;                        
        }        
    }

    if ( from == -1 ) {
        ROS_INFO_STREAM( "Z : " + to_string(_traj.back().z ));
        return false;
    }
    
    int n = to - from + 1;
    ROS_INFO_STREAM("predictTrajectory - Valid Trajectory #: " << n);
    if ( n < 3 )
        return false;

    // equi-acceleration 
    const double G2 = 9.80665 / 2.5;
    const float TargetZ = 0.4;

    Mat pts( n, 3, CV_32F );    
    double t0nsec = _stamp[from].nsec;
    double dt;
    for ( i = from, j = 0 ; i <= to ; i++, j++ )
    {
        const Point3d & p = _traj[i];
        const Time & t = _stamp[i];
        dt = (t.nsec - t0nsec) / 1000000000.0;
        if (dt < 0) dt += 1.0;        

        pts.at<float>( j, 0 ) = float( p.x );
        pts.at<float>( j, 1 ) = float( p.y ) - dt*dt*G2;
        pts.at<float>( j, 2 ) = float( p.z );        

        ROS_INFO_STREAM("   : " << makeString(p) << " " << makeString(t) << " - dt: " << dt << ", pts.y: " << float(float(p.y)-dt*dt*G2) );
    }

    Mat mpts, m0pts;
    reduce( pts, mpts, 0, REDUCE_AVG);
    m0pts = pts - repeat( mpts, n, 1 );
    ROS_INFO_STREAM("mpts: " << mpts.at<float>(0,0) << ", " << mpts.at<float>(0,1) << ", " << mpts.at<float>(0,2));

    PCA pc(m0pts, noArray(), PCA::DATA_AS_ROW);
    //ROS_ASSERT( pc.eigenvalues.at<float>(0) > pc.eigenvalues.at<float>(1) * 40);

    Mat V = pc.eigenvectors.row(0);
    float t1 = (TargetZ - mpts.at<float>(2)) / V.at<float>(2);

    ROS_INFO_STREAM("V: " << V.at<float>(0,0) << ", " << V.at<float>(0,1) << ", " << V.at<float>(0,2) << "   t1: " << t1);

    Mat newP = mpts + t1 * V;
    double dT1 = dt * (TargetZ - pts.at<float>(0,2))
                / (pts.at<float>(n-1,2) - pts.at<float>(0,2));
    double gdT12 = dT1*dT1*G2;

    newP.col(1) += gdT12;
    predP.x = newP.at<float>(0);
    predP.y = newP.at<float>(1);
    predP.z = newP.at<float>(2);

    Mat vt1 = linspace(1.5f, -1.5f, 100);
    _predTraj = repeat(mpts, vt1.rows, 1) + vt1 * V;
    Mat vdT1 = dt * ( _predTraj.col(2) - pts.at<float>(0,2)) 
                / (pts.at<float>(n-1,2) - pts.at<float>(0,2));
    Mat vgdT12 = vdT1.mul(vdT1) * G2;
    _predTraj.col(1) += vgdT12;
    
    return true;
}

void BulletTracker::changeState(const STATE state)
{
    if ( _state == state )  return;

    _state = state;

    switch( state )
    {
    case _STATE_LOST:
        _controller.manipulate(_state);
        _gunLostCount = 0;   
        _gunStableCount = 0;
        _cleanupCount = 0;     
        _isHumanStable = false;
        _isGunStable = false;
        _sImgStamp = _sImgs.getLastStamp();
        _gunROI[0] = _gunROI[1] = Rect(-1,-1,-1,-1);
        break;

    case _STATE_HT:
        _controller.manipulate(_state);
        _gunLostCount = 0; 
        _gunStableCount = 0;
        _cleanupCount = 0;       
        _isGunStable = false;
        _sImgStamp = _sImgs.getLastStamp();
        _gunROI[0] = _gunROI[1] = Rect(-1,-1,-1,-1);
        break;

    case _STATE_GT:       
        _controller.manipulate(_state);
        _gunStableCount = 0;         
        break;

    case _STATE_AIM:
        _controller.manipulate(_state);
        break;

    case _STATE_BT:
        _controller.manipulate(_state);
        _trackStartCount = 0;
        break;   

    case _STATE_CLEANUP:
        _controller.manipulate(_state, &_predEndPoint);
        _cleanupCount = 30; 
        _traj.clear();
        _stamp.clear();
        break;

    default:
        ROS_ASSERT(0);                     
    }    
    drawStatus(true);
}


////////////////////////////////
//
//
void BulletTracker::drawStatus(bool prnStable)
{
    prnStatus();
    draw(0, prnStable);
    draw(1, prnStable);
    _eventImgs.setNextStamp(_sImgStamp);
}

void BulletTracker::prnStatus()
{
    ROS_INFO( getStateString().c_str() );
    ROS_INFO( _lastmsg[0].c_str() );
    ROS_INFO( _lastmsg[1].c_str() );
    ROS_INFO( _lastmsg[2].c_str() );
    ROS_INFO( _lastmsg[3].c_str() );
}

void BulletTracker::draw(const int v, bool stable)
{
    int idx = _sImgs.getIndex(_sImgStamp);

    Mat img = _sImgs[idx].getImage(v).clone();
    string st = makeString(Time::now());

    rectangle( img, _humanROIs._ROI[v], Scalar(0, 255, 0), 2);
    if ( _state != _STATE_BT )
        rectangle( img, _gunROI[v], Scalar(0, 0, 255), 2 );

    putText( img, getStateString() + " [" + st + "]", Point(30, 30),
            FONT_HERSHEY_PLAIN, 1.5, Scalar(255, 0, 255));    
    
    putText( img, "Human:" + makeString(_humanROIs._ROI[v]) + makeString(_humanROIs._ROI[!v]) + 
                  " STABLE: " + to_string(stable), Point(30, 55),
                  FONT_HERSHEY_PLAIN, 1.5, Scalar(255, 0, 255));   
    putText( img, "Gun:" + makeString(_gunROI[v]) + makeString(_gunROI[!v])  + 
                  " STABLE: " + to_string(_isGunStable), Point(30, 80),
                  FONT_HERSHEY_PLAIN, 1.5, Scalar(255, 0, 255));   
    
    if ( _state >= _STATE_GT )
    {
        double d = _PG.depthFromDisparity( center(_gunROI[0]).x - center(_gunROI[1]).x );
        putText( img, "Gun depth: " + to_string(d) + " m",
        Point(30, 105), FONT_HERSHEY_PLAIN, 1.5, Scalar(0, 255, 0));   
    }

    static int imgnum = 0;            
    string path1;
    path1 = OUT_FOLDER + std::to_string(v);
    path1 = path1 + "-" + std::to_string(imgnum) + getStateString();
    path1 = path1 + ".png";
    //imwrite( path1, img );

    img.copyTo(_eventImgs.getNextImageSlot(v));    

    imgnum++;    
}

void BulletTracker::updateMsg(const std::string & str, int i)
{
    ROS_ASSERT( 0 <= i && i < 4 );
    _lastmsg[i] = str;
}

string BulletTracker::getStateString()
{
    string statestr;
    switch( _state )
    {
    case _STATE_LOST:
        statestr = "== STATE LOST ==";        
        break;
    case _STATE_HT:
        statestr = "== STATE HUMAN TRACKING ==";        
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
    case _STATE_CLEANUP:
        statestr = "== STATE CLEANUP ==";        
        break;                
    default:
        ROS_ASSERT(0);
    }

    return statestr;
}

void BulletTracker::makeEventImage( int n )
{
    // string str;
    // _eventImgs[n][0] = getImage(CUR,0).clone();
    // _eventImgs[n][1] = getImage(CUR,1).clone();
    // if ( n == 0 )   // Bullet Detected
    // {
    //     ROS_ASSERT( _traj.size() >= 2 );
    //     Point3d gunPos = _traj[0], bulPos = _traj[1];
    //     Point2d gunPt[2], bulPt[2];

    //     _PG.projPoint( gunPt[0], gunPos, 0 );
    //     _PG.projPoint( gunPt[1], gunPos, 1 );
    //     _PG.projPoint( bulPt[0], bulPos, 0 );
    //     _PG.projPoint( bulPt[1], bulPos, 1 );
        
    //     for ( int i = 0 ; i < 2 ; i++ )  {
    //         Mat & img = _eventImgs[n][i];
    //         str = "Bullet Detected";
    //         putText( img, str, Point(5, 30), FONT_HERSHEY_PLAIN, 1.0, Scalar(0,255,0));
    //         str = "Gun Pos: " + makeString(gunPos) + makeString(Point(gunPt[i]));
    //         putText( img, str, Point(5, 60), FONT_HERSHEY_PLAIN, 1.0, Scalar(0,255,0));
    //         str = "Bullet Pos: " + makeString(bulPos) + makeString(Point(bulPt[i]));
    //         putText( img, str, Point(5, 90), FONT_HERSHEY_PLAIN, 1.0, Scalar(0,255,0));
    //         crossLine(img, gunPt[i], Scalar(255,0,0));
    //         crossLine(img, bulPt[i], Scalar(255,0,0));
    //     }
    // }
}

void BulletTracker::saveEventImages()
{
    string fn = OUT_FOLDER + "ALERT_event";
    _eventImgs.write(fn);

    fn = OUT_FOLDER + "ALERT_sImgs";
    _sImgs.write(fn);

    ROS_INFO_STREAM(makeString(_traj.back()) << center(_BSR[0].getRect(0)) << center(_BSR[1].getRect(0)) );

    string dstr; Mat img;
    dstr = OUT_FOLDER + "ALERT_bullet_00.png";
    img = _bulletROIImg[0][CMV_L].clone(); imwrite( dstr, img );      
    dstr = OUT_FOLDER + "ALERT_bullet_01.png";
    img = _bulletROIImg[0][CMV_R].clone(); imwrite( dstr, img );
    dstr = OUT_FOLDER + "ALERT_bullet_10.png";
    img = _bulletROIImg[1][CMV_L].clone(); imwrite( dstr, img );      
    dstr = OUT_FOLDER + "ALERT_bullet_11.png";
    img = _bulletROIImg[1][CMV_R].clone(); imwrite( dstr, img );
    dstr = OUT_FOLDER + "ALERT_bullet_20.png";
    img = _bulletROIImg[2][CMV_L].clone(); imwrite( dstr, img );      
    dstr = OUT_FOLDER + "ALERT_bullet_21.png";
    img = _bulletROIImg[2][CMV_R].clone(); imwrite( dstr, img );

    dstr = OUT_FOLDER + "ALERT_diff_00.png";
    imwrite( dstr, _difBulletROIImg[0][CMV_L] );
    dstr = OUT_FOLDER + "ALERT_diff_01.png";
    imwrite( dstr, _difBulletROIImg[0][CMV_R] );
    dstr = OUT_FOLDER + "ALERT_diff_10.png";
    imwrite( dstr, _difBulletROIImg[1][CMV_L] );
    dstr = OUT_FOLDER + "ALERT_diff_11.png";
    imwrite( dstr, _difBulletROIImg[1][CMV_R] );     
}
