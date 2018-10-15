#include <lims2_vision/BulletTrack.h>
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

void maxLocStereo(cv::Mat * imgs, Point * Locs)
{
    cv::Mat sumMax(imgs[0].rows, 1, CV_32F);
    cv::Mat maxIndex;

    double maxVal[2];
    Point maxLoc;
    for (int r = 0; r < imgs[0].rows; r++)
    {
        minMaxLoc(imgs[0].row(r), NULL, &maxVal[0], NULL, &Locs[0]);
        minMaxLoc(imgs[1].row(r), NULL, &maxVal[1], NULL, &Locs[1]);
        sumMax.at<float>(r, 0) = maxVal[0] + maxVal[1];
        maxIndex.push_back(Vec2i(Locs[0].x, Locs[1].x));
    }
    minMaxLoc(sumMax, NULL, NULL, NULL, &maxLoc);
    Vec2i xx = maxIndex.at<Vec2i>(maxLoc.y, 0);
    Locs[0].x = xx(0);
    Locs[1].x = xx(1);
    Locs[0].y = Locs[1].y = maxLoc.y;
}

void minLocStereo(cv::Mat * imgs, Point * Locs)
{
    cv::Mat sumMin(imgs[0].rows, 1, CV_32F);
    cv::Mat minIndex;

    double minVal[2];
    Point minLoc;
    for (int r = 0; r < imgs[0].rows; r++)
    {
        minMaxLoc(imgs[0].row(r), &minVal[0], NULL, &Locs[0]);
        minMaxLoc(imgs[1].row(r), &minVal[1], NULL, &Locs[1]);
        sumMin.at<float>(r, 0) = minVal[0] + minVal[1];
        minIndex.push_back(Vec2i(Locs[0].x, Locs[1].x));
    }
    minMaxLoc(sumMin, NULL, NULL, &minLoc);
    Vec2i xx = minIndex.at<Vec2i>(minLoc.y, 0);
    Locs[0].x = xx(0);
    Locs[1].x = xx(1);
    Locs[0].y = Locs[1].y = minLoc.y;
}

BulletTrack::BulletTrack()    
{
    ROS_INFO( "BulletTrack Constructor!!" );    
    Rect initRect(-1,-1,-1,-1);

    _isHumanStable = false;
    _isGunStable = false;
    _humanDepth = 0.0;
    _humanROI[0] = initRect;
    _humanROI[1] = initRect;

    _gunROI[0] = initRect;
    _gunROI[1] = initRect;
    _isImageReady = false;    

    _simg_ptrs.clear();
    _img_ptrs[0].clear();
    _img_ptrs[1].clear();

    _state = _STATE_LOST;
    
    _BSR[0] = CoRects(IMG_RECT);
    _BSR[1] = CoRects(IMG_RECT);

    _trackImgs[0].create( Size(IMG_WIDTH, IMG_HEIGHT), CV_8UC3 );
    _trackImgs[1].create( Size(IMG_WIDTH, IMG_HEIGHT), CV_8UC3 );

    _gunLostCount = 0;
    _positionCount = 0;

    _predTraj = Mat(1, 3, CV_32F);

    _PG.setCamParams( KArr, Size(IMG_WIDTH, IMG_HEIGHT));
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

const ros::Time & BulletTrack::getLastStamp() const
{
    ROS_ASSERT( _simg_ptrs.size() );

    return _simg_ptrs.back()._stamp; 
}

bool BulletTrack::updateImagePtrs()
{
    if ( _img_ptrs[0].size() == 0 || _img_ptrs[1].size() == 0 ) return false;
    
    boost::mutex::scoped_lock lock(_stereo_mutex);
    bool res = false;
    
    const ros::Time & t = getLastStamp(0);
    
    if ( eqStamp( t, getLastStamp(1) ) ) {
        StereoImagePtrs simgPtr;
        simgPtr._ptrs[0] = _img_ptrs[0].back();
        simgPtr._ptrs[1] = _img_ptrs[1].back();
        simgPtr._stamp = t;
        _simg_ptrs.push_back(simgPtr);

        while ( !_img_ptrs[0].empty() ) _img_ptrs[0].pop_front();
        while ( !_img_ptrs[1].empty() ) _img_ptrs[1].pop_front();

        res = true;
    }

    // ready for bullet detection, but for bullet tracking three streo images are required
    if ( _simg_ptrs.size() == 2 )   
        _isImageReady = true;
    
    while ( _simg_ptrs.size() > 3 ) _simg_ptrs.pop_front();
    
    return res;
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

    return updateImagePtrs();
}

bool BulletTrack::setHumanROI(int v, const cv::Rect & humanROI)
{
    ROS_ASSERT( 0 <= v && v < 2 );    
    
    static float overlapRatio[2] = { -1.0f, -1.0f };
    
    boost::mutex::scoped_lock lock(_humanROI_mutex[v]);

    _isHumanStable = false;

    overlapRatio[v] = float((_humanROI[v] & humanROI).area()) / humanROI.area();

    _humanROI[v] = humanROI;
    
    if ( isValid(_humanROI[0]) && isValid(_humanROI[1]) ) {
        // stability check
        double d = _PG.depthFromDisparity( double(center(_humanROI[0]).x - center(_humanROI[1]).x) );
        if ( overlapRatio[0] > STABLE_LIMIT && overlapRatio[1] > STABLE_LIMIT &&  _state >= _STATE_HT ) // human motion is stable
        {
            _isHumanStable = true;
        }        
        _humanDepth = _humanDepth * 0.8 + d * 0.2;

        if ( _state < _STATE_HT ) {
            changeState( _STATE_HT );
            return _isHumanStable;    
        }
    }
    else {
        changeState( _STATE_LOST );        
    }        
    return _isHumanStable;
}

bool BulletTrack::detectGun()
{
    if ( ! condDetectGun() ) return false;

    boost::mutex::scoped_lock lock(_stereo_mutex);

    Point prvGunCtr[2] = { center(_gunROI[0]), center(_gunROI[1]) };

    string str = "detectGun: Stable: " + to_string(_isGunStable) + ", STATE: " + to_string(_state);    

    if ( _isGunStable == false )
    {
        bool gunDetected = false;
        if ( _state == _STATE_HT )
        {
            str += "         : Get_Gun -- STATE_HT  ";
            if ( (gunDetected = Get_Gun( getImage(CUR,0), getImage(CUR,1), 
                                        extendRectHor(_humanROI[0],150), extendRectHor(_humanROI[1],150),
                                        _gunROI[0], _gunROI[1] )) == true ) 
            {
                changeState( _STATE_GT ); 
                //drawStatus(_isHumanStable);
            }
        }
        else if ( _state >= _STATE_GT )   // _state >= _STATE_GT ; _STATE_GT, _STATE_BT
        {
            str += "         : Get_Gun -- STATE_GT  ";
            int ext[2] = { max( _gunROI[0].width, _gunROI[0].height ), max( _gunROI[1].width, _gunROI[1].height ) };
            if ( (gunDetected = Get_Gun( getImage(CUR,0), getImage(CUR,1), 
                                        extendRect(_gunROI[0], Size(ext[0],ext[0])), extendRect(_gunROI[1], Size(ext[1],ext[1])),
                                        _gunROI[0], _gunROI[1] )) == false )
            {
                _gunLostCount++;

                if ( _gunLostCount > 60 )
                    changeState( _STATE_LOST );
            }        
        }

        float d0 = dist(prvGunCtr[0], center(_gunROI[0]));
        float d1 = dist(prvGunCtr[1], center(_gunROI[1]));
        _isGunStable = d0 + d1 < 5.0f;
    }
    else
    {
        ROS_ASSERT( _state >= _STATE_GT );
        str += "         : Gun Mask Matching  ";

        Rect newROI[2] = { extendRect(_gunROI[0], Size(10,10)), extendRect(_gunROI[1], Size(10,10))};
        Mat result[2];
        matchTemplate( getImage(CUR,0)(newROI[0]), _gunMask[0], result[0], TM_SQDIFF );
        matchTemplate( getImage(CUR,1)(newROI[1]), _gunMask[1], result[1], TM_SQDIFF );        
        result[0] /= _gunROI[0].area();
        result[1] /= _gunROI[1].area();

        Point Locs[2];
        minLocStereo(result, Locs);
        float SQDiff[2] = { result[0].at<float>(Locs[0]), result[1].at<float>(Locs[1]) };
        
        if ( SQDiff[0] < 70.0f && SQDiff[1] < 70.0f )
        {
            // gun positions are refined
            _gunROI[0] += Locs[0] - Point(5,5);
            _gunROI[1] += Locs[1] - Point(5,5);
            float d0 = dist(prvGunCtr[0], center(_gunROI[0]));
            float d1 = dist(prvGunCtr[1], center(_gunROI[1]));
            _isGunStable = d0 + d1 < 5.0f;
        }
        else
            _isGunStable = false;

        str += to_string(SQDiff[0]) + makeString(Locs[0]) + "," + to_string(SQDiff[1]) + makeString(Locs[1]);
    }    
    
    if ( condDetectBullet() )
    {
        _gunMask[0] = getImage(CUR,0)(_gunROI[0]);
        _gunMask[1] = getImage(CUR,1)(_gunROI[1]);
        return true;
    }
    else
        return false;
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

void BulletTrack::predictBulletPosition()
{
    if ( _state == _STATE_GT )
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
            d0 = getStamp(CUR) - _stamp[nTraj-1];
            d1 = _stamp[nTraj-1] - _stamp[nTraj-2];

            predBulletPos = _traj[nTraj-1] + (_traj[nTraj-1] - _traj[nTraj-2]) * (double(d0.nsec) / double(d1.nsec));
        }
        else    // nTraj == 2
        {
            predBulletPos = _traj.back();
        }
        _PG.projPoint( _predBulletPos2[0], predBulletPos, 0 );
        _PG.projPoint( _predBulletPos2[1], predBulletPos, 1 );

        ROS_INFO_STREAM( "predBPos: " << makeString(predBulletPos) << makeString(_predBulletPos2[0]) << makeString(_predBulletPos2[1]));
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

int BulletTrack::getBulletROIImages()
{
    predictBulletPosition();

    list<Size> CoRectSizes;
    static int count = 0;

    if ( _state == _STATE_GT )
    {
        ROS_ASSERT( _gunROI[0].height == _gunROI[1].height );

        const int WR = 7;
        const int HR = 5;
        const float WRf = 3;
        const float HRf = 2;                
        
        CoRectSizes.push_back( Size(_gunROI[0].width*WR,  _gunROI[0].height*HR) );
        CoRectSizes.push_back( Size(_gunROI[0].width*WRf, _gunROI[0].height*HRf) );
        CoRectSizes.push_back( Size(_gunROI[0].width, _gunROI[0].height) );
        _BSR[0].setCoRects( CoRectSizes, _predBulletPos2[0] );

        CoRectSizes.clear();
        CoRectSizes.push_back( Size(_gunROI[1].width*WR,  _gunROI[1].height*HR) );
        CoRectSizes.push_back( Size(_gunROI[1].width*WRf, _gunROI[1].height*HRf) );
        CoRectSizes.push_back( Size(_gunROI[1].width, _gunROI[1].height) );
        _BSR[1].setCoRects( CoRectSizes, _predBulletPos2[1] );

        // left
        _bulletROIImg[0][CMV_L] = getImage(CUR, CMV_L)(_BSR[0].getRect(1)).clone(); // WRf, HRf are used 
        _bulletROIImg[1][CMV_L] = getImage(PRV, CMV_L)(_BSR[0].getRect(1)).clone(); 
        // right
        _bulletROIImg[0][CMV_R] = getImage(CUR, CMV_R)(_BSR[1].getRect(1)).clone(); 
        _bulletROIImg[1][CMV_R] = getImage(PRV, CMV_R)(_BSR[1].getRect(1)).clone(); 

        // difference
        _difBulletROIImg[0][CMV_L] = abs( _bulletROIImg[0][CMV_L] - _bulletROIImg[1][CMV_L] );
        _difBulletROIImg[0][CMV_R] = abs( _bulletROIImg[0][CMV_R] - _bulletROIImg[1][CMV_R] );        

        _ROIStamp[0] = getStamp(CUR);
        _ROIStamp[1] = getStamp(PRV);

        // string dstr;
        // dstr = OUT_FOLDER + "bullet" + to_string(count) + "_00.png";
        //rectangle( _bulletROIImg[0][CMV_L], _BSR[0].get0BaseRect(2), Scalar(0,0,255));
        //imwrite( dstr, getImage(CUR, CMV_L)(_BSR[0].getRect(0)).clone() );

        // dstr = OUT_FOLDER + "diff" + to_string(count) + "_0.png";
        //rectangle( _difBulletROIImg[CMV_L], _BSR[0].get0BaseRect(2), Scalar(0,0,255));
        //imwrite( dstr, _difBulletROIImg[0][CMV_L] );
    }
    else
    {
        ROS_ASSERT( _state == _STATE_BT );
        
        CoRectSizes.push_back( Size( 200, 200 ) );
        _BSR[0].setCoRects( CoRectSizes, _predBulletPos2[0] );
        _BSR[1].setCoRects( CoRectSizes, _predBulletPos2[1] );

        _bulletROIImg[0][CMV_L] = getImage(CUR, CMV_L)(_BSR[0].getRect(0)).clone();
        _bulletROIImg[1][CMV_L] = getImage(PRV, CMV_L)(_BSR[0].getRect(0)).clone();
        _bulletROIImg[2][CMV_L] = getImage(PR2, CMV_L)(_BSR[0].getRect(0)).clone();
        _bulletROIImg[0][CMV_R] = getImage(CUR, CMV_R)(_BSR[1].getRect(0)).clone();
        _bulletROIImg[1][CMV_R] = getImage(PRV, CMV_R)(_BSR[1].getRect(0)).clone();
        _bulletROIImg[2][CMV_R] = getImage(PR2, CMV_R)(_BSR[1].getRect(0)).clone();
        
        _difBulletROIImg[0][CMV_L] = abs( _bulletROIImg[0][CMV_L] - _bulletROIImg[1][CMV_L] );
        _difBulletROIImg[0][CMV_R] = abs( _bulletROIImg[0][CMV_R] - _bulletROIImg[1][CMV_R] );  
        _difBulletROIImg[1][CMV_L] = abs( _bulletROIImg[1][CMV_L] - _bulletROIImg[2][CMV_L] );
        _difBulletROIImg[1][CMV_R] = abs( _bulletROIImg[1][CMV_R] - _bulletROIImg[2][CMV_R] );  

        _ROIStamp[0] = getStamp(CUR);
        _ROIStamp[1] = getStamp(PRV);
        _ROIStamp[2] = getStamp(PR2);

        // string dstr;
        // dstr = OUT_FOLDER + "BT_bullet" + to_string(count) + "_00.png";
        // imwrite( dstr, _bulletROIImg[0][CMV_L] );      
        // dstr = OUT_FOLDER + "BT_diff" + to_string(count) + "_0.png";
        // imwrite( dstr, _difBulletROIImg[0][CMV_L] );
        // dstr = OUT_FOLDER + "BT_diff" + to_string(count) + "_1.png";
        // imwrite( dstr, _difBulletROIImg[1][CMV_L] );
    }
    count++;

    return count - 1;
}

void BulletTrack::detectBullet()
{
    if ( ! condDetectBullet() ) return;
    
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
        
        threshold(sch[i], blob[i], 60, 255, THRESH_BINARY);

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

    Locs[CMV_L] += _BSR[0].getRect(1).tl(); 
    Locs[CMV_R] += _BSR[1].getRect(1).tl(); 

    Point3d pos;
    pos = _PG.triangulate(Point2d(center(_gunROI[0])), Point2d(center(_gunROI[1])));
    _traj.push_back(pos);
    _stamp.push_back(_ROIStamp[0]);
    
    string str1 = "Gun Pos: " + makeString(pos);

    pos = _PG.triangulate(Point2d(Locs[CMV_L]), Point2d(Locs[CMV_R]));    
    _traj.push_back(pos);
    _stamp.push_back(_ROIStamp[1]);

    ROS_INFO_STREAM( str1 << "   Initial Bllet Pos: " << makeString(pos) << ", Gun: " << 
                     makeString(_gunROI[0]) << makeString(_gunROI[1]) <<
                     ", Bullet: " << makeString(Locs[0]) << makeString(Locs[1]) );
    
    _trackImgs[0] = getImage(CUR,0).clone();
    _trackImgs[1] = getImage(CUR,1).clone();

    changeState( _STATE_BT );
}

void preProcRegion(Mat & blob)
{
    static Mat SE3 = getStructuringElement(MORPH_RECT, Size(3,3));
    static Mat SE7 = getStructuringElement(MORPH_RECT, Size(7,7));
    erode( blob, blob, SE3 );
    dilate( blob, blob, SE7 );
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

bool BulletTrack::trackBullet()
{
    static int funcCount = 0;

    string str = "trackBullet: " + to_string(condTrackBullet()) + ", #: " + to_string(_simg_ptrs.size()) + " --  ";
    for ( auto & st : _simg_ptrs )
    {
        str += makeString(st._stamp) + " - ";
    }
    ROS_INFO_STREAM( str );

    if ( ! condTrackBullet() )  return false;

    Mat sch[2][2], blob[2][2];
    Mat blur[2][CMV_NUM];
    int i;
    int count;
    count = getBulletROIImages();    

    //ROS_ASSERT( funcCount < 2 );

    for ( i = CMV_L ; i < CMV_NUM ; i++) {
        int maxS;
        sumChannels<unsigned char>(sch[0][i], _difBulletROIImg[0][i]);
        sumChannels<unsigned char>(sch[1][i], _difBulletROIImg[1][i]);
        
        GaussianBlur(sch[0][i], blur[0][i], Size(3, 3), 1.0, 1.0);
        GaussianBlur(sch[1][i], blur[1][i], Size(3, 3), 1.0, 1.0);

        threshold(blur[0][i], blob[0][i], 60, 255, THRESH_BINARY);
        threshold(blur[1][i], blob[1][i], 60, 255, THRESH_BINARY);        

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
    Locs[0][CMV_L] += _BSR[0].getRect(0).tl(); 
    Locs[0][CMV_R] += _BSR[1].getRect(0).tl(); 

    ///
    Blob[1][0] = blob[0][0] & blob[1][0]; preProcRegion(Blob[1][0]);
    Blob[1][1] = blob[0][1] & blob[1][1]; preProcRegion(Blob[1][1]);

    getBulletSal( bSal[1][0], _bulletROIImg[1][0], (blur[0][0] + blur[1][0])/2, Blob[1][0] );
    getBulletSal( bSal[1][1], _bulletROIImg[1][1], (blur[0][1] + blur[1][0])/2, Blob[1][1] );

    maxLocStereo( bSal[1], Locs[1]);
    Locs[1][CMV_L] += _BSR[0].getRect(0).tl(); 
    Locs[1][CMV_R] += _BSR[1].getRect(0).tl(); 

    Blob[2][0] = blob[1][0] - blob[0][0]; preProcRegion(Blob[2][0]);
    Blob[2][1] = blob[1][1] - blob[0][1]; preProcRegion(Blob[2][1]);

    getBulletSal( bSal[2][0], _bulletROIImg[2][0], blur[1][0], Blob[2][0] );
    getBulletSal( bSal[2][1], _bulletROIImg[2][1], blur[1][1], Blob[2][1] );

    maxLocStereo( bSal[2], Locs[2]);
    Locs[2][CMV_L] += _BSR[0].getRect(0).tl(); 
    Locs[2][CMV_R] += _BSR[1].getRect(0).tl(); 

    if ( funcCount == 0 )
    {
        ROS_ASSERT( _traj.size() == 2 );
        _traj[0] = _PG.triangulate( Locs[2][0], Locs[2][1] );
        _traj[1] = _PG.triangulate( Locs[1][0], Locs[1][1] );
        _traj.push_back( _PG.triangulate( Locs[0][0], Locs[0][1] ));

        _stamp[0] = _ROIStamp[2];
        _stamp[1] = _ROIStamp[1];
        _stamp.push_back( _ROIStamp[0] );

        _bulletROIImg[0][0].copyTo(_trackImgs[0](_BSR[0].getRect(0)), Blob[0][0]);
        _bulletROIImg[1][0].copyTo(_trackImgs[0](_BSR[0].getRect(0)), Blob[1][0]);
        _bulletROIImg[2][0].copyTo(_trackImgs[0](_BSR[0].getRect(0)), Blob[2][0]);

        _bulletROIImg[0][1].copyTo(_trackImgs[1](_BSR[1].getRect(0)), Blob[0][1]);
        _bulletROIImg[1][1].copyTo(_trackImgs[1](_BSR[1].getRect(0)), Blob[1][1]);
        _bulletROIImg[2][1].copyTo(_trackImgs[1](_BSR[1].getRect(0)), Blob[2][1]);
    }
    else
    {
        _traj.push_back( _PG.triangulate( Locs[0][0], Locs[0][1]));
        _stamp.push_back( _ROIStamp[0] );

        _bulletROIImg[0][0].copyTo(_trackImgs[0](_BSR[0].getRect(0)), Blob[0][0]);
        _bulletROIImg[0][1].copyTo(_trackImgs[1](_BSR[1].getRect(0)), Blob[0][1]);
    }

    // string fn;
    // fn = OUT_FOLDER + "trackB_sch_" + to_string(count) + "_0.png";
    // imwrite( fn, sch[0][0] );
    // fn = OUT_FOLDER + "trackB_sch_" + to_string(count) + "_1.png";
    // imwrite( fn, sch[0][1] );
    // fn = OUT_FOLDER + "trackB_blur_" + to_string(count) + "_0.png";
    // imwrite( fn, blur[0][0] );
    // fn = OUT_FOLDER + "trackB_blur_" + to_string(count) + "_1.png";
    // imwrite( fn, blur[0][1] );
    // fn = OUT_FOLDER + "trackB_blob_" + to_string(count) + "_0.png";
    // imwrite( fn, blob[0][0] );
    // fn = OUT_FOLDER + "trackB_blob_" + to_string(count) + "_1.png";
    // imwrite( fn, blob[0][1] );

    // fn = OUT_FOLDER + "trackB_bSal_" + to_string(count) + "_00.png";
    // imwrite( fn, bSal[0][0] / 3 );
    // fn = OUT_FOLDER + "trackB_bSal_" + to_string(count) + "_01.png";
    // imwrite( fn, bSal[0][1] / 3 );

    // fn = OUT_FOLDER + "trackB_bSal_" + to_string(count) + "_10.png";
    // imwrite( fn, bSal[1][0] / 3 );
    // fn = OUT_FOLDER + "trackB_bSal_" + to_string(count) + "_11.png";
    // imwrite( fn, bSal[1][1] / 3 );

    // fn = OUT_FOLDER + "trackB_bSal_" + to_string(count) + "_20.png";
    // imwrite( fn, bSal[2][0] / 3 );
    // fn = OUT_FOLDER + "trackB_bSal_" + to_string(count) + "_21.png";
    // imwrite( fn, bSal[2][1] / 3 );

    // fn = OUT_FOLDER + "trackB_sch_" + to_string(count) + "_01.png";
    // imwrite( fn, sch[1][0] );
    // fn = OUT_FOLDER + "trackB_sch_" + to_string(count) + "_11.png";
    // imwrite( fn, sch[1][1] );
    // fn = OUT_FOLDER + "trackB_blur_" + to_string(count) + "_01.png";
    // imwrite( fn, blur[1][0] );
    // fn = OUT_FOLDER + "trackB_blur_" + to_string(count) + "_11.png";
    // imwrite( fn, blur[1][1] );
    // fn = OUT_FOLDER + "trackB_blob_" + to_string(count) + "_01.png";
    // imwrite( fn, blob[1][0] );
    // fn = OUT_FOLDER + "trackB_blob_" + to_string(count) + "_11.png";
    // imwrite( fn, blob[1][1] );

    string fn;
    // Mat imgl = getImage(CUR,0).clone();
    // Mat imgr = getImage(CUR,1).clone();

    // crossLine( imgl, Locs[0][0], Scalar(255,0,0));
    // crossLine( imgr, Locs[0][1], Scalar(255,0,0));

    // crossLine( imgl, Locs[1][0], Scalar(255,255,0));
    // crossLine( imgr, Locs[1][1], Scalar(255,255,0));

    // crossLine( imgl, Locs[2][0], Scalar(255,0,0));
    // crossLine( imgr, Locs[2][1], Scalar(255,0,0));

    // fn = OUT_FOLDER + "BLT_" + to_string(count) + "_0.png";
    // imwrite( fn, imgl );
    // fn = OUT_FOLDER + "BLT_" + to_string(count) + "_1.png";
    // imwrite( fn, imgr );

    funcCount++;

    if ( _traj.back().z < 1.0 )
    {
        predictTrajectory();
        for ( int i = 0 ; i < _predTraj.rows; i++ )
        {
            Point3d pt( _predTraj.at<float>(i,0), _predTraj.at<float>(i,1), _predTraj.at<float>(i,2));
            Point2d pt2;
            
            if ( pt.z < 0.1 ) break;

            _PG.projPoint( pt2, pt, 0 );
            crossLine( _trackImgs[0], Point(pt2), Scalar(255,0,0));

            _PG.projPoint( pt2, pt, 1 );
            crossLine( _trackImgs[1], Point(pt2), Scalar(255,0,0));
            
            ROS_INFO_STREAM( makeString( pt ));
        }
        fn = OUT_FOLDER + "TrackHist_" + to_string(funcCount) + "_0.png";
        imwrite( fn, _trackImgs[0]);
        fn = OUT_FOLDER + "TrackHist_" + to_string(funcCount) + "_1.png";
        imwrite( fn, _trackImgs[1]);
    }

    ROS_ASSERT( funcCount < 60 );

    return true;
}

bool BulletTrack::predictTrajectory()
{
    int i, j;
    int from = -1, to = _traj.size() - 1;

    for ( i = 0 ; i <= to ; i++ )
    {
        const Point3d & p = _traj[i];
        if ( 0.0 < p.z && p.z < 2.0 ) {
            from = i;
            break;                        
        }        
    }

    if ( from == -1 )
        ROS_INFO_STREAM( "Z : " + to_string(_traj.back().z ));

    ROS_ASSERT( from != -1 );
    int n = to - from + 1;
    Mat pts( n, 3, CV_32F );

    for ( i = from, j = 0 ; i <= to ; i++, j++ )
    {
        const Point3d & p = _traj[i];
        pts.at<float>( j, 0 ) = float( p.x );
        pts.at<float>( j, 1 ) = float( p.y );
        pts.at<float>( j, 2 ) = float( p.z );
    }

    Mat mpts, m0pts;
    reduce( pts, mpts, 0, REDUCE_AVG);
    m0pts = pts - repeat( mpts, n, 1 );

    PCA pc(m0pts, noArray(), PCA::DATA_AS_ROW);
    if ( pc.eigenvectors.at<float>(2,0) < 0 )
        pc.eigenvectors.col(0) *= -1.0f;

    Mat XY( n, 3, CV_32F);
    for ( j = 0 ; j < n ; j++ )
        pc.project( m0pts.row(j), XY.row(j) );
        

    Mat oX( n, 3, CV_32F, Scalar(1.0f));
    XY.col(0).copyTo(oX.col(1));
    oX.col(2) = XY.col(0).mul(XY.col(0));

    Mat oY(XY.col(1));
    Mat coefs = (oX.t() * oX).inv() * oX.t() * oY;

    float farXY = XY.at<float>(0,0);
    Mat newX = linspace(farXY, farXY - 5.0f, 100);
    Mat newX3(newX.rows, 3, CV_32F, Scalar(1.0f));
    newX.copyTo(newX3.col(1));
    newX3.col(2) = newX.mul(newX);
    Mat newY = newX3 * coefs;

    Mat newp(newX.rows, 3, CV_32F, Scalar(0.0f));
    newX.copyTo(newp.col(0));
    newY.copyTo(newp.col(1));

    _predTraj.resize( newX.rows );

    for ( j = 0 ; j < newX.rows; j++ )
    {
        Mat p = pc.backProject(newp.row(j)) + mpts;
        p.copyTo(_predTraj.row(j));        
    }

    return true;
}

cv::Point3d BulletTrack::predictTrajEnd()
{
    predictTrajectory();
    for ( int i = 0 ; i < _predTraj.rows ; i++ )
    {
        if ( _predTraj.at<float>(i,3) < 0.5 ) {
            Point3d pt( _predTraj.at<float>(i,0), _predTraj.at<float>(i,1), _predTraj.at<float>(i,2)); 
            return pt;
        }
    }
    ROS_ASSERT(0);
}

// cv::Point3d BulletTrack::predictTrajEnd(const cv::Point3d & p1, const cv::Point3d & p2)
// {
//     double t = Frame_dif / FPS; // [s]
// 	double limit_Range;

// 	if (p2.z >= 0.5)
// 		limit_Range = 0.5;
// 	else
// 		limit_Range = 0.0;

// 	t = t / 10;

// 	double kd = ((Cd * Rho * A) / (m * 2.)) * t;
// 	double gt = -(g * t);

// 	int Mat_count = 0;

// 	cv::Mat1d F(6, 6);
// 	cv::Mat1d mul_Mat(6, 6);

// 	cv::Mat1d v_(3, 1);

// 	v_.at<double>(0, 0) = (p2.x - p1.x) / t;
// 	v_.at<double>(1, 0) = (p2.y - p1.y) / t;
// 	v_.at<double>(2, 0) = (p2.z - p1.z) / t;

// 	v_ /= 10;

// 	for (Mat_count = 0; Mat_count < 999; Mat_count++)
// 	{
// 		if (Mat_count == 0)
// 		{
// 			Tpred_Mat[Mat_count].create(6, 1);
// 			Tpred_Mat[Mat_count].at<double>(0, 0) = p1.x;
// 			Tpred_Mat[Mat_count].at<double>(1, 0) = p1.y;
// 			Tpred_Mat[Mat_count].at<double>(2, 0) = p1.z;
// 			Tpred_Mat[Mat_count].at<double>(3, 0) = v_.at<double>(0, 0);
// 			Tpred_Mat[Mat_count].at<double>(4, 0) = v_.at<double>(1, 0);
// 			Tpred_Mat[Mat_count].at<double>(5, 0) = v_.at<double>(2, 0);

// 			for (int i = 0; i < 6; i++)
// 			{
// 				mul_Mat.at<double>(i, 0) = p1.x;
// 				mul_Mat.at<double>(i, 1) = p1.y;
// 				mul_Mat.at<double>(i, 2) = p1.z;
// 				mul_Mat.at<double>(i, 3) = v_.at<double>(0, 0);
// 				mul_Mat.at<double>(i, 4) = v_.at<double>(1, 0);
// 				mul_Mat.at<double>(i, 5) = v_.at<double>(2, 0);
// 			}
// 		}

// 		double v_norm = cv::norm(v_, cv::NORM_L2);
// 		F = 0.;

// 		F.at<double>(0, 0) = 1.;
// 		F.at<double>(1, 1) = 1.;
// 		F.at<double>(2, 2) = 1.;

// 		F.at<double>(0, 3) = t;
// 		F.at<double>(3, 3) = 1 - kd*v_norm;

// 		F.at<double>(1, 4) = t;
// 		F.at<double>(4, 4) = 1 - kd*v_norm;

// 		F.at<double>(2, 5) = t;
// 		F.at<double>(5, 5) = 1 - kd*v_norm;

// 		mul_Mat = F.mul(mul_Mat);

// 		cv::reduce(mul_Mat, Tpred_Mat[Mat_count + 1], 1, cv::REDUCE_SUM);

// 		Tpred_Mat[Mat_count + 1].at<double>(4, 0) += gt;

// 		if (Tpred_Mat[Mat_count + 1].at<double>(2, 0) < limit_Range)
// 			break;
// 		else
// 		{
// 			for (int i = 0; i < 6; i++)
// 			{
// 				mul_Mat.at<double>(i, 0) = Tpred_Mat[Mat_count + 1].at<double>(0, 0);
// 				mul_Mat.at<double>(i, 1) = Tpred_Mat[Mat_count + 1].at<double>(1, 0);
// 				mul_Mat.at<double>(i, 2) = Tpred_Mat[Mat_count + 1].at<double>(2, 0);
// 				mul_Mat.at<double>(i, 3) = Tpred_Mat[Mat_count + 1].at<double>(3, 0);
// 				mul_Mat.at<double>(i, 4) = Tpred_Mat[Mat_count + 1].at<double>(4, 0);
// 				mul_Mat.at<double>(i, 5) = Tpred_Mat[Mat_count + 1].at<double>(5, 0);
// 			}
// 		}

// 	}

// 	return cv::Point3d(Tpred_Mat[Mat_count].at<double>(0, 0), Tpred_Mat[Mat_count].at<double>(1, 0), -Tpred_Mat[Mat_count].at<double>(2, 0)); 
// }



/////////////
//
//
//
//

// bool BulletTrack::getBulletEnds(const Mat & bulletImg, Point * endPnts)
// {
//     ROS_ASSERT( endPnts != NULL );

//     int sum =0, count = 0;
//     int y = 0, x = 0;
//     int w = bulletImg.cols;
//     int h = bulletImg.rows;
//     bool res = false;

//     vector<Mat> chs;
//     split(bulletImg, chs);

//     threshold( chs[2], chs[2], 10, 1, THRESH_BINARY );
//     filter2D( chs[2], chs[0], chs[0].depth(), _bulletKernel );    // 25 - 4 = 21
//     threshold( chs[0], chs[0], 20, 255, THRESH_BINARY);         // detected if the circular neighbor is over 20

//     Mat colvec;
//     reduce( chs[0], colvec, 1, REDUCE_MAX);
    
//     for ( y = h - 1 ; y >= 0 ; y-- ) {
//         if ( colvec.at<uchar>(y,0) != 0 ) {
//             res = true;
//             break;
//         }
//     }

//     if ( res == false ) {
//         endPnts[0] = endPnts[1] = Point(-1, -1);
//         return res;
//     }

//     endPnts[0].y = y;    
//     for ( sum = 0, count = 0, x = 0 ; x < w ; x++ ) {
//         if ( chs[0].at<uchar>(endPnts[0].y, x) != 0 )
//         {
//             sum += x;
//             count++;
//         }
//     }
//     endPnts[0].x = sum / count;

//     //////
//     for ( y = 0 ; y < h ; y++ ) {
//         if ( colvec.at<uchar>(y,0) != 0 )
//             break;
//     }

//     if ( y == endPnts[0].y ) {
//         endPnts[1] = endPnts[0];
//         return res;
//     }

//     endPnts[1].y = y;
//     for ( sum = 0, count = 0, x = 0 ; x < w ; x++ ) {
//         if ( chs[0].at<uchar>(endPnts[1].y, x) != 0 )
//         {
//             sum += x;
//             count++;
//         }
//     }
//     endPnts[1].x = sum / count;
    
//     return res;
// }

void BulletTrack::draw(const int v, bool stable)
{
    Mat img = getImage(CUR, v).clone();
    string st = makeString(getLastStamp());

    rectangle( img, _humanROI[v], Scalar(0, 255, 0), 2);
    rectangle( img, _gunROI[v], Scalar(0, 0, 255), 2 );
    putText( img, getStateString() + "[" + st + "]", Point(30, 30),
            FONT_HERSHEY_PLAIN, 2.0, Scalar(0, 255, 0));    
    
    putText( img, "Human:" + makeString(_humanROI[v]) + makeString(_humanROI[!v]) + " STABLE: " + to_string(stable), Point(30, 60),
            FONT_HERSHEY_PLAIN, 2.0, Scalar(0, 255, 0));   
    putText( img, "Gun:" + makeString(_gunROI[v]) + makeString(_gunROI[!v])  + " STABLE: " + to_string(_isGunStable), Point(30, 90),
            FONT_HERSHEY_PLAIN, 2.0, Scalar(0, 255, 0));   
    
    if ( _state >= _STATE_GT )
    {
        double d = _PG.depthFromDisparity( center(_gunROI[0]).x - center(_gunROI[1]).x );
        putText( img, "Gun depth: " + to_string(d) + " m",
        Point(30, 120), FONT_HERSHEY_PLAIN, 2.0, Scalar(0, 255, 0));   
    }

    if ( _state == _STATE_BT )
    {
        crossLine( img, _bulletPos2[v], Scalar(255,0,0) );        
    }

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
        _gunLostCount = 0;
        break;
    case _STATE_HT:
        break;
    case _STATE_GT:
        _gunLostCount = 0;
        break;
    case _STATE_BT:
        break;   
    default:
        ROS_ASSERT(0);                     
    }
    //drawStatus();
}

void BulletTrack::drawStatus(bool stable)
{
    ROS_INFO( getStateString().c_str() );
    ROS_INFO( _lastmsg[0].c_str() );
    ROS_INFO( _lastmsg[1].c_str() );
    ROS_INFO( _lastmsg[2].c_str() );
    ROS_INFO( _lastmsg[3].c_str() );
    draw(0, stable);
    draw(1, stable);
}

string BulletTrack::getStateString()
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
    case _STATE_BT:
        statestr = "== STATE BULLET TRACKING ==";
        break;                
    default:
        ROS_ASSERT(0);
    }

    return statestr;
}

void BulletTrack::makeEventImage( int n )
{
    string str;
    _eventImgs[n][0] = getImage(CUR,0).clone();
    _eventImgs[n][1] = getImage(CUR,1).clone();
    if ( n == 0 )   // Bullet Detected
    {
        ROS_ASSERT( _traj.size() >= 2 );
        Point3d gunPos = _traj[0], bulPos = _traj[1];
        Point2d gunPt[2], bulPt[2];

        _PG.projPoint( gunPt[0], gunPos, 0 );
        _PG.projPoint( gunPt[1], gunPos, 1 );
        _PG.projPoint( bulPt[0], bulPos, 0 );
        _PG.projPoint( bulPt[1], bulPos, 1 );
        
        for ( int i = 0 ; i < 2 ; i++ )  {
            Mat & img = _eventImgs[n][i];
            str = "Bullet Detected";
            putText( img, str, Point(5, 30), FONT_HERSHEY_PLAIN, 1.0, Scalar(0,255,0));
            str = "Gun Pos: " + makeString(gunPos) + makeString(Point(gunPt[i]));
            putText( img, str, Point(5, 60), FONT_HERSHEY_PLAIN, 1.0, Scalar(0,255,0));
            str = "Bullet Pos: " + makeString(bulPos) + makeString(Point(bulPt[i]));
            putText( img, str, Point(5, 90), FONT_HERSHEY_PLAIN, 1.0, Scalar(0,255,0));
            crossLine(img, gunPt[i], Scalar(255,0,0));
            crossLine(img, bulPt[i], Scalar(255,0,0));
        }
    }
}

void BulletTrack::saveEventImages()
{
    string fn;
    fn = OUT_FOLDER + "det_0.png";
    imwrite( fn, _eventImgs[0][0] );
    fn = OUT_FOLDER + "det_1.png";
    imwrite( fn, _eventImgs[0][1] );
}
