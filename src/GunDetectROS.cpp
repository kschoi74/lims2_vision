
#include <lims2_vision/lims2_vision_global.h>
#include <lims2_vision/GunDetectROS.h>
#include <lims2_vision/GunDetect.h>
#include <cv_bridge/cv_bridge.h>

using namespace lims2_vision;

GunDetectROS::GunDetectROS(ros::NodeHandle& n, ros::NodeHandle& pnh):_pnh(pnh), _it(n) {
    boost::mutex::scoped_lock lock(_connect_mutex);

    _pnh.getParam("position", _camPos);

    ROS_INFO_STREAM("GunDetectROS constructor!,  CAM# : " << _camPos);
    _pub = n.advertise<lims2_vision::bbox>("gbbox", 2, boost::bind(&GunDetectROS::connectCb, this, _1), 
                                                       boost::bind(&GunDetectROS::disconnectCb, this, _1));

    ros::Time::init();
    _prvT = ros::Time::now();

    _humanROIs.clear();
}

GunDetectROS::~GunDetectROS() {
    _sub.shutdown();
}

void GunDetectROS::imgCb(const sensor_msgs::ImageConstPtr& img_msg, 
                         const sensor_msgs::CameraInfoConstPtr& info_msg) 
{
    const ros::Duration T_THR(1,0);
    ros::Time start_t = ros::Time::now();
    ros::Duration tdiff = start_t - _prvT;
    
    cv_bridge::CvImageConstPtr imgPtr = cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::BGR8);

    // FOR DEBUGGING
    //ROS_INFO_STREAM("imgCb: [" << imgPtr->header.stamp.sec << "." << imgPtr->header.stamp.nsec );

    // For each humanROI, detect a gun
    Rect humanROI = _hdt.getBestHumanROI();
    Rect gunROI;
    bbox gbbox;
    gbbox.stamp = img_msg->header.stamp;
    gbbox.p = _camPos;

    if ( isValid(humanROI) ) {
        gunROI = Get_Gun(imgPtr->image, humanROI);
        //ROS_INFO( makeString(gunROI).c_str() );
        gbbox.p += 2;   // humanROI is valid
        gbbox.x = gunROI.x;
        gbbox.y = gunROI.y;
        gbbox.width = gunROI.width;
        gbbox.height = gunROI.height;        
    }
    else {
        gbbox.x = -1;
        gbbox.y = -1;
        gbbox.width = -1;
        gbbox.height = -1;        
    }

    _pub.publish(gbbox);        

    if (tdiff < T_THR)  return;
    
    // human detection using tensorflow
    try {
        int nHuman = _hdt.detectHuman(img_msg, _humanROIs);
    }
    catch (std::runtime_error& e) {
        ROS_ERROR_THROTTLE(1.0, "Could not detect human: %s", e.what());
    }
    _prvT = start_t;

    _hdt.buildHumanTracks( _humanROIs );
    
#ifdef _DEBUG_BT_    
    // Draw bounding boxes for human (green) and gun (red)
    Mat img = imgPtr->image.clone();
    _hdt.drawHumanTracks( img, _camPos );
    if ( isValid(gunROI) )
    {
        rectangle( img, gunROI, Scalar(0, 0, 255), 2 );
        static int imgnum = 0;            
        std::string path1;
        path1 = OUT_FOLDER + std::to_string(_camPos);
        path1 = path1 + "-";
        path1 = path1 + std::to_string(imgnum);
        path1 = path1 + ".png";
        imwrite( path1, img );

        imgnum++;
    }
#endif  // _DEBUG_BT_    
}

// if anyone connects to me, I subscribe camera
void GunDetectROS::connectCb(const ros::SingleSubscriberPublisher& pub) {
  boost::mutex::scoped_lock lock(_connect_mutex);
  if (!_sub && _pub.getNumSubscribers() > 0) {
    ROS_DEBUG("Connecting to image topic.");
    image_transport::TransportHints hints("raw", ros::TransportHints(), _pnh);
    _sub = _it.subscribeCamera("image", 2, &GunDetectROS::imgCb, this, hints);
  }
}

void GunDetectROS::disconnectCb(const ros::SingleSubscriberPublisher& pub) {
  boost::mutex::scoped_lock lock(_connect_mutex);
  if (_pub.getNumSubscribers() == 0) {
    ROS_DEBUG("Unsubscribing from image topic.");
    _sub.shutdown();
  }
}
