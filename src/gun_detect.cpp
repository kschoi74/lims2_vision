#include <lims2_vision/GunDetectROS.h>

int main(int argc, char **argv){
  ros::init(argc, argv, "gun_detect");
  ros::NodeHandle n;
  ros::NodeHandle pnh("~");
  
  lims2_vision::GunDetectROS hdROS(n, pnh);
  
  ros::spin();

  return 0;
}