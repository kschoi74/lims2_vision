#include <lims2_vision/HumanDetectROS.h>

int main(int argc, char **argv){
  ros::init(argc, argv, "human_detect");
  ros::NodeHandle n;
  ros::NodeHandle pnh("~");
  
  lims2_vision::HumanDetectROS hdROS(n, pnh);
  
  ros::spin();

  return 0;
}