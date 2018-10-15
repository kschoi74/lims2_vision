#include <lims2_vision/BulletTrackROS.h>

int main(int argc, char **argv){
  ros::init(argc, argv, "bullet_track");
  ros::NodeHandle n;
  ros::NodeHandle pnh("~");
  
  lims2_vision::BulletTrackROS hdROS(n, pnh);
  
  ros::spin();

  return 0;
}