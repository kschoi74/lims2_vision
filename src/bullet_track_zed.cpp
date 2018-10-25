#include <lims2_vision/lims2_vision_global.h>
#include <lims2_vision/MiniZEDWrapper.hpp>
#include <lims2_vision/HumanDetect.hpp>
#include <lims2_vision/BulletTracker.hpp>

#include <signal.h>

using namespace lims2_vision;

MiniZEDWrapper * pZED = NULL;
HumanDetector  * pHDT = NULL;
BulletTracker  * pBLT = NULL;

void mySigintHandler(int sig)
{
    if ( pBLT != NULL ) pBLT->shutdown();
    if ( pHDT != NULL ) pHDT->shutdown();
    if ( pZED != NULL ) pZED->shutdown();

    ros::shutdown();
}

int main(int argc, char **argv){
    ros::init(argc, argv, "bullet_track_zed");
    ros::NodeHandle pnh;    

    signal(SIGINT, mySigintHandler);

    StereoROI hRegion;

    MiniZEDWrapper btZED(pnh);
    HumanDetector hdt(btZED.getStereoImageQueue(), hRegion);
    BulletTracker blt(btZED.getStereoImageQueue(), hRegion);
    pZED = &btZED;
    pHDT = &hdt;
    pBLT = &blt;

    ros::spin();

    return 0;
}

