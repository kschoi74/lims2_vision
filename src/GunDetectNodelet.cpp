#include <lims2_vision/GunDetectROS.h>
#include <nodelet/nodelet.h>

namespace lims2_vision
{
class GunDetectNodelet : public nodelet::Nodelet
{
    public:
    GunDetectNodelet() {}

    ~GunDetectNodelet() {}

    private:
    virtual void onInit()
    {
        ROS_INFO("GunDetectNodelet onInit()");
        hdROS.reset(new GunDetectROS(getNodeHandle(), getPrivateNodeHandle()));
    }

    ros::NodeHandle nhNs;
    boost::shared_ptr<GunDetectROS> hdROS;
};
}

#include <pluginlib/class_list_macros.h>
PLUGINLIB_DECLARE_CLASS(lims2_vision, GunDetectNodelet, lims2_vision::GunDetectNodelet, nodelet::Nodelet);