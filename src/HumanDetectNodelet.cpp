#include <lims2_vision/HumanDetectROS.h>
#include <nodelet/nodelet.h>

namespace lims2_vision
{
class HumanDetectNodelet : public nodelet::Nodelet
{
    public:
    HumanDetectNodelet() {}

    ~HumanDetectNodelet() {}

    private:
    virtual void onInit()
    {
        ROS_INFO("HumanDetectNodelet onInit()");
        hdROS.reset(new HumanDetectROS(getNodeHandle(), getPrivateNodeHandle()));
    }

    ros::NodeHandle nhNs;
    boost::shared_ptr<HumanDetectROS> hdROS;
};
}

#include <pluginlib/class_list_macros.h>
PLUGINLIB_DECLARE_CLASS(lims2_vision, HumanDetectNodelet, lims2_vision::HumanDetectNodelet, nodelet::Nodelet);