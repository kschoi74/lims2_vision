#include <lims2_vision/BulletTrackROS.h>
#include <nodelet/nodelet.h>

namespace lims2_vision
{
class BulletTrackNodelet : public nodelet::Nodelet
{
    public:
    BulletTrackNodelet() {}

    ~BulletTrackNodelet() {}

    private:
    virtual void onInit()
    {
        ROS_INFO("BulletTrackNodelet onInit()");
        btROS.reset(new BulletTrackROS(getNodeHandle(), getPrivateNodeHandle()));
    }

    boost::shared_ptr<BulletTrackROS> btROS;
};
}

#include <pluginlib/class_list_macros.h>
PLUGINLIB_DECLARE_CLASS(lims2_vision, BulletTrackNodelet, lims2_vision::BulletTrackNodelet, nodelet::Nodelet);