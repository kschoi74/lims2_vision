#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/Image.h>
#include <lims2_vision/bbox.h>
#include <vector>

namespace lims2_vision
{
    class HumanDetect
    {
    private:
        //constant tensor names for tensorflow object detection api
        const std::string IMAGE_TENSOR = "image_tensor:0";
        const std::string DETECTION_BOXES = "detection_boxes:0";
        const std::string DETECTION_SCORES = "detection_scores:0";
        const std::string DETECTION_CLASSES = "detection_classes:0";
        const std::string NUM_DETECTIONS = "num_detections:0";

    protected:
        float                   _threshold; 
        tensorflow::Session*    _session; 
        tensorflow::GraphDef    _graph_def;
        tensorflow::Tensor      _input_tensor;

    public:
        HumanDetect();
        ~HumanDetect();

        std::vector<lims2_vision::bbox> getBoundingBox(const sensor_msgs::ImageConstPtr& img_msg);
    };
}