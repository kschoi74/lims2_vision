#include <lims2_vision/HumanDetect.h>

using namespace lims2_vision;
using namespace tensorflow;

HumanDetect::HumanDetect() : _input_tensor(DT_UINT8, TensorShape({1, 720, 1280, 3})), _threshold(0.7) 
{
    // Initialize a tensorflow session
    ROS_INFO( "TF Session" );
    SessionOptions options = SessionOptions();
    options.config.mutable_gpu_options()->set_allow_growth(true);
    Status status = NewSession(options, &_session);
    if (!status.ok()) {
        ROS_INFO_STREAM( "TF Session failed " << status.ToString() << "\n" );
        return;
    }

    ROS_INFO( "TF Graph" );
    status = ReadBinaryProto(Env::Default(), ros::package::getPath("lims2_vision") + 
                "/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb", &_graph_def);
    if (!status.ok()) {
        ROS_INFO_STREAM( "TF Graph read failed " << status.ToString() << "\n" );
        return;
    }

    ROS_INFO( "TF Create" );
    status = _session->Create(_graph_def);
    if (!status.ok()) {
        ROS_INFO_STREAM( "TF Session Create failed " << status.ToString() << "\n" );
        return;
    }

    ros::Time old_t = ros::Time::now();
    std::vector<Tensor> outputs;
    auto run_status = _session->Run({{IMAGE_TENSOR, _input_tensor}},
                                    {DETECTION_BOXES, DETECTION_SCORES, DETECTION_CLASSES, NUM_DETECTIONS}, {}, &outputs);

    if (!run_status.ok()) {
        ROS_INFO_STREAM( "TF Initial Run - Failed to run interference model: " << run_status.ToString() );        
    }
    else {
        ros::Time t = ros::Time::now();
        ros::Duration tdiff = t - old_t;
        ROS_INFO_STREAM( "TF Initial Run : Duration " << tdiff.nsec / 1000000 << "ms" );
    }
}

HumanDetect::~HumanDetect() {
}

// TODO: header setting
std::vector<bbox> HumanDetect::getBoundingBox(const sensor_msgs::ImageConstPtr& img_msg)
{
    auto image = img_msg->data; // std::vector<uint8>
    const auto rows = img_msg->height;
    const auto cols = img_msg->width;
    
    auto image_data = _input_tensor.shaped<uint8_t, 3>({rows, cols, 3});
    int i = 0;
    for (auto y = 0 ; y < rows; y++)
        for (auto x = 0; x < cols; ++x)
            for (auto c = 0; c < 3; ++c, ++i)
                image_data(y,x,c) = image[i];    
    
    std::vector<Tensor> outputs;
    auto run_status = _session->Run({{IMAGE_TENSOR, _input_tensor}},
                                    {DETECTION_BOXES, DETECTION_SCORES, DETECTION_CLASSES, NUM_DETECTIONS}, {}, &outputs);

    if (!run_status.ok()) {
        ROS_INFO_STREAM( "Failed to run interference model: " << run_status.ToString() );        
    }

    const auto boxes_tensor = outputs[0].shaped<float, 2>({100, 4});     //shape={1, 100, 4}
    const auto scores_tensor = outputs[1].shaped<float, 1>({100});       //shape={1, 100}
    const auto classes_tensor = outputs[2].shaped<float, 1>({100});      //shape={1, 100}
    const auto num_detections_tensor = outputs[3].shaped<float, 1>({1}); //shape={1}
    
    //retrieve and format valid results
    std::vector<bbox> results;
    
    for(int i = 0; i < num_detections_tensor(0); ++i) {
        const float score = scores_tensor(i);
        const int label_index = classes_tensor(i);
        if (score < _threshold || label_index != 1) {
            continue;
        }
    
        bbox bbox_msg;
        bbox_msg.l = boxes_tensor(i, 1) * cols;
        bbox_msg.t = boxes_tensor(i, 0) * rows;
        bbox_msg.r = boxes_tensor(i, 3) * cols;
        bbox_msg.b = boxes_tensor(i, 2) * rows;
        results.push_back(bbox_msg);
    }
    ROS_INFO_STREAM("# human: " << results.size() << ", # object: " << num_detections_tensor(0) );

    return results;
}