#include <lims2_vision/MiniZEDWrapper.hpp>
#include <lims2_vision/lims2_vision_global.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "sl_tools.h"

// >>>>> Backward compatibility
#define COORDINATE_SYSTEM_IMAGE static_cast<sl::COORDINATE_SYSTEM>(0)
#define COORDINATE_SYSTEM_RIGHT_HANDED_Z_UP                                    \
    static_cast<sl::COORDINATE_SYSTEM>(3)
#define COORDINATE_SYSTEM_RIGHT_HANDED_Z_UP_X_FWD                              \
    static_cast<sl::COORDINATE_SYSTEM>(5)
// <<<<< Backward compatibility

using namespace lims2_vision;

MiniZEDWrapper::MiniZEDWrapper(ros::NodeHandle& pnh)
{
    nhNs = pnh;
    onInit();
}

void MiniZEDWrapper::onInit()
{
    mStopNode = false;

    resolution = sl::RESOLUTION_HD720;
    frameRate = 60;
    gpuId = -1;
    zedId = 0;
    serial_number = 0;
    matResizeFactor = 1.0;

    param.camera_fps = frameRate;
    param.camera_resolution = static_cast<sl::RESOLUTION>(resolution);

    if (serial_number == 0) {
        param.camera_linux_id = zedId;
    } else {
        bool waiting_for_camera = true;
        while (waiting_for_camera) {

            if (!nhNs.ok()) {
                mStopNode = true; // Stops other threads
                zed.close();
                ROS_DEBUG("ZED pool thread finished");
                return;
            }

            sl::DeviceProperties prop = sl_tools::getZEDFromSN(serial_number);
            if (prop.id < -1 ||
                prop.camera_state == sl::CAMERA_STATE::CAMERA_STATE_NOT_AVAILABLE) {
                std::string msg = "ZED SN" + to_string(serial_number) +
                                  " not detected ! Please connect this ZED";
                ROS_INFO_STREAM(msg.c_str());
                std::this_thread::sleep_for(std::chrono::milliseconds(2000));
            } else {
                waiting_for_camera = false;
                param.camera_linux_id = prop.id;
            }
        } 
    }

    std::string ver = sl_tools::getSDKVersion(verMajor, verMinor, verSubMinor);
    ROS_INFO_STREAM("SDK version : " << ver);

    param.coordinate_system = COORDINATE_SYSTEM_RIGHT_HANDED_Z_UP_X_FWD;
    param.coordinate_units = sl::UNIT_METER;

    param.sdk_verbose = verbose;
    param.sdk_gpu_id = gpuId;    

    sl::ERROR_CODE err = sl::ERROR_CODE_CAMERA_NOT_DETECTED;
    while (err != sl::SUCCESS) {
        err = zed.open(param);
        ROS_INFO_STREAM(toString(err));
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));

        if (!nhNs.ok()) {
            mStopNode = true; // Stops other threads
            zed.close();
            ROS_DEBUG("ZED pool thread finished");
            return;
        }
    }

    realCamModel = zed.getCameraInformation().camera_model;

    std::string camModelStr = "LAST";
    if (realCamModel == sl::MODEL_ZED) {
        camModelStr = "ZED";
        if (userCamModel != 0) {
            ROS_WARN("Camera model does not match user parameter. Please modify "
                          "the value of the parameter 'camera_model' to 0");
        }
    } else if (realCamModel == sl::MODEL_ZED_M) {
        camModelStr = "ZED M";
        if (userCamModel != 1) {
            ROS_WARN("Camera model does not match user parameter. Please modify "
                          "the value of the parameter 'camera_model' to 1");
        }
    }

    ROS_INFO_STREAM("CAMERA MODEL : " << realCamModel);

    serial_number = zed.getCameraInformation().serial_number;

    // Start pool thread
    devicePollThread = std::thread(&MiniZEDWrapper::device_poll, this);
}

void MiniZEDWrapper::device_poll()
{
    ros::Rate loop_rate(frameRate);

    ros::Time old_t =
        sl_tools::slTime2Ros(zed.getTimestamp(sl::TIME_REFERENCE_CURRENT));    

    sl::ERROR_CODE grab_status;    

    // Get the parameters of the ZED images
    camWidth = zed.getResolution().width;
    camHeight = zed.getResolution().height;
    ROS_INFO_STREAM("Camera Frame size : " << camWidth << "x" << camHeight);

    matWidth = static_cast<int>(camWidth * matResizeFactor);
    matHeight = static_cast<int>(camHeight * matResizeFactor);
    ROS_DEBUG_STREAM("Data Mat size : " << matWidth << "x" << matHeight);

    cv::Size cvSize(matWidth, matWidth);
    _simgs.resize( 30, cvSize );  // 30 stereo images    
    
    zed.setCameraSettings(sl::CAMERA_SETTINGS_EXPOSURE, 0, true);

    sl::RuntimeParameters runParams;
    runParams.sensing_mode = sl::SENSING_MODE_STANDARD;
    runParams.enable_point_cloud = false;
    runParams.enable_depth = false;

    sl::Mat leftZEDMat, rightZEDMat;
    _bRun = true;    

    // Main loop
    while ( ros::ok() && _bRun ) {
            
        // Timestamp
        ros::Time t =
            sl_tools::slTime2Ros(zed.getTimestamp(sl::TIME_REFERENCE_IMAGE));

        grab_status = zed.grab(runParams); 
        
        if (grab_status !=
            sl::ERROR_CODE::SUCCESS) { // Detect if a error occurred (for example:
            // the zed have been disconnected) and
            // re-initialize the ZED

            if (grab_status == sl::ERROR_CODE_NOT_A_NEW_FRAME) {
                ROS_DEBUG_THROTTLE(1.0, "Wait for a new image to proceed");
            } else {
                ROS_INFO_STREAM_ONCE(toString(grab_status));
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(2));

            if ((t - old_t).toSec() > 5) {
                zed.close();

                ROS_INFO("Re-opening the ZED");
                sl::ERROR_CODE err = sl::ERROR_CODE_CAMERA_NOT_DETECTED;
                while (err != sl::SUCCESS) {
                    if (!nhNs.ok()) {
                        mStopNode = true;
                        zed.close();
                        ROS_DEBUG("ZED pool thread finished");
                        return;
                    }

                    int id = sl_tools::checkCameraReady(serial_number);
                    if (id > 0) {
                        param.camera_linux_id = id;
                        err = zed.open(param); // Try to initialize the ZED
                        ROS_INFO_STREAM(toString(err));
                    } else {
                        ROS_INFO("Waiting for the ZED to be re-connected");
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
                }                        
            }
            continue;
        }

        // Time update
        old_t =
            sl_tools::slTime2Ros(zed.getTimestamp(sl::TIME_REFERENCE_CURRENT));

        // if (autoExposure) {
        //     // getCameraSettings() can't check status of auto exposure
        //     // triggerAutoExposure is used to execute setCameraSettings() only once
        //     if (triggerAutoExposure) {
        //         zed.setCameraSettings(sl::CAMERA_SETTINGS_EXPOSURE, 0, true);
        //         triggerAutoExposure = false;
        //     }
        // } else {
        //     int actual_exposure =
        //         zed.getCameraSettings(sl::CAMERA_SETTINGS_EXPOSURE);
        //     if (actual_exposure != exposure) {
        //         zed.setCameraSettings(sl::CAMERA_SETTINGS_EXPOSURE, exposure);
        //     }

        //     int actual_gain = zed.getCameraSettings(sl::CAMERA_SETTINGS_GAIN);
        //     if (actual_gain != gain) {
        //         zed.setCameraSettings(sl::CAMERA_SETTINGS_GAIN, gain);
        //     }
        // }

        dataMutex.lock();

        // Retrieve RGBA Left image
        zed.retrieveImage(leftZEDMat, sl::VIEW_LEFT, sl::MEM_CPU, matWidth, matHeight);        
        cv::cvtColor(sl_tools::toCVMat(leftZEDMat), _simgs.getNextImageSlot(0), CV_RGBA2RGB);                 
        
        // Retrieve RGBA Right image
        zed.retrieveImage(rightZEDMat, sl::VIEW_RIGHT, sl::MEM_CPU, matWidth, matHeight);       
        cv::cvtColor(sl_tools::toCVMat(rightZEDMat), _simgs.getNextImageSlot(1), CV_RGBA2RGB);                
        _simgs.setNextStamp(t);

        dataMutex.unlock();

        static int rateWarnCount = 0;
        if (!loop_rate.sleep()) {
            rateWarnCount++;

            if (rateWarnCount == 10) {
                ROS_DEBUG_THROTTLE(
                    1.0,
                    "Working thread is not synchronized with the Camera frame rate");
                ROS_DEBUG_STREAM_THROTTLE(
                    1.0, "Expected cycle time: " << loop_rate.expectedCycleTime()
                    << " - Real cycle time: "
                    << loop_rate.cycleTime());
                ROS_WARN_THROTTLE(10.0, "Elaboration takes longer than requested "
                                        "by the FPS rate. Please consider to "
                                        "lower the 'frame_rate' setting.");
            }
        } else {
            rateWarnCount = 0;
        }            
    } // while loop

    mStopNode = true; // Stops other threads
    zed.close();

    ROS_DEBUG("ZED pool thread finished");
}