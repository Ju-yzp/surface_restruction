#ifndef VIEW_H_
#define VIEW_H_

#include <opencv2/opencv.hpp>

#include "../../Tracker/include/Tracker/cameraParams.h"

struct View {
    View(RGBDCalibrationParams calibrationParams_) : calibrationParams(calibrationParams_) {}

    cv::Mat depth;
    cv::Mat rgb;
    cv::Mat rgb_prev;

    // 相机标定参数
    RGBDCalibrationParams calibrationParams;
};
#endif
