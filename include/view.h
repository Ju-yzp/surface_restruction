#ifndef VIEW_H_
#define VIEW_H_

#include <cstdint>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

#include <calibrationParams.h>
#include <opencv2/core/hal/interface.h>

namespace surface_reconstruction {
struct View {
    View(cv::Size2i rgb_size, cv::Size2i depth_size) {
        depth = cv::Mat(depth_size.height, depth_size.width, CV_32F);
        rgb = cv::Mat(rgb_size.height, rgb_size.width, CV_8UC3);
        prev_rgb = cv::Mat(rgb_size.height, rgb_size.width, CV_8UC3);
    }

    View() = default;

    // 深度图像
    cv::Mat depth;

    // 彩色图像
    cv::Mat rgb;

    // 上一帧的彩色图像
    cv::Mat prev_rgb;

    // 相机标定参数
    RGBDCalibrationParams calibrationParams;

    void processDepth(cv::Mat origin_depth) {
        int rows = origin_depth.rows;
        int cols = origin_depth.cols;

        float* depthPtr = (float*)depth.data;
        uint16_t* originDepthPtr = (uint16_t*)origin_depth.data;
        const float scale = calibrationParams.scale;

        for (int y{0}; y < rows; ++y)
            for (int x{0}; x < cols; ++x) {
                int offset = x + y * cols;
                depthPtr[offset] = (float)originDepthPtr[offset] / scale;

                if (depthPtr[offset] <= 1e-3)
                    depthPtr[offset] = std::numeric_limits<float>::quiet_NaN();
            }
    }
};
}  // namespace surface_reconstruction

#endif  // VIEW_H_
