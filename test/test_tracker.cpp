#include <calibrationParams.h>
#include <pixelUtils.h>
#include <settings.h>
#include <trackingState.h>
#include <view.h>

// cpp
#include <memory>

// opencv
#include <opencv2/core/hal/interface.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

int main() {
    using namespace surface_reconstruction;
    cv::Mat orginDepth = cv::imread(
        "/home/adrewn/surface_reconstruction/data/1305031102.160407.png", cv::IMREAD_UNCHANGED);

    int rows = orginDepth.rows;
    int cols = orginDepth.cols;

    RGBDCalibrationParams calibrationParams;
    calibrationParams.scale = 5000.0f;

    std::shared_ptr<View> view = std::make_shared<View>();
    view->calibrationParams = calibrationParams;
    view->depth = cv::Mat(orginDepth.rows, orginDepth.cols, CV_32F);
    view->processDepth(orginDepth);

    std::shared_ptr<Settings> settings = std::make_shared<Settings>();
    std::shared_ptr<TrackingState> trackingState = std::make_shared<TrackingState>(
        orginDepth.rows, orginDepth.cols, settings->orientation_weight,
        settings->translation_weight, settings->regenerate_pointcloud_threahold);

    auto pointsMap = trackingState->get_pointcloud();

    Intrinsic depthIntrinsics(Eigen::Vector4f{525.f, 525.f, 310.5f, 239.5f});

    view->calibrationParams.depth = depthIntrinsics;

    Eigen::Matrix3f k_inv = calibrationParams.depth.k_inv;

    for (int y{0}; y < rows; ++y) {
        for (int x{0}; x < cols; ++x) {
            Eigen::Vector4f& point = (*pointsMap)[x + y * cols];
            float depth_measure = view->depth.at<float>(y, x);

            point(3) = -1.f;
            if (depth_measure < 1e-5) continue;
            point << depth_measure * Eigen::Vector3f(x, y, 1.0f), 1.0f;
        }
    }

    computeNormalMap(rows, cols, trackingState->get_pointcloud(), trackingState->get_normals());

    cv::Mat subsmapleDepth;

    filterSubsampleWithHoles(view->depth, subsmapleDepth, Eigen::Vector2i(rows, cols));
    cv::namedWindow("depth", cv::WINDOW_AUTOSIZE);

    cv::imshow("depth", subsmapleDepth);

    cv::waitKey();
    return 0;
}
