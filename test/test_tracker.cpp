#include <calibrationParams.h>
#include <pixelUtils.h>
#include <settings.h>
#include <tracker.h>
#include <trackingState.h>
#include <view.h>

// cpp
#include <Eigen/Geometry>
#include <cmath>
#include <memory>

// opencv
#include <opencv2/core/hal/interface.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

int main() {
    using namespace surface_reconstruction;
    cv::Mat orginDepth =
        cv::imread("/home/adrewn/surface_reconstruction/data/depth3.png", cv::IMREAD_UNCHANGED);

    int rows = orginDepth.rows;
    int cols = orginDepth.cols;

    Eigen::Quaternionf q1(1, 0, 0, 0.0), q2(1.0, 0.0, 0.0, 0.0);

    Eigen::Matrix4f t1 = Eigen::Matrix4f::Identity(), t2 = Eigen::Matrix4f::Identity();
    t1.col(3) << 0.0f, 0.0f, 0.0f, 1.0f;
    t2.col(3) << 0.0f, 0.0f, 0.0, 1.0f;
    t1.block(0, 0, 3, 3) = q1.matrix();
    t2.block(0, 0, 3, 3) = q2.matrix();

    RGBDCalibrationParams calibrationParams;
    calibrationParams.scale = 1000.0f;

    std::shared_ptr<View> view = std::make_shared<View>();
    view->calibrationParams = calibrationParams;
    view->depth = cv::Mat(orginDepth.rows, orginDepth.cols, CV_32F);
    view->processDepth(orginDepth);

    std::shared_ptr<Settings> settings = std::make_shared<Settings>();
    std::shared_ptr<TrackingState> trackingState = std::make_shared<TrackingState>(
        orginDepth.rows, orginDepth.cols, settings->orientation_weight,
        settings->translation_weight, settings->regenerate_pointcloud_threahold);

    auto pointsMap = trackingState->get_pointclouds();

    Intrinsic depthIntrinsics(Eigen::Vector4f{504.867, 504.958, 333.731, 339.597});

    view->calibrationParams.depth = depthIntrinsics;
    view->calibrationParams.viewFrustum_max = 2.f;
    view->calibrationParams.viewFrustum_min = 0.2f;

    Eigen::Matrix3f k_inv = view->calibrationParams.depth.k_inv;
    trackingState->set_generate_camera_in_localmap(t1.inverse());
    trackingState->set_current_camera_in_localmap(t1.inverse());
    for (int y{0}; y < rows; ++y) {
        for (int x{0}; x < cols; ++x) {
            Eigen::Vector4f& point = (*pointsMap)[x + y * cols];
            float depth_measure = view->depth.at<float>(y, x);
            if (std::isnan(depth_measure)) {
                point = Eigen::Vector4f::Constant(std::numeric_limits<float>::quiet_NaN());
                continue;
            }
            point << depth_measure * (k_inv * Eigen::Vector3f(x, y, 1.0f)), 1.0f;
            point = t1 * point;
        }
    }

    computeNormalMap(rows, cols, trackingState->get_pointclouds(), trackingState->get_normals());

    Tracker tracker(
        settings, settings->nPyramidLevel, settings->maxNLMIteration, settings->minNLMIteration,
        settings->maxSpaceThreshold, settings->minSpaceThreshold);

    orginDepth =
        cv::imread("/home/adrewn/surface_reconstruction/data/depth4.png", cv::IMREAD_UNCHANGED);
    view->processDepth(orginDepth);
    tracker.track(view, trackingState);

    std::cout << "-------ApproxPose is -------" << std::endl;
    std::cout << trackingState->get_current_camera_in_localmap().inverse() << std::endl;
    std::cout << "------Actual Pose is ------" << std::endl;
    std::cout << t2 << std::endl;
    return 0;
}
