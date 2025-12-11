// ros2
#include <memory>
#include <rclcpp/executors.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/utilities.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <pixelUtils.h>
#include <Eigen/Eigen>
#include "calibrationParams.h"
#include "settings.h"
#include "tracker.h"
#include "trackingState.h"
#include "view.h"
#include "visualisationEngine.h"

class VisualVoxel : public rclcpp::Node {
public:
    VisualVoxel() : Node("Test") {
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "voxel_blocks_markers", 10);
        RCLCPP_INFO(
            this->get_logger(),
            "VisualVoxel node initialized. Publishing to /voxel_blocks_markers.");
    }

    void publish(std::shared_ptr<surface_reconstruction::Scene> scene) {
        std::cout << "wow" << std::endl;
        visualization_msgs::msg::MarkerArray marker_array;
        visualization_msgs::msg::Marker marker;

        marker.header.frame_id = "map";
        marker.header.stamp = rclcpp::Clock().now();
        marker.ns = "visible_voxels";

        marker.type = visualization_msgs::msg::Marker::CUBE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.id = 0;

        marker.pose.orientation.w = 1.0;

        float voxelSize = scene->get_settings().voxelSize;

        marker.scale.x = voxelSize;
        marker.scale.y = voxelSize;
        marker.scale.z = voxelSize;

        marker.color.a = 0.4;
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;

        const int SDF_BLOCK_SIZE = scene->get_settings().sdf_block_size;

        std::set<int> currentVisibleVoxelBlock = scene->get_currentFrameVisibleVoxelBlockList();
        for (std::set<int>::iterator it = currentVisibleVoxelBlock.begin();
             it != currentVisibleVoxelBlock.end(); ++it) {
            surface_reconstruction::HashEntry& entry = scene->get_entry(*it);
            surface_reconstruction::Voxel* localVoxelBlock = scene->get_voxelBolck(entry.ptr);
            geometry_msgs::msg::Point center;
            Eigen::Vector3i globalPos = entry.pos * SDF_BLOCK_SIZE;
            for (int z = 0; z < SDF_BLOCK_SIZE; ++z)
                for (int y = 0; y < SDF_BLOCK_SIZE; ++y)
                    for (int x = 0; x < SDF_BLOCK_SIZE; ++x) {
                        int localId = x + y * SDF_BLOCK_SIZE + z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;

                        surface_reconstruction::Voxel* localVoxel = &localVoxelBlock[localId];
                        if (localVoxel->sdf < 0.7f && localVoxel->sdf > -0.7f) {
                            //计算世界坐标系下的位置，需要从体素块的表达方式进行计算,转换至笛卡尔坐标系
                            Eigen::Vector4f point_in_world;
                            center.x = (globalPos(0) + x) * voxelSize + voxelSize / 2.0f;
                            center.y = (globalPos(1) + y) * voxelSize + voxelSize / 2.0f;
                            center.z = (globalPos(2) + z) * voxelSize + voxelSize / 2.0f;

                            marker.points.push_back(center);
                        }
                    }
        }

        std::cout << marker.points.size() << std::endl;
        marker_array.markers.push_back(marker);
        marker_pub_->publish(marker_array);
    }

private:
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
};

void generatePointsMap(
    std::shared_ptr<surface_reconstruction::TrackingState> trackingState,
    std::shared_ptr<surface_reconstruction::View> view) {
    int rows = trackingState->get_height();
    int cols = trackingState->get_width();
    auto pointsMap = trackingState->get_pointclouds();
    Eigen::Matrix3f k_inv = view->calibrationParams.depth.k_inv;
    for (int y{0}; y < rows; ++y) {
        for (int x{0}; x < cols; ++x) {
            Eigen::Vector4f& point = (*pointsMap)[x + y * cols];
            float depth_measure = view->depth.at<float>(y, x);
            if (std::isnan(depth_measure)) {
                point = Eigen::Vector4f::Constant(std::numeric_limits<float>::quiet_NaN());
                continue;
            }
            point << depth_measure * (k_inv * Eigen::Vector3f(x, y, 1.0f)), 1.0f;
            point = trackingState->get_current_camera_in_localmap() * point;
        }
    }

    trackingState->set_generate_camera_in_localmap(trackingState->get_current_camera_in_localmap());
    computeNormalMap(rows, cols, trackingState->get_pointclouds(), trackingState->get_normals());
}

int main(int argc, char* argv[]) {
    RGBDCalibrationParams calibrationParams;
    calibrationParams.scale = 1000.0f;
    Intrinsic depthIntrinsics(Eigen::Vector4f{504.867, 504.958, 333.731, 339.597});
    calibrationParams.depth = depthIntrinsics;
    calibrationParams.viewFrustum_max = 2.0f;
    calibrationParams.viewFrustum_min = 0.0f;

    std::shared_ptr<surface_reconstruction::Settings> settings =
        std::make_shared<surface_reconstruction::Settings>();

    auto tracker = std::make_shared<surface_reconstruction::Tracker>(
        settings, settings->nPyramidLevel, settings->maxNLMIteration, settings->minNLMIteration,
        settings->maxSpaceThreshold, settings->minSpaceThreshold);

    auto view = std::make_shared<surface_reconstruction::View>(
        cv::Size2i(1920, 1080), cv::Size2i(640, 576));
    view->calibrationParams = calibrationParams;

    auto scene = std::make_shared<surface_reconstruction::Scene>(settings);

    auto trackingState =
        std::make_shared<surface_reconstruction::TrackingState>(576, 640, 0.0, 0.0, 0.0);

    cv::Mat orginDepth =
        cv::imread("/home/adrewn/surface_reconstruction/data/depth3.png", cv::IMREAD_UNCHANGED);

    view->processDepth(orginDepth);
    surface_reconstruction::allocateVoxelFormDepth(scene, view, trackingState);
    surface_reconstruction::intergrateIntoScene(scene, view, trackingState);
    surface_reconstruction::processFrame(scene, view, trackingState);

    // orginDepth =
    //     cv::imread("/home/adrewn/surface_reconstruction/data/depth4.png", cv::IMREAD_UNCHANGED);
    // view->processDepth(orginDepth);

    // generatePointsMap(trackingState, view);
    // tracker->track(view, trackingState);

    // surface_reconstruction::processFrame(scene, view, trackingState);
    rclcpp::init(argc, argv);
    auto visualer = std::make_shared<VisualVoxel>();
    visualer->publish(scene);
    while (rclcpp::ok()) {
    }
    rclcpp::shutdown();
    return 0;
}
