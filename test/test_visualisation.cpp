// cpp
#include <cstdint>
#include <memory>

// opencv
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>

// tracker
#include <Tracker/cameraParams.h>
#include <Tracker/pixelUtils.h>

// visualsation
#include <VisualisationEngine/TrackingState.h>
#include <VisualisationEngine/View.h>
#include <VisualisationEngine/VisualisationEngine.h>
#include <VisualisationEngine/VoxelBlockHash.h>
#include <opencv2/core/hal/interface.h>
// #include <rcl/node_options.h>

// ros2
// #include <rclcpp/executors.hpp>
// #include <rclcpp/rclcpp.hpp>
// #include <rclcpp/utilities.hpp>
// #include <visualization_msgs/msg/marker_array.hpp>

cv::Mat getDepth(std::string file_path) {
    cv::Mat origin = cv::imread(file_path, cv::IMREAD_UNCHANGED);
    cv::Mat convert;
    convertShortToFloat(&origin, &convert, 5000.0f);
    return convert;
}

cv::Mat computeDepthFromPointsMap(cv::Mat& pointsMap, Eigen::Matrix4f& world_to_camera) {
    cv::Mat depth(pointsMap.rows, pointsMap.cols, CV_32F);
    int rows = pointsMap.rows;
    int cols = pointsMap.cols;

    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x) {
            cv::Vec4f point = pointsMap.at<cv::Vec4f>(y, x);
            if (point(3) == 1.0f) {
                Eigen::Vector4f point_in_world(point(0), point(1), point(2), 1.0f);

                Eigen::Vector4f point_in_camera = world_to_camera * point_in_world;
                depth.at<float>(y, x) = point_in_camera(2);

            } else
                depth.at<float>(y, x) = 0.0f;
        }
    return depth;
}

class VisualVoxel : public rclcpp::Node {
public:
    VisualVoxel() : Node("Test") {
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "voxel_blocks_markers", 10);
        RCLCPP_INFO(
            this->get_logger(),
            "VisualVoxel node initialized. Publishing to /voxel_blocks_markers.");
    }

    void publish(std::shared_ptr<Scene> scene) {
        visualization_msgs::msg::MarkerArray marker_array;
        visualization_msgs::msg::Marker marker;

        marker.header.frame_id = "map";
        marker.header.stamp = rclcpp::Clock().now();
        marker.ns = "visible_voxels";

        marker.type = visualization_msgs::msg::Marker::CUBE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.id = 0;

        marker.pose.orientation.w = 1.0;

        float voxelSize = scene->get_sceneParams().voxelSize;

        marker.scale.x = voxelSize;
        marker.scale.y = voxelSize;
        marker.scale.z = voxelSize;

        marker.color.a = 0.4;
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;

        std::set<int> currentVisibleVoxelBlock = scene->get_currentFrameVisibleVoxelBlockList();
        for (std::set<int>::iterator it = currentVisibleVoxelBlock.begin();
             it != currentVisibleVoxelBlock.end(); ++it) {
            HashEntry& entry = scene->get_entry(*it);
            Voxel* localVoxelBlock = scene->get_voxelBolck(entry.ptr);
            geometry_msgs::msg::Point center;
            Eigen::Vector3i globalPos = entry.pos * SDF_BLOCK_SIZE;
            for (int z = 0; z < SDF_BLOCK_SIZE; ++z)
                for (int y = 0; y < SDF_BLOCK_SIZE; ++y)
                    for (int x = 0; x < SDF_BLOCK_SIZE; ++x) {
                        int localId = x + y * SDF_BLOCK_SIZE + z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;

                        Voxel* localVoxel = &localVoxelBlock[localId];
                        if (localVoxel->sdf < 0.4 && localVoxel->sdf > -0.4f) {
                            //计算世界坐标系下的位置，需要从体素块的表达方式进行计算,转换至笛卡尔坐标系
                            Eigen::Vector4f point_in_world;
                            center.x = (globalPos(0) + x) * voxelSize + voxelSize / 2.0f;
                            center.y = (globalPos(1) + y) * voxelSize + voxelSize / 2.0f;
                            center.z = (globalPos(2) + z) * voxelSize + voxelSize / 2.0f;

                            marker.points.push_back(center);
                        }
                    }
        }

        marker_array.markers.push_back(marker);
        marker_pub_->publish(marker_array);
    }

private:
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
};

int main(int argc, char* argv[]) {
    // 相机参数
    Intrinsic depth(525.0f, 525.0f, 319.5f, 239.5f);
    RGBDCalibrationParams calibrationParams(depth, depth, Eigen::Matrix4f(), 0.3f, 4.0f, 5000.0f);

    // 深度图
    std::string file1_path = "/home/adrewn/surface_reconstruction/data/1305031102.194330.png";
    std::shared_ptr<View> view = std::make_shared<View>(calibrationParams);
    view->depth = getDepth(file1_path);

    SceneParams sceneParams{0.02f, 100.0f, 0.005f};

    std::shared_ptr<TrackingState> ts =
        std::make_shared<TrackingState>(cv::Size2i(view->depth.cols, view->depth.rows));
    std::shared_ptr<RenderState> rs =
        std::make_shared<RenderState>(cv::Size2i(view->depth.cols, view->depth.rows));
    {
        // 相机坐标系转世界坐标系
        Eigen::Vector3f translation(1.3352f, 0.6261f, 1.6519f);
        Eigen::Quaternionf rotation_q(-0.3231f, 0.6564f, 0.6139f, -0.2963f);
        rotation_q.normalize();
        Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
        T.block<3, 3>(0, 0) = rotation_q.toRotationMatrix();
        T.block<3, 1>(0, 3) = translation;
        ts->pose_d = T.inverse();
    }

    std::shared_ptr<Scene> scene = std::make_shared<Scene>(sceneParams);
    // scene->reserveVisibleVoxelBlockList(view->depth.rows * view->depth.cols);

    //可视化引擎
    VisualisationEngine ve;
    ve.processFrame(scene, view, ts);
    ve.prepare(scene, view, ts, rs);
    rclcpp::init(argc, argv);
    auto visual_node = std::make_shared<VisualVoxel>();
    visual_node->publish(scene);
    rclcpp::spin(visual_node);
    rclcpp::shutdown();
    return 0;
    scene->swapVisibleList();
}
