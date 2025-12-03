#ifndef TRACKING_STATE_H_
#define TRACKING_STATE_H_

// eigen
#include <Eigen/Core>
#include <Eigen/Eigen>
#include <Eigen/Geometry>

// cpp
#include <cstdint>
#include <memory>

namespace surface_reconstruction {
class TrackingState {
public:
    enum class TrackingResult : uint8_t {
        TRACKING_GOOD = 0,
        TRACKING_POOR = 1,
        TRACKING_FAILED = 2
    };

    TrackingState(
        int height, int weight, float orientation_weight, float translation_weight,
        float regenerate_pointcloud_threahold)
        : height_(height),
          width_(weight),
          orientation_weight_(orientation_weight),
          translation_weight_(translation_weight),
          regenerate_pointcloud_threahold_(regenerate_pointcloud_threahold) {
        pointclouds_ = std::make_shared<std::vector<Eigen::Vector4f>>(height_ * width_);
        normals_ = std::make_shared<std::vector<Eigen::Vector4f>>(height_ * width_);
    }

    bool NeedRegenerateNewPointcloud() {
        Eigen::Matrix4f transform =
            generate_camera_in_localmap_.inverse() * current_camera_in_localmap_;

        Eigen::AngleAxisf angleAxis(transform.block<3, 3>(0, 0));
        float score = std::abs(angleAxis.angle()) * orientation_weight_ +
                      transform.block<3, 1>(0, 3).norm() * translation_weight_;
        return score > regenerate_pointcloud_threahold_;
    }

    std::shared_ptr<std::vector<Eigen::Vector4f>> get_pointcloud() { return pointclouds_; }

    std::shared_ptr<std::vector<Eigen::Vector4f>> get_normals() { return normals_; }

    int get_height() const { return height_; }

    int get_width() const { return width_; }

    Eigen::Matrix4f get_current_camera_in_localmap() const { return current_camera_in_localmap_; }

    void set_current_camera_in_localmap(const Eigen::Matrix4f pose) {
        current_camera_in_localmap_ = pose;
    }

    Eigen::Matrix4f get_generate_camera_in_localmap() { return generate_camera_in_localmap_; }

    void set_generate_camera_in_localmap(const Eigen::Matrix4f pose) {
        generate_camera_in_localmap_ = pose;
    }

    TrackingResult get_tracking_result() const { return tracking_result_; }

    void set_tracking_result(TrackingResult result) { tracking_result_ = result; }

private:
    // 点云数据以及对应的图像和高度信息
    std::shared_ptr<std::vector<Eigen::Vector4f>> pointclouds_;
    std::shared_ptr<std::vector<Eigen::Vector4f>> normals_;
    int height_;
    int width_;

    // 生成的点云所对应的相机在子地图中的位姿
    Eigen::Matrix4f generate_camera_in_localmap_ = Eigen::Matrix4f::Identity();

    // 相机在子地图中的位姿
    Eigen::Matrix4f current_camera_in_localmap_ = Eigen::Matrix4f::Identity();

    // 相机位姿中朝向变化对于是否需要重新进行光线投射生成新点云的线性权重影响
    float orientation_weight_;

    // 相机位姿中位置变化对于是否需要重新进行光线投射生成新点云的线性权重影响
    float translation_weight_;

    // 判断是否需要重新生成点云的阈值
    float regenerate_pointcloud_threahold_;

    // 当前跟踪结果
    TrackingResult tracking_result_{TrackingResult::TRACKING_GOOD};
};
}  // namespace surface_reconstruction

#endif  // TRACKING_STATE_H_
