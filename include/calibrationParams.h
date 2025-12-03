#ifndef CAMERA_PARAMS_H_
#define CAMERA_PARAMS_H_

// opencv
#include <opencv2/opencv.hpp>

// eigen
#include <Eigen/Eigen>

struct Intrinsic {
    Intrinsic() = default;

    Intrinsic(float fx, float fy, float cx, float cy) {
        k(0, 0) = fx;
        k(1, 1) = fy;
        k(0, 2) = cx;
        k(1, 2) = cy;
        k_inv = k.inverse();

        params(0) = fx;
        params(1) = fy;
        params(2) = cx;
        params(3) = cy;
    }

    Intrinsic(Eigen::Vector4f params_) {
        k(0, 0) = params_(0);
        k(1, 1) = params_(1);
        k(0, 2) = params_(2);
        k(1, 2) = params_(3);
        k_inv = k.inverse();

        params = params_;
    }

    Eigen::Matrix3f k = Eigen::Matrix3f::Identity();
    Eigen::Vector4f params = Eigen::Vector4f::Zero();
    Eigen::Matrix3f k_inv = Eigen::Matrix3f::Identity();

    Intrinsic subIntrisic() { return Intrinsic(params * 0.5f); }
};

struct RGBDCalibrationParams {
    RGBDCalibrationParams(
        Intrinsic rgb_, Intrinsic depth_, Eigen::Matrix4f depth_to_rgb_, float viewFrustum_min_,
        float viewFrustum_max_, float scale_)
        : rgb(rgb_),
          depth(depth_),
          depth_to_rgb(depth_to_rgb_),
          viewFrustum_max(viewFrustum_max_),
          viewFrustum_min(viewFrustum_min_),
          scale(scale_) {
        rgb_to_depth = depth_to_rgb.inverse();
    }

    RGBDCalibrationParams() {}

    // rgb相机内参
    Intrinsic rgb;

    // 深度相机内参
    Intrinsic depth;

    // rgb转depth
    Eigen::Matrix4f rgb_to_depth = Eigen::Matrix4f::Identity();

    // depth转rgb
    Eigen::Matrix4f depth_to_rgb = Eigen::Matrix4f::Identity();

    // 有效深度范围
    float viewFrustum_min, viewFrustum_max;

    // 缩放至m的尺度因子
    float scale;
};

#endif
