// surface_reconstruction
#include <pixelUtils.h>
#include <tracker.h>

// eigen
#include <Eigen/Core>

// cpp
#include <algorithm>
#include <limits>

// opencv
#include <opencv2/opencv.hpp>

namespace surface_reconstruction {

Tracker::Tracker(int nPyramidLevel, int maxNIteration, int minNIteration)
    : nPyramidLevel_(nPyramidLevel) {
    rgbPyramid_.resize(nPyramidLevel);
    depthPyramid_.resize(nPyramidLevel);
    rgbIntrinsicsPyramid_.resize(nPyramidLevel);
    depthIntrinsicsPyramid_.reserve(nPyramidLevel);
    nIterationPyramid_.resize(nPyramidLevel);

    float step = (float)(maxNIteration - minNIteration) / (float)(nPyramidLevel - 1);
    for (int i{0}; i < nPyramidLevel; ++i) nIterationPyramid_[i] = maxNIteration - step * i;
}

void Tracker::track(std::shared_ptr<View> view, std::shared_ptr<TrackingState> trackingState) {
    prepare(view);

    Eigen::Matrix<float, 6, 6> hessian_good;
    Eigen::Vector<float, 6> nabla_good;

    for (int level = nPyramidLevel_ - 1; level >= 0; --level) {
        Eigen::Matrix4f approxInvPose = trackingState->get_current_camera_in_localmap();
        Eigen::Matrix4f old_pose = trackingState->get_current_camera_in_localmap();
        float old_f = std::numeric_limits<float>::infinity();
        int old_nVaildPoints{0};
        float lamdba{1.0f};

        for (int i = 0; i < nIterationPyramid_[level]; ++i) {
            float local_f;
            Eigen::Matrix<float, 6, 6> local_hessian;
            Eigen::Vector<float, 6> local_nabla;
            int local_nVaildPoints{0};

            computeHessianAndGradient(
                level, local_hessian, local_nabla, local_f, local_nVaildPoints, view,
                trackingState);

            if (local_nVaildPoints > settings_->minNVaildPoints) {
                local_hessian /= (float)local_nVaildPoints;
                local_nabla /= (float)local_nVaildPoints;
                local_f /= (float)local_nVaildPoints;
            } else
                local_f = std::numeric_limits<float>::infinity();

            if (local_nVaildPoints < settings_->minNVaildPoints || local_f > old_f) {
                trackingState->set_current_camera_in_localmap(old_pose);
                lamdba *= settings_->lamdbaScale;
                continue;
            } else {
                old_f = local_f;
                hessian_good = local_hessian;
                nabla_good = local_nabla;
                lamdba /= settings_->lamdbaScale;
            }

            local_hessian = hessian_good + Eigen::Matrix<float, 6, 6>::Identity() * lamdba;
            local_nabla = nabla_good;

            Eigen::Vector<float, 6> delta;
            computeDelta(local_hessian, local_nabla, delta);
            applyDelta(approxInvPose, delta);
            trackingState->set_current_camera_in_localmap(approxInvPose.inverse());
            updateQualityOfTracking();
        }
    }
}

void Tracker::prepare(std::shared_ptr<View> view) {
    // 获取图像金字塔以及相机内参金字塔的第一层
    rgbPyramid_[0] = view->rgb;
    depthPyramid_[0] = view->depth;
    if (rgbIntrinsicsPyramid_.empty()) {
        rgbIntrinsicsPyramid_[0] = view->calibrationParams.rgb;
        depthIntrinsicsPyramid_[0] = view->calibrationParams.depth;
        for (int i = 1; i < nPyramidLevel_; ++i) {
            rgbIntrinsicsPyramid_[i] = rgbIntrinsicsPyramid_[i - 1].subIntrisic();
            depthIntrinsicsPyramid_[i] = depthIntrinsicsPyramid_[i - 1].subIntrisic();
        }
    }

    for (int i = 1; i < nPyramidLevel_; ++i) {
        // 彩色图像金字塔下采样
        cv::pyrDown(rgbPyramid_[i - 1], rgbPyramid_[i]);
        // 深度和法向量图像金字塔下采样
        Eigen::Vector2i mapSize{depthPyramid_[i - 1].rows, depthPyramid_[i - 1].cols};
        filterSubsampleWithHoles(depthPyramid_[i - 1], depthPyramid_[i], mapSize);
        filterSubsampleWithHoles(normalPyramid_[i - 1], normalPyramid_[i], mapSize, true);
    }
}

inline float rho(float r, float huber_r) {
    float tmp = std::abs(r) - huber_r;
    tmp = std::max(tmp, 0.0f);
    return r * r - tmp * tmp;
}

void Tracker::computeHessianAndGradient(
    int id, Eigen::Matrix<float, 6, 6>& hessian, Eigen::Vector<float, 6> nabla, float& f,
    int& nVaildPoints, std::shared_ptr<View> view, std::shared_ptr<TrackingState> trackingState) {
    int rows = depthPyramid_[id].rows;
    int cols = depthPyramid_[id].cols;
    float* depth = (float*)depthPyramid_[id].data;
    const auto normalMap = normalPyramid_[id];
    const auto pointsMap = pointcloudPyramid_[id];

    Eigen::Matrix4f invApproximatePose = trackingState->get_current_camera_in_localmap().inverse();
    Eigen::Matrix4f generatePose = trackingState->get_generate_camera_in_localmap();
    Eigen::Matrix3f k_inv = view->calibrationParams.depth.k_inv;
    Eigen::Matrix3f k = k_inv.inverse();

    Eigen::Vector<float, 6> A;
    const float spaceThreshold = spaceThresholds_[id];

    for (int y = 1; y < rows - 1; ++y) {
        int offset = y * cols + rows;
        for (int x = 1; x < cols - 1; ++x) {
            Eigen::Vector<float, 6> local_hessian;
            Eigen::Vector<float, 6> local_nabla;
            float local_f{0.0f};

            float b;
            float currentDepth = depth[x + offset];
            if (currentDepth < 1e-4) continue;
            Eigen::Vector3f currentPointcloud(x, y, 1.0f);
            currentPointcloud = currentDepth * k_inv * currentPointcloud;

            Eigen::Vector3f point_in_last_view =
                k * (generatePose * invApproximatePose *
                     (Eigen::Vector4f() << currentPointcloud, 1.0f).finished())
                        .head(3);

            point_in_last_view /= point_in_last_view(2);

            Eigen::Vector4f point =
                interpolateBilinear_withHoles(pointsMap, point_in_last_view.head(2), cols);
            if (point(3) < 0.0f) continue;
            Eigen::Vector4f normal =
                interpolateBilinear_withHoles(normalMap, point_in_last_view.head(2), cols);

            b = point.head(3).transpose() * normal.head(3);

            A(0) = currentPointcloud(2) * normal(1) - currentPointcloud(1) * normal(2);
            A(0) = -currentPointcloud(2) * normal(0) + currentPointcloud(0) * normal(2);
            A(0) = currentPointcloud(1) * normal(0) - currentPointcloud(0) * normal(1);
            A(3) = normal(0);
            A(4) = normal(1);
            A(5) = normal(2);

            local_f = rho(b, spaceThreshold);

            hessian += A * A.inverse();
            nabla += A * local_f;
            nVaildPoints++;
        }
    }
}

}  // namespace surface_reconstruction
