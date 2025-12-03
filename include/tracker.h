#ifndef TRACKER_H_
#define TRACKER_H_

// opencv
#include <memory>
#include <opencv2/opencv.hpp>

// surface_reconstruction
#include <calibrationParams.h>
#include <settings.h>
#include <trackingState.h>
#include <view.h>

// eigen
#include <Eigen/Eigen>

namespace surface_reconstruction {
class Tracker {
public:
    Tracker(int nPyramidLevel, int maxNIteration, int minNIteration);

    void track(std::shared_ptr<View>, std::shared_ptr<TrackingState> trackingState);

private:
    // 获取图像金字塔
    void prepare(std::shared_ptr<View> view);

    // 写入追踪质量
    void updateQualityOfTracking();

    // 计算深度图像的hessian矩阵和梯度值
    void computeHessianAndGradient(
        int id, Eigen::Matrix<float, 6, 6>& hessian, Eigen::Vector<float, 6> nabla, float& f,
        int& nVaildPoints, std::shared_ptr<View> view,
        std::shared_ptr<TrackingState> trackingState);

    // 计算变化量
    void computeDelta(
        Eigen::Matrix<float, 6, 6> hessian, Eigen::Vector<float, 6> nabla,
        Eigen::Vector<float, 6>& delta);

    // 应用变化量
    void applyDelta(Eigen::Matrix4f& pose, Eigen::Vector<float, 6> delta);

    std::vector<cv::Mat> rgbPyramid_;
    std::vector<cv::Mat> depthPyramid_;

    std::vector<Intrinsic> rgbIntrinsicsPyramid_;
    std::vector<Intrinsic> depthIntrinsicsPyramid_;

    std::vector<std::shared_ptr<std::vector<Eigen::Vector4f>>> pointcloudPyramid_;
    std::vector<std::shared_ptr<std::vector<Eigen::Vector4f>>> normalPyramid_;

    std::vector<int> nIterationPyramid_;
    std::vector<float> spaceThresholds_;
    int nPyramidLevel_;

    std::shared_ptr<Settings> settings_;
};
}  // namespace surface_reconstruction

#endif  // TRACKER_H_
