#ifndef COARSE_MATCHER_H_
#define COARSE_MATCHER_H_

// eigen
#include <Eigen/Eigen>

// cpp
#include <opencv2/core/types.hpp>
#include <optional>

// opencv
#include <opencv2/opencv.hpp>

// pcl
#include <pcl/console/time.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/ia_fpcs.h>
#include <pcl/registration/ia_kfpcs.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace surface_reconstruction {

class GridFilter {
public:
    GridFilter(int rows, int cols, int gridLenght, int gridWidth);

    void filter(std::vector<cv::Point2f>& sourcePoints, std::vector<cv::Point2f>& targetPoints);

private:
    bool isBetterPoint(cv::Point2f gridCenter, cv::Point2f p1, cv::Point2f p2);

    // 图像尺寸信息
    int rows_, cols_;

    // 棋格参数
    int gridLenght_, gridWidth_;
};

using PointT = pcl::PointXYZ;
using pointcloud = pcl::PointCloud<PointT>;
using pointnormal = pcl::PointCloud<pcl::Normal>;
using fpfhFeature = pcl::PointCloud<pcl::FPFHSignature33>;

class CoarseMatcher {
public:
    CoarseMatcher(
        int rows, int cols, int gridLenght, int gridWidth, int maxIteration,
        float lnlierArtioThreshold, float diff, float distanceThreshold, int minNPoints,
        float fpfh_matched_threahold);

    // 过滤错误的匹配特征点
    void filterIncorrectMatchedPoints(
        std::vector<Eigen::Vector3f>& source, std::vector<Eigen::Vector3f>& target, int axis,
        std::set<int>& vaildId, bool merge = true);

    // 求解相似变换矩阵
    Eigen::Matrix4f solveSimilarityTransform(
        std::vector<Eigen::Vector3f>& source, std::vector<Eigen::Vector3f>& target);

    // 随机挑选指定数量的点
    std::vector<int> randomSelectPoints(int pointsNum, int selectedNum);

    // 将图像二维点转换为三维点
    void convertFeaturePointsToObjectPoints(
        cv::Mat& depth, std::vector<cv::Point2f> keypoints,
        std::vector<Eigen::Vector3f>& objectPoints, const Eigen::Matrix3f k);

    std::optional<Eigen::Matrix4f> ransacEvalue(
        std::vector<Eigen::Vector3f>& source, std::vector<Eigen::Vector3f>& target);

    Eigen::Matrix3f skew(const Eigen::Vector3f v) {
        Eigen::Matrix3f m;
        m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
        return m;
    }

    // 提取fpfh需要的点云，避免密集提取深度图所对应的点云
    void extractTargetPointCloud(
        const cv::Mat& depth, const Eigen::Matrix3f k, std::vector<Eigen::Vector3f>& pointclouds);

    // 提取通过fpfh提取到的匹配点云对
    void extractFPFHMatchedPointCloud(
        const std::vector<Eigen::Vector3f>& unmatched_source,
        const std::vector<Eigen::Vector3f>& unmatched_taregt,
        std::vector<Eigen::Vector3f>& matched_source, std::vector<Eigen::Vector3f>& matched_target);

    pointcloud::Ptr convertStyle(const std::vector<Eigen::Vector3f>& input);

    fpfhFeature::Ptr compueFPFHFeature(pointcloud::Ptr pointcloud);

    // 内点比例阈值
    double lnlierArtioThreshold_;

    // 最小关键点数量
    double minNumFeaturePoint_ = 8;

    // ransac最大迭代次数
    int maxIteration_;

    // 相似变换求解的输入点最小数量要求
    int minNPoints_;

    GridFilter gridFilter_;

    // 序列最长差值
    float diff_;

    //
    float distanceThreshold_;

    float fpfh_matched_threahold_;
};
}  // namespace surface_reconstruction

#endif
