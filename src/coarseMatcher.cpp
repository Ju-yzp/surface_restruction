#include <coarseMatcher.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <random>
#include <set>
#include <vector>

namespace surface_reconstruction {

GridFilter::GridFilter(int rows, int cols, int gridLenght, int gridWidth)
    : rows_(rows), cols_(cols), gridLenght_(gridLenght), gridWidth_(gridWidth) {}

void GridFilter::filter(
    std::vector<cv::Point2f>& sourcePoints, std::vector<cv::Point2f>& targetPoints) {
    assert(sourcePoints.size() == targetPoints.size());

    int gridRows = std::ceil(rows_ / (float)gridLenght_);
    int gridCols = std::ceil(cols_ / (float)gridWidth_);
    std::vector<int> sgrids(gridRows * gridCols, -1), tgrids(gridRows * gridCols, -1);

    for (int i{0}; i < sourcePoints.size(); ++i) {
        cv::Point2f sourcePoint = sourcePoints[i], targetPoint = targetPoints[i];
        int sy = sourcePoint.y / gridLenght_;
        int sx = sourcePoint.x / gridWidth_;
        int ty = targetPoint.y / gridLenght_;
        int tx = targetPoint.x / gridWidth_;

        int &sgrid = sgrids[sy * gridCols + sx], tgrid = tgrids[ty * gridCols + tx];
        if (sgrid == -1 && tgrid == -1)
            sgrid = tgrid = i;
        else if (sgrid != -1 && tgrid != -1) {
            cv::Point2f sgridCenter, tgridCenter;
            sgridCenter.y = sy * gridLenght_ + gridLenght_ / 2.0f;
            sgridCenter.x = sx * gridWidth_ + gridWidth_ / 2.0f;
            tgridCenter.y = ty * gridLenght_ + gridLenght_ / 2.0f;
            tgridCenter.x = tx * gridWidth_ + gridWidth_ / 2.0f;
            if (isBetterPoint(sgridCenter, sourcePoint, sourcePoints[sgrid]) &&
                isBetterPoint(tgridCenter, targetPoint, targetPoints[tgrid]))
                sgrid = tgrid = i;
        }
    }

    std::vector<cv::Point2f> newSourcePoints, newTargetPoints;
    newSourcePoints.reserve(sourcePoints.size());
    newTargetPoints.reserve(targetPoints.size());

    for (int i{0}; i < sgrids.size(); ++i)
        if (sgrids[i] != -1) {
            newSourcePoints.emplace_back(sourcePoints[sgrids[i]]);
            newTargetPoints.emplace_back(targetPoints[sgrids[i]]);
        }

    sourcePoints.swap(newSourcePoints);
    targetPoints.swap(newTargetPoints);
}

bool GridFilter::isBetterPoint(cv::Point2f gridCenter, cv::Point2f p1, cv::Point2f p2) {
    float distance1_sq = (p1 - gridCenter).dot(p1 - gridCenter);
    float distance2_sq = (p2 - gridCenter).dot(p2 - gridCenter);

    return distance1_sq < distance2_sq;
}

CoarseMatcher::CoarseMatcher(
    int rows, int cols, int gridLenght, int gridWidth, int maxIteration, float lnlierArtioThreshold,
    float diff, float distanceThreshold, int minNPoints, float fpfh_matched_threahold)
    : gridFilter_(rows, cols, gridLenght, gridWidth),
      maxIteration_(maxIteration),
      lnlierArtioThreshold_(lnlierArtioThreshold),
      diff_(diff),
      distanceThreshold_(distanceThreshold),
      minNPoints_(minNPoints),
      fpfh_matched_threahold_(fpfh_matched_threahold) {}

std::optional<Eigen::Matrix4f> CoarseMatcher::ransacEvalue(
    std::vector<Eigen::Vector3f>& source, std::vector<Eigen::Vector3f>& target) {
    assert(source.size() == target.size());

    std::set<int> final_ids;
    filterIncorrectMatchedPoints(source, target, 2, final_ids, false);
    filterIncorrectMatchedPoints(source, target, 1, final_ids);
    filterIncorrectMatchedPoints(source, target, 0, final_ids);

    if (final_ids.size() < minNumFeaturePoint_) {
        std::cout << "The points is too few" << std::endl;
        return std::nullopt;
    }

    int num = final_ids.size();
    std::vector<int> ids_copy(final_ids.begin(), final_ids.end());

    for (auto id : ids_copy) {
        std::cout << "source" << std::endl;
        std::cout << source[id] << std::endl;
        std::cout << "target" << std::endl;
        std::cout << target[id] << std::endl;
    }

    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    float old_artio = -1.0f;
    for (int i = 0; i < maxIteration_; ++i) {
        float f{0.0f};
        auto ids = randomSelectPoints(num, minNPoints_);
        std::vector<Eigen::Vector3f> source_copy, target_copy;
        for (auto id : ids) {
            source_copy.emplace_back(source[ids_copy[id]]);
            target_copy.emplace_back(target[ids_copy[id]]);
        }
        Eigen::Matrix4f t = solveSimilarityTransform(source_copy, target_copy);

        int nVaildPoints{0};

        for (int j{0}; j < num; ++j) {
            Eigen::Vector3f point =
                (t * (Eigen::Vector4f() << source[ids_copy[j]], 1.0f).finished()).head(3);
            f = (target[ids_copy[j]] - point).norm();
            if (f < distanceThreshold_) ++nVaildPoints;
        }

        float artio = (float)nVaildPoints / (float)num;
        if (artio > old_artio) {
            transform = t;
            old_artio = artio;
        }
    }

    if (old_artio > lnlierArtioThreshold_) return transform;
    return std::nullopt;
}

Eigen::Matrix4f CoarseMatcher::solveSimilarityTransform(
    std::vector<Eigen::Vector3f>& source, std::vector<Eigen::Vector3f>& target) {
    assert(source.size() == target.size() && source.size() >= minNPoints_);

    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();

    int num = source.size();

    constexpr int maxIter = 5;

    float max_distance = std::numeric_limits<float>::lowest(),
          min_distance = std::numeric_limits<float>::infinity();

    for (int i{0}; i < num; ++i) {
        if (target[i](2) > max_distance) max_distance = target[i](2);

        if (target[i](2) < min_distance) min_distance = target[i](2);
    }

    Eigen::Matrix4f old_matrix = Eigen::Matrix4f::Identity();
    float old_f = std::numeric_limits<float>::infinity();
    float lamdba{1.0f};
    Eigen::Matrix<float, 6, 6> hessian_good = Eigen::Matrix<float, 6, 6>::Zero();
    Eigen::Vector<float, 6> nabla_good = Eigen::Vector<float, 6>::Zero();
    for (int i{0}; i < maxIter; ++i) {
        float f{0.0f};
        Eigen::Matrix<float, 6, 6> local_hessian = Eigen::Matrix<float, 6, 6>::Zero();
        Eigen::Vector<float, 6> local_nabla = Eigen::Vector<float, 6>::Zero();

        for (int j{0}; j < num; ++j) {
            Eigen::Vector3f transformPoint =
                (transform * (Eigen::Vector4f() << source[j], 1.0f).finished()).head(3);
            Eigen::Vector3f diff = transformPoint - target[j];

            Eigen::Matrix<float, 3, 6> A;
            A.block<3, 3>(0, 0) = -skew(transformPoint);
            A.block<3, 3>(0, 3) = Eigen::Matrix3f::Identity();

            f += diff.norm();
            local_hessian += A.transpose() * A;
            local_nabla += -A.transpose() * diff;
        }

        if (f < old_f) {
            lamdba /= 15.0;
            old_f = f;
            old_matrix = transform;
            hessian_good = local_hessian;
            nabla_good = local_nabla;
        } else {
            lamdba *= 2.0;
            transform = old_matrix;
            continue;
        }

        local_hessian += Eigen::Matrix<float, 6, 6>::Identity() * lamdba;
        Eigen::Matrix<float, 1, 6> delta = local_hessian.ldlt().solve(local_nabla);
        Eigen::Vector3f dw = delta.head<3>();
        Eigen::Vector3f dv = delta.tail<3>();
        Eigen::Matrix3f dR = (Eigen::AngleAxisf(dw.norm(), dw.normalized())).toRotationMatrix();
        transform.block(0, 0, 3, 3) = dR * transform.block(0, 0, 3, 3);
        transform.block(0, 3, 3, 1) = dR * transform.block(0, 3, 3, 1) + dv;
    }
    return transform;
}

void CoarseMatcher::filterIncorrectMatchedPoints(
    std::vector<Eigen::Vector3f>& source, std::vector<Eigen::Vector3f>& target, int axis,
    std::set<int>& vaildIds, bool merge) {
    std::vector<float> diff_v;
    for (size_t i{0}; i < source.size(); ++i) {
        diff_v.emplace_back(target[i](axis) - source[i](axis));
    }

    std::vector<int> sorted_indices(diff_v.size());
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(), [&diff_v](int i1, int i2) {
        return diff_v[i1] < diff_v[i2];
    });

    int max_len = 0;
    int best_start_index = -1;
    int best_end_index = -1;
    for (int i = 0, j = 0; i < sorted_indices.size(); ++i) {
        while (j < sorted_indices.size() &&
               std::abs(diff_v[sorted_indices[j]] - diff_v[sorted_indices[i]]) <= diff_) {
            j++;
        }

        int current_len = j - i;

        if (current_len > max_len) {
            max_len = current_len;
            best_start_index = i;
            best_end_index = j;
        }
    }

    std::set<int> ids;
    if (best_start_index != -1) {
        for (int k = best_start_index; k < best_end_index; ++k) {
            merge ? ids.emplace(sorted_indices[k]) : vaildIds.emplace(sorted_indices[k]);
        }
    }

    if (merge) {
        std::set<int> ids_intersection;
        std::set_intersection(
            ids.begin(), ids.end(), vaildIds.begin(), vaildIds.end(),
            std::inserter(ids_intersection, ids_intersection.begin()));

        vaildIds = ids_intersection;
    }
}

std::vector<int> CoarseMatcher::randomSelectPoints(int pointsNum, int selectedNum) {
    std::vector<int> ids;
    std::vector<int> numbers(pointsNum);
    std::iota(numbers.begin(), numbers.end(), 0);
    std::random_device rd;
    std::mt19937 generator(rd());
    int current_size = pointsNum;
    for (int i{0}; i < selectedNum; ++i) {
        std::uniform_int_distribution<> distrib(0, numbers.size() - 1);
        int random_id = distrib(generator);
        ids.emplace_back(numbers[random_id]);
        std::swap(numbers[random_id], numbers[current_size - 1]);
        current_size--;
    }
    return ids;
}

void CoarseMatcher::convertFeaturePointsToObjectPoints(
    cv::Mat& depth, std::vector<cv::Point2f> keypoints, std::vector<Eigen::Vector3f>& objectPoints,
    const Eigen::Matrix3f k) {
    for (int i{0}; i < keypoints.size(); ++i) {
        cv::Point2f keypoint = keypoints[i];
        float depth_measure = depth.at<float>(keypoint.y, keypoint.x);
        Eigen::Vector3f point = depth_measure * k * Eigen::Vector3f(keypoint.x, keypoint.y, 1.0f);
        objectPoints.emplace_back(point);
    }
}

void CoarseMatcher::extractTargetPointCloud(
    const cv::Mat& depth, const Eigen::Matrix3f k, std::vector<Eigen::Vector3f>& pointclouds) {
    cv::Mat processed_depth, mask;
}

void CoarseMatcher::extractFPFHMatchedPointCloud(
    const std::vector<Eigen::Vector3f>& unmatched_source,
    const std::vector<Eigen::Vector3f>& unmatched_taregt,
    std::vector<Eigen::Vector3f>& matched_source, std::vector<Eigen::Vector3f>& matched_target) {
    pointcloud::Ptr source, target;
    source = convertStyle(unmatched_source);
    target = convertStyle(unmatched_taregt);
    if (source->empty() || target->empty()) {
        std::cout << "Has no vaild points" << std::endl;
        return;
    }
    fpfhFeature::Ptr source_fpfh, target_fpfh;
    source_fpfh = compueFPFHFeature(source);
    target_fpfh = compueFPFHFeature(target);
    if (source_fpfh->empty() || target_fpfh->empty()) {
        std::cout << "Has no fpfh feature point" << std::endl;
        return;
    }

    std::cout << source_fpfh->size() << std::endl;
    std::cout << target_fpfh->size() << std::endl;
    pcl::KdTreeFLANN<pcl::FPFHSignature33>::Ptr tree(new pcl::KdTreeFLANN<pcl::FPFHSignature33>);
    tree->setInputCloud(target_fpfh);

    std::vector<Eigen::Vector3f> source_matched_points;
    std::vector<Eigen::Vector3f> target_matched_points;

    for (size_t i = 0; i < source_fpfh->size(); ++i) {
        std::vector<int> nn_indices(1);
        std::vector<float> nn_distances(1);

        if (tree->nearestKSearch(source_fpfh->points[i], 1, nn_indices, nn_distances) > 0) {
            if (nn_distances[0] < fpfh_matched_threahold_) {
                int target_index = nn_indices[0];
                const pcl::PointXYZ& p_source = source->points[i];
                const pcl::PointXYZ& p_target = target->points[target_index];

                source_matched_points.emplace_back(p_source.x, p_source.y, p_source.z);
                target_matched_points.emplace_back(p_target.x, p_target.y, p_target.z);
            }
        }
    }
}

pointcloud::Ptr CoarseMatcher::convertStyle(const std::vector<Eigen::Vector3f>& input) {
    pointcloud::Ptr cloud(new pointcloud);
    cloud->points.reserve(input.size());

    for (const auto& vec : input) {
        PointT p;
        p.x = vec[0];
        p.y = vec[1];
        p.z = vec[2];
        cloud->points.push_back(p);
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    return cloud;
}

fpfhFeature::Ptr CoarseMatcher::compueFPFHFeature(pointcloud::Ptr pointcloud) {
    pointnormal::Ptr normals(new pointnormal);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> n;
    n.setInputCloud(pointcloud);
    n.setNumberOfThreads(8);
    n.setSearchMethod(tree);
    n.setKSearch(30);
    n.compute(*normals);

    fpfhFeature::Ptr fpfh(new fpfhFeature);
    pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fest;
    fest.setNumberOfThreads(8);
    fest.setInputCloud(pointcloud);
    fest.setInputNormals(normals);
    fest.setSearchMethod(tree);
    fest.setKSearch(40);
    fest.compute(*fpfh);

    return fpfh;
}
}  // namespace surface_reconstruction
