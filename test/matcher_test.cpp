#include <calibrationParams.h>
#include <opencv2/core/hal/interface.h>
#include <pixelUtils.h>
#include <settings.h>
#include <tracker.h>
#include <view.h>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

// ros2
#include <coarseMatcher.h>
#include <pixelUtils.h>
#include <scene.h>
#include <Eigen/Eigen>

#include <cmath>

void computePointCloudFromDepth(
    const cv::Mat& originDepth, std::vector<Eigen::Vector3f>& pointcloud, const float scale,
    const Eigen::Matrix3f k) {
    int rows = originDepth.rows;
    int cols = originDepth.cols;

    assert(originDepth.type() == CV_16U);

    Eigen::Matrix3f k_inv = k.inverse();

    for (int y{0}; y < rows; ++y) {
        for (int x{0}; x < cols; ++x) {
            uint16_t depth_origin = originDepth.at<uint16_t>(y, x);
            if (depth_origin > 0) {
                float depth_measure = (float)depth_origin / scale;
                Eigen::Vector3f point = depth_measure * k_inv * Eigen::Vector3f(x, y, 1.0f);
                pointcloud.emplace_back(point);
            }
        }
    }
}

int main(int argc, char* argv[]) {
    // 1. 读取原始深度图 (16位)
    std::string path1 = "/home/adrewn/surface_reconstruction/data/depth3.png";
    std::string path2 = "/home/adrewn/surface_reconstruction/data/depth4.png";

    cv::Mat depth1_16u = cv::imread(path1, cv::IMREAD_UNCHANGED);
    cv::Mat depth2_16u = cv::imread(path2, cv::IMREAD_UNCHANGED);

    if (depth1_16u.empty() || depth2_16u.empty()) {
        std::cerr << "Error: Failed to load images!" << std::endl;
        return -1;
    }

    cv::Mat orb_img1, orb_img2;
    cv::Mat norm1, norm2;

    Intrinsic depthIntrinsics(Eigen::Vector4f{504.867, 504.958, 333.731, 339.597});

    std::vector<Eigen::Vector3f> sourcePoints, targetPoints, matched_source, matched_target;
    computePointCloudFromDepth(depth1_16u, sourcePoints, 1000.0, depthIntrinsics.k);
    computePointCloudFromDepth(depth2_16u, targetPoints, 1000.0, depthIntrinsics.k);

    std::cout << sourcePoints.size() << std::endl;
    std::cout << targetPoints.size() << std::endl;
    surface_reconstruction::CoarseMatcher coarseMatcher(
        576, 640, 40, 50, 1000, 0.7, 0.06, 0.03, 6, 100);
    coarseMatcher.extractFPFHMatchedPointCloud(
        sourcePoints, targetPoints, matched_source, matched_target);
    std::cout << matched_source.size() << std::endl;
    // cv::normalize(depth1_16u, norm1, 0, 255, cv::NORM_MINMAX, CV_32F);
    // cv::normalize(depth2_16u, norm2, 0, 255, cv::NORM_MINMAX, CV_32F);

    // norm1.convertTo(orb_img1, CV_8UC1);
    // norm2.convertTo(orb_img2, CV_8UC1);

    // cv::Mat mask1 = createSafeMask(depth1_16u, 5);
    // cv::Mat mask2 = createSafeMask(depth2_16u, 5);

    // cv::Ptr<cv::ORB> orb = cv::ORB::create(10000, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31,
    // 3);

    // std::vector<cv::KeyPoint> keypoints1, keypoints2;
    // cv::Mat descriptors1, descriptors2;

    // orb->detectAndCompute(orb_img1, mask1, keypoints1, descriptors1);
    // orb->detectAndCompute(orb_img2, mask2, keypoints2, descriptors2);

    // std::cout << "Detected keypoints inside safe region: " << keypoints1.size() << ", "
    //           << keypoints2.size() << std::endl;

    // if (descriptors1.empty() || descriptors2.empty()) {
    //     std::cerr << "Error: No valid descriptors found." << std::endl;
    //     return -1;
    // }

    // // 4. 匹配
    // cv::Ptr<cv::BFMatcher> bf = cv::BFMatcher::create(cv::NORM_HAMMING, true);
    // std::vector<cv::DMatch> matches;
    // bf->match(descriptors1, descriptors2, matches);

    // // 排序
    // std::sort(matches.begin(), matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) {
    //     return a.distance < b.distance;
    // });

    // int num_good = std::min((int)matches.size(), 30);
    // std::vector<cv::DMatch> good_matches(matches.begin(), matches.begin() + num_good);

    // std::vector<cv::Point2f> src_pts_2d, dst_pts_2d;
    // extractMatchedCoordinates_Filtered(
    //     good_matches, keypoints1, keypoints2, src_pts_2d, dst_pts_2d, 50.0);

    // binary_mask.convertTo(keypoint_mask, CV_8UC1);

    // cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    // cv::morphologyEx(keypoint_mask, keypoint_mask, cv::MORPH_OPEN, kernel);
    // convertDepth(depth2_16u, depth2_, 1000.0f);
    // surface_reconstruction::GridFilter gridFilter(576, 640, 40, 50);

    // std::vector<cv::DMatch> filtered_matches_to_draw;

    // gridFilter.filter(src_pts_2d, dst_pts_2d);
    // ConvertImagePointsToObjectPoints(depth1_, src_pts_2d, sourcePoints, depthIntrinsics.k);

    // std::cout << "-----------------" << std::endl;
    // ConvertImagePointsToObjectPoints(depth2_, dst_pts_2d, targetPoints, depthIntrinsics.k);

    // auto T = coarseMatcher.ransacEvalue(sourcePoints, targetPoints);

    // 5. 准备显示 (转为 BGR 避免报错)

    // 遍历原始 good_matches
    // for (const auto& match : good_matches) {
    //     cv::Point2f pt1 = keypoints1[match.queryIdx].pt;
    //     cv::Point2f pt2 = keypoints2[match.trainIdx].pt;
    //     auto it = std::find_if(src_pts_2d.begin(), src_pts_2d.end(), [&pt1](const
    //     cv::Point2f& p)
    //     {
    //         return std::hypot(pt1.x - p.x, pt1.y - p.y) < 1e-4;
    //     });

    //     if (it != src_pts_2d.end()) {
    //         filtered_matches_to_draw.push_back(match);
    //     }
    // }

    // cv::Mat disp1, disp2;
    // cv::cvtColor(orb_img1, disp1, cv::COLOR_GRAY2BGR);
    // cv::cvtColor(orb_img2, disp2, cv::COLOR_GRAY2BGR);

    // // 8. 绘制过滤后的匹配点
    // cv::Mat img_matches;
    // cv::drawMatches(
    //     disp1, keypoints1, disp2, keypoints2, filtered_matches_to_draw, img_matches,
    //     cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
    //     cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // cv::namedWindow("ORB Matches (Filtered for RANSAC)", cv::WINDOW_NORMAL);
    // cv::imshow("ORB Matches (Filtered for RANSAC)", processed_depth);
    // cv::waitKey(0);

    return 0;
}
