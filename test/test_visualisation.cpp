// cpp
#include <memory>

// opencv
#include <opencv2/opencv.hpp>

// tracker
#include <Tracker/cameraParams.h>
#include <Tracker/pixelUtils.h>

// visualsation
#include <VisualisationEngine/TrackingState.h>
#include <VisualisationEngine/View.h>
#include <VisualisationEngine/VisualisationEngine.h>
#include <VisualisationEngine/VoxelBlockHash.h>

cv::Mat getDepth(std::string file_path) {
    cv::Mat origin = cv::imread(file_path, cv::IMREAD_UNCHANGED);
    cv::Mat convert;
    convertShortToFloat(&origin, &convert, 5000.0f);
    return convert;
}

int main() {
    // 相机参数
    Intrinsic depth(525.0f, 525.0f, 319.5f, 239.5f);
    RGBDCalibrationParams calibrationParams(depth, depth, Eigen::Matrix4f(), 0.3f, 4.0f, 5000.0f);

    // 深度图
    std::string file1_path = "/home/adrewn/surface_restruction/data/1305031102.194330.png";
    std::shared_ptr<View> view = std::make_shared<View>(calibrationParams);
    view->depth = getDepth(file1_path);

    // 哈希表
    std::shared_ptr<VoxelBlockHash> vbh = std::make_shared<VoxelBlockHash>();
    vbh->mu_ = 0.02f;
    vbh->max_w_ = 100.0f;
    vbh->voxelSize_ = 0.05f;

    std::shared_ptr<TrackingState> ts = std::make_shared<TrackingState>();
    {
        Eigen::Vector3f translation(1.3352f, 0.6261f, 1.6519f);
        Eigen::Quaternionf rotation_q(-0.3231f, 0.6564f, 0.6139f, -0.2963f);
        rotation_q.normalize();
        Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
        T.block<3, 3>(0, 0) = rotation_q.toRotationMatrix();
        T.block<3, 1>(0, 3) = translation;
        ts->pose_d = T;
    }

    //可视化引擎
    VisualisationEngine ve;
    ve.reset(view->depth.rows * view->depth.cols);
    ve.processFrame(vbh, view, ts);
    // {
    // Eigen::Vector3f translation(1.3434,0.6271,1.6606);
    // Eigen::Quaternionf rotation_q( -0.3266,0.6583,0.6112,-0.2938);
    // rotation_q.normalize();
    // Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    // T.block<3, 3>(0, 0) = rotation_q.toRotationMatrix();
    // T.block<3, 1>(0, 3) = translation;
    // ts->pose_d = T;
    // }
    // file1_path = "/home/adrewn/surface_restruction/data/1305031102.160407.png";
    // view->depth = getDepth(file1_path);
    // ve.processFrame(vbh, view, ts);
}
