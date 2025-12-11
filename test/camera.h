#ifndef CAMERA_H_
#define CAMEERA_H_

// orbbec sdk
#include <libobsensor/h/ObTypes.h>
#include <libobsensor/ObSensor.hpp>
#include <libobsensor/hpp/Context.hpp>
#include <libobsensor/hpp/Device.hpp>
#include <libobsensor/hpp/Frame.hpp>
#include <libobsensor/hpp/Pipeline.hpp>
#include <libobsensor/hpp/Sensor.hpp>
// cpp
#include <atomic>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

// opencv
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

// eigen
#include <pixelUtils.h>
#include <Eigen/Eigen>
#include "calibrationParams.h"
#include "settings.h"
#include "tracker.h"
#include "trackingState.h"
#include "view.h"
#include "visualisationEngine.h"

#include <scene.h>

class Camera {
public:
    Camera(
        std::shared_ptr<surface_reconstruction::Settings> settings,
        RGBDCalibrationParams calibrationParams);
    ~Camera();

    void start();

    void displayThread();

    std::shared_ptr<surface_reconstruction::Scene> get_scene() { return scene_; }

    std::mutex scene_mutex_;

private:
    void processFrame(std::shared_ptr<ob::FrameSet> frames);
    struct FrameData {
        FrameData(
            std::shared_ptr<ob::ColorFrame>& frame_, std::shared_ptr<cv::Mat>& img_,
            std::shared_ptr<ob::DepthFrame>& depth_)
            : frame(frame_), img(img_), depth(depth_) {}

        std::shared_ptr<ob::ColorFrame> frame = nullptr;
        std::shared_ptr<cv::Mat> img = nullptr;
        std::shared_ptr<ob::DepthFrame> depth = nullptr;
    };

    struct SpinLock {
        std::atomic_flag flag = ATOMIC_FLAG_INIT;
        void lock() {
            while (flag.test_and_set(std::memory_order_acquire)) {
            }
        }

        void unlock() { flag.clear(std::memory_order_release); }
    };

    std::shared_ptr<ob::Device> dev_;

    std::shared_ptr<ob::Pipeline> pipe_;

    std::unique_ptr<std::thread> display_thread_;

    std::queue<std::shared_ptr<FrameData>> rgb_produce_queue_;

    std::queue<std::shared_ptr<FrameData>> rgb_resume_queue_;

    std::atomic<bool> is_stop_{false};

    SpinLock spin_lock_;

    std::shared_ptr<surface_reconstruction::Scene> scene_;

    std::shared_ptr<surface_reconstruction::View> view_;

    std::shared_ptr<surface_reconstruction::TrackingState> trackingState_;

    std::shared_ptr<surface_reconstruction::Tracker> tracker_;

    bool isFirstFrame{true};

    bool trackingFaild{false};
};

#endif
