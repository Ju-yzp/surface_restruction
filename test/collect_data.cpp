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
#include <queue>
#include <string>
#include <thread>

// opencv
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

class Camera {
public:
    Camera() {
        // get the device list
        try {
            ob::Context cxt;
            std::shared_ptr<ob::DeviceList> dev_list = cxt.queryDeviceList();
            if (dev_list->deviceCount() == 0) {
                return;
            }

            dev_ = dev_list->getDevice(0);

            pipe_ = std::make_shared<ob::Pipeline>(dev_);

            std::shared_ptr<ob::Config> config = std::make_shared<ob::Config>();

            config->enableVideoStream(OB_STREAM_COLOR, 1920, 1080, 30, OB_FORMAT_BGRA);

            config->enableVideoStream(OB_STREAM_DEPTH, 640, 576, 30, OB_FORMAT_Y16);

            // depth_ = new cv::Mat(cv::Size(640,576),CV_32F);

            pipe_->start(config, std::bind(&Camera::processFrame, this, std::placeholders::_1));

        } catch (ob::Error& e) {
            std::cerr << "function:" << e.getName() << "\nargs:" << e.getArgs()
                      << "\nmessage:" << e.getMessage() << "\ntype:" << e.getExceptionType()
                      << std::endl;
        }
    }

    ~Camera() {
        if (pipe_) pipe_->stop();

        is_stop_.store(true);

        if (display_thread_->joinable()) display_thread_->join();
    }

    void processFrame(std::shared_ptr<ob::FrameSet> frames) {
        auto depth_frame = frames->depthFrame();
        auto rgb_frame = frames->colorFrame();

        if (rgb_frame && depth_frame) {
            std::shared_ptr<cv::Mat> rgb =
                std::make_shared<cv::Mat>(cv::Size(1920, 1080), CV_8UC4, rgb_frame->data());
            std::shared_ptr<FrameData> fd =
                std::make_shared<FrameData>(rgb_frame, rgb, depth_frame);
            fd->frame = rgb_frame;
            fd->img = rgb;
            rgb_produce_queue_.emplace(fd);
        }

        if (rgb_produce_queue_.size() > 2) {
            spin_lock_.lock();
            rgb_produce_queue_.swap(rgb_resume_queue_);
            spin_lock_.unlock();
        }
    }

    void start() { display_thread_ = std::make_unique<std::thread>(&Camera::displayThread, this); }

    void displayThread() {
        cv::namedWindow("realtime", cv::WINDOW_AUTOSIZE);

        while (!is_stop_.load()) {
            spin_lock_.lock();

            if (!rgb_resume_queue_.empty()) {
                std::shared_ptr<FrameData> fd = rgb_resume_queue_.front();
                rgb_resume_queue_.pop();
                cv::imshow("realtime", *((*fd).img));

                int key = cv::waitKey(2);
                if (key == 's') {
                    cv::Mat depth = cv::Mat(cv::Size(640, 576), CV_16U, fd->depth->data());
                    cv::imwrite("Frame" + std::to_string(count) + ".png", depth);
                    count++;
                }
            }

            spin_lock_.unlock();
        }

        cv::destroyWindow("realtime");
    }

private:
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

    int count{3};
};

int main() {
    Camera camera;

    camera.start();

    while (true) {
        std::this_thread::sleep_for((std::chrono::milliseconds(10)));
    }

    // cv::Mat img = cv::imread("Frame3.png", cv::IMREAD_UNCHANGED);
    // if (!img.empty()) {
    //     cv::namedWindow("realtime", cv::WINDOW_AUTOSIZE);
    //     cv::Mat depth = cv::Mat(img.rows, img.cols, CV_32F);
    //     for (int i{0}; i < img.rows; ++i) {
    //         for (int j{0}; j < img.rows; ++j) {
    //             depth.at<float>(i, j) = (int)img.at<unsigned short>(i, j) / 30.0;
    //         }
    //     }

    //     cv::imshow("realtime", depth);
    //     cv::waitKey();
    //     cv::destroyWindow("realtime");
    // }
}
