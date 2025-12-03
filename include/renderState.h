#ifndef RENDER_STATE_H_
#define RENDER_STATE_H_

#include <Eigen/Eigen>

#include <opencv2/opencv.hpp>

#include <memory>

namespace surface_reconstruction {
struct RenderState {
    RenderState(cv::Size imgSize) {
        raycastResult =
            std::make_shared<std::vector<Eigen::Vector4f>>(imgSize.height * imgSize.width);
        rows = imgSize.height;
        cols = imgSize.width;
    }

    std::shared_ptr<std::vector<Eigen::Vector4f>> raycastResult{nullptr};

    int rows;

    int cols;

    bool smoothing{true};

    bool flipNormals{false};
};
}  // namespace surface_reconstruction
#endif
