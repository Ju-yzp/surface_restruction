#ifndef PIXEL_UTILS_H_
#define PIXEL_UTILS_H_

// opencv
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/matx.hpp>
#include <opencv2/opencv.hpp>

// eigen
#include <Eigen/Core>
#include <Eigen/Eigen>

// cpp
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <type_traits>

using NormlsAndPointsMap = std::shared_ptr<std::vector<Eigen::Vector4f>>;
using DepthImage = cv::Mat;

// 仅用于点云图、法向量图、深度图的下采样，因为彩色图不存在空洞(无效值或者不连续平面)，所以可以使用opencv接口
// 仅支持CV_32F、CV_32FC4格式
template <typename T>
inline void filterSubsampleWithHoles(
    T& input, T& output, Eigen::Vector2i input_size, bool isNormal = false) {
    int output_rows = input_size(0) / 2.0f;
    int output_cols = input_size(1) / 2.0f;

    if constexpr (std::is_same_v<T, NormlsAndPointsMap>) {
        if (!input || output)
            throw std::runtime_error("This input must be not empty and output have no memory");
        output = std::make_shared<std::vector<Eigen::Vector4f>>(output_rows * output_cols);
    } else if constexpr (std::is_same_v<T, DepthImage>) {
        if (input.empty() || !output.empty())
            throw std::runtime_error("This input must be not empty and output have no memory");
        output.create(output_rows, output_cols, CV_32F);
    }

    for (int y{0}; y < output_rows; ++y) {
        for (int x{0}; x < output_cols; ++x) {
            if constexpr (std::is_same_v<T, DepthImage>) {
                float pixel_in[4], pixel_out{0.0f};
                pixel_in[0] = input.template at<float>(y * 2, x * 2);
                pixel_in[1] = input.template at<float>(y * 2 + 1, x * 2 + 1);
                pixel_in[2] = input.template at<float>(y * 2 + 1, x * 2);
                pixel_in[3] = input.template at<float>(y * 2, x * 2 + 1);

                int nVaildPoints{0};
                for (int k{0}; k < 4; ++k)
                    if (pixel_in[k] >= 0.0f) {
                        pixel_out += pixel_in[k];
                        ++nVaildPoints;
                    }

                if (nVaildPoints > 0) pixel_out /= (float)nVaildPoints;
                output.template at<float>(y, x) = pixel_out;
            } else if constexpr (std::is_same_v<T, NormlsAndPointsMap>) {
                Eigen::Vector4f pixel_in[4], pixel_out;
                pixel_in[0] = (*input)[y * 2 * input_size(1) + x * 2];
                pixel_in[1] = (*input)[(y * 2 + 1) * input_size(1) + x * 2 + 1];
                pixel_in[2] = (*input)[(y * 2 + 1) * input_size(1) + x * 2];
                pixel_in[3] = (*input)[(y * 2) * input_size(1) + x * 2 + 1];

                int nVaildPoints{0};
                for (int k{0}; k < 4; ++k)
                    if (pixel_in[k](3) >= 0.0f) {
                        pixel_out += pixel_in[k];
                        ++nVaildPoints;
                    }

                if (nVaildPoints == 0) {
                    pixel_out(3) = -1.0f;

                    (*output)[x + y * output_cols] = pixel_out;
                    continue;
                }

                pixel_out /= (float)nVaildPoints;
                if (isNormal) {
                    float norm = pixel_out.head(3).norm();
                    pixel_out(0) /= norm;
                    pixel_out(1) /= norm;
                    pixel_out(2) /= norm;
                }
                (*output)[x + y * output_cols] = pixel_out;
            }
        }
    }
}

// 将无符号16位的原始深度图转换为以m为单位的float图像
inline void convertShortToFloat(cv::Mat* input, cv::Mat* output, float scale) {
    output->create(input->rows, input->cols, CV_32F);

    for (int y{0}; y < output->rows; ++y) {
        for (int x{0}; x < output->cols; ++x) {
            int32_t pixel_in = input->at<uint16_t>(y, x);
            if (pixel_in > 0) output->at<float>(y, x) = (float)pixel_in / scale;
        }
    }
}

// 深度图、法向量图、点云图插值
inline Eigen::Vector4f interpolateBilinear_withHoles(
    std::shared_ptr<std::vector<Eigen::Vector4f>> map, Eigen::Vector2f coorinate, int cols) {
    Eigen::Vector2i imgPoint((int)floor(coorinate(0)), (int)floor(coorinate(1)));

    auto a = (*map)[coorinate(0) + coorinate(1) * cols];
    auto b = (*map)[coorinate(0) + 1 + coorinate(1) * cols];
    auto c = (*map)[coorinate(0) + (coorinate(1) + 1) * cols];
    auto d = (*map)[coorinate(0) + 1 + (coorinate(1) + 1) * cols];
    Eigen::Vector4f result;
    Eigen::Vector2f delta{coorinate(0) - imgPoint(0), coorinate(1) - imgPoint(1)};

    if (a(3) < 0.0f || a(3) < 0.0f || c(3) < 0.0f || d(3) < 0.0f) {
        result(0) = 0.0f;
        result(1) = 0.0f;
        result(2) = 0.0f;
        result(3) = -1.0f;
        return result;
    }

    result(0) = a(0) * (1.0f - delta(0)) * (1.0f - delta(1)) + b(0) * delta(0) * (1.0f - delta(1)) +
                c(0) * (1.0f - delta(0)) * delta(1) + d(0) * delta(0) * delta(1);

    result(1) = a(1) * (1.0f - delta(0)) * (1.0f - delta(1)) + b(1) * delta(0) * (1.0f - delta(1)) +
                c(1) * (1.0f - delta(0)) * delta(1) + d(1) * delta(0) * delta(1);

    result(2) = a(2) * (1.0f - delta(0)) * (1.0f - delta(1)) + b(2) * delta(0) * (1.0f - delta(1)) +
                c(2) * (1.0f - delta(0)) * delta(1) + d(2) * delta(0) * delta(1);

    result(3) = a(3) * (1.0f - delta(0)) * (1.0f - delta(1)) + b(3) * delta(0) * (1.0f - delta(1)) +
                c(3) * (1.0f - delta(0)) * delta(1) + d(3) * delta(0) * delta(1);
    return result;
}

inline void computeNormalMap(
    int rows, int cols, const std::shared_ptr<std::vector<Eigen::Vector4f>> pointcloudMap,
    std::shared_ptr<std::vector<Eigen::Vector4f>> normalMap) {
    Eigen::Vector4f points[4];
    for (int y{0}; y < rows; ++y)
        for (int x{0}; x < cols; ++x) {
            Eigen::Vector4f& normal = (*normalMap)[x + y * cols];
            normal(3) = -1.0f;

            Eigen::Vector4f diff_x, diff_y;

            if (x <= 2 || x >= cols - 3 || y <= 2 || y >= rows - 3) continue;

            points[0] = (*pointcloudMap)[x + 2 + y * cols];
            points[1] = (*pointcloudMap)[x + (y + 2) * cols];
            points[2] = (*pointcloudMap)[x - 2 + y * cols];
            points[3] = (*pointcloudMap)[x + (y - 2) * cols];

            bool doPlus{false};

            if (points[0](3) < 0.0f || points[1](3) < 0.0f || points[2](3) < 0.0f ||
                points[3](3) < 0.0f)
                doPlus = true;
            if (doPlus) {
                points[0] = (*pointcloudMap)[x + 1 + y * cols];
                points[1] = (*pointcloudMap)[x + (y + 1) * cols];
                points[2] = (*pointcloudMap)[x - 1 + y * cols];
                points[3] = (*pointcloudMap)[x + (y - 1) * cols];
            }
            diff_x = points[0] - points[2];
            diff_y = points[1] - points[3];

            normal(0) = -(diff_x(1) * diff_y(2) - diff_x(2) * diff_y(1));
            normal(1) = -(diff_x(2) * diff_y(0) - diff_x(0) * diff_y(2));
            normal(2) = -(diff_x(0) * diff_y(1) - diff_x(1) * diff_y(0));

            float norm = sqrt(pow(normal(0), 2) + pow(normal(1), 2) + pow(normal(2), 2));

            if (normal.head(3).norm() < 1e-5) continue;
            normal(0) /= norm;
            normal(1) /= norm;
            normal(2) /= norm;
            normal(3) = 0.0f;
        }
}
#endif
