#ifndef SETTINGS_H_
#define SETTINGS_H_

#include <cstdint>
#include <opencv2/opencv.hpp>

namespace surface_reconstruction {
struct Settings {
    // 彩色图像尺寸
    cv::Size2i rgb_imageSize{0, 0};

    // 深度图像尺寸
    cv::Size2i depth_imageSize{0, 0};

    // 体素分辨率
    float voxelSize{0.005f};

    // 一个体素块所对应的体素边长
    int sdf_block_size{8};

    // 一个体素块所包含的体素数量
    int sdf_block_size3{512};

    // 哈希表掩码
    uint32_t sdf_hash_mask{0xfffff};

    // 哈希表桶数量
    uint32_t sdf_bucket_num{0x100000};

    // 哈希表冲突列表大小
    uint32_t sdf_excess_list_size{0x20000};

    // 截断距离
    float mu{0.02f};

    // 最大深度权重
    float maxWeight{100.0f};

    // 判断是否需要重新生成点云的阈值
    float regenerate_pointcloud_threahold{0.1f};

    // 朝向权重
    float orientation_weight{1.0f};

    // 平移权重
    float translation_weight{0.6f};

    // 图像金字塔层数
    int nPyramidLevel{4};

    // 重定位成功次数
    int nRelocSucess{30};

    // 重定位尝试次数最大次数
    int nRelocTrials{20};

    // 追踪最少有效点
    int minNVaildPoints{1000};

    // LM的lamdba尺度因子
    float lamdbaScale{7.0f};
};
}  // namespace surface_reconstruction
#endif  // SETTINGS_H_
