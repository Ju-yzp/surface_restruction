#ifndef UTILS_H_
#define UTILS_H_

#include <scene.h>
#include <settings.h>

namespace surface_reconstruction {

inline void checkVoxelVisibility(
    bool& isVisible, Eigen::Vector4f pt, Eigen::Matrix3f& project_m, Eigen::Matrix4f& pose_m,
    Eigen::Vector2i& imgSize) {
    pt = pose_m * pt;
    if (pt(2) < 1e-5) return;
    Eigen::Vector3f pointImage = project_m * pt.head(3);
    pointImage /= pointImage(2);

    if (pointImage(0) >= 0 && pointImage(0) < imgSize(0) && pointImage(1) >= 0 &&
        pointImage(1) < imgSize(1))
        isVisible = true;
}

inline bool checkVoxelBlockvisibility(
    Eigen::Vector4f blockPos, Eigen::Matrix3f& project_m, Eigen::Matrix4f& pose_m,
    Eigen::Vector2i& imgSize, float voxelSize, int sdf_block_size) {
    Eigen::Vector4f point;
    float factor = voxelSize * sdf_block_size;
    bool isVisible = false;

    for (int i = 0; i < 8; ++i) {
        point = blockPos * factor;
        if (i & 1) point(0) += factor;
        if (i & 2) point(1) += factor;
        if (i & 4) point(2) += factor;

        point(3) = 1.0f;
        isVisible = false;
        checkVoxelVisibility(isVisible, point, project_m, pose_m, imgSize);
        if (isVisible) return true;
    }
    return false;
}

inline int pointToVoxelBlockPos(
    Eigen::Vector3i& point, Eigen::Vector3i& blockPos, const int sdf_block_size) {
    blockPos(0) = (point(0) < 0 ? point(0) - sdf_block_size + 1 : point(0)) / sdf_block_size;
    blockPos(1) = (point(1) < 0 ? point(1) - sdf_block_size + 1 : point(1)) / sdf_block_size;
    blockPos(2) = (point(2) < 0 ? point(2) - sdf_block_size + 1 : point(2)) / sdf_block_size;

    Eigen::Vector3i locPos = point - blockPos * sdf_block_size;
    return locPos(0) + locPos(1) * sdf_block_size + locPos(2) * sdf_block_size * sdf_block_size;
}

inline Voxel readVoxel(std::shared_ptr<Scene> scene, Eigen::Vector3i point) {
    Eigen::Vector3i blockPos;
    int linearId = pointToVoxelBlockPos(point, blockPos, scene->get_settings().sdf_block_size);

    int hashId = Scene::getHashIndex(blockPos);

    HashEntry hashEntry;

    while (true) {
        hashEntry = scene->get_entry(hashId);
        if (hashEntry.pos == blockPos && hashEntry.isUsed) {
            Voxel* voxelBlock = scene->get_voxelBolck(hashEntry.ptr);
            return voxelBlock[linearId];
        }

        if (hashEntry.offset < 0) return Voxel();

        hashId = hashEntry.offset;
    }
}

inline Eigen::Vector3i IntToFloor(Eigen::Vector3f& in, Eigen::Vector3f& other) {
    Eigen::Vector3i tmp = in.array().floor().cast<int>();
    other = in - tmp.cast<float>();
    return tmp;
}

inline float readFromSDFInterpolated(std::shared_ptr<Scene> scene, Eigen::Vector3f point) {
    Eigen::Vector3f coeff;
    Eigen::Vector3i position = IntToFloor(point, coeff);

    float v1, v2, res1, res2;
    const float cx = coeff(0);
    const float cy = coeff(1);
    const float cz = coeff(2);

    v1 = readVoxel(scene, position + Eigen::Vector3i(0, 0, 0)).sdf;
    v2 = readVoxel(scene, position + Eigen::Vector3i(0, 0, 0)).sdf;
    res1 = (1.0f - cx) * v1 + cx * v2;

    v1 = readVoxel(scene, position + Eigen::Vector3i(0, 1, 0)).sdf;
    v2 = readVoxel(scene, position + Eigen::Vector3i(1, 1, 0)).sdf;
    res1 = (1.0f - cy) * res1 + cy * ((1.0f - cx) * v1 + cx * v2);

    v1 = readVoxel(scene, position + Eigen::Vector3i(0, 0, 1)).sdf;
    v2 = readVoxel(scene, position + Eigen::Vector3i(1, 0, 1)).sdf;
    res2 = (1.0f - cx) * v1 + cx * v2;

    v1 = readVoxel(scene, position + Eigen::Vector3i(0, 1, 1)).sdf;
    v2 = readVoxel(scene, position + Eigen::Vector3i(1, 1, 1)).sdf;
    res2 = (1.0f - cy) * res2 + cy * ((1.0f - cx) * v1 + cx * v2);

    return (1.0f - cz) * res1 + cz * res2;
}

inline float readFromSDFUninterpolated(std::shared_ptr<Scene> scene, Eigen::Vector3f point) {
    return readVoxel(scene, Eigen::Vector3i(point.array().floor().cast<int>())).sdf;
}

inline float readWithConfidenceFromSDFInterpolated(
    std::shared_ptr<Scene> scene, Eigen::Vector3f point, float& confidence) {
    Voxel voxel;
    float res1, res2, v1, v2;
    float res1_c, res2_c, v1_c, v2_c;

    Eigen::Vector3f coeff;
    Eigen::Vector3i position = IntToFloor(point, coeff);

    const float cx = coeff(0);
    const float cy = coeff(1);
    const float cz = coeff(2);

    voxel = readVoxel(scene, position + Eigen::Vector3i(0, 0, 0));
    v1 = voxel.sdf;
    v1_c = voxel.w_depth;
    voxel = readVoxel(scene, position + Eigen::Vector3i(1, 0, 0));
    v2 = voxel.sdf;
    v2_c = voxel.w_depth;
    res1 = (1.0f - cx) * v1 + cx * v2;
    res1_c = (1.0f - cx) * v1_c + cx * v2_c;

    voxel = readVoxel(scene, position + Eigen::Vector3i(0, 1, 0));
    v1 = voxel.sdf;
    v1_c = voxel.w_depth;
    voxel = readVoxel(scene, position + Eigen::Vector3i(1, 1, 0));
    v2 = voxel.sdf;
    v2_c = voxel.w_depth;
    res1 = (1.0f - cy) * res1 + cy * ((1.0f - cx) * v1 + cx * v2);
    res1_c = (1.0f - cy) * res1_c + cy * ((1.0f - cx) * v1_c + cx * v2_c);

    voxel = readVoxel(scene, position + Eigen::Vector3i(0, 0, 1));
    v1 = voxel.sdf;
    v1_c = voxel.w_depth;
    voxel = readVoxel(scene, position + Eigen::Vector3i(1, 0, 1));
    v2 = voxel.sdf;
    v2_c = voxel.w_depth;
    res2 = (1.0f - cx) * v1 + cx * v2;
    res2_c = (1.0f - cx) * v1_c + cx * v2_c;

    voxel = readVoxel(scene, position + Eigen::Vector3i(0, 1, 1));
    v1 = voxel.sdf;
    v1_c = voxel.w_depth;
    voxel = readVoxel(scene, position + Eigen::Vector3i(1, 1, 1));
    v2 = voxel.sdf;
    v2_c = voxel.w_depth;
    res2 = (1.0f - cy) * res2 + cy * ((1.0f - cx) * v1 + cx * v2);
    res2_c = (1.0f - cy) * res2_c + cy * ((1.0f - cx) * v1_c + cx * v2_c);

    confidence = (1.0f - cz) * res1_c + cz * res2_c;

    return (1.0f - cz) * res1 + cz * res2;
}

inline bool computeNormalAndAngle(
    bool smoothing, bool flipNormals, Eigen::Vector3f lightSource, cv::Vec2i point, float voxelSize,
    Eigen::Vector4f* pointRay, cv::Size2i imgSize, Eigen::Vector3f& output_normal, float& angle) {
    Eigen::Vector4f rayPoints[4];
    if (smoothing) {
        if (point(0) <= 2 || point(1) >= imgSize.height - 3 || point(1) <= 2 ||
            point(0) >= imgSize.width - 3)
            return false;

        rayPoints[0] = pointRay[(point(0) + 2) + point(1) * imgSize.width];
        rayPoints[1] = pointRay[point(0) + (point(1) + 2) * imgSize.width];
        rayPoints[2] = pointRay[(point(0) - 2) + point(1) * imgSize.width];
        rayPoints[3] = pointRay[point(0) + (point(1) - 2) * imgSize.width];
    } else {
        if (point(0) <= 1 || point(1) >= imgSize.height - 2 || point(1) <= 1 ||
            point(0) >= imgSize.width - 2)
            return false;

        rayPoints[0] = pointRay[(point(0) + 1) + point(1) * imgSize.width];
        rayPoints[1] = pointRay[point(0) + (point(1) + 1) * imgSize.width];
        rayPoints[2] = pointRay[(point(0) - 1) + point(1) * imgSize.width];
        rayPoints[3] = pointRay[point(0) + (point(1) - 1) * imgSize.width];
    }

    Eigen::Vector4f diff_x, diff_y;

    bool doPuls{false};
    if (rayPoints[0](3) < 0.0f || rayPoints[1](3) < 0.0f || rayPoints[2](3) < 0.0f ||
        rayPoints[3](3) < 0.0f)
        doPuls = true;

    else {
        diff_x = rayPoints[0] - rayPoints[2];
        diff_y = rayPoints[1] - rayPoints[3];

        float diff = std::max(
            Eigen::Vector4f(diff_x(0), diff_x(1), diff_x(2), diff_x(3)).norm(),
            Eigen::Vector4f(diff_y(0), diff_y(1), diff_y(2), diff_y(3)).norm());

        if ((diff * voxelSize * voxelSize) > (std::pow(0.15f, 2))) doPuls = true;
    }

    if (doPuls) {
        if (smoothing) {
            rayPoints[0] = pointRay[(point(0) + 1) + point(1) * imgSize.width];
            rayPoints[1] = pointRay[point(0) + (point(1) + 1) * imgSize.width];
            rayPoints[2] = pointRay[(point(0) - 1) + point(1) * imgSize.width];
            rayPoints[3] = pointRay[point(0) + (point(1) - 1) * imgSize.width];

            diff_x = rayPoints[0] - rayPoints[2];
            diff_y = rayPoints[1] - rayPoints[3];
            if (rayPoints[0](3) <= 0.0f || rayPoints[1](3) <= 0.0f || rayPoints[2](3) <= 0.0f ||
                rayPoints[3](3) <= 0.0f)
                return false;
        }
    }

    output_normal(0) = -(diff_x(1) * diff_y(2) - diff_x(2) * diff_y(1));
    output_normal(1) = -(diff_x(2) * diff_y(0) - diff_x(0) * diff_y(2));
    output_normal(2) = -(diff_x(0) * diff_y(1) - diff_x(1) * diff_y(0));

    if (flipNormals) output_normal *= -1.0f;
    Eigen::Vector3f temp(output_normal[0], output_normal[1], output_normal[2]);
    output_normal *= (1.0f / temp.norm());

    angle = temp.transpose() * lightSource;
    if (!(angle > 0.0)) return false;
    return true;
}
}  // namespace surface_reconstruction

#endif
