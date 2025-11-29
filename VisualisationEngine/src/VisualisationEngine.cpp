// VisualisationEigene
#include <VisualisationEngine/VisualisationEngine.h>

// eigen
#include <Eigen/Core>

// cpp
#include <algorithm>
#include <cmath>
#include <iterator>
#include <memory>
#include <optional>

void checkVoxelVisibility(
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

bool checkVoxelBlockvisibility(
    Eigen::Vector4f blockPos, Eigen::Matrix3f& project_m, Eigen::Matrix4f& pose_m,
    Eigen::Vector2i& imgSize, float voxelSize) {
    Eigen::Vector4f point;
    float factor = voxelSize * SDF_BLOCK_SIZE;
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

void VisualisationEngine::processFrame(
    std::shared_ptr<Scene> scene, std::shared_ptr<View> view,
    std::shared_ptr<TrackingState> trackingState) {
    allocateMemoryFromDepth(scene, view, trackingState);

    /* TODO:在查看上一帧的体素能在当前相机视锥体内被观察到部分的话，就把索引添加至当前可视帧体素块索引列表中
            我们或许该引进删除重复的元素，避免后续重复操作，同时如果跟踪的比较好，就说明笛卡尔系运动速度较慢，
            同时在李代姿态下变化也比较小
    */
    std::set<int>& lastFrametVisibleVoxelBlockList = scene->get_lastFrameVisibleVoxelBlockList();
    std::set<int>& currentFrameVisibleVoxelBlockIdList =
        scene->get_currentFrameVisibleVoxelBlockList();

    std::set<int> result;

    std::set_difference(
        lastFrametVisibleVoxelBlockList.begin(), lastFrametVisibleVoxelBlockList.end(),
        currentFrameVisibleVoxelBlockIdList.begin(), currentFrameVisibleVoxelBlockIdList.end(),
        std::inserter(result, result.begin()));

    Eigen::Matrix3f project_m = view->calibrationParams.depth.k;
    Eigen::Matrix4f pose_m = trackingState->pose_d;
    Eigen::Vector2i imageSize(view->depth.cols, view->depth.rows);
    float voxelSize = scene->get_sceneParams().voxelSize;

    for (auto entryId : result) {
        HashEntry hashEntry = scene->get_entry(entryId);
        Eigen::Vector4f p;
        p << hashEntry.pos(0), hashEntry.pos(1), hashEntry.pos(2), 1.0f;
        if (checkVoxelBlockvisibility(p, project_m, pose_m, imageSize, voxelSize))
            currentFrameVisibleVoxelBlockIdList.emplace(entryId);
    }

    integrateIntoScene(scene, view, trackingState);
}

// TODO：相对于上一个版本，我们直接在函数内部进行分配内存，同时更新可视化entry列表
void VisualisationEngine::allocateMemoryFromDepth(
    std::shared_ptr<Scene> scene, std::shared_ptr<View> view,
    std::shared_ptr<TrackingState> trackingState) {
    // 图像尺寸
    int rows = view->depth.rows;
    int cols = view->depth.cols;

    // 视锥体范围
    float viewFrustum_max = view->calibrationParams.viewFrustum_max;
    float viewFrustum_min = view->calibrationParams.viewFrustum_min;

    // 场景参数
    SceneParams sceneParams = scene->get_sceneParams();
    const float mu = sceneParams.mu;
    const float oneOverVoxelSize = 1.0f / (sceneParams.voxelSize * SDF_BLOCK_SIZE);

    // 步长次数
    int nstep, norm;

    Eigen::Vector3f point_in_camera, direction;
    Eigen::Matrix3f inv_depth = view->calibrationParams.depth.k_inv;
    Eigen::Matrix4f pose_inv = trackingState->pose_d.inverse();

    // 深度图对应像素的深度值
    float depth_measure;

    // 深度图像数据指针
    float* depth = (float*)view->depth.data;

    // 当前帧可视化哈希表entry列表
    std::set<int>& currentFrameVisibleVoxelBlockList =
        scene->get_currentFrameVisibleVoxelBlockList();

    // 遍历深度图像
    for (int y = 0; y < rows; ++y) {
        int offset = y * cols;
        for (int x = 0; x < cols; ++x) {
            depth_measure = depth[offset + x];
            if (depth_measure < 1e-4 || (depth_measure - mu) < viewFrustum_min ||
                (depth_measure + mu) > viewFrustum_max)
                continue;

            // 获取在depth相机下的点云数据
            point_in_camera(2) = 1.0f;
            point_in_camera(0) = x;
            point_in_camera(1) = y;
            point_in_camera = inv_depth * point_in_camera * depth_measure;

            norm = point_in_camera.norm();

            // 获取从该点云延伸的截断线段的起始点和终点
            Eigen::Vector3f point_s =
                (pose_inv *
                 (Eigen::Vector4f() << point_in_camera * (1.0f - mu / norm), 1.0f).finished())
                    .head(3) *
                oneOverVoxelSize;
            Eigen::Vector3f point_e =
                (pose_inv *
                 (Eigen::Vector4f() << point_in_camera * (1.0f + mu / norm), 1.0f).finished())
                    .head(3) *
                oneOverVoxelSize;

            direction = point_e - point_s;
            nstep = (int)ceil(2.0f * direction.norm());
            direction /= (float)(nstep - 1);

            for (int i = 0; i < nstep; ++i) {
                Eigen::Vector3i blockPos(
                    (int)std::floor(point_s(0)), (int)std::floor(point_s(1)),
                    (int)std::floor(point_s(2)));

                int hashId = Scene::getHashIndex(blockPos);

                bool isFound{false};

                HashEntry hashEntry = scene->get_entry(hashId);

                if (hashEntry.pos == blockPos && hashEntry.isUsed) {
                    currentFrameVisibleVoxelBlockList.emplace(hashId);
                    isFound = true;
                } else if (!hashEntry.isUsed) {  // 在哈希表上进行分配
                    hashEntry.pos = blockPos;
                    hashEntry.isUsed = true;
                    scene->set_entry(hashEntry, hashId);
                    currentFrameVisibleVoxelBlockList.emplace(hashId);
                    isFound = true;
                } else {  // 在冲突列表上进行分配
                    int currentEntryId = hashId;
                    while (scene->get_entry(currentEntryId).offset > -1) {
                        currentEntryId = scene->get_entry(currentEntryId).offset;
                        HashEntry currentHashEntry = scene->get_entry(currentEntryId);
                        if (currentHashEntry.pos == blockPos && currentHashEntry.isUsed) {
                            currentFrameVisibleVoxelBlockList.emplace(currentEntryId);
                            isFound = true;
                            break;
                        }
                    }

                    if (!isFound) {
                        std::optional<HashEntry*> childHashEntry = scene->get_freeExcessEntry();
                        if (!childHashEntry.has_value()) continue;
                        childHashEntry.value()->pos = blockPos;
                        childHashEntry.value()->isUsed = true;
                        scene->set_entryOffset(currentEntryId, childHashEntry.value()->ptr);
                        currentFrameVisibleVoxelBlockList.emplace(childHashEntry.value()->ptr);
                    }
                }

                point_s += direction;
            }
        }
    }
}

void VisualisationEngine::integrateIntoScene(
    std::shared_ptr<Scene> scene, std::shared_ptr<View> view,
    std::shared_ptr<TrackingState> trackingState) {
    float* depth = (float*)view->depth.data;

    const std::set<int>& currentFrameVisibleVoxelBlockIdList =
        scene->get_constCurrentFrameVisibleVoxelBlockList();

    Eigen::Matrix3f k = view->calibrationParams.rgb.k;

    int rows = view->depth.rows;
    int cols = view->depth.cols;

    SceneParams sceneParams = scene->get_sceneParams();

    float old_sdf, new_sdf, old_weight, new_weight, depth_measure, eta;
    float mu = sceneParams.mu;
    float max_weight = sceneParams.maxWeight;
    float voxelSize = sceneParams.voxelSize;

    Eigen::Matrix4f pose = trackingState->pose_d;

    std::vector<int> entryIds;
    for (std::set<int>::iterator it = currentFrameVisibleVoxelBlockIdList.begin();
         it != currentFrameVisibleVoxelBlockIdList.end(); ++it)
        entryIds.push_back(*it);
#pragma omp parallel for
    for (int i = 0; i < entryIds.size(); ++i) {
        HashEntry currentHashEntry = scene->get_entry(entryIds[i]);
        Voxel* localVoxelBlock = scene->get_voxelBolck(currentHashEntry.ptr);

        Eigen::Vector3i globalPos = currentHashEntry.pos;
        globalPos *= SDF_BLOCK_SIZE;

        for (int z = 0; z < SDF_BLOCK_SIZE; ++z)
            for (int y = 0; y < SDF_BLOCK_SIZE; ++y)
                for (int x = 0; x < SDF_BLOCK_SIZE; ++x) {
                    int localId = x + y * SDF_BLOCK_SIZE + z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;

                    // 计算世界坐标系下的位置，需要从体素块的表达方式进行计算,转换至笛卡尔坐标系
                    Eigen::Vector4f point_in_world;
                    point_in_world(0) = (globalPos(0) + x) * voxelSize;
                    point_in_world(1) = (globalPos(1) + y) * voxelSize;
                    point_in_world(2) = (globalPos(2) + z) * voxelSize;
                    point_in_world(3) = 1.0f;

                    // 在相机坐标系下的坐标
                    Eigen::Vector4f point_in_camera;
                    point_in_camera = pose * point_in_world;

                    if (point_in_camera(2) < 1e-5) continue;

                    Eigen::Vector3f pointImage = k * point_in_camera.head(3);
                    pointImage /= pointImage(2);

                    if (pointImage(0) < 1 || pointImage(0) > cols - 2 || pointImage(1) < 1 ||
                        pointImage(1) > rows - 2)
                        continue;

                    depth_measure =
                        depth[((int)(pointImage(0) + 0.5) + (int)(pointImage(1) + 0.5) * cols)];
                    if (depth_measure < 1e-6) continue;
                    eta = depth_measure - point_in_camera(2);

                    // 不在截断区域内,跳过不更新
                    if (std::abs(eta) > mu) continue;

                    // 更新sdf值
                    old_sdf = localVoxelBlock[localId].sdf;
                    old_weight = localVoxelBlock[localId].w_depth;
                    if (old_weight == max_weight) continue;
                    new_sdf = std::max(-1.0f, std::min(1.0f, eta / mu));
                    new_weight = 1.0f;

                    new_sdf = old_weight * old_sdf + new_weight * new_sdf;
                    new_weight = new_weight + old_weight;
                    new_sdf /= new_weight;

                    new_weight = std::min(new_weight, max_weight);

                    localVoxelBlock[localId].sdf = new_sdf;
                    localVoxelBlock[localId].w_depth = new_weight;
                }
    }
}

// 获取点所对应的体素在相对应的体素块中偏移量，同时获取体素块的坐标，用于哈希查找
int pointToVoxelBlockPos(Eigen::Vector3i& point, Eigen::Vector3i& blockPos) {
    blockPos(0) = (point(0) < 0 ? point(0) - SDF_BLOCK_SIZE + 1 : point(0)) / SDF_BLOCK_SIZE;
    blockPos(1) = (point(1) < 0 ? point(1) - SDF_BLOCK_SIZE + 1 : point(1)) / SDF_BLOCK_SIZE;
    blockPos(2) = (point(2) < 0 ? point(2) - SDF_BLOCK_SIZE + 1 : point(2)) / SDF_BLOCK_SIZE;

    Eigen::Vector3i locPos = point - blockPos * SDF_BLOCK_SIZE;
    return locPos(0) + locPos(1) * SDF_BLOCK_SIZE + locPos(2) * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
}

Voxel readVoxel(std::shared_ptr<Scene> scene, Eigen::Vector3i point) {
    Eigen::Vector3i blockPos;
    int linearId = pointToVoxelBlockPos(point, blockPos);

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

Eigen::Vector3i IntToFloor(Eigen::Vector3f& in, Eigen::Vector3f& other) {
    Eigen::Vector3i tmp = in.array().floor().cast<int>();
    other = in - tmp.cast<float>();
    return tmp;
}

float readFromSDFInterpolated(std::shared_ptr<Scene> scene, Eigen::Vector3f point) {
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

float readFromSDFUninterpolated(std::shared_ptr<Scene> scene, Eigen::Vector3f point) {
    return readVoxel(scene, Eigen::Vector3i(point.array().floor().cast<int>())).sdf;
}

// 三线性插值法
float readWithConfidenceFromSDFInterpolated(
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

void VisualisationEngine::raycast(
    std::shared_ptr<Scene> scene, std::shared_ptr<View> view,
    std::shared_ptr<TrackingState> trackingState, std::shared_ptr<RenderState> renderState) {
    int rows = renderState->raycastResult.rows;
    int cols = renderState->raycastResult.cols;

    // 相机重投影
    Eigen::Matrix3f k_inv = view->calibrationParams.depth.k_inv;

    // 视锥体范围
    float viewFrustum_max = view->calibrationParams.viewFrustum_max;
    float viewFrustum_min = view->calibrationParams.viewFrustum_min;

    // 转换至全局坐标系
    Eigen::Matrix4f inv_m = trackingState->pose_d.inverse();

    float oneOverVoxelSize = 1.0f / scene->get_sceneParams().voxelSize;

    float stepScale = scene->get_sceneParams().mu * oneOverVoxelSize;

    cv::Vec4f* data = (cv::Vec4f*)renderState->raycastResult.data;

#pragma omp parallel for
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            float stepLen;
            float totalLenght, totalLenghtMax;
            // 前进方向
            Eigen::Vector3f rayDirection;

            Eigen::Vector3f pointImage(x, y, 1.0f);
            Eigen::Vector3f temp = k_inv * pointImage;
            Eigen::Vector3f pointE = temp * viewFrustum_max;

            Eigen::Vector3f point_e =
                (inv_m * (Eigen::Vector4f() << pointE, 1.0f).finished()).head(3) * oneOverVoxelSize;

            Eigen::Vector3f pointS = temp * viewFrustum_min;
            Eigen::Vector3f point_s =
                (inv_m * (Eigen::Vector4f() << pointS, 1.0f).finished()).head(3) * oneOverVoxelSize;

            totalLenght = pointS.norm() * oneOverVoxelSize;
            totalLenghtMax = pointE.norm() * oneOverVoxelSize;

            rayDirection = point_e - point_s;
            rayDirection.normalize();

            Eigen::Vector3f pt_result = point_s;

            float sdf_v;

            bool pointFound{false};

            float confidence;

            // TODO：这里的sdf_v判断阈值不应该使用硬编码，这样子的形式不仅需要程序员细心，
            // 还需要对于tsdf有着比较深入的理解，后面会改成软编码形式
            while (totalLenght < totalLenghtMax) {
                sdf_v = readFromSDFUninterpolated(scene, pt_result);
                if (sdf_v < 0.6f && sdf_v > -0.6f)
                    sdf_v = readFromSDFInterpolated(scene, pt_result);
                if (sdf_v < 0.0f) break;
                stepLen = std::max(sdf_v * stepScale, 1.0f);

                pt_result += stepLen * rayDirection;
                totalLenght += stepLen;
            }

            if (sdf_v < 0.0f) {
                stepLen = sdf_v * stepScale;
                pt_result += stepLen * rayDirection;

                sdf_v = readWithConfidenceFromSDFInterpolated(scene, pt_result, confidence);

                stepLen = sdf_v * stepScale;
                pt_result += stepLen * rayDirection;
                pointFound = true;
            }

            data[x + y * cols] = cv::Vec4f(
                pt_result(0), pt_result(1), pt_result(2), pointFound ? confidence + 1.0f : 0.0f);
        }
    }
}

bool computeNormalAndAngle(
    bool smoothing, bool flipNormals, Eigen::Vector3f lightSource, cv::Vec2i point, float voxelSize,
    cv::Vec4f* pointRay, cv::Size2i imgSize, cv::Vec3f& output_normmal, float& angle) {
    cv::Vec4f rayPoints[4];
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

    cv::Vec4f diff_x, diff_y;

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

    output_normmal(0) = -(diff_x(1) * diff_y(2) - diff_x(2) * diff_y(1));
    output_normmal(1) = -(diff_x(2) * diff_y(0) - diff_x(0) * diff_y(2));
    output_normmal(2) = -(diff_x(0) * diff_y(1) - diff_x(1) * diff_y(0));

    if (flipNormals) output_normmal *= -1.0f;
    Eigen::Vector3f temp(output_normmal[0], output_normmal[1], output_normmal[2]);
    output_normmal *= (1.0f / temp.norm());

    angle = temp.transpose() * lightSource;
    if (!(angle > 0.0)) return false;
    return true;
}

void VisualisationEngine::produceNeedICP(
    Eigen::Vector3f lightSource, float voxelSize, std::shared_ptr<RenderState> renderState,
    std::shared_ptr<TrackingState> trackingState) {
    int rows = trackingState->pointsMap.rows;
    int cols = trackingState->pointsMap.cols;

    cv::Vec4f* point = (cv::Vec4f*)trackingState->pointsMap.data;
    cv::Vec4f* normal = (cv::Vec4f*)trackingState->normalMap.data;
    cv::Vec4f* rayPoint = (cv::Vec4f*)renderState->raycastResult.data;

    cv::Size2i imgSize(cols, rows);

    cv::Vec3f output_normmal;
#pragma omp parallel for
    for (int y = 0; y < rows; ++y) {
        int offset = y * cols;
        for (int x = 0; x < cols; ++x) {
            float angle;
            bool found = computeNormalAndAngle(
                renderState->smoothing, renderState->flipNormals, lightSource, cv::Vec2i(x, y),
                voxelSize, rayPoint, imgSize, output_normmal, angle);

            int id = x + offset;
            if (found) {
                point[id] = rayPoint[id] * voxelSize;
                point[id](3) = 1.0f;
                normal[id] =
                    cv::Vec4f{output_normmal(0), output_normmal(1), output_normmal(2), 0.0f};
            } else {
                cv::Vec4f output;
                output(3) = -1.0f;
                point[id] = output;
                normal[id] = output;
            }
        }
    }
}

void VisualisationEngine::prepare(
    std::shared_ptr<Scene> scene, std::shared_ptr<View> view,
    std::shared_ptr<TrackingState> trackingState, std::shared_ptr<RenderState> renderState) {
    raycast(scene, view, trackingState, renderState);

    Eigen::Vector3f lightSource = trackingState->pose_d.inverse().col(2).head(3);

    produceNeedICP(lightSource, scene->get_sceneParams().voxelSize, renderState, trackingState);
}
