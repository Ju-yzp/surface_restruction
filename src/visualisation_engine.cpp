#include <visualisationEngine.h>

#include <utils.h>

namespace surface_reconstruction {

void allocateVoxelFormDepth(
    std::shared_ptr<Scene> scene, std::shared_ptr<View> view,
    std::shared_ptr<TrackingState> trackingState, bool updateVisibleList) {
    int rows = view->depth.rows;
    int cols = view->depth.cols;

    float viewFrustum_max = view->calibrationParams.viewFrustum_max;
    float viewFrustum_min = view->calibrationParams.viewFrustum_min;

    Settings settings = scene->get_settings();
    const float mu = settings.mu;
    const float oneOverVoxelSize = 1.0f / (settings.voxelSize * settings.sdf_block_size);
    std::cout << oneOverVoxelSize << std::endl;

    Eigen::Matrix3f inv_depth = view->calibrationParams.depth.k_inv;
    Eigen::Matrix4f pose_inv = trackingState->get_current_camera_in_localmap().inverse();

    std::cout << pose_inv << std::endl;

    float* depth = (float*)view->depth.data;

    std::set<int>& currentFrameVisibleVoxelBlockList =
        scene->get_currentFrameVisibleVoxelBlockList();

    for (int y = 0; y < rows; ++y) {
        int offset = y * cols;
        for (int x = 0; x < cols; ++x) {
            float depth_measure;
            depth_measure = depth[offset + x];
            if (std::isnan(depth_measure) || (depth_measure - mu) < viewFrustum_min ||
                (depth_measure + mu) > viewFrustum_max)
                continue;
            int nstep;
            float norm;
            Eigen::Vector3f point_in_camera, direction;
            // 获取在depth相机下的点云数据
            point_in_camera(2) = 1.0f;
            point_in_camera(0) = x;
            point_in_camera(1) = y;
            point_in_camera = inv_depth * point_in_camera * depth_measure;
            norm = point_in_camera.norm();
            const float NORM_EPSILON = 1e-6f;

            if (norm < NORM_EPSILON) {
                continue;
            }
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

                int hashId = scene->getHashIndex(blockPos);

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
                    scene->allocateVoxelBlock();
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
                        scene->hasFreeScene();
                    }
                }

                point_s += direction;
            }
        }
    }
}

void intergrateIntoScene(
    std::shared_ptr<Scene> scene, std::shared_ptr<View> view,
    std::shared_ptr<TrackingState> trackingState) {
    float* depth = (float*)view->depth.data;

    const std::set<int>& currentFrameVisibleVoxelBlockIdList =
        scene->get_constCurrentFrameVisibleVoxelBlockList();

    std::cout << currentFrameVisibleVoxelBlockIdList.size() << std::endl;
    Eigen::Matrix3f k = view->calibrationParams.depth.k;

    int rows = view->depth.rows;
    int cols = view->depth.cols;

    Settings settings = scene->get_settings();

    float old_sdf, new_sdf, old_weight, new_weight, depth_measure, eta;
    float mu = settings.mu;
    float max_weight = settings.maxWeight;
    float voxelSize = settings.voxelSize;

    Eigen::Matrix4f pose = trackingState->get_current_camera_in_localmap();

    std::vector<int> entryIds;
    for (std::set<int>::iterator it = currentFrameVisibleVoxelBlockIdList.begin();
         it != currentFrameVisibleVoxelBlockIdList.end(); ++it)
        entryIds.push_back(*it);
#pragma omp parallel for
    for (int i = 0; i < entryIds.size(); ++i) {
        HashEntry currentHashEntry = scene->get_entry(entryIds[i]);
        Voxel* localVoxelBlock = scene->get_voxelBolck(currentHashEntry.ptr);

        Eigen::Vector3i globalPos = currentHashEntry.pos;
        globalPos *= settings.sdf_block_size;

        for (int z = 0; z < settings.sdf_block_size; ++z)
            for (int y = 0; y < settings.sdf_block_size; ++y)
                for (int x = 0; x < settings.sdf_block_size; ++x) {
                    int localId = x + y * settings.sdf_block_size +
                                  z * settings.sdf_block_size * settings.sdf_block_size;

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
                    if (std::isnan(depth_measure)) continue;
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

void raycast(
    std::shared_ptr<Scene> scene, std::shared_ptr<View> view,
    std::shared_ptr<TrackingState> trackingState, std::shared_ptr<RenderState> renderState) {
    int rows = renderState->rows;
    int cols = renderState->cols;

    Eigen::Matrix3f k_inv = view->calibrationParams.depth.k_inv;

    float viewFrustum_max = view->calibrationParams.viewFrustum_max;
    float viewFrustum_min = view->calibrationParams.viewFrustum_min;

    Eigen::Matrix4f inv_m = trackingState->get_current_camera_in_localmap().inverse();

    float oneOverVoxelSize = 1.0f / scene->get_settings().voxelSize;

    float stepScale = scene->get_settings().mu * oneOverVoxelSize;

    Eigen::Vector4f* data = renderState->raycastResult->data();

#pragma omp parallel for
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            float stepLen;
            float totalLenght, totalLenghtMax;

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

            data[x + y * cols] << pt_result, pointFound ? confidence + 1.0f : 0.0f;
        }
    }
}

void generatePointCloudsAndNormals(
    Eigen::Vector3f lightSource, float voxelSize, std::shared_ptr<RenderState> renderState,
    std::shared_ptr<TrackingState> trackingState) {
    int rows = trackingState->get_height();
    int cols = trackingState->get_width();

    Eigen::Vector4f* pointcloud = trackingState->get_pointclouds()->data();
    Eigen::Vector4f* normal = trackingState->get_normals()->data();
    Eigen::Vector4f* rayPoint = renderState->raycastResult->data();

    cv::Size2i imgSize(cols, rows);

    Eigen::Vector3f output_normmal;
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
                pointcloud[id] << rayPoint[id].head(3) * voxelSize, 1.0f;
                normal[id] << output_normmal, 0.0f;
            } else {
                pointcloud[id](3) = -1.0f;
                normal[id](3) = -1.0f;
            }
        }
    }
}

void processFrame(
    std::shared_ptr<Scene> scene, std::shared_ptr<View> view,
    std::shared_ptr<TrackingState> trackingState) {
    allocateVoxelFormDepth(scene, view, trackingState);

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
    Eigen::Matrix4f pose_m = trackingState->get_current_camera_in_localmap();
    Eigen::Vector2i imageSize(view->depth.cols, view->depth.rows);
    float voxelSize = scene->get_settings().voxelSize;
    int sdf_block_size = scene->get_settings().sdf_block_size;

    cv::Size2i depthSize(view->depth.rows, view->depth.cols);

    for (auto entryId : result) {
        HashEntry hashEntry = scene->get_entry(entryId);
        Eigen::Vector4f p;
        p << hashEntry.pos(0), hashEntry.pos(1), hashEntry.pos(2), 1.0f;
        if (checkVoxelBlockvisibility(p, project_m, pose_m, imageSize, voxelSize, sdf_block_size))
            currentFrameVisibleVoxelBlockIdList.emplace(entryId);
    }

    intergrateIntoScene(scene, view, trackingState);
}
}  // namespace surface_reconstruction
