#include <VisualisationEngine/VisualisationEngine.h>

#include <Eigen/Core>
#include <algorithm>
#include <cmath>

void VisualisationEngine::processFrame(
    std::shared_ptr<VoxelBlockHash> vbh, std::shared_ptr<View> view,
    std::shared_ptr<TrackingState> trackingState) {
    // 获取分配内存请求
    allocateMemoryFromDepth(vbh, view, trackingState);

    // 哈希表项
    HashEntry* hashTable = vbh->get_entryData();
    #pragma omp parallel for
    for(int i = 0 ; i  < allocateRequests_.size(); ++i){
        auto request = allocateRequests_[i];
        HashEntry hashEntry;
        hashEntry.pos = request.head(3);
        hashEntry.ptr = request(3);
        hashTable[request(3)] = hashEntry;
    }

    allocateRequests_.clear();
    integrateIntoScene(vbh, view, trackingState);
}

void VisualisationEngine::raycast(
    std::shared_ptr<VoxelBlockHash> vbh, RenderState* renderState,
    RGBDCalibrationParams* cameraParams, const Eigen::Matrix4f inv_m) {
    int rows = renderState->raycastResult.rows;
    int cols = renderState->raycastResult.cols;

    float mu;                // tsdf截断距离
    float oneOverVoxelSize;  // 一米所能容纳的体素个数
    float stepScale = mu * oneOverVoxelSize;

    Eigen::Vector3f imagePoint;

    // 视锥体范围
    float viewFrustum_max = cameraParams->viewFrustum_max;
    float viewFrustum_min = cameraParams->viewFrustum_min;

    // 重投影
    Eigen::Matrix3f k_inv_depth = cameraParams->depth.k_inv;

    // 光线投射的方向
    Eigen::Vector3f rayDirection = Eigen::Vector3f::Zero();

    float stepLen;
#pragma omp parallel for
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            // 图像齐次坐标
            imagePoint(2) = 1.0f;
            imagePoint(0) = x;
            imagePoint(1) = y;

            Eigen::Vector3f point_s =
                (inv_m *
                 (Eigen::Vector4f() << k_inv_depth * imagePoint * viewFrustum_min, 1.0f).finished())
                    .head(3) *
                oneOverVoxelSize;

            Eigen::Vector3f point_e =
                (inv_m *
                 (Eigen::Vector4f() << k_inv_depth * imagePoint * viewFrustum_max, 1.0f).finished())
                    .head(3) *
                oneOverVoxelSize;

            rayDirection = (point_e - point_s).normalized();

            u_char sdf_v;

            while (true) {  // 沿着光线出发，在这个范围内寻找等值面

                if (sdf_v < 0.0f) {  // 此时需要回退，同时步长乘以
                }
            }
        }
    }
}

void VisualisationEngine::allocateMemoryFromDepth(
    std::shared_ptr<VoxelBlockHash> vbh, std::shared_ptr<View> view,
    std::shared_ptr<TrackingState> trackingState) {
    int rows = view->depth.rows;
    int cols = view->depth.cols;

    // 视锥体范围
    float viewFrustum_max = view->calibrationParams.viewFrustum_max;
    float viewFrustum_min = view->calibrationParams.viewFrustum_min;

    // 截断距离
    float mu = vbh->mu_;

    float oneOverVoxelSize = 1.0f / (vbh->voxelSize_ * SDF_BLOCK_SIZE);

    float norm{0.0f};

    int nstep;

    Eigen::Vector3f point_in_camera;
    Eigen::Matrix3f inv_depth = view->calibrationParams.depth.k_inv;
    Eigen::Matrix4f pose_inv = trackingState->pose_d.inverse();
    Eigen::Vector3f direction;

    HashEntry* hashTable = vbh->get_entryData();

    float* depth = (float*)view->depth.data;

    for (int y = 0; y < rows; ++y) {
        int offset = y * cols;
        for (int x = 0; x < cols; ++x) {
            // 获取深度值
            float depth_measure = depth[offset + x];
            if (depth_measure < 1e-4 || (depth_measure - mu) < viewFrustum_min ||
                (depth_measure - mu) < viewFrustum_min || (depth_measure + mu) > viewFrustum_max)
                continue;

            // 获取在depth下的点云数据
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
                Eigen::Vector3i blockPos(point_s(0), point_s(1), point_s(2));

                int hashId = VoxelBlockHash::getHashIndex(blockPos);

                bool isFound{false};

                HashEntry hashEntry = hashTable[hashId];

                // 如果已经分配相应的体素块，那么就直接退出
                if (hashEntry.pos == blockPos && hashEntry.ptr > -1) {
                    visibleEntryIds_.push_back(hashId);
                    isFound = true;
                }

                if (!isFound) {
                    if (hashEntry.ptr > -1) {
                        while (hashEntry.offset >= 1) {
                            hashId = SDF_BUCKET_NUM + hashEntry.offset - 1;
                            hashEntry = hashTable[hashId];
                            if (hashEntry.pos == blockPos && hashEntry.ptr > -1) {
                                isFound = true;
                                break;
                            }
                        }
                    }

                    if (!isFound) {
                        allocateRequests_.push_back(
                            (Eigen::Vector4i() << blockPos, hashId).finished());
                        visibleEntryIds_.push_back(hashId);
                    }
                }

                point_s += direction;
            }
        }
    }
}

void VisualisationEngine::integrateIntoScene(
    std::shared_ptr<VoxelBlockHash> vbh, std::shared_ptr<View> view,
    std::shared_ptr<TrackingState> trackingState) {
    HashEntry* hashTable = vbh->get_entryData();
    Voxel* localVoxels = vbh->get_voxelData();

    float voxelSize = vbh->voxelSize_;

    Eigen::Matrix4f pose = trackingState->pose_d;

    float* depth = (float*)view->depth.data;
    
    #pragma omp parallel for
    for(int i = 0; i < visibleEntryIds_.size(); ++i){
        int entryId = visibleEntryIds_[i];
        HashEntry currentHashEntry = hashTable[entryId];

        Eigen::Vector3i globalPos = currentHashEntry.pos;
        globalPos *= SDF_BLOCK_SIZE;

        Voxel* localVoxelBlock = &(localVoxels[currentHashEntry.ptr * SDF_BLOCK_SIZE3]);
        Eigen::Matrix3f k = view->calibrationParams.rgb.k;
        int rows = view->depth.rows;
        int cols = view->depth.cols;

        float old_sdf, new_sdf, old_weight, new_weight, depth_measure, eta, mu = vbh->mu_;
        float max_weight;
        for (int z = 0; z < SDF_BLOCK_SIZE; ++z) {
            for (int y = 0; y < SDF_BLOCK_SIZE; ++y) {
                for (int x = 0; x < SDF_BLOCK_SIZE; ++x) {
                    int localId = x + y * SDF_BLOCK_SIZE + z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;

                    // 计算世界坐标系下的位置，需要从体素的表达方式进行计算
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

                    depth_measure = depth[(int)(pointImage(0) + pointImage(1) * cols)];
                    eta = depth_measure - point_in_camera(2);

                    if (eta < -mu) continue;
                    // 更新sdf值
                    old_sdf = localVoxelBlock[localId].sdf;
                    old_weight = localVoxelBlock[localId].w_depth;

                    new_sdf = std::min(1.0f, eta / mu);

                    new_sdf = old_sdf * old_sdf + new_sdf * new_sdf;
                    new_weight = new_weight + old_weight;
                    new_sdf /= new_weight;
                    new_weight = std::min(new_weight, max_weight);

                    localVoxelBlock[localId].sdf = new_sdf;
                    localVoxels[localId].w_depth = new_weight;
                }
            }
        }
    }

    visibleEntryIds_.clear();
}
