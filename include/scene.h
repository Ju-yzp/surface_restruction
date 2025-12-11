#ifndef SCENE_H_
#define SCENE_H_

// eigen
#include <Eigen/Eigen>

// cpp
#include <climits>
#include <cmath>
#include <memory>
#include <optional>
#include <set>

#include <settings.h>

namespace surface_reconstruction {
// 哈希表实体
struct HashEntry {
    // 指向的体素块的坐标
    Eigen::Vector3i pos{Eigen::Vector3i::Zero()};
    // 发生冲突时，指向的冲突列表元素的全局偏移量
    int offset{-1};
    // 指向的体素块地址偏移量
    int ptr;
    // 该所对应的体素块是否被使用标志
    bool isUsed{false};
};

// 体素
struct Voxel {
    float sdf = 1.0f;
    float w_depth = 0;
};

// TODO：当前版本不支持动态内存
class Scene {
public:
    Scene(std::shared_ptr<Settings> settings) {
        settings_ = settings;

        hashEntries_ = new HashEntry[settings_->sdf_bucket_num + settings_->sdf_excess_list_size];

        freeVoxelBlockNum = settings_->sdf_bucket_num + settings_->sdf_excess_list_size;

        voxel_ = new Voxel
            [(settings_->sdf_bucket_num + settings_->sdf_excess_list_size) *
             settings_->sdf_block_size3];
#pragma omp parallel for
        for (int i = 0; i < (settings_->sdf_bucket_num + settings_->sdf_excess_list_size); ++i)
            hashEntries_[i].ptr = i;
        freeExcessEntries_.resize(settings_->sdf_excess_list_size);
        for (int i = 0; i < settings_->sdf_excess_list_size; ++i)
            freeExcessEntries_[i] = settings_->sdf_bucket_num + i + 1;
    }

    ~Scene() {
        delete[] voxel_;
        delete[] hashEntries_;
    }

    // 哈希表索引函数
    int getHashIndex(Eigen::Vector3i voxelBlockPos) {
        return (((uint)voxelBlockPos(0) * 73856093u) ^ ((uint)voxelBlockPos(1) * 19349669u) ^
                ((uint)voxelBlockPos(2) * 83492791u)) &
               (uint)settings_->sdf_hash_mask;
    }

    // 获取体素块地址
    Voxel* get_voxelBolck(int id) { return voxel_ + id * settings_->sdf_block_size3; }

    // 清除上一帧的可见体素块索引并交换上一帧和当前帧的数据指针以及其他数据
    void swapVisibleList() {
        lastFrametVisibleVoxelBlockList_.clear();
        lastFrametVisibleVoxelBlockList_.swap(currentFrameVisibleVoxelBlockIdList_);
    }

    // 设置哈希表entry
    void set_entry(HashEntry hashEntry, int id) { hashEntries_[id] = hashEntry; }

    // 获取相对的哈希表entry
    HashEntry& get_entry(int id) { return hashEntries_[id]; }

    void set_entryOffset(int id, int offset) { hashEntries_[id].offset = offset; }

    // 获取空闲的冲突列表所对应的Entry指针
    std::optional<HashEntry*> get_freeExcessEntry() {
        if (freeExcessEntries_.empty()) return std::nullopt;
        int id = freeExcessEntries_.back();
        freeExcessEntries_.pop_back();
        return &hashEntries_[id];
    }

    Settings get_settings() const { return *settings_; }

    const std::set<int>& get_constLastFrameVisibleVoxelBlockList() const {
        return lastFrametVisibleVoxelBlockList_;
    }

    std::set<int>& get_lastFrameVisibleVoxelBlockList() { return lastFrametVisibleVoxelBlockList_; }

    const std::set<int>& get_constCurrentFrameVisibleVoxelBlockList() const {
        return currentFrameVisibleVoxelBlockIdList_;
    }

    std::set<int>& get_currentFrameVisibleVoxelBlockList() {
        return currentFrameVisibleVoxelBlockIdList_;
    }

    void freeExcessEntry(int id) {
        if (id - settings_->sdf_bucket_num + 1 >= 0) {
            freeExcessEntries_.push_back(id);
            hashEntries_[id].isUsed = false;
            hashEntries_[id].offset = -1;
        }
    }

    void resetEntry(int id) {
        hashEntries_[id].isUsed = false;
        hashEntries_[id].offset = -1;
    }

    bool hasFreeScene() { return freeVoxelBlockNum >= 0; }

    void allocateVoxelBlock() {
        if (hasFreeScene()) --freeVoxelBlockNum;
    }

private:
    Voxel* voxel_{nullptr};

    HashEntry* hashEntries_{nullptr};

    std::vector<int> freeExcessEntries_;

    std::shared_ptr<Settings> settings_;

    std::set<int> lastFrametVisibleVoxelBlockList_;

    std::set<int> currentFrameVisibleVoxelBlockIdList_;

    int freeVoxelBlockNum;
};
}  // namespace surface_reconstruction

#endif  // SCENE_H_
