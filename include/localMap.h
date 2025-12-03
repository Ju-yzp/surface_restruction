#ifndef LOCAL_MAP_H_
#define LOCAL_MAP_H_

#include <scene.h>
#include <memory>

#include <settings.h>

namespace surface_reconstruction {
class LocalMap {
public:
    LocalMap(Eigen::Matrix4f estimatedGlobalPoseconst, const std::shared_ptr<Settings> settings);

    std::shared_ptr<Scene> get_scene() { return scene_; }

private:
    // 子地图所拥有的场景数据
    std::shared_ptr<Scene> scene_;

    // 子地图在全局坐标系中的位姿估计
    Eigen::Matrix4f estimatedGlobalPose_;
};
}  // namespace surface_reconstruction

#endif  // LOCAL_MAP_H_
