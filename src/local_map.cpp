#include <localMap.h>
#include <memory>

namespace surface_reconstruction {
LocalMap::LocalMap(Eigen::Matrix4f estimatedGlobalPose, const std::shared_ptr<Settings> settings)
    : estimatedGlobalPose_(estimatedGlobalPose) {
    scene_ = std::make_shared<Scene>(settings);
}
}  // namespace surface_reconstruction
