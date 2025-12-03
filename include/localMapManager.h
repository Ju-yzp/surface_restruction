#ifndef LOCAL_MAP_MANAGER_H_
#define LOCAL_MAP_MANAGER_H_

// cpp
#include <settings.h>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <localMap.h>
#include <trackingState.h>

namespace surface_reconstruction {

enum class LocalMapState : uint8_t {
    PRIMARY_LOCAL_MAP = 0,
    NEW_LOCAL_MAP = 1,
    NEW_LOST = 2,
    LOST = 3,
    LOOPCOUSED = 4,
    RELOCALISATION = 5
};

struct ActiveMapDescriptor {
    LocalMapState state{LocalMapState::NEW_LOCAL_MAP};
    int id{-1};
    int trackingAttempts{0};
    int continueSucessTrackAfterReloc{0};
};

class LocalMapManager {
public:
    ActiveMapDescriptor createNewLocalMap(bool isPrimary = false);

    bool isStartNewArea();

    std::optional<int> getPrimaryLocalMapIndex() const;

    void recordTrackingResult(int localMapId, std::shared_ptr<TrackingState> tracking_state);

    bool maintainActiveData();

private:
    enum class RelocalisationResult : uint8_t {
        RELOCALISATION_FAILD = 0,
        RELOCALISATION_TRYING = 1,
        RELOCALISATION_SUCESS = 2,
    };

    std::shared_ptr<Settings> settings_;

    RelocalisationResult checkSuccessNewLink(int id);

    std::vector<ActiveMapDescriptor> activeLocalMaps_;

    std::vector<std::shared_ptr<LocalMap>> allLocalMaps_;
};
}  // namespace surface_reconstruction

#endif  // LOCAL_MAP_MANAGER_H_
