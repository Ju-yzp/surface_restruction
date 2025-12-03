#include <localMapManager.h>
#include <cassert>
#include <optional>

namespace surface_reconstruction {
ActiveMapDescriptor LocalMapManager::createNewLocalMap(bool isPrimary) {
    ActiveMapDescriptor descriptor;
    descriptor.id = static_cast<int>(activeLocalMaps_.size());
    descriptor.state = isPrimary ? LocalMapState::PRIMARY_LOCAL_MAP : LocalMapState::NEW_LOCAL_MAP;
    activeLocalMaps_.emplace_back(descriptor);
    return descriptor;
}

std::optional<int> LocalMapManager::getPrimaryLocalMapIndex() const {
    for (const auto& map : activeLocalMaps_)
        if (map.state == LocalMapState::PRIMARY_LOCAL_MAP) return map.id;
    return std::nullopt;
}

void LocalMapManager::recordTrackingResult(
    int localMapId, std::shared_ptr<TrackingState> tracking_state) {
    assert(localMapId <= static_cast<int>(activeLocalMaps_.size()));

    ActiveMapDescriptor& currentLocalMap = activeLocalMaps_[localMapId];

    std::optional<int> primaryLocalMapId = getPrimaryLocalMapIndex();

    if (tracking_state->get_tracking_result() == TrackingState::TrackingResult::TRACKING_GOOD) {
        if (currentLocalMap.state == LocalMapState::RELOCALISATION)
            ;
        else if (
            (currentLocalMap.state == LocalMapState::NEW_LOCAL_MAP ||
             currentLocalMap.state == LocalMapState::LOOPCOUSED) &&
            primaryLocalMapId.has_value()) {
        }
    } else if (
        tracking_state->get_tracking_result() == TrackingState::TrackingResult::TRACKING_FAILED) {
        if (currentLocalMap.state == LocalMapState::PRIMARY_LOCAL_MAP)
            for (auto& localMap : activeLocalMaps_) {
                if (localMap.state == LocalMapState::NEW_LOCAL_MAP)
                    localMap.state = LocalMapState::NEW_LOST;
                else
                    localMap.state = LocalMapState::LOST;
            }
    }
}

LocalMapManager::RelocalisationResult LocalMapManager::checkSuccessNewLink(int id) {
    if (activeLocalMaps_[id].continueSucessTrackAfterReloc >= settings_->nRelocSucess)
        return RelocalisationResult::RELOCALISATION_SUCESS;
    if ((settings_->nRelocTrials - activeLocalMaps_[id].trackingAttempts) <
        (settings_->nRelocSucess - activeLocalMaps_[id].continueSucessTrackAfterReloc))
        return RelocalisationResult::RELOCALISATION_FAILD;

    return RelocalisationResult::RELOCALISATION_TRYING;
}

bool LocalMapManager::maintainActiveData() {
    std::optional<int> moveLocalMapId;

    for (int i = 0; i < activeLocalMaps_.size(); ++i) {
        ActiveMapDescriptor& localMap = activeLocalMaps_[i];
        if (localMap.state == LocalMapState::RELOCALISATION) {
            RelocalisationResult relocResult = checkSuccessNewLink(i);
            if (relocResult == RelocalisationResult::RELOCALISATION_SUCESS) {
                if (!moveLocalMapId.has_value())
                    moveLocalMapId = i;
                else
                    localMap.state = LocalMapState::LOST;
            } else if (relocResult == RelocalisationResult::RELOCALISATION_FAILD)
                localMap.state = LocalMapState::LOST;
        }
    }

    std::vector<int> restartLinksToLocalMaps;
    std::optional<int> primaryId;

    for (int i = 0; i < activeLocalMaps_.size(); ++i) {
        ActiveMapDescriptor& localMap = activeLocalMaps_[i];

        if (moveLocalMapId.has_value() && moveLocalMapId.value() == i)
            localMap.state = LocalMapState::PRIMARY_LOCAL_MAP;

        if (localMap.state == LocalMapState::PRIMARY_LOCAL_MAP && moveLocalMapId.has_value() &&
            i != moveLocalMapId.value()) {
            localMap.state = LocalMapState::LOST;
            restartLinksToLocalMaps.emplace_back(localMap.id);
        }
    }
}
}  // namespace surface_reconstruction
