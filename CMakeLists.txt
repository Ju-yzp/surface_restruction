cmake_minimum_required(VERSION 3.10)
project(surface_retruction VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Debug
      CACHE STRING "Choose the type of build." FORCE)
endif()

cmake_policy(SET CMP0072 NEW)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(g2o REQUIRED)
find_package(Sophus REQUIRED)
find_package(OpenMP REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(visualization_msgs REQUIRED)

# set(OrbbecSDK_DIR "/home/adrewn/OrbbecSDK") find_package(OrbbecSDK REQUIRED)

add_subdirectory(VisualisationEngine)
# # add_subdirectory(Engines) add_subdirectory(FernRelocLib)
# add_subdirectory(Tracker) add_executable(collect_data test/collect_data.cpp)

# # target_include_directories(collect_data PRIVATE  ${OpenCV_INCLUDE_DIRS})
# target_link_libraries(collect_data PRIVATE OrbbecSDK::OrbbecSDK
# ${OpenCV_LIBS})

# add_executable(test_relocaiser test/test_relocaiser.cpp)
# target_link_libraries(test_relocaiser PRIVATE FernRelocLib)

add_executable(test_visualisation test/test_visualisation.cpp)
target_include_directories(
  test_visualisation
  PUBLIC ${PROJECT_SOURCE_DIR}/Tracker/include
         ${PROJECT_SOURCE_DIR}/VisualisationEngine/include
         ${Eigen3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${G2O_INCLUDE_DIRS}
  # ${rclcpp_INCLUDE_DIRS} ${visualization_msgs_INCLUDE_DIRS}
)

target_link_libraries(
  test_visualisation
  PUBLIC ${OpenCV_LIBS} ${G2O_LIBRARIES} Sophus::Sophus
         # rclcpp::rclcpp # ${visualization_msgs_LIBRARIES}
         VisualisationEngine)

# add_executable(compute_pointcloud test/compute_pointcloud.cpp)
# target_include_directories( compute_pointcloud PUBLIC
# ${PROJECT_SOURCE_DIR}/Tracker/include ${Eigen3_INCLUDE_DIRS}
# ${OpenCV_INCLUDE_DIRS} ${G2O_INCLUDE_DIRS})

# target_link_libraries(compute_pointcloud PUBLIC ${OpenCV_LIBS}
# ${G2O_LIBRARIES} Sophus::Sophus)

# add_executable(test_tracker test/test_tracker.cpp) target_include_directories(
# test_tracker PUBLIC ${PROJECT_SOURCE_DIR}/Tracker/include
# ${Eigen3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${G2O_INCLUDE_DIRS})

# target_link_libraries(test_tracker PUBLIC ${OpenCV_LIBS} ${G2O_LIBRARIES}
# Eigen3::Eigen Sophus::Sophus tracker)
