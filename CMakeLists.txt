cmake_minimum_required(VERSION 3.10)
project(surface_retruction VERSION 1.0)

set(CMAKE_CXX_STANDARD 20)
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

file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cxx")

# add_library(surface_retruction ${SOURCES}) target_include_directories(
# surface_retruction PUBLIC ${PROJECT_SOURCE_DIR}/include ${Eigen3_INCLUDE_DIRS}
# ${OpenCV_INCLUDE_DIRS} ${G2O_INCLUDE_DIRS})

# target_link_libraries(surface_retruction PUBLIC ${OpenCV_LIBS}
# ${G2O_LIBRARIES} Eigen3::Eigen Sophus::Sophus)

# add_executable(test_visualisation test/test_visualisation.cpp)
# target_include_directories( test_visualisation PUBLIC
# ${PROJECT_SOURCE_DIR}/Tracker/include
# ${PROJECT_SOURCE_DIR}/VisualisationEngine/include ${Eigen3_INCLUDE_DIRS}
# ${OpenCV_INCLUDE_DIRS} ${G2O_INCLUDE_DIRS} # ${rclcpp_INCLUDE_DIRS}
# ${visualization_msgs_INCLUDE_DIRS} )

# target_link_libraries( test_visualisation PUBLIC ${OpenCV_LIBS}
# ${G2O_LIBRARIES} Sophus::Sophus # rclcpp::rclcpp #
# ${visualization_msgs_LIBRARIES} VisualisationEngine)

add_executable(test_tracker test/test_tracker.cpp)
target_include_directories(
  test_tracker PUBLIC include ${Eigen3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS}
                      ${G2O_INCLUDE_DIRS} ${visualization_msgs_INCLUDE_DIRS})

target_link_libraries(test_tracker PUBLIC ${OpenCV_LIBS} ${G2O_LIBRARIES}
                                          Sophus::Sophus)
