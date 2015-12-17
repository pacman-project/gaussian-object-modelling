cmake_minimum_required(VERSION 2.8.3)
project(gp_regression)

add_definitions (-fpermissive -std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  message_generation
  std_msgs
  geometry_msgs
  sensor_msgs
  visualization_msgs
  cmake_modules
  pacman_vision_comm
)

#ToDo: port to 1.7.2
#TODO:  1.7.1 is  fine for  just loading  clouds, readding  it back  at whatever
#       version it finds on system (Fri 06 Nov 2015 04:48:02 PM CET -- Tabjones)
find_package(PCL REQUIRED)

find_package(Eigen REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_service_files(
    FILES
    StartProcess.srv
    GetToExploreTrajectory.srv
    )

add_message_files(
    FILES
    SampleToExplore.msg
    Chart.msg
    Atlas.msg
    MapCollection.msg
    )

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
  visualization_msgs
)

catkin_package(
    CATKIN_DEPENDS roscpp message_runtime pacman_vision_comm message_generation
    )

file(GLOB_RECURSE gp_INCLUDES RELATIVE ${PROJECT_SOURCE_DIR} include/*.h include/*.hpp)

include_directories(
    include
    src/include
    ${catkin_INCLUDE_DIRS}
    ${Eigen_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

# ROS node
add_executable(gaussian_process
    src/gp_node.cpp
    src/gp_node_main.cpp
    ${gp_INCLUDES}
)

add_dependencies(gaussian_process
  gp_regression_generate_messages_cpp
)

target_link_libraries(gaussian_process
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

## Tests

# add_executable(test_gaussian
#   tests/test_gaussian.cpp
#   ${gp_INCLUDES}
# )

# target_link_libraries(test_gaussian
#   ${catkin_LIBRARIES}
# )


