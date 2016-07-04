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
  shape_msgs
  cmake_modules
  pacman_vision_comm
  roslib
  tf
)

#ToDo: port to 1.7.2
#TODO:  1.7.1 is  fine for  just loading  clouds, readding  it back  at whatever
#       version it finds on system (Fri 06 Nov 2015 04:48:02 PM CET -- Tabjones)
find_package(PCL 1.7.2 REQUIRED)

find_package(Eigen REQUIRED)

find_package(octomap REQUIRED)

# find_package(OpenMP)
# if (OPENMP_FOUND)
#     set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# endif()

add_service_files(
    FILES
    StartProcess.srv
    GetNextBestPath.srv
    Update.srv
    # StopExploration.srv
    )

add_message_files(
    FILES
    Path.msg
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
  shape_msgs
)

catkin_package(
    CATKIN_DEPENDS roscpp message_runtime pacman_vision_comm message_generation
    )

file(GLOB_RECURSE gp_INCLUDES RELATIVE ${PROJECT_SOURCE_DIR} include/*.h include/*.hpp)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${Eigen_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${OCTOMAP_INCLUDE_DIRS}
)

# ROS node
add_executable(gaussian_process
    ${gp_INCLUDES}
    src/gp_node.cpp
)

#i ROS test node for surfaces
add_executable(surface_gp
    ${gp_INCLUDES}
    src/surface_approx.cpp
)

add_dependencies(gaussian_process
  gp_regression_generate_messages_cpp
)
add_dependencies(surface_gp
  gp_regression_generate_messages_cpp
)

target_link_libraries(gaussian_process
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OCTOMAP_LIBRARIES}
)

target_link_libraries(surface_gp
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

## Tests

# add_executable(test_gaussian
#   tests/test_gaussian.cpp
#   ${gp_INCLUDES}
# )

# add_executable(test_gp
#   tests/test_gp.cpp
#   src/gp/SampleSet.cpp
#   # ${gp_INCLUDES}
# )

# add_executable(test_eigen
#   tests/test_eigen.cpp
# )

# add a target to generate API documentation with Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
add_custom_target(doc
${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
COMMENT "Generating API documentation with Doxygen" VERBATIM
)
endif(DOXYGEN_FOUND)
