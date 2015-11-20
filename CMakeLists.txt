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

# Golem
#SET(GOLEM_INCLUDE /usr/include CACHE PATH "Path prefix for Golem include")
#SET(GOLEM_LIBRARY /usr/lib CACHE PATH "Path prefix for Golem library")
#SET(GOLEM_BINARIES /usr/bin CACHE PATH "Path prefix for Golem binaries")

add_service_files(
    FILES
    start_process.srv
    )

add_message_files(
    FILES
    SampleToExplore.msg
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

include_directories(
    include
    node/include
    ${catkin_INCLUDE_DIRS}
    ${Eigen_INCLUDE_DIRS}
    #${GOLEM_INCLUDE}
    ${PCL_INCLUDE_DIRS}
)

add_executable(regression_server
  src/regression_server.cpp
)

# add_dependencies(regression_server
#   gp_regression_generate_messages_cpp
# )

target_link_libraries(regression_server
  ${catkin_LIBRARIES}
  #${GOLEM_LIBRARY}
  #${PCL_LIBRARIES}
)


## Tests

add_executable(test_gaussian
  tests/test_gaussian.cpp
)

# add_dependencies(test_gaussian
#   gp_regression_generate_messages_cpp
# )

target_link_libraries(test_gaussian
  ${catkin_LIBRARIES}
  #${PCL_LIBRARIES}
)

add_executable(test_gp
  tests/test_gp.cpp
  src/gp/SampleSet.cpp
)

# add_dependencies(test_gp
#   gp_regression_generate_messages_cpp
# )

target_link_libraries(test_gp
  ${catkin_LIBRARIES}
  # ${PCL_LIBRARIES}
)

#Add ROS node
add_executable(gaussian_process
    node/include/gp_node.h
    node/gp_node.cpp
    node/gp_node_main.cpp
    src/gp/SampleSet.cpp)

add_dependencies(gaussian_process
  gp_regression_generate_messages_cpp
)

target_link_libraries(gaussian_process
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)
