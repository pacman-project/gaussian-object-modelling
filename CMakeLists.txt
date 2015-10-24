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
)

# ToDo: port to 1.7.2
# find_package(PCL REQUIRED)

find_package(Eigen REQUIRED)

generate_messages(
  DEPENDENCIES
  std_msgs
  std_msgs
  geometry_msgs
  sensor_msgs
  visualization_msgs
)


catkin_package()

include_directories(include
  ${catkin_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  # ${PCL_INCLUDE_DIRS}
)

add_executable(regression_server
  src/regression_server.cpp
)

add_dependencies(regression_server 
  gp_regression_generate_messages_cpp
)

target_link_libraries(regression_server 
  ${catkin_LIBRARIES}
  # ${PCL_LIBRARIES}
)


## Tests

add_executable(test_gaussian
  tests/test_gaussian.cpp
)

add_dependencies(test_gaussian 
  gp_regression_generate_messages_cpp
)

target_link_libraries(test_gaussian 
  ${catkin_LIBRARIES}
  # ${PCL_LIBRARIES}
)