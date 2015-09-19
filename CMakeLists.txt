cmake_minimum_required(VERSION 2.8.3)
project(gaussian_object_modelling)

find_package(catkin REQUIRED COMPONENTS 
	roscpp
	message_generation
	std_msgs
	geometry_msgs
	sensor_msgs
	visualization_msgs
	urdf
	tf
	pcl_ros
	pacman_vision
)

# ToDo: port to 1.7.2
find_package(PCL REQUIRED)

add_service_files(
  FILES
  PickPoint.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)


catkin_package()

include_directories(include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_executable(object_model_node 
  src/object_model_node.cpp
)

add_dependencies(object_model_node 
  gaussian_object_modelling_generate_messages_cpp
)

target_link_libraries(object_model_node 
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)