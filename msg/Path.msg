#Contains ref. frame and timestamp
std_msgs/Header header
#Contains coordinates of the point to explore
geometry_msgs/PointStamped[] points
#Contains the suggested direction of approach to the point
geometry_msgs/Vector3Stamped[] directions
