// ROS headers
#include <ros/ros.h>
#include <ros/console.h>
#include <ros/package.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
//PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
// General Utils
#include <cmath>
#include <fstream>
#include <string>
#include <stdlib.h>
// Vision services
#include <pacman_vision_comm/get_cloud_in_hand.h>
// This node service
#include <gp_regression/start_process.h>

/**\brief Class GaussianProcessNode
 * {Wraps Gaussian process into a ROS node}
*/
class GaussianProcessNode
{
    public:
        /**\brief Constructor */
        GaussianProcessNode (): nh(ros::NodeHandle("gaussian_process")), start_gp(false)
        {
            srv_start = nh.advertiseService("start_process", &GaussianProcessNode::cb_start, this);
            //TODO: Added a  publisher to republish point cloud  with new points
            //from  gaussian process,  right now  it's unused  (Fri 06  Nov 2015
            //05:18:41 PM CET -- tabjones)
            pub_cloud = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB>> ("estimated_cloud", 1);
            cloud_ptr.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        }
        /**\brief Destructor */
        virtual ~GaussianProcessNode (){}
        /**\brief Node Handle*/
        ros::NodeHandle nh;
    private:
        //input point cloud
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr;
        //Services and publishers
        ros::ServiceServer srv_start;
        ros::Publisher pub_cloud;
        //control if we can start processing
        bool start_gp;
        //callback to start process service, executes when service is called
        bool cb_start(gp_regression::start_process::Request& req, gp_regression::start_process::Response& res)
        {
            if(req.cloud.empty()){
            //Request was empty, means we have to call pacman vision service to
            //get a cloud.
                std::string service_name = nh.resolveName("/pacman_vision/listener/get_cloud_in_hand");
                pacman_vision_comm::get_cloud_in_hand service;
                service.request.save = "false";
                //TODO:  Service requires  to know  which hand  is grasping  the
                //object we  dont have a way to tell inside here.  Assuming it's
                //the right  hand for now.(Fri  06 Nov  2015 05:11:45 PM  CET --
                //tabjones)
                service.request.right = true;
                if (!ros::service::call<pacman_vision_comm::get_cloud_in_hand>(service_name, service))
                {
                    ROS_ERROR("Get cloud in hand service call failed!");
                    return (false);
                }
                //obj cloud is saved into class, while hand cloud is wasted
                pcl::fromROSMsg (service.response.obj, *cloud_ptr);
                //pcl::fromROSMsg (service.response.hand, *hand_ptr); //snippet to save hand too
            }
            else{
            //User told us to load a cloud from disk instead.
            //TODO: Add some path checks perhaps? Lets hope the path is valid
            //      for now!  (Fri 06 Nov 2015 05:03:19 PM CET -- tabjones)
                if (pcl::io::loadPCDFile(req.cloud, *cloud_ptr) != 0){
                    ROS_ERROR("Error loading cloud from %s",req.cloud.c_str());
                    return (false);
                }
            }
            start_gp = true;
            return (true);
        }
};

int main (int argc, char *argv[])
{
    ros::init(argc, argv, "gaussian_process");
    GaussianProcessNode node;
    ros::Rate rate(20); //try to go at 20hz
    while (node.nh.ok())
    {
        //gogogo!
        ros::spinOnce();
        //TODO do something in the loop if we recived a start? i.e. start_gp == true
        rate.sleep();
    }
    //someone killed us :(
    return 0;
}
