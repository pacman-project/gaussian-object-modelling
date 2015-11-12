#ifndef _INCL_GP_NODE_H
#define _INCL_GP_NODE_H

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
//GP
#include <gp/GaussianProcess.h>
#include <gp/SampleSet.h>

using namespace gp;

/*
 *              PLEASE LOOK at TODOs by searching "TODO" to have an idea of
 *              what is still missing or is improvable!
 */


/**\brief Class GaussianProcessNode
 * {Wraps Gaussian process into a ROS node}
*/
class GaussianProcessNode
{
    public:
        /**\brief Constructor */
        GaussianProcessNode ();
        /**\brief Destructor */
        virtual ~GaussianProcessNode (){}

        /**\brief Node Handle*/
        ros::NodeHandle nh;

        //Sample some points and query the gp, build a point cloud reconstructed
        //model and publish it.
        void sampleAndPublish ();

    private:
        //input point cloud
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr hand_ptr;
        //Services and publishers
        ros::ServiceServer srv_start;
        ros::Publisher pub_model;
        //control if we can start processing
        bool start_gp;
        //reconstructed model cloud to republish
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_ptr;
        //Laplace regressor for the model
        //this ideally gets updated with new points when they arrive
        LaplaceRegressor::Ptr gp;


        //callback to start process service, executes when service is called
        bool cb_start(gp_regression::start_process::Request& req, gp_regression::start_process::Response& res);

        //gp computation
        void compute();
        //update gaussian model with new points from probe
        void update();
        //Republish cloud method
        void publishCloudModel();
};
#endif
