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
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/WrenchStamped.h>

//PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/io.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/search/kdtree.h>
// General Utils
#include <cmath>
#include <fstream>
#include <string>
#include <stdlib.h>
// Vision services
#include <pacman_vision_comm/get_cloud_in_hand.h>
// This node services (includes custom messages)
#include <gp_regression/StartProcess.h>
#include <gp_regression/GetToExploreTrajectory.h>

//GP
#include <gp_regression/gp_regressors.h>
#include <gp_regression/gp_planner.h>

using namespace gp_regression;

/* PLEASE LOOK at  TODOs by searching "TODO" to have an idea  of * what is still
missing or is improvable! */

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
        //input object point cloud, this gets updated with new points from probe
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr object_ptr;
        //input hand point cloud
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr hand_ptr;
        //Services, publishers and subscribers
        ros::ServiceServer srv_start;
        ros::ServiceServer srv_sample;
        ros::Publisher pub_model, pub_point, pub_point_marker, pub_direction_marker;
        ros::Subscriber sub_points;
        //control if we can start processing, i.e. we have a model and clouds
        bool start;
        //control if gp model was updated with new points and thus we need to
        //republish a new point cloud estimation
        bool need_update;
        //control how many new discovered points we need before updating the model
        int how_many_discoveries;
        //reconstructed model cloud to republish
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr model_ptr;
        //Gaussian Model object and Lapalce regressor
        Model* object_gp;
        LaplaceRegressor regressor;
        //stored variances of sample points
        std::vector<double> samples_var;
        //stored samples
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr samples_ptr;
        //kdtree for object
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr object_tree;
        //kdtree for object, used by isSampleVisible method
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr viewpoint_tree;
        //kdtree for hand
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr hand_tree;
        //sample to explore
        gp_regression::SampleToExplore sample_to_explore;
        //vector of discovered points
        std::vector<pcl::PointXYZRGB> discovered;

        //test sample for occlusion, i.e tells if the sample can reach the camera
        //without "touching" other object points
        int isSampleVisible(const pcl::PointXYZRGB sample, const float min_z) const;
        //callback to start process service, executes when service is called
        bool cb_start(gp_regression::StartProcess::Request& req, gp_regression::StartProcess::Response& res);
        //callback to sample process service, executes when service is called
        bool cb_sample(gp_regression::GetToExploreTrajectory::Request& req, gp_regression::GetToExploreTrajectory::Response& res);
        //callback for sub point subscriber
        // TODO: Convert this callback if  needed to accept probe points and not
        // rviz clicked points, as it is now. (tabjones on Wednesday 18/11/2015)
        void cb_point(const geometry_msgs::PointStamped::ConstPtr &msg);
        //gp computation
        bool compute();
        //update gaussian model with new points from probe
        void update();
        //Republish cloud method
        void publishCloudModel() const;
        //Publish sample to explore
        void publishSampleToExplore() const;
};
#endif
