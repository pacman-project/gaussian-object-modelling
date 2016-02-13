#ifndef _INCL_GP_NODE_H
#define _INCL_GP_NODE_H

// General Utils
#include <cmath>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <mutex>

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
#include <std_msgs/ColorRGBA.h>
#include <visualization_msgs/MarkerArray.h>

//PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/io.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/features/normal_3d_omp.h>
// #include <pcl/filters/voxel_grid.h>
// #include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/search/kdtree.h>

// Vision services
#include <pacman_vision_comm/get_cloud_in_hand.h>

// This node services (includes custom messages)
#include <gp_regression/StartProcess.h>
#include <gp_regression/GetNextBestPath.h>

// Gaussian Process library
#include <gp_regression/gp_regressors.h>

//Atlas
#include <atlas/atlas.hpp>
#include <atlas/explorer.hpp>
#include <atlas/atlas_variance.hpp>
#include <atlas/atlas_collision.hpp>
#include <atlas/exp_single_path.hpp>
#include <atlas/exp_multibranch.hpp>

/* PLEASE LOOK at  TODOs by searching "TODO" to have an idea  of * what is still
missing or is improvable! */

// EVERYTHING IS IMPROVABLE UP TO ONES AND ZEROS

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

        /** \brief  Publish the object  model if there  is one along  with other
         *  markers.
         *
         * Points belonging to object are blue, points  belonging to external
         * sphere are purple,  internal points are cyan.
         */
        void Publish();

        /**
         * \brief Check exploration status and store the solution if successful
         */
        void checkExploration();
        typedef pcl::PointCloud<pcl::PointXYZRGB> PtC;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:

        /**************
         * PARAMETERS *
         * ************
         *
         */
        //control if we can start processing, i.e. we have a model and clouds
        bool start, fake_sampling, exploration_started;
        const double out_sphere_rad;

        /***************
         * VAR HOLDERS *
         ***************
         *
         */
        // input object point cloud, this gets updated with new points from probe
        // but keeps the history of the raw input data
        PtC::Ptr object_ptr;
        // same as object data but normalized by R_ and deMean'ed by centroid everytime
        PtC::Ptr data_ptr_;
        // input hand point cloud
        PtC::Ptr hand_ptr;
        Eigen::Vector4d current_offset_;
        double current_scale_;

        // regressor and model
        gp_regression::ThinPlateRegressor::Ptr reg_;
        gp_regression::Model::Ptr obj_gp;
        std::shared_ptr<gp_regression::ThinPlate> my_kernel;

        //atlas and explorer
        gp_atlas_rrt::AtlasCollision::Ptr atlas;
        gp_atlas_rrt::ExplorerMultiBranch::Ptr explorer;
        // gp_atlas_rrt::AtlasVariance::Ptr atlas;
        // gp_atlas_rrt::ExplorerSinglePath::Ptr explorer;
        //exploration solution
        std::vector<std::size_t> solution;

        /***********
         * METHODS *
         ***********
         *
         */
        // Helpers
        void deMeanAndNormalizeData(const PtC::Ptr &data_ptr, PtC::Ptr &out);
        void reMeanAndDenormalizeData(Eigen::Vector3d &data);
        // Compute a Gaussian Process from object and store it
        bool computeGP();
        // start the RRT exploration
        bool startExploration();

        /***********
         * ROS API *
         ***********
         *
         *
         */
        // visualization of atlas and explorer
        visualization_msgs::MarkerArrayPtr markers;
        // and its mutex
        std::shared_ptr<std::mutex> mtx_marks;

        //Services, publishers and subscribers
        ros::ServiceServer srv_start;
        ros::ServiceServer srv_get_next_best_path_;

        // ros::ServiceServer srv_sample;
        ros::Publisher pub_markers; //, pub_point_marker, pub_direction_marker;
        ros::Subscriber sub_update_;

        // Publish last computed atlas
        void publishAtlas () const;

        //callback to start process service, executes when service is called
        bool cb_start(gp_regression::StartProcess::Request& req, gp_regression::StartProcess::Response& res);

        //callback to sample process service, executes when service is called
        bool cb_get_next_best_path(gp_regression::GetNextBestPath::Request& req, gp_regression::GetNextBestPath::Response& res);

        // callback for sub point subscriber
        // TODO: Convert this callback if  needed to accept probe points and not
        // rviz clicked points, as it is now. (tabjones on Wednesday 18/11/2015)
        // I don't think it is bad to have geometry_msgs::PointStamped, we'll see
        // later how we deal with the information streaming, or may to pass back from the exploration
        // a vector of geometry_msgs::PointStamped that can be a trajectory or one single poke
        void cb_update(const geometry_msgs::PointStamped::ConstPtr &msg);

        /*****************
         * DEBUG MEMBERS *
         *****************
         *
         */

        //reconstructed model cloud to republish including centroid and sphere
        PtC::Ptr model_ptr;
        // ros::ServiceServer srv_rnd_tests_;
        ros::Publisher pub_model;

        // Publish object model
        void publishCloudModel() const;

        // the grid plotting
        void fakeDeterministicSampling(const double scale=1.0, const double pass=0.08);

};
#endif
