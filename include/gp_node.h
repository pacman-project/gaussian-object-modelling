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
// General Utils
#include <cmath>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <map>
// Vision services
#include <pacman_vision_comm/get_cloud_in_hand.h>
// This node services (includes custom messages)
#include <gp_regression/StartProcess.h>
#include <gp_regression/GetToExploreTrajectory.h>
//GP
#include <gp/GaussianProcess.h>
#include <gp/SampleSet.h>

    #include <gp_regression/gp_modelling.h>

using namespace gp;

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

        /** \brief  Publish the object  model if there  is one along  with other
         *  markers.
         *
         * Points belonging to object are blue, points belonging to the hand are
         * cyan, points  belonging to external  sphere are red,  internal points
         * are yellow.
         */
        void Publish();
        typedef pcl::PointCloud<pcl::PointXYZRGB> PtC;

    private:
        //control if we can start processing, i.e. we have a model and clouds
        bool start;
        //input object point cloud, this gets updated with new points from probe
        PtC::Ptr object_ptr;
        //input hand point cloud
        PtC::Ptr hand_ptr;
        //reconstructed model cloud to republish including centroid and sphere
        PtC::Ptr model_ptr;
        //Services, publishers and subscribers
        ros::ServiceServer srv_start;
        // ros::ServiceServer srv_sample;
        ros::Publisher pub_model,  pub_markers; //, pub_point_marker, pub_direction_marker;
        ros::Subscriber sub_points;
        //GP regressor and dataset
        LaplaceRegressor::Ptr gp;
        SampleSet::Ptr data;

            gp_regression::GaussianRegressor reg;
            gp_regression::Model::Ptr obj_gp;
            gp_regression::Atlas::Ptr gp_atlas;
        //Atlas TODO temp until it is implemented in gp
        struct Chart
        {
            uint8_t id;
            Vec3 center;
            F32 radius;
            Eigen::Vector3d N;
            Eigen::Vector3d Tx;
            Eigen::Vector3d Ty;
            uint8_t parent; //parent id, along with its depth level uniquely identifies branches
        };
        //key is the depth level
        typedef std::multimap<uint8_t, Chart> Atlas;
        std::shared_ptr<Atlas> atlas;
        /////////////
        //atlas visualization
        visualization_msgs::MarkerArrayPtr markers;
        //kdtree for object, used by isSampleVisible method
        // pcl::search::KdTree<pcl::PointXYZRGB>::Ptr viewpoint_tree;
        //kdtree for hand
        // pcl::search::KdTree<pcl::PointXYZRGB>::Ptr hand_tree;

        //test sample for occlusion, i.e tells if the sample can reach the camera
        //without "touching" other object points
        // int isSampleVisible(const pcl::PointXYZRGB sample, const float min_z) const;
        //callback to start process service, executes when service is called
        bool cb_start(gp_regression::StartProcess::Request& req, gp_regression::StartProcess::Response& res);
        //callback to sample process service, executes when service is called
        // bool cb_sample(gp_regression::GetToExploreTrajectory::Request& req, gp_regression::GetToExploreTrajectory::Response& res);
        //callback for sub point subscriber
        // TODO: Convert this callback if  needed to accept probe points and not
        // rviz clicked points, as it is now. (tabjones on Wednesday 18/11/2015)
        void cb_point(const geometry_msgs::PointStamped::ConstPtr &msg);
        /** \brief Compute a Gaussian Process from object and store it */
        bool computeGP();
        /** \brief Compute Atlas from a random starting point */
        bool computeAtlas();
        /** \brief compute markers that compose an atlas */
        void createAtlasMarkers();
        /** \brief Update the Gaussian Process with new points */
        void update(Vec3Seq &points);
        /** \brief Publish object model */
        void publishCloudModel() const;
        /** \brief Publish last computed atlas */
        void publishAtlas () const;
};
#endif
