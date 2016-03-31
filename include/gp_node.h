#ifndef _INCL_GP_NODE_H
#define _INCL_GP_NODE_H

// General Utils
#include <cmath>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <mutex>
#include <thread>

// ROS headers
#include <ros/ros.h>
#include <ros/console.h>
#include <ros/package.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/WrenchStamped.h>
#include <shape_msgs/Mesh.h>
#include <std_msgs/ColorRGBA.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_listener.h>

//PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/io.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/octree/octree_pointcloud.h>
// #include <pcl/features/normal_3d_omp.h>
// #include <pcl/filters/voxel_grid.h>
// #include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl_ros/transforms.h>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/poisson.h>
// #include <pcl/surface/marching_cubes_hoppe.h>
// #include <pcl/surface/marching_cubes_hoppe.h>
#include <pcl/surface/gp3.h>

// Vision services
#include <pacman_vision_comm/get_cloud_in_hand.h>

// This node services (includes custom messages)
#include <gp_regression/StartProcess.h>
#include <gp_regression/Update.h>
#include <gp_regression/GetNextBestPath.h>
#include <gp_regression/Path.h>

// Gaussian Process library
#include <gp_regression/gp_regressors.h>

//Atlas
#include <atlas/atlas.hpp>
#include <atlas/explorer.hpp>
#include <atlas/atlas_variance.hpp>
#include <atlas/atlas_collision.hpp>
#include <atlas/exp_single_path.hpp>
#include <atlas/exp_multibranch.hpp>

//octomap
#include <octomap/octomap.h>
#include <octomap/OcTree.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>
#include <octomap_ros/conversions.h>

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

        //automatically call get_next_best_path if synthetic touch is enabled
        void automatedSynthTouch();

        typedef pcl::PointCloud<pcl::PointXYZRGB> PtC;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:

        /**************
         * PARAMETERS *
         * ************
         *
         */
        //control if we can start processing, i.e. we have a model and clouds
        bool start, exploration_started, simulate_touch;
        const double out_sphere_rad;
        int synth_type;

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
        PtC::Ptr hand_ptr;  //actually unused
        //reconstructed model cloud to republish including centroid and sphere
        // this is not true, model_ptr contains the trainint data in PCL format!
        PtC::Ptr model_ptr;
        // internal explicit (or reconstructed) model in the real world
        // lets abuse the intensity channel to store variance
        // it's a float so who cares!!
        pcl::PointCloud<pcl::PointXYZI>::Ptr real_explicit_ptr;
        Eigen::Vector4d current_offset_;
        double current_scale_;
        shape_msgs::Mesh predicted_shape_;
        //Total exploration steps
        std::size_t steps;
        //last point touched
        Eigen::Vector3d last_touched;

        // regressor, model and covariance
        gp_regression::ThinPlateRegressor::Ptr reg_;
        gp_regression::Model::Ptr obj_gp;
        std::shared_ptr<gp_regression::ThinPlate> my_kernel;
        //gp obj data
        gp_regression::Data::Ptr cloud_gp;
        //gp external/internal data
        gp_regression::Data::Ptr ext_gp;
        //external data size, for model resizing
        size_t ext_size;
        //data noise parameter
        double sigma2;


        //atlas and explorer
        gp_atlas_rrt::AtlasCollision::Ptr atlas;
        gp_atlas_rrt::ExplorerMultiBranch::Ptr explorer;
        // gp_atlas_rrt::AtlasVariance::Ptr atlas;
        // gp_atlas_rrt::ExplorerSinglePath::Ptr explorer;
        //exploration solution
        std::vector<std::size_t> solution;

        //octomap
        std::shared_ptr<octomap::OcTree> octomap;

        //marching sampling stuff
        pcl::octree::OctreePointCloud<pcl::PointXYZ>::Ptr s_oct;
        pcl::PointCloud<pcl::PointXYZ>::Ptr oct_cent;
        std::mutex mtx_samp;
        double sample_res;
        //min max variance found on samples
        double min_v, max_v;
        //synthetic touch data needed
        pcl::PointCloud<pcl::PointXYZ>::Ptr full_object, full_object_real;
        pcl::search::KdTree<pcl::PointXYZ> kd_full;
        double synth_var_goal;
        double current_goal;
        std::string obj_name, test_name;

        //Final gloabl variance goal
        double goal;

        /***********
         * METHODS *
         ***********
         *
         */
        // Helpers
        void deMeanAndNormalizeData(const PtC::Ptr &data_ptr, PtC::Ptr &out);
        void deMeanAndNormalizeData(Eigen::Vector3d &data);
        void reMeanAndDenormalizeData(Eigen::Vector3d &data);
        template<typename PT>
        void reMeanAndDenormalizeData(const typename pcl::PointCloud<PT>::Ptr &data_ptr, pcl::PointCloud<PT> &out) const;
        //prepare the data for gp computation
        void prepareExtData();
        bool prepareData();
        // Compute a Gaussian Process from object and store it
        bool computeGP();
        // start the RRT exploration
        bool startExploration(const float v_des, Eigen::Vector3d &start);
        // compute octomap from real explicit cloud
        void computeOctomap();
        // compute the predicted shape from real explicit cloud as a message
        void computePredictedShapeMsg();
        // the grid plotting
        void fakeDeterministicSampling(const bool first_time, const double scale=1.0, const double pass=0.08);
        //sample at a point
        void samplePoint(const double x, const double y, const double z, visualization_msgs::Marker &samp);
        // alternative hopefully faster sampling
        void marchingSampling(const bool first_time, const float leaf_size=0.15, const float leaf_pass=0.03);
        // cube sampling for marchingSampling (nested therads)
        void marchingCubes(const pcl::PointXYZ start, const float leaf, const float pass, visualization_msgs::Marker &samp);
        // simulated synthetic touch
        void synthTouch(const gp_regression::Path &sol);
        Eigen::Vector3d raycast(Eigen::Vector3d &point, const Eigen::Vector3d &normal, gp_regression::Path &touched, bool no_external=false);
        /**
         * \brief Check exploration status and store the solution if successful
         */
        void checkExploration();
        //create arrows to view touches
        void createTouchMarkers(const Eigen::MatrixXd &pts);
        //calculate euclidean norm
        double inline L2(const Eigen::Vector3d &a, const Eigen::Vector3d &b) const
        {
            return (std::sqrt( std::pow(a[0]+b[0],2) + std::pow(a[1]+b[1],2) + std::pow(a[2]+b[2],2) ));
        }


        /***********
         * ROS API *
         ***********
         * members and methods
         *
         */
        // visualization of atlas and explorer
        visualization_msgs::MarkerArrayPtr markers;
        visualization_msgs::MarkerArrayPtr gt_marks;
        // and its mutex
        std::shared_ptr<std::mutex> mtx_marks;

        //Services, publishers and subscribers
        ros::ServiceServer srv_start;
        ros::ServiceServer srv_update;
        ros::ServiceServer srv_get_next_best_path_;

        // ros::ServiceServer srv_sample;
        ros::Publisher pub_markers; //, pub_point_marker, pub_direction_marker;
        ros::Publisher pub_ground_truth;
        ros::Subscriber sub_update_;
        ros::Publisher pub_model;
        ros::Publisher pub_real_explicit;
        ros::Publisher pub_octomap;

        //transform listener
        tf::TransformListener listener;
        //processing frame
        std::string proc_frame, anchor;

        // Publish last computed atlas and ground truth meshes
        void publishAtlas () const;
        void publishGroundTruth () const;

        //callback to start process service, executes when service is called
        bool cb_start(gp_regression::StartProcess::Request& req, gp_regression::StartProcess::Response& res);

        //callback to sample process service, executes when service is called
        bool cb_get_next_best_path(gp_regression::GetNextBestPath::Request& req, gp_regression::GetNextBestPath::Response& res);

        void cb_update(const gp_regression::Path::ConstPtr &msg);
        bool cb_updateS(gp_regression::Update::Request &req, gp_regression::Update::Response &res);

        // Publish object model
        void publishCloudModel() const;

        // Publish octomap
        void publishOctomap() const;
};
#endif
