#ifndef _INCL_SURFACE_APPROX_H
#define _INCL_SURFACE_APPROX_H

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
#include <std_msgs/ColorRGBA.h>
#include <visualization_msgs/MarkerArray.h>

//PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/io.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid.h>
// #include <pcl/filters/voxel_grid.h>
// #include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl_ros/transforms.h>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d_omp.h>

// This node services (includes custom messages)
#include <gp_regression/StartProcess.h>

// Gaussian Process library
#include <gp_regression/gp_regressors.h>

/**\brief Class SurfaceEstimator
 * {Wraps Gaussian process into a ROS node for surface approximation}
*/
class SurfaceEstimator
{
    public:
        /**\brief Constructor */
        SurfaceEstimator ();
        /**\brief Destructor */
        virtual ~SurfaceEstimator (){}

        /**\brief Node Handle*/
        ros::NodeHandle nh;

        /** \brief  Publish the object  model if there  is one along  with other
         *  markers.
         *
         * Points belonging to surface are blue, external points are orange,
         * internal points are purple.
         */
        void Publish();

        typedef pcl::PointCloud<pcl::PointXYZRGB> PtC;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:

        /**************
         * PARAMETERS *
         * ************
         *
         */
        //control if we can start processing, i.e. we have a model and clouds
        bool start;

        /***************
         * VAR HOLDERS *
         ***************
         *
         */
        // input surface point cloud
        PtC::Ptr input_ptr;
        // training data point cloud  (input plus lower / upper bound)
        PtC::Ptr training_ptr;
        // cloud of samples
        pcl::PointCloud<pcl::PointXYZI>::Ptr model_ptr;
        // regressor, model and covariance
        gp_regression::ThinPlateRegressor::Ptr reg;
        gp_regression::Model::Ptr obj_gp;
        //kerneò function
        std::shared_ptr<gp_regression::ThinPlate> my_kernel;
        // R parameter of thinplate
        double R;
        //gp obj data
        gp_regression::Data::Ptr input_gp;
        //gp external/internal data
        gp_regression::Data::Ptr intext_gp;
        //external/internal data size, for model resizing
        size_t intext_size;
        //data noise parameter
        double sigma2;
        //min max variance found on samples
        double min_v, max_v;
        //reference frame
        std::string proc_frame;
        //sampling resolution
        double sample_res;
        //mutex for samples
        std::mutex mtx_samp;
        /***********
         * METHODS *
         ***********
         *
         */
        // Helpers
        //prepare the data for gp computation
        void prepareExtData();
        bool prepareData();
        // Compute a Gaussian Process and store it
        bool computeGP();
        // the grid plotting
        void fakeDeterministicSampling(const bool first_time, const double scale=1.0, const double pass=0.08);
        //sample at a point
        void samplePoint(const double x, const double y, const double z, visualization_msgs::Marker &samp, visualization_msgs::Marker &arrow);
        //publish samples
        void publishSamples () const;
        // Publish surface training model
        void publishTraining() const;

        /***********
         * ROS API *
         ***********
         * members and methods
         *
         */
        // Samples visualization
        visualization_msgs::MarkerArrayPtr samples_marks;

        //Services, publishers and subscribers
        ros::ServiceServer srv_start;
        ros::Publisher pub_samples;
        ros::Publisher pub_training;


        //callback to start process service, executes when service is called
        bool cb_start(gp_regression::StartProcess::Request& req, gp_regression::StartProcess::Response& res);
};
#endif
