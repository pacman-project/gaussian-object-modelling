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
            hand_ptr.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        }
        /**\brief Destructor */
        virtual ~GaussianProcessNode (){}
        /**\brief Node Handle*/
        ros::NodeHandle nh;

        //Republish cloud method
        void republish_cloud ()
        {
            if (start_gp && cloud_ptr)
                if(!cloud_ptr->empty() && pub_cloud.getNumSubscribers()>0)
                    pub_cloud.publish(*cloud_ptr);
        }
    private:
        //input point cloud
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr hand_ptr;
        //Services and publishers
        ros::ServiceServer srv_start;
        ros::Publisher pub_cloud;
        //control if we can start processing
        bool start_gp;
        //callback to start process service, executes when service is called
        bool cb_start(gp_regression::start_process::Request& req, gp_regression::start_process::Response& res)
        {
            if(req.cloud_dir.empty()){
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
                pcl::fromROSMsg (service.response.hand, *hand_ptr);
            }
            else{
            //User told us to load a cloud from disk instead.
            //TODO: Add some path checks perhaps? Lets hope the path is valid
            //      for now!  (Fri 06 Nov 2015 05:03:19 PM CET -- tabjones)
                if (pcl::io::loadPCDFile((req.cloud_dir+"/obj.pcd"), *cloud_ptr) != 0){
                    ROS_ERROR("Error loading cloud from %s",(req.cloud_dir+"obj.pcd").c_str());
                    return (false);
                }
                if (pcl::io::loadPCDFile((req.cloud_dir+"/hand.pcd"), *hand_ptr) != 0){
                    ROS_ERROR("Error loading cloud from %s",(req.cloud_dir + "/hand.pcd").c_str());
                    return (false);
                }
            }
            start_gp = true;
            compute();
            return (true);
        }
        //gp computation
        void compute()
        {
            if(!start_gp || !cloud_ptr)
                return;
            Vec3Seq cloud;
            Vec targets;
            const size_t size_cloud = cloud_ptr->size();
            const size_t size_hand = hand_ptr->size();
            targets.resize(size_cloud + size_hand);
            cloud.resize(size_cloud + size_hand);
            for(size_t i=0; i<size_cloud; ++i)
            {
                Vec3 point(cloud_ptr->points[i].x, cloud_ptr->points[i].y, cloud_ptr->points[i].z);
                cloud[i]=point;
                targets[i]=0;
            }
            for(size_t i=0; i<size_hand; ++i)
            {
                Vec3 point(hand_ptr->points[i].x, hand_ptr->points[i].y, hand_ptr->points[i].z);
                cloud[size_cloud +i]=point;
                targets[size_cloud+i]=1;
            }
            /*****  Create the model  *********************************************/
            SampleSet::Ptr trainingData(new SampleSet(cloud, targets));
            LaplaceRegressor::Desc laplaceDesc;
            laplaceDesc.noise = 0.001;
            LaplaceRegressor::Ptr gp = laplaceDesc.create();
            printf("Regressor created %s\n", gp->getName().c_str());
            gp->set(trainingData);
            /*****  Query the model with a point  *********************************/
            Vec3 q(cloud[0]);
            const double qf = gp->f(q);
            const double qVar = gp->var(q);
            std::cout << "y = " << targets[0] << " -> qf = " << qf << " qVar = " << qVar << std::endl << std::endl;

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
        node.republish_cloud();
        rate.sleep();
    }
    //someone killed us :(
    return 0;
}
