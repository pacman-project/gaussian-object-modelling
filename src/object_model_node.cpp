#include <algorithm>    // std::min_element, std::max_element

// BOOST INTEGRATION
#include <boost/array.hpp>
#include <boost/numeric/odeint.hpp>

// ROS headers
#include <ros/ros.h>
#include <ros/message_operations.h>
#include <geometry_msgs/WrenchStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/MarkerArray.h>
#include <std_srvs/Empty.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <pcl_conversions/pcl_conversions.h>

// PCL headers
#include <pcl_ros/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/pcl_base.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/intensity_gradient.h>
#include <pcl/filters/voxel_grid.h>

// Messages
#include "gaussian_object_modelling/PickPoint.h"

// integration vars
int itera;
typedef boost::array< float , 6 > state_type;
using namespace std;
using namespace boost::numeric::odeint;
using namespace Eigen;

// point at surface 
MatrixXf p_;
std::vector<float> x_;
std::vector<float> y_;
std::vector<float> z_;
std::vector<float> tx_;
std::vector<float> ty_;
std::vector<float> tz_;

// THIS IS PARTICULAR FOR A FUNCTION TO BE EVALUATED
// this is for kernel density function, instead of copying the whole vector, it is setted as a member
VectorXf w_;
float sigma_;
MatrixXf mean_;
// and this is for the ellipsoidal surface
const float a = 5.0;
const float b = 5.0;
const float c = 5.0;
const float R = 1.0;

// HELPER FUNCTIONS FOR KERNEL COLLECTION
// kernel
float evaluateKernel(Vector3f p, Vector3f m, float s)
{
  // triangular isotropic  kernel
  float k;
  Vector3f v;
  v = p - m;
  float d;
  d = v.dot(v);
  d = sqrt(d);
   
  if ( d>(2*s) )
  {
    k = 0;
  }
  else
  {
    k = 1 - d/(2*s);
  }
  return k;
}

// kernel collection
float evaluateKernels(Vector3f p)
{
  float f = 0;
  float k = 0;

  for (int i=0; i<w_.size(); i++)
  {
    k = evaluateKernel(p, mean_.row(i), sigma_);
    f += w_(i)*k;
  }

  return f;
}

// kernel first derivative
Vector3f evaluateKernelDiff(Vector3f p, Vector3f m, float s)
{
  // triangular kernel
  Vector3f k3;
  Vector3f v;
  v = p - m;
  float d;
  d = v.dot(v);
  d = sqrt(d);

  if ( d>(2*s) )
  {
    k3.setZero(3);
  }
  else
  {
    k3 = (-1/(2*s*d))*v;  
  }
  return k3;
}

// kernel collection first derivative
Vector3f evaluateKernelsDiffs(Vector3f p)
{
  Vector3f ff, kk;
  ff.setZero(3);
  kk.setZero(3);

  for (int i=0; i<w_.size(); i++)
  {
      kk = evaluateKernelDiff(p, mean_.row(i), sigma_);
      ff += kk*w_(i);
  }

  return ff;
}

// kernel second derivative
Matrix3f evaluateKernelDiff2(Vector3f p, Vector3f m, float s)
{
  // triangular kernel
  Matrix3f k33;
  Vector3f v;
  v = p - m;
  float d;
  d = v.dot(v);
  d = sqrt(d);
  
  Matrix3f I3;
  I3.setIdentity(3,3);

  Matrix3f M3;
  M3(0,0) = v(0)*v(0);
  M3(1,1) = v(1)*v(1);
  M3(2,2) = v(2)*v(2);
  M3(0,1) = v(0)*v(1);
  M3(1,0) = v(0)*v(1);
  M3(0,2) = v(0)*v(2);
  M3(2,0) = v(0)*v(2);
  M3(1,2) = v(1)*v(2);
  M3(2,1) = v(1)*v(2);
    
  if ( d>(2*s) )
  {
    k33.setZero(3,3);
  }
  else
  {
    k33 = ( -1/(2*s*d*d*d) )*( I3*d*d - M3 );
  }
  return k33;
}

// kernel collection second derivative
Matrix3f evaluateKernelsDiffs2(Vector3f p)
{
  Matrix3f fff, kkk;
  fff.setZero(3,3);
  kkk.setZero(3,3);

  for (int i=0; i<w_.size(); i++)
  {
    kkk = evaluateKernelDiff2(p, mean_.row(i), sigma_);
    fff += w_(i)*kkk;
  }

  return fff;
}

// THIS ROUTINES ARE USED IN THE GEODESIC EQUATIONS
// modify if you want another function f

// HESSIAN
// kernel-based surface
Matrix3f evaluateHessian(Vector3f p)
{
  Matrix3f fff;
  fff = evaluateKernelsDiffs2(p);
  return fff;
}
// ellipsoidal surface
// Matrix3f evaluateHessian(Vector3f p)
// {
//   Matrix3f fff;
//   fff << 2/a, 0, 0, 0, 2/b, 0, 0, 0, 2/c ; 
//   return fff;
// }

// GRADIENT
// kernel-based surface
Vector3f evaluateGradient(Vector3f p)
{
  Vector3f ff;
  ff = evaluateKernelsDiffs(p);
  return ff;
}
// ellipsoidal surface
// Vector3f evaluateGradient(Vector3f p)
// {
//   Vector3f ff;
//   ff = Vector3f(2*p(0)/a, 2*p(1)/b, 2*p(2)/c) ;
//   return ff;
// }

// FUNCTION
// modify if you want another function f
// kernel-based surface
float evaluateFunction(Vector3f p)
{
  float f;
  f = evaluateKernels(p);
  return f;
}
// ellipsoidal surface
// float evaluateFunction(Vector3f p)
// {
//   float f;
//   f = p(0)*p(0)/a + p(1)*p(1)/b + p(2)*p(2)/c - R;
//   return f;
// }

// model function of the integrator
// GEODESIC EQUATION FOR AN IMPLICIT SURFACE, taken from http://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node196.html
void geodesicModel( const state_type &x , state_type &dxdt , float t )
{
  // cout << "geodesic model" << endl; // just to check how many times this function is called

  // temporary variables in matrix and element forms
  Matrix3f hessian;
  Vector3f gradient;   
  Vector3f point;
  float fxx, fyy, fzz, fxy, fyz, fxz, fx, fy, fz, p, q, r, D, L, N;

  // convert the current point to internal types
  p = x[0]; 
  q = x[1];
  r = x[2];
  point(0) = x[3];
  point(1) = x[4];
  point(2) = x[5];

  // get the hessian at the current point
  hessian = evaluateHessian( point );

  fxx = hessian(0,0);
  fyy = hessian(1,1);
  fzz = hessian(2,2);
  fxy = hessian(0,1); // or hessian(1,0)
  fyz = hessian(2,1); // or hessian(1,2)
  fxz = hessian(0,2); // or hessian(2,0)
  
  // get the gradient at the current point
  gradient = evaluateGradient( point );

  fx = gradient(0);
  fy = gradient(1);
  fz = gradient(2);
  
  // the normalizing factor
  D = (r*fy - q*fz)*(r*fy - q*fz) + (p*fz - r*fx)*(p*fz - r*fx) + (q*fx - p*fy)*(q*fx - p*fy);

  // the constant factor
  L = fxx*p*p + fyy*q*q + fzz*r*r + 2*(fxy*p*q + fyz*q*r + fxz*p*r);

  // how the tangent vector and the point evolve
  dxdt[0] = ((p*fz - r*fx)*r + (p*fy - q*fx)*q)*L/D;
  dxdt[1] = ((q*fz - r*fy)*r + (q*fx - p*fy)*p)*L/D;
  dxdt[2] = ((r*fy - q*fz)*q + (r*fx - p*fz)*p)*L/D;
  dxdt[3] = p; 
  dxdt[4] = q;
  dxdt[5] = r;

}

// null function of the integrator
void writeGeodesic( const state_type &x , const float t )
{

    // cout << "write geoedesic" << endl; // just to check how many times this function is called

    // norm of tangent vector
    float N = sqrt( x[0]*x[0] + x[1]*x[1] + x[2]*x[2] );
    
    // print state
    cout << t << '\t' << x[0] << '\t' << x[1] << '\t' << x[2] << '\t' << x[3] << '\t' << x[4] << '\t' << x[5] << '\t' << N << endl;

    // save the tangent vector
    tx_.push_back(x[0]);
    ty_.push_back(x[1]);
    tz_.push_back(x[2]);
    // and save the point
    x_.push_back(x[3]);
    y_.push_back(x[4]);
    z_.push_back(x[5]);
    itera++;
}


// class object for the ROS node
namespace gaussian_object_modelling {

class ObjectModeller
{
  private:
    //! The node handle
    ros::NodeHandle nh_;

    //! Node handle in the private namespace
    ros::NodeHandle priv_nh_;

    //! Service server for object model
    ros::ServiceServer object_model_srv_;
    
    //! Publisher for the markers
    ros::Publisher pub_geodesic_curve_;

    //! The touch frame 
    tf::Transform touch_frame_;

    //! Markers
    visualization_msgs::MarkerArray geodesic_curve_mrk_;
    visualization_msgs::Marker geodesic_point_mrk_;

    // surface parameters
    // Kernel widths, for position and orientation:
    double locH_; // in the same unit as the datapoints forming the density
    double oriH_; // in radians
    // number of points
    int N_; 
    // points on, outside and inside
    MatrixXf XYZ_; 
    // computed weights
    VectorXf W_;

    // There always should be a listener and a broadcaster!
    //! A tf transform listener
    tf::TransformListener tf_listener_;

    //! A tf transform broadcaster
    tf::TransformBroadcaster tf_broadcaster_;

    // processing frame for everything, even printing the cart trajectory
    std::string processing_frame_;
        
  public:
    //------------------ Callbacks -------------------
    // Callback to perform the object modelling
    bool modelObject(gaussian_object_modelling::PickPoint::Request& request, gaussian_object_modelling::PickPoint::Response& response);
    void publishTouchFrame();

    //void updateObjectModel(); // the object model and another point cloud (filled with the tactile info, for instance)

    //! Subscribes to and advertises topics
    ObjectModeller(ros::NodeHandle nh) : nh_(nh), priv_nh_("~")
    {

      processing_frame_ = "/world";

      // service to create the object model using a gaussian process implemented by nuklei
      object_model_srv_ = nh_.advertiseService(nh_.resolveName("object_model_srv"), &ObjectModeller::modelObject, this);

      pub_geodesic_curve_ = nh_.advertise<visualization_msgs::MarkerArray>(nh_.resolveName("geodeisc_curve"), 10);

      // MARKER initialization
      geodesic_point_mrk_.header.frame_id = processing_frame_;
      geodesic_point_mrk_.type = visualization_msgs::Marker::SPHERE;
      geodesic_point_mrk_.action = visualization_msgs::Marker::ADD;
      geodesic_point_mrk_.scale.x = 0.003;
      geodesic_point_mrk_.scale.y = 0.003;
      geodesic_point_mrk_.scale.z = 0.003;
      geodesic_point_mrk_.pose.orientation.x = 0.0;
      geodesic_point_mrk_.pose.orientation.y = 0.0;
      geodesic_point_mrk_.pose.orientation.z = 0.0;
      geodesic_point_mrk_.pose.orientation.w = 1.0;
      geodesic_point_mrk_.color.r = 1.0f;
      geodesic_point_mrk_.color.g = 0.0f;
      geodesic_point_mrk_.color.b = 0.0f;
      geodesic_point_mrk_.color.a = 1.0;
      geodesic_point_mrk_.lifetime = ros::Duration();

      // initial touch frame, just to have something to publish
      touch_frame_.setOrigin(tf::Vector3( 0.0,0.0,0.0 ));
      touch_frame_.setBasis(tf::Matrix3x3( 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 ));

      //density parameters
      locH_ = 100.0;
      oriH_ = 1.57;
      N_ = 0; 
    }

    //! Empty stub
    ~ObjectModeller() {}

};

//bool ObjectModeller::modelObject(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response)
bool ObjectModeller::modelObject(gaussian_object_modelling::PickPoint::Request& request, gaussian_object_modelling::PickPoint::Response& response)
{

  // density_.clear(); // not needed anymore

  ROS_INFO("geodesic service called on point: %li", request.pick);
  x_.clear();
  y_.clear();
  z_.clear();
  tx_.clear();
  ty_.clear();
  tz_.clear();

  //-------------------------------------------------------------------------
  // STEP 1 GET THE OBJECT POINT CLOUT USING TABLETOP SEGMENTATION AND EUCLIDEAN CLUSTERING
  //-------------------------------------------------------------------------

  // THIS NEEDS TO CHANGE TO WAIT FOR A SEGMENTED OBJECT IN THE FORM OF A POINT CLOUD!

  // First, call the tabletop segmentator to get the pointcloud of the object, which should be set to listen to a voxel filtered point cloud
  /*std::string service_name("/tabletop_segmentation");
  // Wait infinitely until the service is up
  while ( !ros::service::waitForService(service_name, ros::Duration()))
  {
    ROS_INFO("Waiting for service %s...", service_name.c_str());
  }
  // call the service
  tabletop_object_segmentation::TabletopSegmentation segmentation_srv;
  if (!ros::service::call(service_name, segmentation_srv))
  {
    ROS_ERROR("Call to segmentation service failed");
    return false;
  }
  // check the result of calling the service
  if (segmentation_srv.response.result != segmentation_srv.response.SUCCESS)
  {
    ROS_ERROR("Segmentation service returned error %d", segmentation_srv.response.result);
    return false;
  }
  ROS_INFO("Segmentation service succeeded. Detected %d clusters", (int)segmentation_srv.response.clusters.size());
  if (segmentation_srv.response.clusters.empty()) return false;
  
  // get the object point cloud in the desired frame
  sensor_msgs::PointCloud transformedCluster;
  // MODIFIED
  // tf_listener_.transformPointCloud(processing_frame_, segmentation_srv.response.clusters[0], transformedCluster);

  // concert from sensor_msgs to a tractable PCL object
  
  // MODIFIED
  // sensor_msgs::convertPointCloudToPointCloud2 (transformedCluster, cluster2);
  sensor_msgs::convertPointCloudToPointCloud2 (segmentation_srv.response.clusters[0], cluster2);
  */

  sensor_msgs::PointCloud2 cluster2;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB> cluster;
  pcl::fromROSMsg(cluster2, *cluster_ptr);
  cluster = *cluster_ptr;
  ROS_INFO("Cluster size is %li", cluster.points.size());

  //-------------------------------------------------------------------------
  // STEP 2 COMPUTE THE NORMALS OF THE CLUSTER POINT CLOUD
  //-------------------------------------------------------------------------

  ROS_INFO("Computing normals of the cluster...");
  // UPDATE pcl::search STUFF OR ASSUME POINT CLOUD WITH NORMALS IS GIVEN
  /*
  // PCL NORMAL ESTIMATION ROUTINES
  pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> ne;
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
  ne.setInputCloud (cluster_ptr);
  // Create an empty kdtree representation, and pass it to the normal estimation object.
  // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
  ne.setSearchMethod (tree);
  // create the output cloud which contains the normal
  pcl::PointCloud<pcl::Normal>::Ptr object_normals_ptr (new pcl::PointCloud<pcl::Normal> ());
  // Use all neighbors in a sphere of radius 3cm
  ne.setRadiusSearch (0.1);
  // Compute the features
  ne.compute (*object_normals_ptr);
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr object_with_normals_ptr (new pcl::PointCloud<pcl::PointXYZRGBNormal> ());
  pcl::concatenateFields( *cluster_ptr, *object_normals_ptr, *object_with_normals_ptr );
  */
  pcl::PointCloud<pcl::PointXYZRGBNormal> object_with_normals;
  // object_with_normals = *object_with_normals_ptr;

  //-------------------------------------------------------------------------
  // STEP 3 CHOOSE A STARTING POINT AND BUILD A TOUCHING FRAME
  //-------------------------------------------------------------------------
  
  tf::Vector3 touchAt(object_with_normals.points[request.pick].x+0.00005, object_with_normals.points[request.pick].y+0.00005, object_with_normals.points[request.pick].z+0.00005);
  Vector3f E_touchAt(touchAt.x(), touchAt.y(), touchAt.z());
   
  Vector3f n(object_with_normals.points[request.pick].normal_x, object_with_normals.points[request.pick].normal_y, object_with_normals.points[request.pick].normal_z); 
  n.normalize();
  
  // find orthonormal vectors to form a base  
  Vector3f v;
  v(1) = -1;
  v(2) = -1;
  v(0) = (n(2)+n(1))/n(0);
  v.normalize();
  
  Vector3f u;
  u = v.cross(n);
  
  tf::Matrix3x3 touchR( u(0), -v(0), -n(0), u(1), -v(1), -n(1), u(2), -v(2), -n(2) );

  touch_frame_.setOrigin(touchAt);
  touch_frame_.setBasis(touchR);


  // intermediate step, we need to ensure a point beyond the surface
  //tf_listener_.waitForTransform("calib_kimp_right_arm_base_link", "camera_rgb_optical_frame", now, ros::Duration(4));
  //tf_listener_.lookupTransform("calib_kimp_right_arm_base_link", "camera_rgb_optical_frame", now, base_2_camera_stamped);
  

  if (!request.use_same_weights)
  {
    //-------------------------------------------------------------------------
    // STEP 4 SET THE DATA TO BE USED TO TRAIN THE KERNEL COLLECTION
    //-------------------------------------------------------------------------

    Vector4f C;
    pcl::compute3DCentroid( object_with_normals, C );
    ROS_INFO(" Centroid of cluster %f %f %f ", C(0), C(1), C(2) );

    VectorXf L;
    L.resize(object_with_normals.points.size()+object_with_normals.points.size()+1);

    XYZ_.resize(object_with_normals.points.size()+object_with_normals.points.size()+1, 3);
    N_ = object_with_normals.points.size();

    // the first point inside the surface labeled as -1
    int i = 0;
    
    XYZ_(i,0) = touchAt.x() - 0.02f*n(0); //C(0);
    XYZ_(i,1) = touchAt.y() - 0.02f*n(1); //C(1);
    XYZ_(i,2) = touchAt.z() - 0.02f*n(2); //C(2);
    L(i) = -2;

    
    for ( i = 1; i < N_+N_; i++)
    {
      
      if (i<N_+1)
      {
        // points on the surface labeled as 0
        XYZ_(i,0) = object_with_normals.points[i-1].x - 0.0f*object_with_normals.points[i-1].normal_x;
        XYZ_(i,1) = object_with_normals.points[i-1].y - 0.0f*object_with_normals.points[i-1].normal_y;
        XYZ_(i,2) = object_with_normals.points[i-1].z - 0.0f*object_with_normals.points[i-1].normal_z;
        //ROS_INFO("point %f %f %f", XYZ_(i,0), XYZ_(i,1), XYZ_(i,2));
        L(i) = 0;
      }
      else
      {
        // points outside the surface labeled as 1
        XYZ_(i,0) = object_with_normals.points[i-1-N_].x + 0.02f*object_with_normals.points[i-1-N_].normal_x;
        XYZ_(i,1) = object_with_normals.points[i-1-N_].y + 0.02f*object_with_normals.points[i-1-N_].normal_y;
        XYZ_(i,2) = object_with_normals.points[i-1-N_].z + 0.02f*object_with_normals.points[i-1-N_].normal_z;
        //ROS_INFO("point + normal %f %f %f", XYZ_(i,0), XYZ_(i,1), XYZ_(i,2));
        L(i) = 2;
      }
    }

    // just to have the right size of XYZ_
    i = i + 1; 
    ROS_INFO("i = %i, object_with_normals.points.size() = %li", i, object_with_normals.points.size());


    //-------------------------------------------------------------------------
    // STEP 5 COMPUTE THE COVARIANCE MATRIX USING THE KERNEL
    //-------------------------------------------------------------------------

    Vector3f P_I; // tmp point
    Vector3f P_J; // tmp point
    MatrixXf K_IJ; // the kernel matrix
    K_IJ.resize(i, i);
    int p_i;
    int p_j;

    ROS_INFO("Computing the covariance matrix K_IJ...");
    for ( p_i = 0; p_i < i; p_i++)
    {
      for ( p_j = 0; p_j < i; p_j++)
      {
        P_I(0) = XYZ_(p_i,0);
        P_I(1) = XYZ_(p_i,1);
        P_I(2) = XYZ_(p_i,2);
        P_J(0) = XYZ_(p_j,0);
        P_J(1) = XYZ_(p_j,1);
        P_J(2) = XYZ_(p_j,2);
        K_IJ(p_i, p_j) = evaluateKernel(P_I, P_J, locH_);
      }
    }

    ROS_INFO("Computing the inverse of the covariance matrix K_IJ...");

    MatrixXf K_IJ_inv;
    K_IJ_inv.resize(i,i);

    //-------------------------------------------------------------------------
    // STEP 6 COMPUTE THE WEIGHTS AND SET GLOBAL VALUES
    //-------------------------------------------------------------------------

    ROS_INFO("Computing the weights W..");
    W_.resize(i);
    W_ = K_IJ.inverse()*L;

    // copy to global variables
    w_ = W_;
    sigma_ = locH_;
    mean_ = XYZ_;
  }

  //-------------------------------------------------------------------------
  // STEP 7 BUILD THE GEODESIC STARTING AT THE SAMPLED POINT
  //-------------------------------------------------------------------------
   
  ROS_INFO("Surface has %li points!", object_with_normals.points.size() );
  ROS_INFO("Computing geodesic...");
  
  itera = 0;

  //state_type x = { TANGENT, POINT }; // initial conditions
  state_type x = { u(0) , u(1) , u(2) , touchAt.x(), touchAt.y(), touchAt.z() }; // initial conditions
  ROS_INFO("Initial State: %f %f %f %f %f %f", x[0], x[1], x[2], x[3], x[4], x[5]);
  
  // A.- integrate using an adaptive step
  //integrate( geodesicModel , x , 0.0 , 3.0 , 0.01 , writeGeodesic );

  // B.- integrate using a constant step
  integrate_const( runge_kutta4< state_type >(), geodesicModel , x , 0.0 , 0.15 , 0.001 , writeGeodesic );
  

  // publish the result in RVIZ
  ROS_INFO("N points: %i", itera);

  Vector3f normal;
  Vector3f tangent;
  Vector3f cotangent;
  
  //cart_trajectory_mrk_.poses.clear();
  geodesic_curve_mrk_.markers.clear();

  ros::Time now = ros::Time::now();

  for(int j = 0; j < itera; j++)
  {
    //ROS_INFO("Trace points: %f %f %f", x_[j], y_[j], z_[j] );
    geodesic_point_mrk_.header.stamp = now;
    geodesic_point_mrk_.pose.position.x = x_[j];
    geodesic_point_mrk_.pose.position.y = y_[j];
    geodesic_point_mrk_.pose.position.z = z_[j];
    geodesic_point_mrk_.color.r = 1.0 - (float)j/ (float)itera;
    geodesic_point_mrk_.color.g = (float)j/(float)itera;
    geodesic_point_mrk_.color.b = 0.0f;
    geodesic_point_mrk_.color.a = 1.0;
    geodesic_point_mrk_.id = j;
    geodesic_curve_mrk_.markers.push_back(geodesic_point_mrk_);
  }

  // publish both arrays
  pub_geodesic_curve_.publish(geodesic_curve_mrk_);

  //-------------------------------------------------------------------------
  // STEP 8 PRINT THE RESULTS INTO A FILE
  //-------------------------------------------------------------------------
  
  // print file

  // Set the touch end effector pose for the kuka
  // the pose goes like this, similar to the comm protocol specified in FRI
  // ux vx wx x uy vy wy y uz vz wz z
  // that forms the transformation matrix
  // [ ux vx wx x ]
  // [ uy vy wy y ]
  // [ uz vz wz z ]
  // [ 0  0  0  1 ]

  FILE * pFile;
  int f;
  char name [256];

  ROS_INFO("printing file...");
  pFile = fopen ("geodesic.txt","w");

  // print the cart trajectory
  for(int j = 1; j < itera; j++)
  {
    normal = evaluateGradient( Vector3f(x_[j], y_[j], z_[j]) );
    normal.normalize();
    normal = -1*normal;
    tangent = Vector3f( tx_[j], ty_[j], tz_[j] );
    cotangent = normal.cross(tangent);

    // the first line is a pre touch configuration
    if (j==1)
      fprintf (pFile, "%f %f %f %f %f %f %f %f %f %f %f %f\n", tangent(0), cotangent(0), normal(0), x_[j]-0.01*normal(0), tangent(1), cotangent(1), normal(1), y_[j]-0.01*normal(1), tangent(2), cotangent(2), normal(2), z_[j]-0.01*normal(2));
      
    fprintf (pFile, "%f %f %f %f %f %f %f %f %f %f %f %f\n", tangent(0), cotangent(0), normal(0), x_[j], tangent(1), cotangent(1), normal(1), y_[j], tangent(2), cotangent(2), normal(2), z_[j]);
  }

  // close the file
  fclose (pFile);

  // print the surface parameters
  FILE *cFile;
  cFile = fopen ("surface.txt", "w");
  fprintf (cFile, "%s %s %s %s %s\n", "X", "Y", "Z", "W", "locH") ;

  for(int j = 0; j < N_+N_; j++)
  {
    fprintf (cFile, "%f %f %f %f %f\n", XYZ_(j,0), XYZ_(j,1), XYZ_(j,2), W_(j), locH_) ;
  }

  fclose (cFile);

  
  ROS_INFO("Service finished, a file with the cartesian trajectory was saved, including the pre-touch and touch frames at the beggining.");
  return true;
}

void ObjectModeller::publishTouchFrame()
{
  tf_broadcaster_.sendTransform(tf::StampedTransform(touch_frame_, ros::Time::now(), processing_frame_, "touch_frame"));
}

} // namespace gaussian_object_modelling


int main(int argc, char **argv) 
{
  ros::init(argc, argv, "object_modelling_node");
  ros::NodeHandle nh;

  gaussian_object_modelling::ObjectModeller node(nh);

   while(ros::ok())
  {
    node.publishTouchFrame();
    ros::spinOnce();
  } 
  
  return 0;
}





















  ///////////////////// OLDDDD ////////////////////////
  // publishing object with normals and usng nuklei with cloud with normals
  //sensor_msgs::PointCloud2 cloud_w_normal;
  //pcl::toROSMsg(*object_with_normals_ptr, cloud_w_normal);
  //pub_object_cloud_.publish(cloud_w_normal);
  //sensor_msgs::PointCloud2 cloud_normal;
  //pcl::toROSMsg(*object_normals_ptr, cloud_normal);
  //pub_normal_cloud_.publish(cloud_normal);
  // // publish the point cloud with normals 
  // pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_w_normals_ptr (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
  // *cloud_w_normals_ptr = nuklei::pclFromNuklei(density_);
  // sensor_msgs::PointCloud2 cloud_w_normal;
  // pcl::toROSMsg(*cloud_w_normals_ptr, cloud_w_normal);
  // pub_object_cloud_.publish(cloud_w_normal);


  // NUKLEI ! not needed anymore
  // create the gaussian process of the surface
  // nuklei::KernelCollection density;
  // density = nuklei::nukleiFromPcl<pcl::PointXYZRGB>(cluster, false);
  // density_.add(density);
  
  // change from r3 to r3xs2 kernels
  //density_.buildNeighborSearchTree();
  //density_.computeSurfaceNormals();

  // set density parameters
  // density_.setKernelLocH(locH_);
  // density_.setKernelOriH(oriH_);

  // compute statistics
  // density_.normalizeWeights();
  // density_.buildKdTree();

  // from now on, all calls of density functions must be as_const, e.g. as_const(density_).doMethod(ARGS)
  
  // // KERNEL REGRESSION
  // std::vector<int> trainLabels;

  // for (unsigned i = 0; i < as_const(density_).size(); i++)
  // {
  //   trainLabels.push_back(1);
  // }

  // ROS_INFO("as_const(density_).size() %i", as_const(density_).size());
  // ROS_INFO("trainLabels.size() %i", trainLabels.size());

  // klr_.setData(density_, trainLabels);
  
  // // ROS_INFO("Finished, get gramm matrix to check the data set");
  // // nuklei::GMatrix g;
  // // g = klr_.vklr();
  // // ROS_INFO("Finished, got gramm matrix");
  // // int g_rows, g_cols;
  // // g.GetSize(g_rows, g_cols);
  // // ROS_INFO("Gramm matrix elements Rows %d Cols %d", g_rows, g_cols);

  // // ROS_INFO("TRAINING surface model with %d points", density_.size());

  // double delta = 0.0001;
  // unsigned itrNewton = 5;
  // klr_.train( delta, itrNewton);
  // ROS_INFO("TRAINING DONE!");