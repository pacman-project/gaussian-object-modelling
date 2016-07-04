#include <algorithm> //for std::max_element
#include <chrono> //for time measurements
#include <fstream>

#include <node_utils.hpp>
#include <surface_approx.h>

#include <ros/package.h>
#include <boost/algorithm/string/split.hpp>
#include <boost/filesystem/path.hpp>
using namespace gp_regression;

SurfaceEstimator::SurfaceEstimator (): nh(ros::NodeHandle("surface_estimator")), start(false),
    input_ptr(boost::make_shared<PtC>()), training_ptr(boost::make_shared<PtC>()), R(2.0),
    sigma2(1e-1), min_v(0.0), max_v(1.0)
{
    srv_start = nh.advertiseService("start", &SurfaceEstimator::cb_start, this);
    pub_training = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB>> ("training_data", 1);
    pub_samples= nh.advertise<visualization_msgs::MarkerArray> ("samples", 1);
    nh.param<std::string>("/processing_frame", proc_frame, "/camera_rgb_optical_frame");
    nh.param<double>("sample_res", sample_res, 0.02);
}

void SurfaceEstimator::Publish()
{
    if (!start){
        ROS_WARN_THROTTLE(60,"[SurfaceEstimator::%s]\tNo surface model! Call start service to begin.",__func__);
        return;
    }
    //publish the model
    publishTraining();
    publishSamples();
}

// Publish cloud method
void SurfaceEstimator::publishTraining () const
{
    // These checks are  to make sure we are not publishing empty cloud,
    // we have a  gaussian process computed and there's actually someone
    // who listens to us
    if (start && training_ptr)
        if(!training_ptr->empty())
            // publish both the internal/external [-1, 1] training data
            pub_training.publish(*training_ptr);
}
void SurfaceEstimator::publishSamples () const
{
    if (samples_marks)
        if(pub_samples.getNumSubscribers() > 0)
            pub_samples.publish(*samples_marks);
}

//callback to start process service, executes when service is called
bool SurfaceEstimator::cb_start(gp_regression::StartProcess::Request& req, gp_regression::StartProcess::Response& res)
{
    //clean up everything
    start = false;
    input_ptr= boost::make_shared<PtC>();
    training_ptr= boost::make_shared<PtC>();
    reg.reset();
    obj_gp.reset();
    my_kernel.reset();
    samples_marks.reset();
    //////
    if (!req.obj_pcd.empty()){
        // User told us to load a clouds from a dir on disk instead.
        if (pcl::io::loadPCDFile(req.obj_pcd, *input_ptr) != 0){
            ROS_ERROR("[SurfaceEstimator::%s]\tError loading cloud from %s",__func__,req.obj_pcd.c_str());
            return (false);
        }
        input_ptr->header.frame_id=proc_frame;
    }
    training_ptr->header.frame_id=proc_frame;
    colorThem(0,0,255, input_ptr);
    prepareData();
    prepareExtData();
        if (computeGP()){
            //publish training set
            publishTraining();
            ros::spinOnce();
            //perform fake sampling
            fakeDeterministicSampling(true, 1.01, sample_res);
            start = true;
            return true;
        }
    return false;
}

//prepareData surface
bool SurfaceEstimator::prepareData()
{
    if(!input_ptr){
        //This  should never  happen  if compute  is  called from  start_process
        //service callback, however it does not hurt to add this extra check!
        ROS_ERROR("[SurfaceEstimator::%s]\tInput pointer not initialized. Aborting...",__func__);
        start = false;
        return false;
    }
    if (input_ptr->empty()){
        ROS_ERROR("[SurfaceEstimator::%s]\tInput point cloud is empty. Aborting...",__func__);
        start = false;
        return false;
    }

    input_gp = std::make_shared<gp_regression::Data>();

    ROS_INFO("[SurfaceEstimator::%s]\tPreparing Data...",__func__);
    /*****  Prepare the training data  *********************************************/
    //      Surface Points
    // add object points with value 0
    for(size_t i=0; i< input_ptr->points.size(); ++i) {
        input_gp->coord_x.push_back(input_ptr->points[i].x);
        input_gp->coord_y.push_back(input_ptr->points[i].y);
        input_gp->coord_z.push_back(input_ptr->points[i].z);
        input_gp->label.push_back(0.);
        input_gp->sigma2.push_back(sigma2);
    }
    // add object points to rviz in blue
    training_ptr->clear();
    for (const auto& pt: input_ptr->points)
        training_ptr->push_back(pt);
    return true;
}

void SurfaceEstimator::prepareExtData()
{
    pcl::NormalEstimationOMP<pcl::PointXYZRGB,pcl::Normal> ne;
    intext_gp = std::make_shared<gp_regression::Data>();
    ne.setInputCloud(input_ptr);
    ne.useSensorOriginAsViewPoint();
    ne.setRadiusSearch(0.1);
    pcl::PointCloud<pcl::Normal> normals;
    ne.compute(normals);
    if (normals.size() != input_ptr->size()){
        start = false;
        ROS_ERROR("[SurfaceEstimator::%s]\tInput point cloud and normals mismatch. Aborting...",__func__);
        return;
    }
    ROS_INFO("[SurfaceEstimator::%s]\tNormal Projection...",__func__);
    //process external points
    for (std::size_t i=0; i<input_ptr->size(); ++i)
    {
        pcl::PointXYZRGB pt;
        pt.x = input_ptr->points.at(i).x + normals.points.at(i).normal_x;
        pt.y = input_ptr->points.at(i).y + normals.points.at(i).normal_y;
        pt.z = input_ptr->points.at(i).z + normals.points.at(i).normal_z;
        colorIt(180,85,30, pt);
        intext_gp->coord_x.push_back(pt.x);
        intext_gp->coord_y.push_back(pt.y);
        intext_gp->coord_z.push_back(pt.z);
        intext_gp->label.push_back(1.0);
        intext_gp->sigma2.push_back(sigma2);
        training_ptr->push_back(pt);
    }
    //process internal points
    for (std::size_t i=0; i<input_ptr->size(); ++i)
    {
        pcl::PointXYZRGB pt;
        pt.x = input_ptr->points.at(i).x - normals.points.at(i).normal_x;
        pt.y = input_ptr->points.at(i).y - normals.points.at(i).normal_y;
        pt.z = input_ptr->points.at(i).z - normals.points.at(i).normal_z;
        colorIt(150,35,200, pt);
        intext_gp->coord_x.push_back(pt.x);
        intext_gp->coord_y.push_back(pt.y);
        intext_gp->coord_z.push_back(pt.z);
        intext_gp->label.push_back(-1.0);
        intext_gp->sigma2.push_back(sigma2);
        training_ptr->push_back(pt);
    }
}


bool SurfaceEstimator::computeGP()
{
    if(!input_gp || !intext_gp)
        return false;
    ROS_INFO("[SurfaceEstimator::%s]\tStart GP Computation...",__func__);
    auto begin_time = std::chrono::high_resolution_clock::now();

    /*****  Create the gp model  *********************************************/
    //create the model to be stored in class
    gp_regression::Data::Ptr data_gp = std::make_shared<gp_regression::Data>();
    for (size_t i =0; i<input_gp->coord_x.size(); ++i)
    {
        data_gp->coord_x.push_back(input_gp->coord_x[i]);
        data_gp->coord_y.push_back(input_gp->coord_y[i]);
        data_gp->coord_z.push_back(input_gp->coord_z[i]);
        data_gp->label.push_back(input_gp->label[i]);
        data_gp->sigma2.push_back(input_gp->sigma2[i]);
    }
    for (size_t i =0; i<intext_gp->coord_x.size(); ++i)
    {
        data_gp->coord_x.push_back(intext_gp->coord_x[i]);
        data_gp->coord_y.push_back(intext_gp->coord_y[i]);
        data_gp->coord_z.push_back(intext_gp->coord_z[i]);
        data_gp->label.push_back(intext_gp->label[i]);
        data_gp->sigma2.push_back(intext_gp->sigma2[i]);
    }

    obj_gp = std::make_shared<gp_regression::Model>();
    reg = std::make_shared<gp_regression::ThinPlateRegressor>();
    // my_kernel = std::make_shared<gp_regression::ThinPlate>(out_sphere_rad * 2);
    my_kernel = std::make_shared<gp_regression::ThinPlate>(R);
    reg->setCovFunction(my_kernel);
    const bool withoutNormals = false;
    reg->create<withoutNormals>(data_gp, obj_gp);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count();
    ROS_INFO("[SurfaceEstimator::%s]\tRegressor and Model created using %ld training points. Total time consumed: %ld milliseconds.", __func__, data_gp->label.size(), elapsed );
    start = true;
    return true;
}

// for visualization purposes
void SurfaceEstimator::fakeDeterministicSampling(const bool first_time, const double scale, const double pass)
{
    auto begin_time = std::chrono::high_resolution_clock::now();
    samples_marks = boost::make_shared<visualization_msgs::MarkerArray>();
    model_ptr = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    visualization_msgs::Marker samples;
    samples.header.frame_id = proc_frame;
    samples.header.stamp = ros::Time();
    samples.lifetime = ros::Duration(0.5);
    samples.ns = "samples";
    samples.id = 0;
    samples.type = visualization_msgs::Marker::POINTS;
    samples.action = visualization_msgs::Marker::ADD;
    samples.scale.x = 0.003;
    samples.scale.y = 0.003;
    samples.scale.z = 0.003;
    samples_marks->markers.push_back(samples);

    pcl::PointXYZRGB min_pt, max_pt;
    pcl::getMinMax3D(*input_ptr, min_pt, max_pt);
    size_t count(0);
    const auto total = std::pow( std::floor((2*scale+1)/pass), 3);
    ROS_INFO("[SurfaceEstimator::%s]\tSampling %g grid points on GP ...",__func__, total);
    for (double x = min_pt.x; x<= max_pt.x; x += pass)
    {
        std::vector<std::thread> threads;
        for (double y = min_pt.y; y<= max_pt.y; y += pass)
        {
            for (double z = min_pt.z; z<= max_pt.z; z += pass)
            {
                ++count;
                std::cout<<" -> "<<count<<"/"<<total<<"\r";
                visualization_msgs::Marker arrow;
                /* arrow.header.frame_id = proc_frame; */
                /* arrow.header.stamp = ros::Time(); */
                /* arrow.lifetime = ros::Duration(0.5); */
                /* arrow.ns = "normals"; */
                /* arrow.id = count; */
                /* arrow.type = visualization_msgs::Marker::ARROW; */
                /* arrow.action = visualization_msgs::Marker::ADD; */
                /* arrow.scale.x = 0.002; */
                /* arrow.scale.y = 0.005; */
                /* arrow.scale.z = 0.005; */
                threads.emplace_back(&SurfaceEstimator::samplePoint, this, x,y,z, std::ref(samples), std::ref(arrow));
            }
        }
        for (auto &t: threads)
            t.join();
        //update visualization
        publishSamples();
        ros::spinOnce();
    }
    std::cout<<std::endl;

    ROS_INFO("[SurfaceEstimator::%s]\tFound %ld points approximately on GP surface.",__func__,
            samples.points.size());

    if (first_time){
        min_v = 10.0;
        max_v = 0.0;
        for (const auto &pt : model_ptr->points)
        {
            if (pt.intensity <= min_v)
                min_v = pt.intensity;
            if (pt.intensity >= max_v)
                max_v = pt.intensity;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end_time - begin_time).count();
    ROS_INFO("[SurfaceEstimator::%s]\tTotal time consumed: %ld seconds.", __func__, elapsed );
    // if (elapsed > 60)
    //     sample_res = sample_res < 0.1 ? sample_res + 0.01 : 0.1;
}
void
SurfaceEstimator::samplePoint(const double x, const double y, const double z, visualization_msgs::Marker &samp,
        visualization_msgs::Marker &arrow)
{
    gp_regression::Data::Ptr qq = std::make_shared<gp_regression::Data>();
    qq->coord_x.push_back(x);
    qq->coord_y.push_back(y);
    qq->coord_z.push_back(z);
    std::vector<double> ff,vv;
    Eigen::MatrixXd nn;
    reg->evaluate(obj_gp, qq, ff, vv, nn);
    if (std::abs(ff.at(0)) <= 0.01) {
        const double mid_v = ( min_v + max_v ) * 0.5;
        geometry_msgs::Point pt;
        pcl::PointXYZI pcl_pt;
        std_msgs::ColorRGBA cl;
        pcl_pt.x = pt.x = x;
        pcl_pt.y = pt.y = y;
        pcl_pt.z = pt.z = z;
        cl.a = 1.0;
        cl.b = 0.0;
        cl.r = (vv[0]<mid_v) ? 1/(mid_v - min_v) * (vv[0] - min_v) : 1.0;
        cl.g = (vv[0]>mid_v) ? -1/(max_v - mid_v) * (vv[0] - mid_v) + 1 : 1.0;
        pcl_pt.intensity = vv[0]; //Use intensity channel to store variance
        //Gradient sampling
        /* Eigen::Vector3d g = nn.row(0); */
        /* if (!g.isMuchSmallerThan(1e3, 1e-1) || g.isZero(1e-6)) */
        /*     g.normalize(); */
        /* arrow.points.push_back(pt); */
        /* geometry_msgs::Point pt_end; */
        /* pt_end.x = pt.x + g[0]*0.1; */
        /* pt_end.y = pt.y + g[1]*0.1; */
        /* pt_end.z = pt.z + g[2]*0.1; */
        /* arrow.points.push_back(pt_end); */
        //locks
        std::lock_guard<std::mutex> lock (mtx_samp);
        //samples_marks->markers.push_back(arrow);
        samp.points.push_back(pt);
        samp.colors.push_back(cl);
        samples_marks->markers[0] = samp;
        model_ptr->push_back(pcl_pt);
    }
}

///// MAIN ////////////////////////////////////////////////////////////////////

int main (int argc, char *argv[])
{
    ros::init(argc, argv, "gaussian_process");
    SurfaceEstimator node;
    ros::Rate rate(10); //try to go at 10hz
    while (node.nh.ok())
    {
        //gogogo!
        ros::spinOnce();
        node.Publish();
        rate.sleep();
    }
    //someone killed us :(
    return 0;
}
