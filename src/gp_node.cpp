#include <algorithm> //for std::max_element
#include <chrono> //for time measurements

#include <node_utils.hpp>
#include <gp_node.h>

#include <ros/package.h>
#include <boost/algorithm/string/split.hpp>
#include <boost/filesystem/path.hpp>
using namespace gp_regression;

GaussianProcessNode::GaussianProcessNode (): nh(ros::NodeHandle("gaussian_process")), start(false),
    object_ptr(boost::make_shared<PtC>()), hand_ptr(boost::make_shared<PtC>()), data_ptr_(boost::make_shared<PtC>()),
    model_ptr(boost::make_shared<PtC>()), real_explicit_ptr(boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>()),
    fake_sampling(true), exploration_started(false), out_sphere_rad(1.8), sigma2(1e-1), min_v(100), max_v(0),
    simulate_touch(true)
{
    mtx_marks = std::make_shared<std::mutex>();
    srv_start = nh.advertiseService("start_process", &GaussianProcessNode::cb_start, this);
    srv_update = nh.advertiseService("update_process", &GaussianProcessNode::cb_updateS, this);
    srv_get_next_best_path_ = nh.advertiseService("get_next_best_path", &GaussianProcessNode::cb_get_next_best_path, this);
    pub_model = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB>> ("training_data", 1);
    pub_real_explicit = nh.advertise<pcl::PointCloud<pcl::PointXYZI> >("estimated_model", 1);
    pub_octomap = nh.advertise<octomap_msgs::Octomap>("octomap",1); //TODO fix correct topic name
    pub_markers = nh.advertise<visualization_msgs::MarkerArray> ("atlas", 1);
    sub_update_ = nh.subscribe(nh.resolveName("/path_log"),1, &GaussianProcessNode::cb_update, this);
    nh.param<std::string>("/processing_frame", proc_frame, "/camera_rgb_optical_frame");

    // pub_point = nh.advertise<gp_regression::SampleToExplore> ("sample_to_explore", 0, true);
    // pub_point_marker = nh.advertise<geometry_msgs::PointStamped> ("point_to_explore", 0, true); // TEMP, should be a trajectory, curve, pose
    // pub_direction_marker = nh.advertise<geometry_msgs::WrenchStamped> ("direction_to_explore", 0, true); // TEMP, should be a trajectory, curve, pose
}

void GaussianProcessNode::Publish ()
{
    if (!start){
        ROS_WARN_THROTTLE(60,"[GaussianProcessNode::%s]\tNo object model found! Call start_process service to begin creating a model.",__func__);
        return;
    }
    //publish the model
    publishCloudModel();
    //publish octomap
    publishOctomap();
    //publish markers
    publishAtlas();
}

// Publish cloud method
void GaussianProcessNode::publishCloudModel () const
{
    // These checks are  to make sure we are not publishing empty cloud,
    // we have a  gaussian process computed and there's actually someone
    // who listens to us
    if (start && model_ptr){
        // if(!model_ptr->empty() && pub_model.getNumSubscribers()>0){
        // I don't care if there are subscribers or not... publish'em all!
        if(!model_ptr->empty()){
            // publish both the internal [-1, 1] model and the ex

            // this actually publishes the training data, not the model!
            pub_model.publish(*model_ptr);

            // and the explicit estimated model back to the real world
            pcl::PointCloud<pcl::PointXYZI> real_explicit;
            real_explicit.header = real_explicit_ptr->header;
            reMeanAndDenormalizeData(real_explicit_ptr, real_explicit);
            pub_real_explicit.publish(real_explicit);
        }
    }
}
//publish atlas markers and other samples
void GaussianProcessNode::publishAtlas () const
{
    if (markers)
        if(pub_markers.getNumSubscribers() > 0){
            std::lock_guard<std::mutex> lock (*mtx_marks);
            for (auto &mark : markers->markers)
            {
                mark.header.frame_id = object_ptr->header.frame_id;
                mark.header.stamp = ros::Time();
            }
            pub_markers.publish(*markers);
        }
}


// simple preparation of data before computing the gp
void GaussianProcessNode::deMeanAndNormalizeData(const PtC::Ptr &data_ptr, PtC::Ptr &out)
{
    // demean and normalize points on cloud(s)...

    // first demean
    Eigen::Vector4f centroid;
    if(pcl::compute3DCentroid<pcl::PointXYZRGB>(*data_ptr, centroid) == 0){
        ROS_ERROR("[GaussianProcessNode::%s]\tFailed to compute object centroid. Aborting...",__func__);
        return;
    }
    current_offset_ = centroid.cast<double>();

    data_ptr_->clear();
    PtC::Ptr tmp (new PtC);
    pcl::demeanPointCloud(*data_ptr, centroid, *tmp);

    // then normalize points to be in box = [-1, 1] x [-1, 1] x [-1, 1]
    current_scale_ = 0.0;
    for (const auto& pt: tmp->points)
    {
        double norm = std::sqrt( pt.x * pt.x + pt.y * pt.y + pt.z* pt.z);
        if (norm >= current_scale_)
            current_scale_ = norm;
    }
    Eigen::Matrix4f sc;
    sc    << 1/current_scale_, 0, 0, 0,
             0, 1/current_scale_, 0, 0,
             0, 0, 1/current_scale_, 0,
             0, 0, 0,          1;
    // note that this writes to class member
    pcl::transformPointCloud(*tmp, *out, sc);
    return;
}

// void GaussianProcessNode::reMeanAndDenormalizeData(const std::vector<Eigen::Vector3d> &in, std::vector<Eigen::Vector3d> &out)
void GaussianProcessNode::reMeanAndDenormalizeData(Eigen::Vector3d &data)
{
    data = current_scale_*data;
    data = data + current_offset_.block(0,0,3,1);
}
template<typename PT>
void GaussianProcessNode::reMeanAndDenormalizeData(const typename pcl::PointCloud<PT>::Ptr &data_ptr, pcl::PointCloud<PT> &out) const
{
    // here it is safe to do it in one single transformation
    Eigen::Matrix4f t;
    t    << current_scale_, 0, 0, current_offset_(0),
             0, current_scale_, 0, current_offset_(1),
             0, 0, current_scale_, current_offset_(2),
             0, 0, 0,          1;
    // note that this writes to class member
    pcl::transformPointCloud(*data_ptr, out, t);
}

//callback to start process service, executes when service is called
bool GaussianProcessNode::cb_get_next_best_path(gp_regression::GetNextBestPath::Request& req, gp_regression::GetNextBestPath::Response& res)
{
    ros::Rate rate(10); //try to go at 10hz, as in the node
    {
        std::lock_guard<std::mutex> lock(*mtx_marks);
        markers = boost::make_shared<visualization_msgs::MarkerArray>();
    }
    //perform fake sampling
    marchingSampling(0.06, 0.06);
    if(startExploration()){
        while(exploration_started){
            // don't like it, cause we loose the actual velocity of the atlas
            // but for now, this is it, repeating the node while loop here
            //gogogo!
            ros::spinOnce();
            Publish();
            checkExploration();
            rate.sleep();
        }
        std_msgs::Header solution_header;
        solution_header.stamp = ros::Time::now();
        solution_header.frame_id = object_ptr->header.frame_id;

        gp_regression::Path next_best_path;
        next_best_path.header = solution_header;
        for (size_t i=0; i<solution.size(); ++i)
        {
            // ToDO: solutionToPath(solution, path) function
            gp_atlas_rrt::Chart chart = atlas->getNode(solution[i]);
            Eigen::Vector3d point_eigen = chart.getCenter();
            Eigen::Vector3d normal_eigen = chart.getNormal();
            if (simulate_touch) //synthetic touch simulation
            {
                synthTouch(point_eigen, normal_eigen);
                return true;
            }
            // modifies the point
            reMeanAndDenormalizeData(point_eigen);
            // normal does not need to be reMeanAndRenormalized for now

            geometry_msgs::PointStamped point_msg;
            geometry_msgs::Vector3Stamped normal_msg;
            point_msg.point.x = point_eigen(0);
            point_msg.point.y = point_eigen(1);
            point_msg.point.z = point_eigen(2);
            point_msg.header = solution_header;
            normal_msg.vector.x = normal_eigen(0);
            normal_msg.vector.y = normal_eigen(1);
            normal_msg.vector.z = normal_eigen(2);
            normal_msg.header = solution_header;

            next_best_path.points.push_back( point_msg );
            next_best_path.directions.push_back( normal_msg );
        }
        res.next_best_path = next_best_path;
        return true;
    }
    return false;
}

//callback to start process service, executes when service is called
bool GaussianProcessNode::cb_start(gp_regression::StartProcess::Request& req, gp_regression::StartProcess::Response& res)
{
    //clean up everything
    start = exploration_started = false;
    mtx_marks = std::make_shared<std::mutex>();
    object_ptr= boost::make_shared<PtC>();
    hand_ptr= boost::make_shared<PtC>();
    data_ptr_=boost::make_shared<PtC>();
    model_ptr= boost::make_shared<PtC>();
    real_explicit_ptr= boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    reg_.reset();
    obj_gp.reset();
    my_kernel.reset();
    atlas.reset();
    explorer.reset();
    solution.clear();
    markers.reset();
    cloud_labels.clear();
    //////
    pcl::PointCloud<pcl::PointXYZ> tmp;
    if(req.obj_pcd.empty()){
        //Request was empty, means we have to call pacman vision service to
        //get a cloud.
        std::string service_name = nh.resolveName("/pacman_vision/listener/get_cloud_in_hand");
        pacman_vision_comm::get_cloud_in_hand service;
        // service.request.save = "false";
        ////////////////////////////////////////////////////////////////////////
        //TODO: Service  requires to know which  hand is grasping the  object we
        //dont have a way  to tell inside here. Assuming it's  the left hand for
        //now.(Fri 06 Nov 2015 05:11:45 PM CET -- tabjones)
        ////////////////////////////////////////////////////////////////////////
        service.request.right = false;
        if (!ros::service::call<pacman_vision_comm::get_cloud_in_hand>(service_name, service))
        {
            ROS_ERROR("[GaussianProcessNode::%s]\tGet cloud in hand service call failed!",__func__);
            return (false);
        }
        //object and hand clouds are saved into class
        sensor_msgs::PointCloud msg, msg_conv;
        sensor_msgs::PointCloud2 msg2;
        sensor_msgs::convertPointCloud2ToPointCloud(service.response.obj, msg);
        // pcl::fromROSMsg (service.response.hand, *hand_ptr);
        listener.transformPointCloud(proc_frame, msg, msg_conv);
        sensor_msgs::convertPointCloudToPointCloud2(msg_conv, msg2);
        pcl::fromROSMsg (msg2, *object_ptr);
    }
    else{
        if(req.obj_pcd.compare("sphere") == 0 || req.obj_pcd.compare("half_sphere") == 0){
            object_ptr = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB> >();
            const int ang_div = 24;
            const int lin_div = 20;
            const double radius = 0.06;
            const double ang_step = M_PI * 2 / ang_div;
            const double lin_step = 2 * radius / lin_div;
            double end_lin = radius;
            if (req.obj_pcd.compare("half_sphere")==0)
                end_lin /= 2;
            int j(0);
            for (double lin=-radius+lin_step/2; lin<end_lin; lin+=lin_step)
                for (double ang=0; ang < 2*M_PI; ang+=ang_step, ++j)
                {
                    double x = sqrt(radius*radius - lin*lin) * cos(ang);
                    double y = sqrt(radius*radius - lin*lin) * sin(ang);
                    double z = lin + 1.0; //add translation along z
                    //add sphere points as blue model
                    pcl::PointXYZRGB sp;
                    sp.x = x;
                    sp.y = y;
                    sp.z = z;
                    colorIt(0,0,255, sp);
                    object_ptr->push_back(sp);
                }
            object_ptr->header.frame_id=proc_frame;
        }
        else{
            // User told us to load a clouds from a dir on disk instead.
            if (pcl::io::loadPCDFile(req.obj_pcd, *object_ptr) != 0){
                ROS_ERROR("[GaussianProcessNode::%s]\tError loading cloud from %s",__func__,req.obj_pcd.c_str());
                return (false);
            }
            if (simulate_touch){
                std::string obj_name;
                std::vector<std::string> vst;
                split(vst, req.obj_pcd, boost::is_any_of("/."), boost::token_compress_on);
                obj_name = vst.at(vst.size()-2);
                std::string models_path (ros::package::getPath("asus_scanner_models"));
                boost::filesystem::path model_path (models_path + "/" + obj_name + "/" + obj_name + ".pcd");
                full_object = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
                if (boost::filesystem::exists(model_path) && boost::filesystem::is_regular_file(model_path))
                {
                    if (pcl::io::loadPCDFile(model_path.c_str(), tmp))
                        ROS_ERROR("[GaussianProcessNode::%s] Error loading model %s",__func__, model_path.c_str());
                }
                else
                    ROS_ERROR("[GaussianProcessNode::%s] Requested model (%s) does not exists in asus_scanner_models package",__func__, model_path.stem().c_str());
            }
            // if (pcl::io::loadPCDFile((req.cloud_dir+"/hand.pcd"), *hand_ptr) != 0)
            //     ROS_WARN("[GaussianProcessNode::%s]\tError loading cloud from %s, ignoring hand",__func__,(req.cloud_dir + "/hand.pcd").c_str());
            // We  need  to  fill  point  cloud header  or  ROS  will  complain  when
            // republishing this cloud. Let's assume it was published by asus kinect.
            // I think it's just need the frame id
            object_ptr->header.frame_id=proc_frame;
            // hand_ptr->header.frame_id=proc_frame;
        }
    }
    model_ptr->header.frame_id=object_ptr->header.frame_id;
    real_explicit_ptr->header.frame_id=object_ptr->header.frame_id;
    for(const auto& pt: object_ptr->points)
        cloud_labels.push_back(0);

    prepareExtData();
    deMeanAndNormalizeData( object_ptr, data_ptr_ );
    if (simulate_touch){
        Eigen::Matrix4f t;
        t    << 1/current_scale_, 0, 0, -current_offset_(0),
             0, 1/current_scale_, 0, -current_offset_(1),
             0, 0, 1/current_scale_, -current_offset_(2),
             0, 0, 0,          1;
        pcl::transformPointCloud(tmp, *full_object, t);
        kd_full.setInputCloud(full_object);
    }
    if (prepareData())
        if (computeGP()){
            //initialize objects involved
            markers = boost::make_shared<visualization_msgs::MarkerArray>();
            //perform fake sampling
            marchingSampling(true, 0.06,0.02);
            computeOctomap();
            return true;
        }
    return false;
}
bool GaussianProcessNode::cb_updateS(gp_regression::Update::Request &req, gp_regression::Update::Response &res)
{
    const gp_regression::Path::ConstPtr &msg = boost::make_shared<gp_regression::Path>(req.explored_points);
    cb_update(msg);
    return true;
}

void GaussianProcessNode::cb_update(const gp_regression::Path::ConstPtr &msg)
{
    //assuming points in processing_frame
    for (size_t i=0; i< msg->points.size(); ++i)
    {
        pcl::PointXYZRGB pt;
        pt.x = msg->points[i].point.x;
        pt.y = msg->points[i].point.y;
        pt.z = msg->points[i].point.z;
        // new data in cyan
        colorIt(0,255,255, pt);
        // model_ptr->push_back(pt);
        object_ptr->push_back(pt);
        if (msg->isOnSurface[i].data)
            cloud_labels.push_back(0);
        else
            cloud_labels.push_back(1);
    }

    /* UPDATE METHOD IS NOT POSSIBLE
     * CENTROID NEEDS TO BE RECOMPUTED EVERY TIME
     * NEW DATA ARRIVES, LEAVING THIS PIECE
     * OF CODE TO HAVE THE UPDATE ROUTINE SOMEWHERE
     *
     * END OF STORY
     *
    gp_regression::Data::Ptr fresh_data = std::make_shared<gp_regression::Data>();
    fresh_data->coord_x.push_back((double)msg->point.x);
    fresh_data->coord_y.push_back((double)msg->point.y);
    fresh_data->coord_z.push_back((double)msg->point.z);
    fresh_data->label.push_back(0.0);
    // this is a more precise measurement than points from the camera
    fresh_data->sigma2.push_back(5e-2);
    */

    deMeanAndNormalizeData( object_ptr, data_ptr_ );
    prepareData();
    computeGP();
    //initialize objects involved
    markers = boost::make_shared<visualization_msgs::MarkerArray>();
    //perform fake sampling
    marchingSampling(false, 0.06,0.06);
    computeOctomap();
    return;
}

void GaussianProcessNode::prepareExtData()
{
    if (!model_ptr->empty())
        model_ptr->clear();
    ext_size = 1;
    ext_gp = std::make_shared<gp_regression::Data>();

    //      Internal Point
    // add centroid as label -1
    ext_gp->coord_x.push_back(0);
    ext_gp->coord_y.push_back(0);
    ext_gp->coord_z.push_back(0);
    ext_gp->label.push_back(-1.0);
    ext_gp->sigma2.push_back(sigma2);
    // add internal point to rviz in cyan
    pcl::PointXYZRGB cen;
    cen.x = 0;
    cen.y = 0;
    cen.z = 0;
    colorIt(0,255,255, cen);
    model_ptr->push_back(cen);

    //      External points
    // add points in a sphere around centroid with label 1
    // sphere bounds computation
    const int ang_div = 8; //divide 360° in 8 pieces, i.e. steps of 45°
    const int lin_div = 4; //divide diameter into 4 pieces
    // This makes 8*6 = 48 points.
    const double ang_step = M_PI * 2 / ang_div; //steps of 45°
    const double lin_step = 2 * out_sphere_rad / lin_div;
    // 8 steps for diameter times 6 for angle, make  points on the sphere surface
    for (double lin=-out_sphere_rad+lin_step/2; lin< out_sphere_rad; lin+=lin_step)
    {
        for (double ang=0; ang < 2*M_PI; ang+=ang_step)
        {
            double x = sqrt(std::pow(out_sphere_rad, 2) - lin*lin) * cos(ang);
            double y = sqrt(std::pow(out_sphere_rad, 2) - lin*lin) * sin(ang);
            double z = lin;

            ext_gp->coord_x.push_back(x);
            ext_gp->coord_y.push_back(y);
            ext_gp->coord_z.push_back(z);
            ext_gp->label.push_back(1.0);
            ext_gp->sigma2.push_back(sigma2);

            // add sphere points to rviz in purple
            pcl::PointXYZRGB sp;
            sp.x = x;
            sp.y = y;
            sp.z = z;
            colorIt(255,0,255, sp);
            model_ptr->push_back(sp);
            ++ext_size;
        }
    }
}

//prepareData in data_ptr_
bool GaussianProcessNode::prepareData()
{
    if(!data_ptr_){
        //This  should never  happen  if compute  is  called from  start_process
        //service callback, however it does not hurt to add this extra check!
        ROS_ERROR("[GaussianProcessNode::%s]\tObject cloud pointer is empty. Aborting...",__func__);
        start = false;
        return false;
    }
    if (data_ptr_->empty()){
        ROS_ERROR("[GaussianProcessNode::%s]\tObject point cloud is empty. Aborting...",__func__);
        start = false;
        return false;
    }

    //size sanity check
    if (data_ptr_->size() != cloud_labels.size()){
        ROS_ERROR("[GaussianProcessNode::%s]\tData labels mismatch. Aborting...",__func__);
        start = false;
        return false;
    }
    cloud_gp = std::make_shared<gp_regression::Data>();

    /*****  Prepare the training data  *********************************************/

    //      Surface Points
    // add object points with label 0 or 1 (not touched)
    for(size_t i=0; i< data_ptr_->points.size(); ++i) {
        cloud_gp->coord_x.push_back(data_ptr_->points[i].x);
        cloud_gp->coord_y.push_back(data_ptr_->points[i].y);
        cloud_gp->coord_z.push_back(data_ptr_->points[i].z);
        cloud_gp->label.push_back(cloud_labels.at(i));
        cloud_gp->sigma2.push_back(sigma2);
        if (cloud_labels.at(i) == 0)
            colorIt(0,0,255, data_ptr_->points[i]);
        else
            colorIt(255,0,255, data_ptr_->points[i]);
    }
    // add object points to rviz in blue
    // resize to ext_size first, so you wont lose external data, but overwrite
    // object data
    model_ptr->resize(ext_size);
    *model_ptr += *data_ptr_;


    return true;
}

bool GaussianProcessNode::computeGP()
{
    if(!cloud_gp || !ext_gp)
        return false;
    auto begin_time = std::chrono::high_resolution_clock::now();

    /*****  Create the gp model  *********************************************/
    //create the model to be stored in class
    gp_regression::Data::Ptr data_gp = std::make_shared<gp_regression::Data>();
    for (size_t i =0; i<cloud_gp->coord_x.size(); ++i)
    {
        data_gp->coord_x.push_back(cloud_gp->coord_x[i]);
        data_gp->coord_y.push_back(cloud_gp->coord_y[i]);
        data_gp->coord_z.push_back(cloud_gp->coord_z[i]);
        data_gp->label.push_back(cloud_gp->label[i]);
        data_gp->sigma2.push_back(cloud_gp->sigma2[i]);
    }
    for (size_t i =0; i<ext_gp->coord_x.size(); ++i)
    {
        data_gp->coord_x.push_back(ext_gp->coord_x[i]);
        data_gp->coord_y.push_back(ext_gp->coord_y[i]);
        data_gp->coord_z.push_back(ext_gp->coord_z[i]);
        data_gp->label.push_back(ext_gp->label[i]);
        data_gp->sigma2.push_back(ext_gp->sigma2[i]);
    }

    obj_gp = std::make_shared<gp_regression::Model>();
    reg_ = std::make_shared<gp_regression::ThinPlateRegressor>();
    my_kernel = std::make_shared<gp_regression::ThinPlate>(out_sphere_rad * 2);
    reg_->setCovFunction(my_kernel);
    const bool withoutNormals = false;
    reg_->create<withoutNormals>(data_gp, obj_gp);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - begin_time).count();
    ROS_INFO("[GaussianProcessNode::%s]\tRegressor and Model created using %ld training points. Total time consumed: %ld nanoseconds.", __func__, cloud_gp->label.size(), elapsed );

    start = true;
    return true;
}

//
bool GaussianProcessNode::startExploration()
{
    //make sure we have a model and an object, we should have if start was called
    if (!object_ptr || object_ptr->empty() || !data_ptr_ || data_ptr_->empty()){
        ROS_ERROR("[GaussianProcessNode::%s]\tNo object initialized, call start service.",__func__);
        return false;
    }
    if (!obj_gp){
        ROS_ERROR("[GaussianProcessNode::%s]\tNo GP model initialized, call start service.",__func__);
        return false;
    }

    //create the atlas
    atlas = std::make_shared<gp_atlas_rrt::AtlasCollision>(obj_gp, reg_);
    //termination condition
    atlas->setVarianceTolGoal( 0.4 );
    //factor to control disc radius
    atlas->setVarRadiusFactor( 0.3 );
    //atlas is ready

    //setup explorer
    explorer = std::make_shared<gp_atlas_rrt::ExplorerMultiBranch>(nh, "explorer");
    explorer->setMarkers(markers, mtx_marks);
    explorer->setAtlas(atlas);
    explorer->setMaxNodes(100);
    //get a starting point from data cloud
    int r_id = getRandIn(0, data_ptr_->points.size()-1 );
    Eigen::Vector3d root;
    root << data_ptr_->points[r_id].x,
            data_ptr_->points[r_id].y,
            data_ptr_->points[r_id].z;
    explorer->setStart(root);
    //explorer is ready, start exploration (this spawns a thread)
    explorer->startExploration();
    exploration_started = true;
    ROS_INFO("[GaussianProcessNode::%s]\tExploration started", __func__);
    return true;
}

void GaussianProcessNode::checkExploration()
{
    if (!exploration_started)
        return;
    if (explorer->hasSolution()){
        solution = explorer->getSolution();
        explorer->stopExploration();
        exploration_started = false;
        ROS_INFO("[GaussianProcessNode::%s]\tSolution Found", __func__);
    }
}

// for visualization purposes
void GaussianProcessNode::fakeDeterministicSampling(const bool first_time, const double scale, const double pass)
{
    auto begin_time = std::chrono::high_resolution_clock::now();
    if(!markers || !fake_sampling)
        return;

    markers->markers.clear();
    visualization_msgs::Marker samples;
    samples.header.frame_id = object_ptr->header.frame_id;
    samples.header.stamp = ros::Time::now();
    samples.lifetime = ros::Duration(0);
    samples.ns = "samples";
    samples.id = 0;
    samples.type = visualization_msgs::Marker::POINTS;
    samples.action = visualization_msgs::Marker::ADD;
    samples.scale.x = 0.01;
    samples.scale.y = 0.01;
    samples.scale.z = 0.01;

    gp_regression::Data::Ptr ss = std::make_shared<gp_regression::Data>();
    std::vector<double> ssvv;

    size_t count(0);
    const auto total = std::floor( std::pow((2*scale+1)/pass, 3) );
    ROS_INFO("[GaussianProcessNode::%s]\tSampling %g grid points on GP ...",__func__, total);
    for (double x = -scale; x<= scale; x += pass)
        for (double y = -scale; y<= scale; y += pass)
            for (double z = -scale; z<= scale; z += pass)
            {
                gp_regression::Data::Ptr qq = std::make_shared<gp_regression::Data>();
                qq->coord_x.push_back(x);
                qq->coord_y.push_back(y);
                qq->coord_z.push_back(z);
                std::vector<double> ff;
                reg_->evaluate(obj_gp, qq, ff);
                if (ff.at(0) <= 0.01 && ff.at(0) >= -0.01) {
                    std::vector<double> vv;
                    reg_->evaluate(obj_gp, qq, ff,vv);
                    if (first_time){
                        if (vv.at(0) <= min_v)
                            min_v = vv.at(0);
                        if (vv.at(0) >= max_v)
                            max_v = vv.at(0);
                    }
                    ss->coord_x.push_back(x);
                    ss->coord_y.push_back(y);
                    ss->coord_z.push_back(z);
                    ssvv.push_back(vv.at(0));
                }
                ++count;
                std::cout<<" -> "<<count<<"/"<<total<<"\r";
            }
    std::cout<<std::endl;

    ROS_INFO("[GaussianProcessNode::%s]\tFound %ld points approximately on GP surface, plotting them.",__func__,
            ssvv.size());
    const double mid_v = ( min_v + max_v ) * 0.5;
    // std::cout<<"min "<<min_v<<" mid "<<mid_v<<" max "<<max_v<<std::endl; //tmp debug
    for (size_t i=0; i< ssvv.size(); ++i)
    {
        geometry_msgs::Point pt;
        std_msgs::ColorRGBA cl;
        pcl::PointXYZI pt_pcl;
        pt.x = ss->coord_x.at(i);
        pt.y = ss->coord_y.at(i);
        pt.z = ss->coord_z.at(i);
        cl.a = 1.0;
        cl.b = 0.0;
        cl.r = (ssvv.at(i)<mid_v) ? 1/(mid_v - min_v) * (ssvv.at(i) - min_v) : 1.0;
        cl.g = (ssvv.at(i)>mid_v) ? -1/(max_v - mid_v) * (ssvv.at(i) - mid_v) + 1 : 1.0;
        samples.points.push_back(pt);
        samples.colors.push_back(cl);
        pt_pcl.x = static_cast<float>(ss->coord_x.at(i));
        pt_pcl.y = static_cast<float>(ss->coord_y.at(i));
        pt_pcl.z = static_cast<float>(ss->coord_z.at(i));
        //this goes linearly from 1 at min_v to 0.1 at max_v
        float p_hit = ( -9*ssvv.at(i) + 9*min_v + 10*(max_v - min_v) ) / (10*(max_v - min_v));
        pt_pcl.intensity = p_hit;
        real_explicit_ptr->push_back(pt_pcl);
    }
    markers->markers.push_back(samples);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(end_time - begin_time).count();
    ROS_INFO("[GaussianProcessNode::%s]\tTotal time consumed: %d minutes.", __func__, elapsed );
}

void
GaussianProcessNode::marchingSampling(const bool first_time, const float leaf_size, const float leaf_pass)
{
    auto begin_time = std::chrono::high_resolution_clock::now();
    if(!markers || !fake_sampling)
        return;

    markers->markers.clear();
    visualization_msgs::Marker samples;
    samples.header.frame_id = object_ptr->header.frame_id;
    samples.header.stamp = ros::Time::now();
    samples.lifetime = ros::Duration(0);
    samples.ns = "samples";
    samples.id = 0;
    samples.type = visualization_msgs::Marker::POINTS;
    samples.action = visualization_msgs::Marker::ADD;
    samples.scale.x = 0.01;
    samples.scale.y = 0.01;
    samples.scale.z = 0.01;

    std::shared_ptr<pcl::PointXYZ> start;

    ROS_INFO("[GaussianProcessNode::%s]\tMarching sampling on GP...",__func__);
    for (double x = -1.1; x<= 1.1; x += 0.1)
    {
        for (double y = -1.1; y<= 1.1; y += 0.1)
        {
            for (double z = -1.1; z<= 1.1; z += 0.1)
            {
                gp_regression::Data::Ptr qq = std::make_shared<gp_regression::Data>();
                qq->coord_x.push_back(x);
                qq->coord_y.push_back(y);
                qq->coord_z.push_back(z);
                std::vector<double> ff;
                reg_->evaluate(obj_gp, qq, ff);
                if (std::abs(ff.at(0)) <= 0.01) {
                    start = std::make_shared<pcl::PointXYZ>();
                    start->x = x;
                    start->y = y;
                    start->z = z;
                    break;
                }
            }
            if (start)
                break;
        }
        if (start)
            break;
    }
    if (!start)
    {
        ROS_ERROR("[GaussianProcessNode::%s]\tNo starting point found. Relax grid pass.",__func__);
        return;
    }
    s_oct = boost::make_shared<pcl::octree::OctreePointCloud<pcl::PointXYZ>>(leaf_size);
    real_explicit_ptr = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    oct_cent = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    s_oct->setInputCloud(oct_cent);
    marchingCubes(*start, leaf_size, leaf_pass); //process will block here until everything is done
    //remove duplicate points
    pcl::VoxelGrid<pcl::PointXYZI> vg;
    vg.setInputCloud(real_explicit_ptr);
    vg.setLeafSize(leaf_pass, leaf_pass, leaf_pass);
    pcl::PointCloud<pcl::PointXYZI> tmp;
    vg.filter(tmp);
    pcl::copyPointCloud(tmp, *real_explicit_ptr);
    ROS_INFO("[GaussianProcessNode::%s]\tFound %ld points approximately on GP surface...",__func__,
            real_explicit_ptr->size());
    //determine min e max variance
    if (first_time){
        //only do this if it is the first time
        for (const auto& pt: real_explicit_ptr->points)
        {
            if (pt.intensity <= min_v)
                min_v = pt.intensity;
            if (pt.intensity >= max_v)
                max_v = pt.intensity;
        }
    }
    const double mid_v = ( min_v + max_v ) * 0.5;
    //create the markers
    for (size_t i=0; i< real_explicit_ptr->size(); ++i)
    {
        geometry_msgs::Point pt;
        std_msgs::ColorRGBA cl;
        pcl::PointXYZI pt_pcl = real_explicit_ptr->points[i];
        pt.x = pt_pcl.x;
        pt.y = pt_pcl.y;
        pt.z = pt_pcl.z;
        cl.a = 1.0;
        cl.b = 0.0;
        cl.r = (pt_pcl.intensity<mid_v) ? 1/(mid_v - min_v) * (pt_pcl.intensity - min_v) : 1.0;
        cl.g = (pt_pcl.intensity>mid_v) ? -1/(max_v - mid_v) * (pt_pcl.intensity - mid_v) + 1 : 1.0;
        samples.points.push_back(pt);
        samples.colors.push_back(cl);
    }
    markers->markers.push_back(samples);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(end_time - begin_time).count();
    ROS_INFO("[GaussianProcessNode::%s]\tTotal time consumed: %d minutes.", __func__, elapsed );
}

//Doesnt fully work!! looks like diagonal adjacency is also needed,
//It does work however for small leaf sizes...
void
GaussianProcessNode::marchingCubes(pcl::PointXYZ start, const float leaf, const float pass)
{
    {//protected section, mark this cube as already explored
        std::lock_guard<std::mutex> lock (mtx_samp);
        s_oct->addPointToCloud(start, oct_cent);
    }//end of protected section
    const size_t steps = std::round(leaf/pass);
    //sample the cube
    std::array<bool,6> where{{false,false,false,false,false,false}};
    for (size_t i = 0; i<= steps; ++i)
        for (size_t j = 0; j<= steps; ++j)
            for (size_t k = 0; k<= steps; ++k)
            {
                gp_regression::Data::Ptr qq = std::make_shared<gp_regression::Data>();
                double x (start.x -leaf/2 + i*pass );
                double y (start.y -leaf/2 + j*pass );
                double z (start.z -leaf/2 + k*pass );
                qq->coord_x.push_back(x);
                qq->coord_y.push_back(y);
                qq->coord_z.push_back(z);
                std::vector<double> ff,vv;
                reg_->evaluate(obj_gp, qq, ff, vv);
                if (std::abs(ff.at(0)) <= 0.01) {
                    pcl::PointXYZI pt;
                    pt.x = x;
                    pt.y = y;
                    pt.z = z;
                    pt.intensity = vv.at(0);
                    {//protected section
                        std::lock_guard<std::mutex> lock (mtx_samp);
                        real_explicit_ptr->push_back(pt);
                    }//end of protected section
                    //decide where to explore
                    if (i == 0 && !where.at(0)) //-x
                        where.at(0) = true;
                    if (i == steps && !where.at(1)) //+x
                        where.at(1) = true;
                    if (j ==0 && !where.at(2)) //-y
                        where.at(2) = true;
                    if (j == steps && !where.at(3)) //+y
                        where.at(3) = true;
                    if (k == 0 && !where.at(4)) //-z
                        where.at(4) = true;
                    if (k == steps && !where.at(5)) //+z
                        where.at(5) = true;
                }
            }

    //Expand in all directions found before
    std::vector<std::thread> threads;
    pcl::PointXYZ pt;
    bool exists;
    for (size_t i=0; i<where.size(); ++i)
    {
        pt = start;
        if (where[i]){
            if (i==0) //-x
                pt.x -= leaf;
            if (i==1) //+x
                pt.x += leaf;
            if (i==2) //-y
                pt.y -= leaf;
            if (i==3) //+y
                pt.y += leaf;
            if (i==4) //-z
                pt.z -= leaf;
            if (i==5) //+z
                pt.z += leaf;
            { //check if we didn't already explore that cube
                std::lock_guard<std::mutex> lock (mtx_samp);
                exists = s_oct->isVoxelOccupiedAtPoint(pt);
            }
            if (!exists)
                threads.emplace_back(&GaussianProcessNode::marchingCubes, this, pt, leaf, pass);
        }
    }
    for (auto& t: threads)
        t.join();
}

void
GaussianProcessNode::computeOctomap()
{
    octomap = std::make_shared<octomap::OcTree>(0.01);
    pcl::PointCloud<pcl::PointXYZI>::Ptr real_explicit =
        boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    pcl::PointCloud<pcl::PointXYZI> real_ds;
    reMeanAndDenormalizeData(real_explicit_ptr, *real_explicit);
    pcl::VoxelGrid<pcl::PointXYZI> vg;
    vg.setInputCloud(real_explicit);
    vg.setLeafSize(0.005, 0.005, 0.005);
    vg.filter(real_ds);
    //store it in a Octomap pointcloud in case is needed in the future
    // octomap::Pointcloud ocl;
    // octomap::pointCloudPCLToOctomap(real_ds, ocl); //Too bad this does only exists in more recent version of octomap!
    // ocl.reserve(real_ds.points.size());
    // for (auto it = real_ds.begin(); it!= real_ds.end(); ++it)
    // {
    //     if (!std::isnan (it->x) && !std::isnan (it->y) && !std::isnan (it->z))
    //         ocl.push_back(it->x, it->y, it->z);
    // }
    //fill the octomap
    for (const auto& pt: real_ds.points)
    {
        float hit = -8/(10*(max_v-min_v))*(pt.intensity- min_v) + 9/10;
        octomap::point3d opt (pt.x, pt.y, pt.z);
        octomap->updateNode(opt, std::log10(hit/(1-hit)), false);
    }

    //TODO what about free(unoccupied) voxels, can they be
    //artificially generated ? or raycasted from pointcloud ?
}

void
GaussianProcessNode::publishOctomap() const
{
    if (!octomap)
        return;
    octomap_msgs::Octomap map;
    map.header.frame_id = proc_frame;
    map.header.stamp = ros::Time::now();
    if (octomap_msgs::fullMapToMsg(*octomap,map))
        pub_octomap.publish(map);
    else
        ROS_ERROR("[GaussianProcessNode::%s]\tError in serializing octomap to publish",__func__);
}

void
GaussianProcessNode::synthTouch(const Eigen::Vector3d &point, const Eigen::Vector3d &normal)
{
    //normal is facing outside
    Eigen::Vector3d p (point +normal*0.1);
    std::vector<int> k_id;
    std::vector<float> k_dist;
    //move along direction
    size_t max_steps(10);
    for (size_t i=0; i<max_steps; ++i)
    {
        pcl::PointXYZ pt;
        pt.x = p[0];
        pt.y = p[1];
        pt.z = p[2];
        if (kd_full.radiusSearch(pt, 0.05, k_id, k_dist, 1) > 0){
            //we intersected the object, aka touch

            break;
        }
        else{ //no touch, external point

        }
        p -= (normal*0.05);
    }
}

///// MAIN ////////////////////////////////////////////////////////////////////

int main (int argc, char *argv[])
{
    ros::init(argc, argv, "gaussian_process");
    GaussianProcessNode node;
    ros::Rate rate(10); //try to go at 10hz
    while (node.nh.ok())
    {
        //gogogo!
        ros::spinOnce();
        node.Publish();
        // node.checkExploration();
        rate.sleep();
    }
    //someone killed us :(
    return 0;
}
