#include <algorithm> //for std::max_element
#include <chrono> //for time measurements

#include <node_utils.hpp>
#include <gp_node.h>

using namespace gp_regression;

/* PLEASE LOOK at  TODOs by searching "TODO" to have an idea  of * what is still
missing or is improvable! */
GaussianProcessNode::GaussianProcessNode (): nh(ros::NodeHandle("gaussian_process")), start(false),
    object_ptr(boost::make_shared<PtC>()), hand_ptr(boost::make_shared<PtC>()), data_ptr_(boost::make_shared<PtC>()),
    model_ptr(boost::make_shared<PtC>()), fake_sampling(true), exploration_started(false),
    out_sphere_rad(1.8)
{
    mtx_marks = std::make_shared<std::mutex>();
    srv_start = nh.advertiseService("start_process", &GaussianProcessNode::cb_start, this);
    srv_get_next_best_path_ = nh.advertiseService("get_next_best_path", &GaussianProcessNode::cb_get_next_best_path, this);
    pub_model = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB>> ("estimated_model", 1);
    pub_markers = nh.advertise<visualization_msgs::MarkerArray> ("atlas", 1);
    sub_update_ = nh.subscribe(nh.resolveName("/path_log"),1, &GaussianProcessNode::cb_update, this);

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
    //publish markers
    publishAtlas();
}

// Publish cloud method
void GaussianProcessNode::publishCloudModel () const
{
    // These checks are  to make sure we are not publishing empty cloud,
    // we have a  gaussian process computed and there's actually someone
    // who listens to us
    if (start && model_ptr)
        if(!model_ptr->empty() && pub_model.getNumSubscribers()>0)
            pub_model.publish(*model_ptr);
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

//callback to start process service, executes when service is called
bool GaussianProcessNode::cb_get_next_best_path(gp_regression::GetNextBestPath::Request& req, gp_regression::GetNextBestPath::Response& res)
{
     if(solution.empty() || exploration_started) {
        ROS_WARN("[GaussianProcessNode::%s]\tSorry, the exploration is WIP",__func__);
        return false;
     }
     else {
         std_msgs::Header solution_header;
         solution_header.stamp = ros::Time::now();
         solution_header.frame_id = object_ptr->header.frame_id;
         std::cout << "solution_header.frame_id: " << solution_header.frame_id << std::endl;

         gp_regression::Path next_best_path;

         // ToDO: solutionToPath(solution, path) function
         gp_atlas_rrt::Chart solution_chart = atlas->getNode(solution.front());
         Eigen::Vector3d point_eigen = solution_chart.getCenter();
         Eigen::Vector3d normal_eigen = solution_chart.getNormal();
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
         next_best_path.header = solution_header;

         res.next_best_path = next_best_path;
         return true;
     }
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
    reg_.reset();
    obj_gp.reset();
    my_kernel.reset();
    atlas.reset();
    explorer.reset();
    solution.clear();
    markers.reset();
    //////
    if(req.cloud_dir.empty()){
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
        pcl::fromROSMsg (service.response.obj, *object_ptr);
        pcl::fromROSMsg (service.response.hand, *hand_ptr);
    }
    else{
        if(req.cloud_dir.compare("sphere") == 0 || req.cloud_dir.compare("half_sphere") == 0){
            object_ptr = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB> >();
            const int ang_div = 24;
            const int lin_div = 20;
            const double radius = 0.06;
            const double ang_step = M_PI * 2 / ang_div;
            const double lin_step = 2 * radius / lin_div;
            double end_lin = radius;
            if (req.cloud_dir.compare("half_sphere")==0)
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
            object_ptr->header.frame_id="/camera_rgb_optical_frame";
            model_ptr->header.frame_id="/camera_rgb_optical_frame";
        }
        else{
            // User told us to load a clouds from a dir on disk instead.
            if (pcl::io::loadPCDFile((req.cloud_dir+"/obj.pcd"), *object_ptr) != 0){
                ROS_ERROR("[GaussianProcessNode::%s]\tError loading cloud from %s",__func__,(req.cloud_dir+"obj.pcd").c_str());
                return (false);
            }
            if (pcl::io::loadPCDFile((req.cloud_dir+"/hand.pcd"), *hand_ptr) != 0)
                ROS_WARN("[GaussianProcessNode::%s]\tError loading cloud from %s, ignoring hand",__func__,(req.cloud_dir + "/hand.pcd").c_str());
            // We  need  to  fill  point  cloud header  or  ROS  will  complain  when
            // republishing this cloud. Let's assume it was published by asus kinect.
            // I think it's just need the frame id
            object_ptr->header.frame_id="/camera_rgb_optical_frame";
            hand_ptr->header.frame_id="/camera_rgb_optical_frame";
            model_ptr->header.frame_id="/camera_rgb_optical_frame";
        }
    }

    ros::Rate rate(10); //try to go at 10hz, as in the node
    deMeanAndNormalizeData( object_ptr, data_ptr_ );
    if (computeGP())
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
            return true;
        }
    return false;
}

// TODO: Convert this callback, if  needed, to accept probe points and not
// rviz clicked points, as it is now. (tabjones on Wednesday 18/11/2015)
// Callback for rviz clicked point to simulate probe
// The only different is the topic name, that can be easily changed with remapping
// Though I think to be general perhaps we need to create our msg of an array of
// PointStamped, that is, a point trajectory
void GaussianProcessNode::cb_update(const geometry_msgs::PointStamped::ConstPtr &msg)
{
    pcl::PointXYZRGB pt;
    pt.x = msg->point.x;
    pt.y = msg->point.y;
    pt.z = msg->point.z;
    // new data in cyan
    colorIt(0,255,255, pt);
    model_ptr->push_back(pt);
    object_ptr->push_back(pt);

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

    //update the model, do not compute normals in the model points
    auto begin_time = std::chrono::high_resolution_clock::now();
    reg_->update<false>(fresh_data, obj_gp);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - begin_time).count();
    ROS_INFO("[GaussianProcessNode::%s]\t Model updated in: %ld nanoseconds.", __func__, elapsed );*/

    deMeanAndNormalizeData( object_ptr, data_ptr_ );
    computeGP();
    startExploration();
    return;
}

bool GaussianProcessNode::computeGP()
{
    auto begin_time = std::chrono::high_resolution_clock::now();
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
    if (!model_ptr->empty())
        model_ptr->clear();

    /*****  Prepare the training data  *********************************************/
    gp_regression::Data::Ptr cloud_gp = std::make_shared<gp_regression::Data>();
    double sigma2 = 1e-1;

    //      Surface Points
    // add object points with label 0
    for(const auto& pt : data_ptr_->points) {
        cloud_gp->coord_x.push_back(pt.x);
        cloud_gp->coord_y.push_back(pt.y);
        cloud_gp->coord_z.push_back(pt.z);
        cloud_gp->label.push_back(0);
        cloud_gp->sigma2.push_back(sigma2);
    }
    // add object points to rviz in blue
    *model_ptr += *data_ptr_;
    colorThem(0,0,255, model_ptr);

    //      Internal Points
    // Now add centroid as label -1
    cloud_gp->coord_x.push_back(0);
    cloud_gp->coord_y.push_back(0);
    cloud_gp->coord_z.push_back(0);
    cloud_gp->label.push_back(-1.0);
    cloud_gp->sigma2.push_back(sigma2);
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

            cloud_gp->coord_x.push_back(x);
            cloud_gp->coord_y.push_back(y);
            cloud_gp->coord_z.push_back(z);
            cloud_gp->label.push_back(1.0);
            cloud_gp->sigma2.push_back(sigma2);

            // add sphere points to rviz in purple
            pcl::PointXYZRGB sp;
            sp.x = x;
            sp.y = y;
            sp.z = z;
            colorIt(255,0,255, sp);
            model_ptr->push_back(sp);
        }
    }

    // TODO: We would probablyy need to add  hand points to the GP in the future.
    // Set them perhaps with target 1 or 0.5 (tabjones on Wednesday 16/12/2015)
    // add hand points to model as slightly different red, if available
    /*if(hand_ptr){
        colorThem(255,125,0, hand_ptr);
        *model_ptr += *hand_ptr;
        for(const auto pt : hand_ptr->points) {
             cloud_gp->coord_x.push_back(pt.x);
             cloud_gp->coord_y.push_back(pt.y);
             cloud_gp->coord_z.push_back(pt.z);
             cloud_gp->label.push_back(0.1);
         }
    }*/
    // I tried, but at least with the offline example it was
    // giving an error. I don't have the force now with me, Luke,
    // maybe another day, but very likely that they are far too many.
    // Thus downsampling the hand cloud could be the answer.
    // (carlosjoserg 7/02/2016)
    //
    // Ok! who cares about the hand! (tabjones 09/02/2016)

    /*****  Create the gp model  *********************************************/
    //create the model to be stored in class
    if (cloud_gp->coord_x.size() != cloud_gp->label.size()){
        ROS_ERROR("[GaussianProcessNode::%s]\tTargets Points size mismatch, something went wrong. Aborting...",__func__);
        start = false;
        return false;
    }

    obj_gp = std::make_shared<gp_regression::Model>();
    reg_ = std::make_shared<gp_regression::ThinPlateRegressor>();
    my_kernel = std::make_shared<gp_regression::ThinPlate>(out_sphere_rad * 2);
    reg_->setCovFunction(my_kernel);
    const bool withoutNormals = false;
    reg_->create<withoutNormals>(cloud_gp, obj_gp);
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

    //initialize objects involved
    markers = boost::make_shared<visualization_msgs::MarkerArray>();
    //perform fake sampling
    fakeDeterministicSampling(1.1, 0.1);

    //create the atlas
    atlas = std::make_shared<gp_atlas_rrt::AtlasCollision>(obj_gp, reg_);
    //termination condition
    atlas->setVarianceTolGoal( 0.4 );
    //factor to control disc radius
    atlas->setVarRadiusFactor( 0.65 );
    //atlas is ready

    //setup explorer
    explorer = std::make_shared<gp_atlas_rrt::ExplorerMultiBranch>(nh, "explorer");
    explorer->setMarkers(markers, mtx_marks);
    explorer->setAtlas(atlas);
    explorer->setMaxNodes(50);
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

        //TODO actually do something with the solution !
        // new service in tha house for that
        // remember everything should have a ROS API to be called from the state machine
    }
}

// for visualization purposes
void GaussianProcessNode::fakeDeterministicSampling(const double scale, const double pass)
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
    samples.scale.x = 0.005;
    samples.scale.y = 0.005;
    samples.scale.z = 0.005;

    gp_regression::Data::Ptr ss = std::make_shared<gp_regression::Data>();
    std::vector<double> ssvv;

    double min_v (100.0);
    double max_v (0.0);
    size_t count(0);
    const auto total = std::lround( std::pow((2*scale+1)/pass, 3) );
    ROS_INFO("[GaussianProcessNode::%s]\tSampling %ld grid points on GP ...",__func__, total);
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
                    if (vv.at(0) <= min_v)
                        min_v = vv.at(0);
                    if (vv.at(0) >= max_v)
                        max_v = vv.at(0);
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
        pt.x = ss->coord_x.at(i);
        pt.y = ss->coord_y.at(i);
        pt.z = ss->coord_z.at(i);
        cl.a = 1.0;
        cl.b = 0.0;
        cl.r = (ssvv.at(i)<mid_v) ? 1/(mid_v - min_v) * (ssvv.at(i) - min_v) : 1.0;
        cl.g = (ssvv.at(i)>mid_v) ? -1/(max_v - mid_v) * (ssvv.at(i) - mid_v) + 1 : 1.0;
        samples.points.push_back(pt);
        samples.colors.push_back(cl);
    }
    markers->markers.push_back(samples);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(end_time - begin_time).count();
    ROS_INFO("[GaussianProcessNode::%s]\tTotal time consumed: %d minutes.", __func__, elapsed );
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
        node.checkExploration();
        rate.sleep();
    }
    //someone killed us :(
    return 0;
}
