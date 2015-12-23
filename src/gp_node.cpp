#include <gp_node.h>
#include <algorithm> //for std::max_element
// #include <chrono> //for time measurements
#include <node_utils.hpp>

using namespace gp_regression;
/* PLEASE LOOK at  TODOs by searching "TODO" to have an idea  of * what is still
missing or is improvable! */
GaussianProcessNode::GaussianProcessNode (): nh(ros::NodeHandle("gaussian_process")), start(false),
    object_ptr(boost::make_shared<PtC>()), hand_ptr(boost::make_shared<PtC>()),
    model_ptr(boost::make_shared<PtC>())
{
    srv_start = nh.advertiseService("start_process", &GaussianProcessNode::cb_start, this);
    // srv_sample = nh.advertiseService("sample_process", &GaussianProcessNode::cb_sample, this); not sure why this is needed
    pub_model = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB>> ("estimated_model", 1);
    pub_markers = nh.advertise<visualization_msgs::MarkerArray> ("atlas", 1);
    sub_points = nh.subscribe(nh.resolveName("/clicked_point"),1, &GaussianProcessNode::cb_point, this);
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

//callback to start process service, executes when service is called
bool GaussianProcessNode::cb_start(gp_regression::StartProcess::Request& req, gp_regression::StartProcess::Response& res)
{
    if(req.cloud_dir.empty()){
        //Request was empty, means we have to call pacman vision service to
        //get a cloud.
        std::string service_name = nh.resolveName("/pacman_vision/listener/get_cloud_in_hand");
        pacman_vision_comm::get_cloud_in_hand service;
        service.request.save = "false";
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
        //User told us to load a clouds from a dir on disk instead.
        if (pcl::io::loadPCDFile((req.cloud_dir+"/obj.pcd"), *object_ptr) != 0){
            ROS_ERROR("[GaussianProcessNode::%s]\tError loading cloud from %s",__func__,(req.cloud_dir+"obj.pcd").c_str());
            return (false);
        }
        if (pcl::io::loadPCDFile((req.cloud_dir+"/hand.pcd"), *hand_ptr) != 0)
            ROS_WARN("[GaussianProcessNode::%s]\tError loading cloud from %s, ignoring hand",__func__,(req.cloud_dir + "/hand.pcd").c_str());
        //We  need  to  fill  point  cloud header  or  ROS  will  complain  when
        //republishing this cloud. Let's assume it was published by asus kinect.
        //I think it's just need the frame id
        object_ptr->header.frame_id="/camera_rgb_optical_frame";
        hand_ptr->header.frame_id="/camera_rgb_optical_frame";
        model_ptr->header.frame_id="/camera_rgb_optical_frame";
    }
    if (computeGP())
        if (computeAtlas())
            return true;
    return false;
}

// TODO: Convert this callback, if  needed, to accept probe points and not
// rviz clicked points, as it is now. (tabjones on Wednesday 18/11/2015)
// Callback for rviz clicked point to simulate probe
void GaussianProcessNode::cb_point(const geometry_msgs::PointStamped::ConstPtr &msg)
{
    pcl::PointXYZRGB pt;
    //get the clicked point
    pt.x = msg->point.x;
    pt.y = msg->point.y;
    pt.z = msg->point.z;
    //color it blue
    colorIt(0,0,255, pt);
    model_ptr->push_back(pt);
    object_ptr->push_back(pt);
    Vec3 p(pt.x, pt.y, pt.z);
    //Predispone the sequence to host multiple points, not just one, for the future.
    Vec3Seq points;
    points.push_back(p);
    //update the model
    this->update(points);
}
//gp computation
bool GaussianProcessNode::computeGP()
{
    // auto begin_time = std::chrono::high_resolution_clock::now();
    if(!object_ptr){
        //This  should never  happen  if compute  is  called from  start_process
        //service callback, however it does not hurt to add this extra check!
        ROS_ERROR("[GaussianProcessNode::%s]\tObject cloud pointer is empty. Aborting...",__func__);
        start = false;
        return false;
    }
    if (object_ptr->empty()){
        ROS_ERROR("[GaussianProcessNode::%s]\tObject point cloud is empty. Aborting...",__func__);
        start = false;
        return false;
    }
    if (!model_ptr->empty())
        //clear previous computation if exists
        model_ptr->clear();

    Vec3Seq cloud;
    Vec targets;

    //Add object points as label 0
    for(const auto pt : object_ptr->points)
    {
        Vec3 point(pt.x ,pt.y ,pt.z);
        targets.push_back(0);
        cloud.push_back(point);
    }
    //add object to published model
    *model_ptr += *object_ptr;
    //color object blue
    colorThem(0,0,255, model_ptr);

    // TODO: We would probalby need to add  hand points to the GP in the future.
    // Set them perhaps with target 1 or 0.5 (tabjones on Wednesday 16/12/2015)
    //add hand points to model as cyan
    colorThem(0,255,255, hand_ptr);
    *model_ptr += *hand_ptr;

    //Now add centroid as label -1
    Eigen::Vector4f centroid;
    if(pcl::compute3DCentroid<pcl::PointXYZRGB>(*object_ptr, centroid) == 0){
        ROS_ERROR("[GaussianProcessNode::%s]\tFailed to compute object centroid. Aborting...",__func__);
        start = false;
        return false;
    }
    Vec3 cent( centroid[0], centroid[1], centroid[2]);
    cloud.push_back(cent);
    targets.push_back(-1);
    pcl::PointXYZRGB cen;
    cen.x = centroid[0];
    cen.y = centroid[1];
    cen.z = centroid[2];
    //add internal point to model as yellow
    colorIt(255,255,0, cen);
    model_ptr->push_back(cen);

    //Add points in a sphere around centroid as label 1
    //sphere bounds computation
    const int ang_div = 8; //divide 360° in 8 pieces, i.e. steps of 45°
    const int lin_div = 6; //divide diameter into 6 pieces
    //this makes 8*6 = 48 points.
    const double radius = 0.4; //40cm
    const double ang_step = M_PI * 2 / ang_div; //steps of 45°
    const double lin_step = 2 * radius / lin_div;
    //8 steps for diameter times 6 for angle, make  points on the sphere surface
    int j(0);
    for (double lin=-radius+lin_step/2; lin< radius; lin+=lin_step)
        for (double ang=0; ang < 2*M_PI; ang+=ang_step, ++j)
        {
            double x = sqrt(radius*radius - lin*lin) * cos(ang) + centroid[0];
            double y = sqrt(radius*radius - lin*lin) * sin(ang) + centroid[1];
            double z = lin + centroid[2]; //add centroid to translate there
            Vec3 sph (x,y,z);
            cloud.push_back(sph);
            targets.push_back(1);
            //add sphere points to model as red
            pcl::PointXYZRGB sp;
            sp.x = x;
            sp.y = y;
            sp.z = z;
            colorIt(255,0,0, sp);
            model_ptr->push_back(sp);
        }
    /*****  Create the gp model  *********************************************/
    //create the model to be stored in class
    data = boost::make_shared<gp::SampleSet>(cloud,targets);
    LaplaceRegressor::Desc ld;
    ld.noise = 0.0;
    gp = ld.create();
    gp->set(data);
    start = true;
    // auto end_time = std::chrono::high_resolution_clock::now();
    // auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - begin_time).count();
    // ROS_INFO("[GaussianProcessNode::%s]\tRegressor and Model created. Total time consumed: %ld nanoseconds.",__func__, elapsed);
    ROS_INFO("[GaussianProcessNode::%s]\tRegressor and Model created.",__func__);
    return true;
}
bool GaussianProcessNode::computeAtlas()
{
    //make sure we have a model and an object, we should have if start was called
    if (!object_ptr || object_ptr->empty()){
        ROS_ERROR("[GaussianProcessNode::%s]\tNo object initialized, call start service.",__func__);
        return false;
    }
    if (!gp){
        ROS_ERROR("[GaussianProcessNode::%s]\tNo GP model initialized, call start service.",__func__);
        return false;
    }
    //right now just create 10 discs at random all depth 0.
    uint8_t N = 10;
    int num_points = object_ptr->size();
    //init atlas
    atlas = std::make_shared<Atlas>();
    for (uint8_t i=0; i<N; ++i)
    {
        int r_id;
        //get a random index
        r_id = getRandIn(0, num_points -1);
        Chart chart;
        chart.center.set(object_ptr->points[r_id].x,
                         object_ptr->points[r_id].y,
                         object_ptr->points[r_id].z);
        Real fx, vx;
        //evaluate N looks wrong  TODO FIX
        gp->evaluate(chart.center, fx, vx, chart.N, chart.Tx, chart.Ty);
        //find a basis with N as new Z axis, dont use tx,ty
        Eigen::Vector3d X,Y;
        Eigen::Vector3d kinX(Eigen::Vector3d::UnitX());
        chart.N.normalize();
        //find a new x as close as possible to x kinect but orthonormal to N
        X = kinX - (chart.N*(chart.N.dot(kinX)));
        X.normalize();
        Y = chart.N.cross(X);
        Y.normalize();
        chart.Tx = X;
        chart.Ty = Y;
        //define a radius of the chart just take 3cm for now
        chart.radius = 0.03f;
        chart.id = r_id; //lets have the id = pointcloud id
        chart.parent = 0; //doesnt have a parent since its root
        atlas->insert(std::pair<uint8_t, Chart>(0,chart));
    }
    createAtlasMarkers();
    return true;
}
//update gaussian model with new points from probe
void GaussianProcessNode::update(Vec3Seq &points)
{
    Vec t;
    t.resize(points.size());
    for (auto &x : t)
        x=0;
    gp->add_patterns(points,t);
    //TODO temp we dont have an updated atlas, just recreate it from scratch
    computeAtlas();
}
//Republish cloud method
void GaussianProcessNode::publishCloudModel () const
{
    //These checks are  to make sure we are not  publishing empty cloud,
    //we have a  gaussian process computed and  there's actually someone
    //who listens to us
    if (start && object_ptr)
        if(!model_ptr->empty() && pub_model.getNumSubscribers()>0)
            pub_model.publish(*model_ptr);
}

void GaussianProcessNode::createAtlasMarkers()
{
    //reset old markers
    markers = boost::make_shared<visualization_msgs::MarkerArray>();
    if (!atlas){
        ROS_WARN("[GaussianProcessNode::%s]\tNo Atlas created, not computing any marker.",__func__);
        return ;
    }
    //fake deterministic sampling
    visualization_msgs::Marker sample;
    sample.header.frame_id = object_ptr->header.frame_id;
    sample.header.stamp = ros::Time();
    sample.lifetime = ros::Duration(1);
    sample.ns = "samples";
    sample.id = 0;
    sample.type = visualization_msgs::Marker::POINTS;
    sample.action = visualization_msgs::Marker::ADD;
    sample.scale.x = 0.001;
    sample.scale.y = 0.001;
    sample.color.a = 0.3;
    sample.color.r = 0.0;
    sample.color.b = 0.0;
    sample.color.g = 1.0;
    pcl::PointXYZRGB min, max;
    pcl::getMinMax3D(*object_ptr, min, max);
    float xm,xM,ym,yM,zm,zM;
    xm = (-2*max.x + 4*min.x)*0.5;
    ym = (-2*max.y + 4*min.y)*0.5;
    zm = (-2*max.z + 4*min.z)*0.5;
    xM = (4*max.x -2*min.x)*0.5;
    yM = (4*max.y -2*min.y)*0.5;
    zM = (4*max.z -2*min.z)*0.5;
    for (float x = xm; x<= xM; x += 0.02)
        for (float y = ym; y<= yM; y += 0.02)
            for (float z = zm; z<= zM*2; z += 0.02)
            {
                Vec3 q(x,y,z);
                const double qf = gp->f(q);
                //test if sample was classified as belonging to obj surface
                if (qf <= 0.001 && qf >= -0.001){
                    //We can  add this sample to visualization
                    geometry_msgs::Point pt;
                    pt.x = x;
                    pt.y = y;
                    pt.z = z;
                    sample.points.push_back(pt);
                }
            }
    markers->markers.push_back(sample);
    //Now show the Atlas
    // for each atlas (we have 1 now TODO loop)
    int a (0); //atlas index
    {
        //for each chart
        for(auto c = atlas->begin(); c != atlas->end(); ++c)
        {
            visualization_msgs::Marker disc;
            disc.header.frame_id = object_ptr->header.frame_id;
            disc.header.stamp = ros::Time();
            disc.lifetime = ros::Duration(1);
            std::string ns("A" + std::to_string(a) + "_D" + std::to_string(c->first));
            disc.ns = ns;
            disc.id = c->second.id;
            disc.type = visualization_msgs::Marker::CYLINDER;
            disc.action = visualization_msgs::Marker::ADD;
            // disc.points.push_back(center);
            disc.scale.x = c->second.radius;
            disc.scale.y = c->second.radius;
            disc.scale.z = 0.001;
            disc.color.a = 0.3;
            disc.color.r = 1.0;
            disc.color.b = 0.8;
            disc.color.g = 0.0;
            Eigen::Matrix3d rot;
            rot.col(0) = c->second.Tx;
            rot.col(1) = c->second.Ty;
            rot.col(2) = c->second.N;
            Eigen::Quaterniond q(rot);
            q.normalize();
            disc.pose.orientation.x = q.x();
            disc.pose.orientation.y = q.y();
            disc.pose.orientation.z = q.z();
            disc.pose.orientation.w = q.w();
            disc.pose.position.x = c->second.center[0];
            disc.pose.position.y = c->second.center[1];
            disc.pose.position.z = c->second.center[2];
            markers->markers.push_back(disc);
            visualization_msgs::Marker aX,aY,aZ;
            aX.header.frame_id = object_ptr->header.frame_id;
            aY.header.frame_id = object_ptr->header.frame_id;
            aZ.header.frame_id = object_ptr->header.frame_id;
            aX.header.stamp = ros::Time();
            aY.header.stamp = ros::Time();
            aZ.header.stamp = ros::Time();
            aX.lifetime = ros::Duration(1);
            aY.lifetime = ros::Duration(1);
            aZ.lifetime = ros::Duration(1);
            std::string nsa("NTxTy" + std::to_string(c->second.id));
            aX.ns = aY.ns = aZ.ns = nsa;
            aX.id = 1;
            aY.id = 2;
            aZ.id = 3;
            aX.type = aY.type = aZ.type = visualization_msgs::Marker::ARROW;
            aX.action = aY.action = aZ.action = visualization_msgs::Marker::ADD;
            geometry_msgs::Point end;
            geometry_msgs::Point start;
            start.x = c->second.center[0];
            start.y = c->second.center[1];
            start.z = c->second.center[2];
            aX.points.push_back(start);
            aY.points.push_back(start);
            aZ.points.push_back(start);
            end.x = start.x + c->second.Tx[0]/100; //since Tx is normalized this makes arrow 1cm long
            end.y = start.y + c->second.Tx[1]/100;
            end.z = start.z + c->second.Tx[2]/100;
            aX.points.push_back(end);
            end.x = start.x + c->second.Ty[0]/100;
            end.y = start.y + c->second.Ty[1]/100;
            end.z = start.z + c->second.Ty[2]/100;
            aY.points.push_back(end);
            end.x = start.x + c->second.N[0]/100;
            end.y = start.y + c->second.N[1]/100;
            end.z = start.z + c->second.N[2]/100;
            aZ.points.push_back(end);
            // std::cout<<"N "<<c->second.N<<std::endl;
            aX.scale.x = aY.scale.x = aZ.scale.x = 0.0002;
            aX.scale.y = aY.scale.y = aZ.scale.y = 0.0008;
            aX.scale.z = aY.scale.z = aZ.scale.z = 0.0008;
            aX.color.a = aY.color.a = aZ.color.a = 0.5;
            aX.color.b = aX.color.g = aY.color.r = aY.color.b = aZ.color.r = aZ.color.g = 0.0;
            aX.color.r = aY.color.g = aZ.color.b = 1.0;
            markers->markers.push_back(aX);
            markers->markers.push_back(aY);
            markers->markers.push_back(aZ);
        }
    }
}

//Publish sample (remove for now, we are publishing atlas markers)
void GaussianProcessNode::publishAtlas () const
{
    if (markers)
        if(pub_markers.getNumSubscribers() > 0)
            pub_markers.publish(*markers);
}
//test for occlusion of samples (now unused)
//return:
//  0 -> not visible
//  1 -> visible
//  -1 -> error
// int GaussianProcessNode::isSampleVisible(const pcl::PointXYZRGB sample, const float min_z) const
// {
//     if(!viewpoint_tree){
//         ROS_ERROR("[GaussianProcessNode::%s]\tObject Viewpoint KdTree is not initialized. Aborting...",__func__);
//         //should never happen if called from sampleAndPublish
//         return (-1);
//     }
//     Eigen::Vector3f camera(0,0,0);
//     Eigen::Vector3f start_point(sample.x, sample.y, sample.z);
//     Eigen::Vector3f direction = camera - start_point;
//     const float norm = direction.norm();
//     direction.normalize();
//     const float step_size = 0.01f;
//     const int nsteps = std::max(1, static_cast<int>(norm/step_size));
//     std::vector<int> k_id;
//     std::vector<float> k_dist;
//     //move along direction
//     Eigen::Vector3f p(start_point[0], start_point[1], start_point[2]);
//     for (size_t i = 0; i<nsteps; ++i)
//     {
//         if (p[2] <= min_z)
//             //don't reach  the sensor, if  we are  outside sample region  we can
//             //stop testing.
//             break;
//         pcl::PointXYZRGB pt;
//         pt.x = p[0];
//         pt.y = p[1];
//         pt.z = p[2];
//         // TODO: This search radius is  hardcoded now, should be adapted somehow
//         // on point density (tabjones on Friday 13/11/2015)
//         if (viewpoint_tree->radiusSearch(pt, 0.005, k_id, k_dist, 1) > 0)
//             //we intersected an object point, this sample cant reach the camera
//             return(0);
//         p += (direction * step_size);
//     }
//     //we didn't intersect anything, this sample is not occluded
//     return(1);
// }
