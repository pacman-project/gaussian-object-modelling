#include <gp_node.h>
#include <algorithm> //for std::max_element

using namespace gp;

/* PLEASE LOOK at  TODOs by searching "TODO" to have an idea  of * what is still
missing or is improvable! */
GaussianProcessNode::GaussianProcessNode (): nh(ros::NodeHandle("gaussian_process")), start(false),
    need_update(false), how_many_discoveries(1)
{
    srv_start = nh.advertiseService("start_process", &GaussianProcessNode::cb_start, this);
    srv_sample = nh.advertiseService("sample_process", &GaussianProcessNode::cb_sample, this);
    pub_model = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB>> ("estimated_model", 1);
    sub_points = nh.subscribe(nh.resolveName("/clicked_point"),1, &GaussianProcessNode::cb_point, this);
    pub_point = nh.advertise<gp_regression::SampleToExplore> ("sample_to_explore", 0, true);
    pub_point_marker = nh.advertise<geometry_msgs::PointStamped> ("point_to_explore", 0, true); // TEMP, should be a trajectory, curve, pose
    pub_direction_marker = nh.advertise<geometry_msgs::WrenchStamped> ("direction_to_explore", 0, true); // TEMP, should be a trajectory, curve, pose
    object_ptr.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    hand_ptr.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    model_ptr.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
}

//Sample some points and query the gp, build a point cloud reconstructed
//model and publish it.
void GaussianProcessNode::sampleAndPublish ()
{
    if (!start){
        ROS_WARN_DELAYED_THROTTLE(80,"[GaussianProcessNode::%s]\tCall start_process service to begin. Not doing anything at the moment...",__func__);
        return;
    }
    if (!need_update)
        //last computed and published model is still fine, let's get out of here
        return;

    //model is reset, refill it
    model_ptr.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    object_tree.reset(new pcl::search::KdTree<pcl::PointXYZRGB>);
    hand_tree.reset(new pcl::search::KdTree<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*object_ptr, *model_ptr);
    model_ptr->header.frame_id="/camera_rgb_optical_frame";
    //color input point cloud blue in the reconstructed model
    uint8_t r = 0, g = 0, b = 255;
    uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
    for (size_t i=0; i < model_ptr->size(); ++i)
        model_ptr->points[i].rgb = *reinterpret_cast<float*>(&rgb);
    //color hand points cyan
    g=255;
    rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
    for (size_t i=0; i< hand_ptr->size(); ++i)
    {
        pcl::PointXYZRGB hp (hand_ptr->points[i]);
        hp.rgb = *reinterpret_cast<float*>(&rgb);
        model_ptr->push_back(hp);
    }
    //We need to sample some points on a grid, query the gp model and add points
    //to cloud model, colored accordingly  to their covariance. Then publish the
    //cloud so the user can see it in rviz. Let's get this job done!
    ROS_INFO("[GaussianProcessNode::%s]\tComputing reconstructed model cloud...",__func__);
    // Get centroid of object and it's rough bounding box
    Eigen::Vector4f obj_cent;
    if(pcl::compute3DCentroid<pcl::PointXYZRGB>(*object_ptr, obj_cent) == 0){
        ROS_ERROR("[GaussianProcessNode::%s]\tFailed to compute object centroid. Aborting...",__func__);
        need_update = false;
        return;
    }
    //initialize samples storage
    samples_var.clear();
    samples_ptr.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    //get minimum and maximum x y z coordinates of object
    Eigen::Vector4f min_coord, max_coord;
    pcl::getMinMax3D(*object_ptr, min_coord, max_coord);
    //construct a kdtree of hand to test if sample point is near the hand
    if (!hand_ptr->empty())
        hand_tree->setInputCloud(hand_ptr);
    //construct  the  kdtree  of  object
    object_tree->setInputCloud(object_ptr);
    //sample from (min  - ext) to (max +  ext) except z that goes  from (min) to
    //(max + 2*ext). No need to sample  in front of the camera,  remember z=0 is
    //kinect plane.
    const float ext = 0.01f;
    //with a step of
    const float step = 0.01f;
    for (float x = min_coord[0] - ext; x<= max_coord[0] + ext; x += step)
        for (float y = min_coord[1] - ext; y<= max_coord[1] + ext; y += step)
            for (float z = min_coord[2]; z<= max_coord[2] + 2*ext; z += step)
            {
                pcl::PointXYZRGB pt;
                pt.x = x;
                pt.y = y;
                pt.z = z;
                std::vector<int> k_id;
                std::vector<float> k_dist;
                if (!hand_ptr->empty()){
                    //first exclude  samples that  could be  near the  hand, we
                    //don't want the probe go near it
                    if(hand_tree->radiusSearch(pt, 0.02, k_id, k_dist, 1) > 0)
                        //this sample is near the hand (2cm), we discard it
                        continue;
                }
                // then exclude  samples too near the object, they probably
                // don't interest much
                if(object_tree->radiusSearch(pt, 0.005, k_id, k_dist, 1) > 0)
                    //this sample is near the obj (5mm), we discard it
                    continue;
                //test if sample point is occluded, if not we don't have to test
                //it since  camera would have seen  it, thus it would  be inside
                //object cloud already
                if (isSampleVisible(pt, min_coord[2]))
                    //camera can see the sample, discard it
                    continue;

                //finally query  the gaussian  model for  the sample,  keep only
                //samples detected as belonging to the object
                Vec3 q(x,y,z);
                const double qf = gp->f(q);
                const double qvar = gp->var(q);
                //test if sample was classified as belonging to obj surface
                if (qf <= 0.02 && qf >= -0.02){
                    //We can  add this sample  to the reconstructed  cloud model
                    //color the sample according to variance. however to do this
                    //we need  the maximum variance  found. So we have  to store
                    //these points  to evaluate it.
                    samples_var.push_back(qvar);
                    samples_ptr->push_back(pt);
                }
            }
    //all generated samples are now tested, good ones are stored into samples.
    const double max_var = *std::max_element(samples_var.begin(), samples_var.end());
    const double min_var = *std::min_element(samples_var.begin(), samples_var.end());
    const double ref_var = (max_var-min_var)*0.5;
    //so color of points goes from green (var = 0) to red (var = max_var)
    //Lots of mumbo-jumbo with bits to do this (convert from double to uint then
    //to float!)

    //sanity check for size
    if(samples_var.size() != samples_ptr->size()){
        ROS_ERROR("[GaussianProcessNode::%s]\tSomething went wrong when computing samples, size mismatch. Aborting...",__func__);
        //this should never happen, however extra checks don't hurt
        return;
    }
    //finally color those sample with their appropriate color
    double red, green;
    uint32_t tmp;
    r = 0; g = 0; b = 0;
    int samp_idx(0);
    double min_var_diff(1e5);
    for (size_t i =0; i<samples_var.size(); ++i)
    {
        if (samples_var[i] <= ref_var){
            red = 255.0*(samples_var[i] - min_var);
            g = 255;
            tmp = uint32_t(red);
            r = tmp & 0x0000ff;
            //This way it goes from green to yellow at half max_var
            double dif_var = ref_var - samples_var[i];
            if (dif_var < min_var_diff){
                min_var_diff = dif_var;
                samp_idx = i;
            }
        }
        if (samples_var[i] > ref_var){
            green = 255.0*(max_var - samples_var[i]);
            r = 255;
            tmp = uint32_t(green);
            g = tmp & 0x0000ff;

            double dif_var = samples_var[i] - ref_var;
            if (dif_var < min_var_diff){
                min_var_diff = dif_var;
                samp_idx = i;
            }
        }
        rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
        samples_ptr->at(i).rgb = *reinterpret_cast<float*>(&rgb);
        model_ptr->push_back(samples_ptr->at(i));
    }
    sample_to_explore.header.frame_id="/camera_rgb_optical_frame";
    sample_to_explore.header.stamp=ros::Time::now();
    sample_to_explore.point.x = samples_ptr->at(samp_idx).x;
    sample_to_explore.point.y = samples_ptr->at(samp_idx).y;
    sample_to_explore.point.z = samples_ptr->at(samp_idx).z;
    Eigen::Vector3f dir(obj_cent[0] - samples_ptr->at(samp_idx).x,
                        obj_cent[1] - samples_ptr->at(samp_idx).y,
                        obj_cent[2] - samples_ptr->at(samp_idx).z);
    dir.normalize();
    sample_to_explore.direction.x = dir[0];
    sample_to_explore.direction.y = dir[1];
    sample_to_explore.direction.z = dir[2];
    //we leave samples filled, so they could be sent to someone if needed
    need_update = false;
    //reset  object  kdtree,   so  no  one  can   call  isSampleVisible  without
    //initializing it again.
    object_tree.reset();
    //publish the model
    publishCloudModel();
    publishSampleToExplore();
    ROS_INFO("[GaussianProcessNode::%s]\tDone computing reconstructed model cloud.",__func__);
}

//callback to start process service, executes when service is called
bool GaussianProcessNode::cb_start(gp_regression::start_process::Request& req, gp_regression::start_process::Response& res)
{
    if(req.cloud_dir.empty()){
        //Request was empty, means we have to call pacman vision service to
        //get a cloud.
        std::string service_name = nh.resolveName("/pacman_vision/listener/get_cloud_in_hand");
        pacman_vision_comm::get_cloud_in_hand service;
        service.request.save = "false";
        //TODO:  Service requires  to know  which hand  is grasping  the
        //object we  dont have a way to tell inside here.  Assuming it's
        //the left  hand for now.(Fri  06 Nov  2015 05:11:45 PM  CET --
        //tabjones)
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
    viewpoint_tree.reset(new pcl::search::KdTree<pcl::PointXYZRGB>);
    viewpoint_tree->setInputCloud(object_ptr);
    discovered.clear();
    return (compute());
}

bool GaussianProcessNode::cb_sample(gp_regression::GetToExploreTrajectory::Request& req, gp_regression::GetToExploreTrajectory::Response& res)
{
    sampleAndPublish();
    res.trajectory = sample_to_explore;
    return true;
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
    uint8_t r = 0, g = 0, b = 255;
    uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
    pt.rgb = *reinterpret_cast<float*>(&rgb);
    discovered.push_back(pt);
    //check if they are enough to update the model
    if (discovered.size() >= how_many_discoveries){
        for (const auto& x: discovered)
            object_ptr->push_back(x);
        //update the model
        this->update();
    }
}
//gp computation
bool GaussianProcessNode::compute()
{
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
    Data cloud; //to be filled later on...

    //Add object points as label 0
    for(const auto pt : object_ptr->points)
    {
        cloud.coord_x.push_back(pt.x);
        cloud.coord_y.push_back(pt.y);
        cloud.coord_z.push_back(pt.z);
        cloud.label.push_back(0);
    }
    //ignore hand in gp model
    // if (size_hand <=0){
    //     ROS_WARN("[GaussianProcessNode::%s]\tLoaded hand cloud is empty, cannot compute a model. Aborting...",__func__);
    //     start = false;
    //     return false;
    // }
    // for(size_t i=0; i<size_hand; ++i)
    // {
    //     //these points are marked as "external", cause they are from the hand
    //     Vec3 point(hand_ptr->points[i].x, hand_ptr->points[i].y, hand_ptr->points[i].z);
    //     cloud[size_cloud +i]=point;
    //     targets[size_cloud+i]=1;
    // }

    //Now add centroid as label -1
    Eigen::Vector4f centroid;
    if(pcl::compute3DCentroid<pcl::PointXYZRGB>(*object_ptr, centroid) == 0){
        ROS_ERROR("[GaussianProcessNode::%s]\tFailed to compute object centroid. Aborting...",__func__);
        start = false;
        return false;
    }
    Vec3 centr(centroid[0], centroid[1], centroid[2]);
    cloud.coord_x.push_back(centroid[0]);
    cloud.coord_y.push_back(centroid[1]);
    cloud.coord_z.push_back(centroid[2]);
    cloud.label.push_back(-1);

    //Add points in a sphere around centroid as label 1
    //sphere bounds computation
    const int ang_div = 8; //divide 360° in 8 pieces, i.e. steps of 45°
    const int lin_div = 6; //divide diameter into 6 pieces
    //this makes 8*6 = 48 points.
    const double radius = 0.3; //30cm
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
            cloud.coord_x.push_back(x);
            cloud.coord_y.push_back(y);
            cloud.coord_z.push_back(z);
            cloud.label.push_back(1);
        }
    /*****  Create the gp model  *********************************************/
    //create the model to be stored in class
    regressor.create(cloud, object_gp);
    ROS_INFO("[GaussianProcessNode::%s]\tRegressor created",__func__);
    start = true;
    //tell the publisher we have a new model, so it can publish it to rviz
    need_update = true;
    return true;

    // TODO: REVISION TILL HERE, BELOW HERE IS NOT CHECKED AND DOES NOT COMPILE!
    // (tabjones on Tuesday 01/12/2015)
}
//update gaussian model with new points from probe
void GaussianProcessNode::update()
{
    gp.reset();
    discovered.clear();
    this->compute();
}
//Republish cloud method
void GaussianProcessNode::publishCloudModel () const
{
    //These checks are  to make sure we are not  publishing empty cloud,
    //we have a  gaussian process computed and  there's actually someone
    //who listens to us
    if (start && model_ptr)
        if(!model_ptr->empty() && pub_model.getNumSubscribers()>0)
            pub_model.publish(*model_ptr);
}
//Publish sample
void GaussianProcessNode::publishSampleToExplore () const
{
    // TMP: The display marker for point and direction, useful for debug ;)
    geometry_msgs::PointStamped point_marker;
    point_marker.header = sample_to_explore.header;
    point_marker.point = sample_to_explore.point;

    geometry_msgs::WrenchStamped direction_marker;
    direction_marker.header = sample_to_explore.header;
    direction_marker.wrench.force = sample_to_explore.direction;

    //These checks are  to make sure we  have a gp and  there's actually someone
    // who listens to us
    if (start && pub_point.getNumSubscribers()>0)
    {
        pub_point.publish(sample_to_explore);
        pub_point_marker.publish(point_marker);
        pub_direction_marker.publish(direction_marker);
    }

}
//test for occlusion of samples
//return:
//  0 -> not visible
//  1 -> visible
//  -1 -> error
int GaussianProcessNode::isSampleVisible(const pcl::PointXYZRGB sample, const float min_z) const
{
    if(!viewpoint_tree){
        ROS_ERROR("[GaussianProcessNode::%s]\tObject Viewpoint KdTree is not initialized. Aborting...",__func__);
        //should never happen if called from sampleAndPublish
        return (-1);
    }
    Eigen::Vector3f camera(0,0,0);
    Eigen::Vector3f start_point(sample.x, sample.y, sample.z);
    Eigen::Vector3f direction = camera - start_point;
    const float norm = direction.norm();
    direction.normalize();
    const float step_size = 0.01f;
    const int nsteps = std::max(1, static_cast<int>(norm/step_size));
    std::vector<int> k_id;
    std::vector<float> k_dist;
    //move along direction
    Eigen::Vector3f p(start_point[0], start_point[1], start_point[2]);
    for (size_t i = 0; i<nsteps; ++i)
    {
        if (p[2] <= min_z)
            //don't reach  the sensor, if  we are  outside sample region  we can
            //stop testing.
            break;
        pcl::PointXYZRGB pt;
        pt.x = p[0];
        pt.y = p[1];
        pt.z = p[2];
        // TODO: This search radius is  hardcoded now, should be adapted somehow
        // on point density (tabjones on Friday 13/11/2015)
        if (viewpoint_tree->radiusSearch(pt, 0.005, k_id, k_dist, 1) > 0)
            //we intersected an object point, this sample cant reach the camera
            return(0);
        p += (direction * step_size);
    }
    //we didn't intersect anything, this sample is not occluded
    return(1);
}
