#include <gp_node.h>
#include <algorithm> //for std::max_element

using namespace gp;

/*
 *              PLEASE LOOK at TODOs by searching "TODO" to have an idea of
 *              what is still missing or is improvable!
 */

GaussianProcessNode::GaussianProcessNode (): nh(ros::NodeHandle("gaussian_process")), start(false),
    need_update(false)
{
    srv_start = nh.advertiseService("start_process", &GaussianProcessNode::cb_start, this);
    //TODO: Added a  publisher to republish point cloud  with new points
    //from  gaussian process,  right now  it's unused  (Fri 06  Nov 2015
    //05:18:41 PM CET -- tabjones)
    pub_model = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB>> ("estimated_model", 1);
    cloud_ptr.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
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

    //We need to sample some points on a grid, query the gp model and add points
    //to cloud model, colored accordingly  to their covariance. Then publish the
    //cloud so the user can see it in rviz. Let's get this job done!
    ROS_INFO("[GaussianProcessNode::%s]\tComputing reconstructed model cloud...",__func__);
    // Get centroid of object and it's rough bounding box
    // TODO: Centroid is not used atm (tabjones on Thursday 12/11/2015)
    /*
     * Eigen::Vector4f obj_cent;
     * if(pcl::compute3DCentroid<pcl::PointXYZRGB>(*cloud_ptr, obj_cent) == 0){
     *     ROS_ERROR("[GaussianProcessNode::%s]\tFailed to compute object centroid. Aborting...",__func__);
     *     need_update = false;
     *     return;
     * }
     */
    //initialize storage
    sample_vars.clear();
    samples.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    //get minimum and maximum x y z coordinates of object
    Eigen::Vector4f min_coord, max_coord;
    pcl::getMinMax3D(*cloud_ptr, min_coord, max_coord);
    //color input point cloud blue in the reconstructed model
    pcl::copyPointCloud(*cloud_ptr, *model_ptr);
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
    //construct a kdtree of hand to test if sample point is near the hand
    pcl::search::KdTree<pcl::PointXYZRGB> tree_hand;
    tree_hand.setInputCloud(hand_ptr);
    //construct  the  kdtree  of  object
    tree_obj.reset(new pcl::search::KdTree<pcl::PointXYZRGB>);
    tree_obj->setInputCloud(cloud_ptr);
    //sample from (min - ext) to (max + ext)
    //except z that  goes from (min + step) to  (max + 2*ext) //no need  to sample in
    //front of the camera, remember z=0 is kinect plane
    float ext = 0.05f;
    //with a step of
    float step = 0.015f;
    for (float x = min_coord[0] - ext; x<= max_coord[0] + ext; x += step)
        for (float y = min_coord[1] - ext; y<= max_coord[1] + ext; y += step)
            for (float z = min_coord[2] + step; z<= max_coord[2] + 2*ext; z += step)
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
                    if(tree_hand.radiusSearch(pt, 0.02, k_id, k_dist, 1) > 0)
                        //this sample is near the hand (2cm), we discard it
                        continue;
                }
                    //then exclude  samples too near the object, they probably
                    //don't interest much
                    if(tree_obj->radiusSearch(pt, 0.005, k_id, k_dist, 1) > 0)
                        //this sample is near the obj (5mm), we discard it
                        continue;
                //test if sample point is occluded, if not we don't have to test
                //it since  camera would have seen  it, thus it would  be inside
                //object cloud already
                if (isSampleVisible(pt, min_coord[2]))
                    //camera can see the sample, discard it
                    continue;

                //finally query  the gaussian  model for  the sample,  keep only
                //samples detected as "internal"
                Vec3 q(x,y,z);
                const double qf = gp->f(q);
                const double qvar = gp->var(q);
                //test if sample was classified internal, with high tolerance
                if (qf <= 0.7){
                    //We can  add this sample  to the reconstructed  cloud model
                    //color the sample according to variance. however to do this
                    //we need  the maximum variance  found. So we have  to store
                    //these points  to evaluate it,  then add them to  the model
                    //later.
                    sample_vars.push_back(qvar);
                    samples->push_back(pt);
                }
            }
    //all generated samples are now tested, good ones are stored into samples.
    const double max_var = *std::max_element(sample_vars.begin(), sample_vars.end());
    //so color of points goes from green (var = 0) to red (var = max_var)
    //Lots of mumbo-jumbo with bits to do this (convert from double to uint then
    //to float!)

    //sanity check for size
    if(sample_vars.size() != samples->size()){
        ROS_ERROR("[GaussianProcessNode::%s]\tSomething went wrong when computing samples, size mismatch. Aborting...",__func__);
        //this should never happen, however extra checks don't hurt
        return;
    }
    //finally color those sample with their appropriate color
    double red, green;
    uint32_t tmp;
    b=0;
    for (size_t i =0; i<samples->size(); ++i)
    {
        if (sample_vars[i] <= max_var*0.5){
            red = 255.0*2.0*sample_vars[i];
            g = 255;
            tmp = uint32_t(red);
            r = tmp & 0x0000ff;
            //This way it goes from green to yellow at half max_var
        }
        if (sample_vars[i] > max_var*0.5){
            green = 255.0*2.0*(max_var - sample_vars[i]);
            r = 255;
            tmp = uint32_t(green);
            g = tmp & 0x0000ff;
        }
        rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
        samples->points[i].rgb = *reinterpret_cast<float*>(&rgb);
        //add it published model
        model_ptr->push_back(samples->points[i]);
    }
    //we leave samples filled, so they can be sent to someone if needed
    need_update = false;
    //reset  object  kdtree,   so  no  one  can   call  isSampleVisible  without
    //initializing it again.
    tree_obj.reset();
    //publish the model
    publishCloudModel();
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
        //the right  hand for now.(Fri  06 Nov  2015 05:11:45 PM  CET --
        //tabjones)
        service.request.right = true;
        if (!ros::service::call<pacman_vision_comm::get_cloud_in_hand>(service_name, service))
        {
            ROS_ERROR("[GaussianProcessNode::%s]\tGet cloud in hand service call failed!",__func__);
            return (false);
        }
        //object and hand clouds are saved into class
        pcl::fromROSMsg (service.response.obj, *cloud_ptr);
        pcl::fromROSMsg (service.response.hand, *hand_ptr);
    }
    else{
        //User told us to load a clouds from a dir on disk instead.
        //TODO: Add  some path checks  perhaps? Lets  hope the user  wrote a
        //valid path for now! (Fri 06 Nov 2015 05:03:19 PM CET -- tabjones)
        if (pcl::io::loadPCDFile((req.cloud_dir+"/obj.pcd"), *cloud_ptr) != 0){
            ROS_ERROR("[GaussianProcessNode::%s]\tError loading cloud from %s",__func__,(req.cloud_dir+"obj.pcd").c_str());
            return (false);
        }
        if (pcl::io::loadPCDFile((req.cloud_dir+"/hand.pcd"), *hand_ptr) != 0){
            ROS_ERROR("[GaussianProcessNode::%s]\tError loading cloud from %s",__func__,(req.cloud_dir + "/hand.pcd").c_str());
            return (false);
        }
        //We  need  to  fill  point  cloud header  or  ROS  will  complain  when
        //republishing this cloud. Let's assume it was published by asus kinect.
        //I think it's just need the frame id
        cloud_ptr->header.frame_id="/camera_rgb_optical_frame";
        hand_ptr->header.frame_id="/camera_rgb_optical_frame";
    }
    return (compute());
}
//gp computation
bool GaussianProcessNode::compute()
{
    if(!cloud_ptr || !hand_ptr){
        //This  should never  happen  if compute  is  called from  start_process
        //service callback, however it does not hurt to add this extra check!
        ROS_ERROR("[GaussianProcessNode::%s]\tObject or Hand cloud pointers are empty. Aborting...",__func__);
        start = false;
        return false;
    }
    Vec3Seq cloud;
    Vec targets;
    const size_t size_cloud = cloud_ptr->size();
    const size_t size_hand = hand_ptr->size();
    targets.resize(size_cloud + size_hand +1);
    cloud.resize(size_cloud + size_hand +1);
    if (size_cloud <=0){
        ROS_ERROR("[GaussianProcessNode::%s]\tLoaded object cloud is empty, cannot compute a model. Aborting...",__func__);
        start = false;
        return false;
    }
    for(size_t i=0; i<size_cloud; ++i)
    {
        Vec3 point(cloud_ptr->points[i].x, cloud_ptr->points[i].y, cloud_ptr->points[i].z);
        cloud[i]=point;
        targets[i]=0;
    }
    if (size_hand <=0){
        ROS_WARN("[GaussianProcessNode::%s]\tLoaded hand cloud is empty, cannot compute a model. Aborting...",__func__);
        start = false;
        return false;
    }
    for(size_t i=0; i<size_hand; ++i)
    {
        //these points are marked as "external", cause they are from the hand
        Vec3 point(hand_ptr->points[i].x, hand_ptr->points[i].y, hand_ptr->points[i].z);
        cloud[size_cloud +i]=point;
        targets[size_cloud+i]=1;
    }
    //Now add centroid as "internal"
    Eigen::Vector4f centroid;
    if(pcl::compute3DCentroid<pcl::PointXYZRGB>(*cloud_ptr, centroid) == 0){
        ROS_ERROR("[GaussianProcessNode::%s]\tFailed to compute object centroid. Aborting...",__func__);
        start = false;
        return false;
    }
    Vec3 centr(centroid[0], centroid[1], centroid[2]);
    cloud[size_hand+size_cloud] =centr;
    targets[size_hand+size_cloud] = -1;
    /*****  Create the model  *********************************************/
    SampleSet::Ptr trainingData(new SampleSet(cloud, targets));
    LaplaceRegressor::Desc laplaceDesc;
    laplaceDesc.noise = 0.01;
    //create the model to be stored in class
    gp = laplaceDesc.create();
    gp->set(trainingData);
    ROS_INFO("[GaussianProcessNode::%s]\tRegressor created: %s",__func__, gp->getName().c_str());
    start = true;
    //tell the publisher we have a new model, so it can publish it to rviz
    need_update = true;
    return true;
}
//update gaussian model with new points from probe
void GaussianProcessNode::update()
{
    // TODO: To  implement entirely. Also  we need a way  to communicate
    // with the probe  package to get the new points.  For instance call
    // this function inside the callback when new points arrive(tabjones
    // on Wednesday 11/11/2015)
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
//test for occlusion of samples
//return:
//  0 -> not visible
//  1 -> visible
//  -1 -> error
int GaussianProcessNode::isSampleVisible(const pcl::PointXYZRGB sample, const float min_z) const
{
    if(!tree_obj){
        ROS_ERROR("[GaussianProcessNode::%s]\tObject KdTree is not initialized. Aborting...",__func__);
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
        if (tree_obj->radiusSearch(pt, 0.005, k_id, k_dist, 1) > 0)
            //we intersected an object point, this sample cant reach the camera
            return(0);
        p += (direction * step_size);
    }
    //we didn't intersect anything, this sample is not occluded
    return(1);
}

