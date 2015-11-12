#include <gp_node.h>

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
    //construct  a  kdtree  of  object  to  test if  sample  point  is  near  it
    pcl::search::KdTree<pcl::PointXYZRGB> tree_obj;
    tree_obj.setInputCloud(cloud_ptr);
    //sample from (min - ext) to (max + ext)
    float ext = 0.03f;
    //with a step of
    float step = 0.02f;
    for (float x = min_coord[0] - ext; x<= max_coord[0] + ext; x += step)
        for (float y = min_coord[1] - ext; y<= max_coord[1] + ext; y += step)
            for (float z = min_coord[2] - ext; z<= max_coord[2] + ext; z += step)
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
                //test if sample point is occluded, if not we don't have to test
                //it since  camera would have seen  it, thus it would  be inside
                //object cloud already
                // TODO:  Implement a test  for occlusion (tabjones  on Thursday
                //12/11/2015)

                //test for object adjacency
                if(tree_obj.radiusSearch(pt, 0.01, k_id, k_dist, 1) > 0)
                    //this sample is near the object (1cm), we discard it
                    continue;
                //finally query  the gaussian  model for  the sample,  keep only
                //samples detected as "internal"
                Vec3 q(x,y,z);
                const double qf = gp->f(q);
                const double qvar = gp->var(q);
                //test if sample was classified internal
                if (qf <= 0.1){
                    //We can add this sample to the reconstructed cloud model
                    //color the sample according to variance
                    double red = 255.0*qvar;
                    double green = 255.0*(1.0 - qvar);
                    uint8_t* tmp = (uint8_t*)&red;
                    b = 0;
                    r = *tmp;
                    tmp = (uint8_t*)&green;
                    g = *tmp;
                    std::cout<<red<<" "<<green<<" -> "<<(unsigned int)r<<" "<<(unsigned int)g<<std::endl;
                    if (r < 0)
                        r = 0;
                    if (r > 255)
                        r = 255;
                    if (g < 0)
                        g = 0;
                    if (g > 255)
                        g = 255;
                    rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
                    pt.rgb = *reinterpret_cast<float*>(&rgb);
                    //add it
                    model_ptr->push_back(pt);
                    // TODO: Add a storage of accepted samples into the class so
                    // they can be  sent to the probe for  example. (tabjones on
                    // Thursday 12/11/2015)
                }
            }
    //All samples tested
    need_update = false;
    //publish it
    ROS_INFO("[GaussianProcessNode::%s]\tDone computing reconstructed model cloud.",__func__);
    publishCloudModel();
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
    targets.resize(size_cloud + size_hand);
    cloud.resize(size_cloud + size_hand);
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
        ROS_WARN("[GaussianProcessNode::%s]\tLoaded hand cloud is empty, using centroid of object as 'external' point for Gaussian model computation...",__func__);
        Eigen::Vector4f centroid;
        if(pcl::compute3DCentroid<pcl::PointXYZRGB>(*cloud_ptr, centroid) == 0){
            ROS_ERROR("[GaussianProcessNode::%s]\tFailed to compute object centroid. Aborting...",__func__);
            start = false;
            return false;
        }
        targets.resize(size_cloud + 1);
        cloud.resize(size_cloud + 1);
        Vec3 ext_point(centroid[0], centroid[1], centroid[2]);
        cloud[size_cloud] = ext_point;
        targets[size_cloud] = 1;
    }
    else{
        for(size_t i=0; i<size_hand; ++i)
        {
            //these points are marked as "external", cause they are from the hand
            Vec3 point(hand_ptr->points[i].x, hand_ptr->points[i].y, hand_ptr->points[i].z);
            cloud[size_cloud +i]=point;
            targets[size_cloud+i]=1;
        }
    }
    /*****  Create the model  *********************************************/
    SampleSet::Ptr trainingData(new SampleSet(cloud, targets));
    LaplaceRegressor::Desc laplaceDesc;
    laplaceDesc.noise = 0.001;
    //create the model to be stored in class
    gp = laplaceDesc.create();
    ROS_INFO("[GaussianProcessNode::%s]\tRegressor created: %s",__func__, gp->getName().c_str());
    gp->set(trainingData);
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
void GaussianProcessNode::publishCloudModel ()
{
    //These checks are  to make sure we are not  publishing empty cloud,
    //we have a  gaussian process computed and  there's actually someone
    //who listens to us
    if (start && model_ptr)
        if(!model_ptr->empty() && pub_model.getNumSubscribers()>0)
            pub_model.publish(*model_ptr);
}

