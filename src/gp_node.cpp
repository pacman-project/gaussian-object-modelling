#include <algorithm> //for std::max_element
#include <chrono> //for time measurements
#include <fstream>

#include <node_utils.hpp>
#include <gp_node.h>

#include <ros/package.h>
#include <boost/algorithm/string/split.hpp>
#include <boost/filesystem/path.hpp>
using namespace gp_regression;

GaussianProcessNode::GaussianProcessNode (): nh(ros::NodeHandle("gaussian_process")), start(false),
    object_ptr(boost::make_shared<PtC>()), hand_ptr(boost::make_shared<PtC>()), data_ptr_(boost::make_shared<PtC>()),
    model_ptr(boost::make_shared<PtC>()), real_explicit_ptr(boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>()),
    exploration_started(false), out_sphere_rad(2.0), sigma2(1e-1), min_v(0.0), max_v(0.5),
    simulate_touch(true), anchor("/mind_anchor"), steps(0), last_touched(Eigen::Vector3d::Zero())
{
    mtx_marks = std::make_shared<std::mutex>();
    srv_start = nh.advertiseService("start_process", &GaussianProcessNode::cb_start, this);
    srv_update = nh.advertiseService("update_process", &GaussianProcessNode::cb_updateS, this);
    srv_get_next_best_path_ = nh.advertiseService("get_next_best_path", &GaussianProcessNode::cb_get_next_best_path, this);
    pub_model = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB>> ("training_data", 1);
    pub_real_explicit = nh.advertise<pcl::PointCloud<pcl::PointXYZI> >("estimated_model", 1);
    pub_octomap = nh.advertise<octomap_msgs::Octomap>("octomap",1);
    pub_markers = nh.advertise<visualization_msgs::MarkerArray> ("atlas", 1);
    sub_update_ = nh.subscribe(nh.resolveName("/path_log"),1, &GaussianProcessNode::cb_update, this);
    nh.param<std::string>("/processing_frame", proc_frame, "/camera_rgb_optical_frame");
    anchor = proc_frame; //TODO remove after fixing of mind anchor
    nh.param<int>("touch_type", synth_type, 2);
    nh.param<double>("global_goal", goal, 0.1);
    nh.param<double>("sample_res", sample_res, 0.07);
    nh.param<bool>("simulate_touch", simulate_touch, true);
    nh.param<bool>("ignore_last_touched", ignore_last_touched, true);
    synth_var_goal = 0.2;
}

void GaussianProcessNode::Publish()
{
    if (!start){
        ROS_WARN_THROTTLE(60,"[GaussianProcessNode::%s]\tNo object model found! Call start_process service to begin creating a model.",__func__);
        return;
    }
    //need to lock here, exploration might interfere
    std::lock_guard<std::mutex> lock (*mtx_marks);
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
            // pub_real_explicit.publish(*full_object);
        }
    }
}
//publish atlas markers and other samples
void GaussianProcessNode::publishAtlas () const
{
    if (markers)
        if(pub_markers.getNumSubscribers() > 0)
            pub_markers.publish(*markers);
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
void GaussianProcessNode::deMeanAndNormalizeData(Eigen::Vector3d &data)
{
    data -= current_offset_.block(0,0,3,1);
    data = data/current_scale_;
}

// void GaussianProcessNode::reMeanAndDenormalizeData(const std::vector<Eigen::Vector3d> &in, std::vector<Eigen::Vector3d> &out)
void GaussianProcessNode::reMeanAndDenormalizeData(Eigen::Vector3d &data)
{
    data = current_scale_*data;
    data += current_offset_.block(0,0,3,1);
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
    current_goal = req.var_desired.data;
    if (simulate_touch){
        nh.getParam("touch_type", synth_type);
        if (synth_type == 0)
            synth_var_goal = goal;
    }
    if(startExploration(req.var_desired.data)){
        while(exploration_started){
            // don't like it, cause we loose the actual velocity of the atlas
            // but for now, this is it, repeating the node while loop here
            //gogogo!
            ros::spinOnce();
            Publish();
            checkExploration();
            rate.sleep();
        }
        if (solution.empty()){
            if ((current_goal > goal) && !simulate_touch){
                ROS_WARN("[GaussianProcessNode::%s]\tNo solution found at requested variance %g, However global goal is set to %g. Call this service again with reduced request!",__func__, current_goal, goal);
                markers = boost::make_shared<visualization_msgs::MarkerArray>();
                visualization_msgs::Marker samples;
                samples.header.frame_id = anchor;
                samples.header.stamp = ros::Time();
                samples.lifetime = ros::Duration(5.0);
                samples.ns = "samples";
                samples.id = 0;
                samples.type = visualization_msgs::Marker::POINTS;
                samples.action = visualization_msgs::Marker::ADD;
                samples.scale.x = 0.025;
                samples.scale.y = 0.025;
                samples.scale.z = 0.025;
                const double mid_v = (min_v + max_v)/2;
                for (const auto &ptc: real_explicit_ptr->points)
                {
                    geometry_msgs::Point pt;
                    std_msgs::ColorRGBA cl;
                    pt.x = ptc.x;
                    pt.y = ptc.y;
                    pt.z = ptc.z;
                    cl.a = 1.0;
                    cl.b = 0.0;
                    cl.r = (ptc.intensity<mid_v) ? 1/(mid_v - min_v) *  (ptc.intensity - min_v) : 1.0;
                    cl.g = (ptc.intensity>mid_v) ? -1/(max_v - mid_v) * (ptc.intensity - mid_v) + 1 : 1.0;
                    samples.points.push_back(pt);
                    samples.colors.push_back(cl);
                }
                markers->markers.push_back(samples);
                publishAtlas();
                ros::spinOnce();
                return true;
            }
            if (simulate_touch && (synth_var_goal > goal)){
                ROS_WARN("[GaussianProcessNode::%s]\tNo solution found at requested variance %g, automatically reducing it.",__func__, synth_var_goal);
                synth_var_goal = (synth_var_goal - 0.1) < goal ? goal : synth_var_goal - 0.1;
                markers = boost::make_shared<visualization_msgs::MarkerArray>();
                visualization_msgs::Marker samples;
                samples.header.frame_id = anchor;
                samples.header.stamp = ros::Time();
                samples.lifetime = ros::Duration(5.0);
                samples.ns = "samples";
                samples.id = 0;
                samples.type = visualization_msgs::Marker::POINTS;
                samples.action = visualization_msgs::Marker::ADD;
                samples.scale.x = 0.025;
                samples.scale.y = 0.025;
                samples.scale.z = 0.025;
                const double mid_v = (min_v + max_v)/2;
                for (const auto &ptc: real_explicit_ptr->points)
                {
                    geometry_msgs::Point pt;
                    std_msgs::ColorRGBA cl;
                    pt.x = ptc.x;
                    pt.y = ptc.y;
                    pt.z = ptc.z;
                    cl.a = 1.0;
                    cl.b = 0.0;
                    cl.r = (ptc.intensity<mid_v) ? 1/(mid_v - min_v) *  (ptc.intensity - min_v) : 1.0;
                    cl.g = (ptc.intensity>mid_v) ? -1/(max_v - mid_v) * (ptc.intensity - mid_v) + 1 : 1.0;
                    samples.points.push_back(pt);
                    samples.colors.push_back(cl);
                }
                markers->markers.push_back(samples);
                publishAtlas();
                ros::spinOnce();
                if (steps == 0)
                    ++steps;
                return true;
            }
            ROS_WARN("[GaussianProcessNode::%s]\tNo solution found, Object shape is reconstructed !",__func__);
            ROS_WARN("[GaussianProcessNode::%s]\tVariance requested: %g, Total number of touches %ld",__func__,req.var_desired.data, steps);
            ROS_WARN("[GaussianProcessNode::%s]\tComputing final shape...",__func__);
            //pause a bit for better visualization
            std::this_thread::sleep_for(std::chrono::seconds(3));
            markers = boost::make_shared<visualization_msgs::MarkerArray>();
            marchingSampling(false, 0.06,0.02);
            res.next_best_path = gp_regression::Path();
            last_touched = Eigen::Vector3d::Zero();
            pcl::PointCloud<pcl::PointXYZI> reconstructed;
            reMeanAndDenormalizeData(real_explicit_ptr, reconstructed);
            double MSE (0.0), RMSE(0.0), SSE(0.0);
            if (simulate_touch){
                test_name = obj_name + "_" + std::to_string(synth_type);
                pcl::PointCloud<pcl::PointXYZ>::Ptr real_recon = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
                pcl::copyPointCloud(reconstructed, *real_recon);
                kd_full.setInputCloud(real_recon);
                std::vector<int> k_id (1);
                std::vector<float> k_dis (1);
                for(const auto &pt : full_object_real->points)
                {
                    kd_full.nearestKSearch(pt, 1, k_id, k_dis);
                    MSE += k_dis[0];
                    RMSE += std::sqrt(k_dis[0]);
                }
                kd_full.setInputCloud(full_object_real);
                for(const auto &pt : real_recon->points)
                {
                    kd_full.nearestKSearch(pt, 1, k_id, k_dis);
                    MSE += k_dis[0];
                    RMSE += std::sqrt(k_dis[0]);
                }
                //reciprocal MSE makes more sense
                MSE /= ( full_object_real->points.size() + real_recon->points.size() );
                RMSE /= ( full_object_real->points.size() + real_recon->points.size() );
                ROS_WARN("[GaussianProcessNode::%s]\tCalculated MSE is %g",__func__, MSE);
                ROS_WARN("[GaussianProcessNode::%s]\tCalculated RMSE is %g",__func__, RMSE);
                //SSE
                for(const auto &p: full_object_real->points)
                {
                    gp_regression::Data::Ptr qq = std::make_shared<gp_regression::Data>();
                    qq->coord_x.push_back(p.x);
                    qq->coord_y.push_back(p.y);
                    qq->coord_z.push_back(p.z);
                    std::vector<double> ff;
                    reg_->evaluate(obj_gp, qq, ff);
                    SSE += std::pow(ff[0], 2);
                }
                ROS_WARN("[GaussianProcessNode::%s]\tCalculated SSE is %g",__func__, SSE);
            }
            else
                obj_name = "object";
            std::string pkg_path (ros::package::getPath("gp_regression"));
            boost::filesystem::path pcd_path (pkg_path + "/results/" + test_name + ".pcd");
            if (pcl::io::savePCDFile(pcd_path.c_str(), reconstructed))
                ROS_ERROR("[GaussianProcessNode::%s] Error saving reconstructed shape %s",__func__, pcd_path.c_str());
            //save a result file (appending)
            std::string file_path (pkg_path + "/results/tests.txt");
            std::ofstream file (file_path.c_str(), std::ios::out | std::ios::app);
            //file is
            //name      steps       goal        RMSE      SSE
            if (file.is_open()){
                if(simulate_touch)
                    file << test_name.c_str() <<"\t\t"<<steps<<"\t"<<req.var_desired.data<<"\t"<<RMSE<<"\t"<<SSE<<std::endl;
                else
                    file << test_name.c_str() <<"\t\t"<<steps<<"\t"<<req.var_desired.data<<"\tNaN\tNaN"<<std::endl; //for real demo we dont have a mesh, hence no comparison
                file.close();
            }
            else
                ROS_ERROR("[GaussianProcessNode::%s] Cannot open file %s for writing",__func__, file_path.c_str());
            steps = 0;
            if (simulate_touch){
                visualization_msgs::Marker mesh;
                mesh.header.frame_id = proc_frame;
                mesh.header.stamp = ros::Time();
                mesh.lifetime = ros::Duration(5.0);
                mesh.ns = "GroundTruth";
                mesh.id = 0;
                mesh.type = visualization_msgs::Marker::MESH_RESOURCE;
                mesh.action = visualization_msgs::Marker::ADD;
                mesh.scale.x = 1.0;
                mesh.scale.y = 1.0;
                mesh.scale.z = 1.0;
                std::string mesh_path ("package://asus_scanner_models/" + obj_name + "/" + obj_name + ".stl");
                mesh.mesh_resource = mesh_path.c_str();
                mesh.pose.position.x = 0.3;
                mesh.color.a=1.0;
                mesh.color.r=0.5;
                mesh.color.g=0.5;
                mesh.color.b=0.5;
                markers->markers.push_back(mesh);
            }
            return true;
        }
        ++steps;
        std_msgs::Header solution_header;
        solution_header.stamp = ros::Time::now();
        solution_header.frame_id = proc_frame;

        gp_regression::Path next_best_path;
        next_best_path.header = solution_header;
        for (size_t i=0; i<solution.size(); ++i)
        {
            // ToDO: solutionToPath(solution, path) function
            gp_atlas_rrt::Chart chart = atlas->getNode(solution[i]);
            Eigen::Vector3d point_eigen = chart.getCenter();
            Eigen::Vector3d normal_eigen = chart.getNormal();
            // modifies the point
            if (!simulate_touch){
                reMeanAndDenormalizeData(point_eigen);
                // normal does not need to be reMeanAndRenormalized for now

                //insert a geodesic intermediate point between nodes
                //Atlas generates always at least two nodes
                if (i>0){
                    gp_atlas_rrt::Chart prev_chart = atlas->getNode(solution[i-1]);
                    Eigen::Vector3d p1 = prev_chart.getCenter();
                    Eigen::Vector3d n1 = prev_chart.getNormal();
                    Eigen::Vector3d p2 = point_eigen;
                    Eigen::Vector3d n2 = normal_eigen;
                    reMeanAndDenormalizeData(p1);
                    Eigen::Vector3d n3 = n1.cross(n2);
                    Eigen::Vector3d Pmean;
                    Eigen::Vector3d Nmean;
                    if (n3.isZero(1e-5)){
                        //Charts are parallel
                        //get mean point from the two centers
                        Pmean << 0.5*(p1[0]+p2[0]), 0.5*(p1[1]+p2[1]), 0.5*(p1[2]+p2[2]);
                        Nmean = n1;
                    }
                    else{
                        n3.normalize();
                        //get coeffs of planes equation
                        double d1,d2,da,db, den;
                        d1 = - (n1.dot(p1));
                        d2 = - (n2.dot(p2));
                        da = - (n3.dot(p1));
                        db = - (n3.dot(p2));
                        den = n1.dot(n2.cross(n3));
                        Eigen::Vector3d Pa = ( - d1*n2.cross(n3) -d2*n3.cross(n1) - da*n1.cross(n2) )/den;
                        Eigen::Vector3d Pb = ( - d1*n2.cross(n3) -d2*n3.cross(n1) - db*n1.cross(n2) )/den;
                        Pmean << 0.5*(Pa[0]+Pb[0]), 0.5*(Pa[1]+Pb[1]), 0.5*(Pa[2]+Pb[2]);
                        Nmean = n1+n2;
                        Nmean.normalize();
                    }
                    geometry_msgs::PointStamped geodesic_msg;
                    geometry_msgs::Vector3Stamped geo_normal_msg;
                    geodesic_msg.point.x = Pmean(0);
                    geodesic_msg.point.y = Pmean(1);
                    geodesic_msg.point.z = Pmean(2);
                    geodesic_msg.header = solution_header;
                    geo_normal_msg.vector.x = Nmean(0);
                    geo_normal_msg.vector.y = Nmean(1);
                    geo_normal_msg.vector.z = Nmean(2);
                    geo_normal_msg.header = solution_header;

                    next_best_path.points.push_back( geodesic_msg );
                    next_best_path.directions.push_back( geo_normal_msg );
                }
            }
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
        if (simulate_touch) //synthetic touch simulation
        {
            //send normalized path
            synthTouch(next_best_path);
        }
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
    predicted_shape_.vertices.clear();
    predicted_shape_.triangles.clear();
    reg_.reset();
    obj_gp.reset();
    my_kernel.reset();
    atlas.reset();
    explorer.reset();
    solution.clear();
    markers.reset();
    steps = 0;
    //////
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
                std::vector<std::string> vst;
                split(vst, req.obj_pcd, boost::is_any_of("/."), boost::token_compress_on);
                obj_name = vst.at(vst.size()-2);
                std::string models_path (ros::package::getPath("asus_scanner_models"));
                boost::filesystem::path model_path (models_path + "/" + obj_name + "/" + obj_name + ".pcd");
                full_object = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
                full_object_real = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
                if (boost::filesystem::exists(model_path) && boost::filesystem::is_regular_file(model_path))
                {
                    if (pcl::io::loadPCDFile(model_path.c_str(), *full_object_real))
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
    model_ptr->header.frame_id=anchor;
    real_explicit_ptr->header.frame_id=proc_frame;
    colorThem(0,0,255, object_ptr);

    prepareExtData();
    deMeanAndNormalizeData( object_ptr, data_ptr_ );
    if (simulate_touch){
        pcl::PointCloud<pcl::PointXYZ> tmp;
        pcl::demeanPointCloud(*full_object_real, current_offset_, tmp);
        Eigen::Matrix4f t;
        t    << 1/current_scale_, 0, 0, 0,
                0, 1/current_scale_, 0, 0,
                0, 0, 1/current_scale_, 0,
                0, 0, 0,                1;
        pcl::transformPointCloud(tmp, *full_object, t);
        full_object->header.frame_id=anchor;
        kd_full.setInputCloud(full_object);
    }
    if (prepareData())
        if (computeGP()){
            //publish training set
            publishCloudModel();
            ros::spinOnce();
            //initialize objects involved
            markers = boost::make_shared<visualization_msgs::MarkerArray>();
            //show ground truth at start
            if (simulate_touch){
                visualization_msgs::Marker mesh;
                mesh.header.frame_id = proc_frame;
                mesh.header.stamp = ros::Time();
                mesh.lifetime = ros::Duration(5.0);
                mesh.ns = "GroundTruth";
                mesh.id = 0;
                mesh.type = visualization_msgs::Marker::MESH_RESOURCE;
                mesh.action = visualization_msgs::Marker::ADD;
                mesh.scale.x = 1.0;
                mesh.scale.y = 1.0;
                mesh.scale.z = 1.0;
                std::string mesh_path ("package://asus_scanner_models/" + obj_name + "/" + obj_name + ".stl");
                mesh.mesh_resource = mesh_path.c_str();
                mesh.pose.position.x = 0.3;
                mesh.color.a=1;
                mesh.color.r=0.5;
                mesh.color.g=0.5;
                mesh.color.b=0.5;
                markers->markers.push_back(mesh);
            }
            // marchingSampling(true, 0.06,0.02);
            //perform fake sampling
            fakeDeterministicSampling(true, 1.01, sample_res);
            computeOctomap();
            computePredictedShapeMsg();
            res.predicted_shape = predicted_shape_;
            return true;
        }
    return false;
}
bool GaussianProcessNode::cb_updateS(gp_regression::Update::Request &req, gp_regression::Update::Response &res)
{
    gp_regression::Path::Ptr msg = boost::make_shared<gp_regression::Path>();
    *msg = req.explored_points;
    cb_update(msg);
    res.predicted_shape = predicted_shape_;
    return true;
}

void GaussianProcessNode::cb_update(const gp_regression::Path::ConstPtr &msg)
{
    //assuming points in processing_frame
    if (!msg)
        return;
    if (msg->points.size() <= 0)
        return;
    Eigen::MatrixXd points;
    points.resize(msg->points.size(), 3);
    for (size_t i=0; i< msg->points.size(); ++i)
    {
        points(i,0) = msg->points[i].point.x;
        points(i,1) = msg->points[i].point.y;
        points(i,2) = msg->points[i].point.z;
        if (msg->isOnSurface[i].data){
            pcl::PointXYZRGB pt;
            pt.x = msg->points[i].point.x;
            pt.y = msg->points[i].point.y;
            pt.z = msg->points[i].point.z;
            colorIt(0,255,255, pt);
            object_ptr->push_back(pt);
        }
        else{
            //we need a new transform to compute externals
            continue;
        }
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
    last_touched[0] = data_ptr_->points[data_ptr_->size()-1].x;
    last_touched[1] = data_ptr_->points[data_ptr_->size()-1].y;
    last_touched[2] = data_ptr_->points[data_ptr_->size()-1].z;
    //now we can add the externals
    model_ptr->resize(ext_size);
    for (size_t i=0; i< msg->points.size(); ++i)
    {
        if (!msg->isOnSurface[i].data){
            Eigen::Vector3d point(
                    msg->points[i].point.x,
                    msg->points[i].point.y,
                    msg->points[i].point.z
                    );
            deMeanAndNormalizeData(point);
            pcl::PointXYZRGB pt;
            pt.x = point[0];
            pt.y = point[1];
            pt.z = point[2];
            double dist = msg->distances[i].data / current_scale_;
            //sanity check
            if (dist >1 || dist < 0)
                ROS_ERROR("[GaussianProcessNode::%s]\tNew point distance is too big...",__func__);
            ext_gp->coord_x.push_back(pt.x);
            ext_gp->coord_y.push_back(pt.y);
            ext_gp->coord_z.push_back(pt.z);
            ext_gp->label.push_back(dist);
            ext_gp->sigma2.push_back(sigma2);
            // add external point to rviz
            colorIt(100,0,50, pt);
            model_ptr->push_back(pt);
            ++ext_size;
        }
    }
    //create touch marker(s)
    for (size_t i=0; i<points.rows(); ++i){
        Eigen::Vector3d p;
        p << points(i,0), points(i,1), points(i,2);
        deMeanAndNormalizeData(p);
        points(i,0) = p[0];
        points(i,1) = p[1];
        points(i,2) = p[2];
    }
    //update full model
    if (simulate_touch){
        pcl::PointCloud<pcl::PointXYZ> tmp;
        pcl::demeanPointCloud(*full_object_real, current_offset_, tmp);
        Eigen::Matrix4f t;
        t    << 1/current_scale_, 0, 0, 0,
                0, 1/current_scale_, 0, 0,
                0, 0, 1/current_scale_, 0,
                0, 0, 0,                1;
        pcl::transformPointCloud(tmp, *full_object, t);
        full_object->header.frame_id=anchor;
        kd_full.setInputCloud(full_object);
    }
    //start recomputing GP
    prepareData();
    computeGP();
    //visualize training data
    publishCloudModel();
    ros::spinOnce();
    //initialize objects involved
    markers = boost::make_shared<visualization_msgs::MarkerArray>();
    // createTouchMarkers(points); //UGLY, no time to beautify it
    //perform fake sampling
    fakeDeterministicSampling(true, 1.01, sample_res);
    computeOctomap();
    computePredictedShapeMsg();
    return;
}
void
GaussianProcessNode::createTouchMarkers(const Eigen::MatrixXd &pts)
{
    if (pts.rows() <= 1)
        return;
    visualization_msgs::Marker lines;
    lines.header.frame_id = anchor;
    lines.header.stamp = ros::Time();
    lines.lifetime = ros::Duration(5.0);
    lines.ns = "last_touch";
    lines.id = 0;
    lines.type = visualization_msgs::Marker::LINE_STRIP;
    lines.action = visualization_msgs::Marker::ADD;
    lines.scale.x = 0.01;
    lines.color.a = 1.0;
    lines.color.r = 0.0;
    lines.color.g = 0.6;
    lines.color.b = 0.7;
    for (size_t i=0; i<pts.rows(); ++i)
    {
        geometry_msgs::Point p;
        p.x = pts(i,0);
        p.y = pts(i,1);
        p.z = pts(i,2);
        lines.points.push_back(p);
    }
    markers->markers.push_back(lines);
}

void GaussianProcessNode::prepareExtData()
{
    if (!model_ptr->empty())
        model_ptr->clear();
    ext_gp = std::make_shared<gp_regression::Data>();

    ext_size = 0;
    /*
     * ext_size = 1;
     * //      Internal Point
     * // add centroid as label -1
     * ext_gp->coord_x.push_back(0);
     * ext_gp->coord_y.push_back(0);
     * ext_gp->coord_z.push_back(0);
     * ext_gp->label.push_back(-1.0);
     * ext_gp->sigma2.push_back(sigma2);
     * // add internal point to rviz in cyan
     * pcl::PointXYZRGB cen;
     * cen.x = 0;
     * cen.y = 0;
     * cen.z = 0;
     * colorIt(255,255,0, cen);
     * model_ptr->push_back(cen);
     */

    //      External points
    // add points in a sphere around centroid with label 1
    // sphere bounds computation
    const int ang_div = 5; //divide 360° in ang_div pieces
    const int lin_div = 3; //divide diameter into lin_div pieces
    const double ang_step = M_PI * 2 / ang_div;
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

    cloud_gp = std::make_shared<gp_regression::Data>();

    /*****  Prepare the training data  *********************************************/

    //      Surface Points
    // add object points with label 0
    for(size_t i=0; i< data_ptr_->points.size(); ++i) {
        cloud_gp->coord_x.push_back(data_ptr_->points[i].x);
        cloud_gp->coord_y.push_back(data_ptr_->points[i].y);
        cloud_gp->coord_z.push_back(data_ptr_->points[i].z);
        cloud_gp->label.push_back(0.0);
        cloud_gp->sigma2.push_back(sigma2);
    }
    // add object points to rviz in blue
    // resize to ext_size first, so you wont lose external data, but overwrite
    // object data
    model_ptr->resize(ext_size);
    for (const auto& pt: data_ptr_->points)
        model_ptr->push_back(pt);
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
    // my_kernel = std::make_shared<gp_regression::ThinPlate>(out_sphere_rad * 2);
    my_kernel = std::make_shared<gp_regression::ThinPlate>(2.0);
    reg_->setCovFunction(my_kernel);
    const bool withoutNormals = false;
    reg_->create<withoutNormals>(data_gp, obj_gp);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count();
    ROS_INFO("[GaussianProcessNode::%s]\tRegressor and Model created using %ld training points. Total time consumed: %ld milliseconds.", __func__, cloud_gp->label.size(), elapsed );
    //make some adjustments to training set, if we are slowing down
    // if (elapsed > 600){
    //     sample_res = sample_res < 0.13 ? sample_res + 0.01 : 0.13;
    //     pcl::VoxelGrid<pcl::PointXYZRGB> vg;
    //     vg.setInputCloud(object_ptr);
    //     vg.setLeafSize(0.1, 0.1, 0.1);
    //     PtC tmp;
    //     vg.filter(tmp);
    //     pcl::copyPointCloud(tmp, *object_ptr);
    //     object_ptr->header.frame_id = proc_frame;
    // }
    start = true;
    return true;
}

//
bool GaussianProcessNode::startExploration(const float v_des)
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
    atlas->setVarianceTolGoal( v_des );
    //factor to control disc radius
    atlas->setVarRadiusFactor( 0.3 );
    //atlas is ready

    //setup explorer
    explorer = std::make_shared<gp_atlas_rrt::ExplorerMultiBranch>(nh, "explorer");
    explorer->setMarkers(markers, mtx_marks, anchor);
    explorer->setAtlas(atlas);
    explorer->setMaxNodes(300);
    explorer->setNoSampleMarkers(true);
    explorer->setBias(0.4); //probability of expanding on an old node
    //get a starting point from data cloud
    if (last_touched.isZero() || ignore_last_touched){
        int r_id = getRandIn(0, data_ptr_->points.size()-1 );
        Eigen::Vector3d root;
        root << data_ptr_->points[r_id].x,
             data_ptr_->points[r_id].y,
             data_ptr_->points[r_id].z;
        explorer->setStart(root);
    }
    else
        explorer->setStart(last_touched);
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
        if (!solution.empty())
            ROS_INFO("[GaussianProcessNode::%s]\tSolution Found", __func__);
    }
}

// for visualization purposes
void GaussianProcessNode::fakeDeterministicSampling(const bool first_time, const double scale, const double pass)
{
    auto begin_time = std::chrono::high_resolution_clock::now();
    if(!markers)
        return;

    visualization_msgs::Marker samples;
    samples.header.frame_id = anchor;
    samples.header.stamp = ros::Time();
    samples.lifetime = ros::Duration(5.0);
    samples.ns = "samples";
    samples.id = 0;
    samples.type = visualization_msgs::Marker::POINTS;
    samples.action = visualization_msgs::Marker::ADD;
    samples.scale.x = 0.025;
    samples.scale.y = 0.025;
    samples.scale.z = 0.025;
    markers->markers.push_back(samples);

    real_explicit_ptr = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    real_explicit_ptr->header.frame_id = proc_frame;
    predicted_shape_.triangles.clear();
    predicted_shape_.vertices.clear();

    size_t count(0);
    const auto total = std::floor( std::pow((2*scale+1)/pass, 3) );
    ROS_INFO("[GaussianProcessNode::%s]\tSampling %g grid points on GP ...",__func__, total);
    for (double x = -scale; x<= scale; x += pass)
    {
        std::vector<std::thread> threads;
        for (double y = -scale; y<= scale; y += pass)
        {
            for (double z = -scale; z<= scale; z += pass)
            {
                ++count;
                std::cout<<" -> "<<count<<"/"<<total<<"\r";
                threads.emplace_back(&GaussianProcessNode::samplePoint, this, x,y,z, std::ref(samples));
            }
        }
        for (auto &t: threads)
            t.join();
        //update visualization
        publishAtlas();
        ros::spinOnce();
    }
    std::cout<<std::endl;

    ROS_INFO("[GaussianProcessNode::%s]\tFound %ld points approximately on GP surface.",__func__,
            real_explicit_ptr->size());

    if (first_time){
        min_v = 10.0;
        max_v = 0.0;
        for (const auto &pt : real_explicit_ptr->points)
        {
            if (pt.intensity <= min_v)
                min_v = pt.intensity;
            if (pt.intensity >= max_v)
                max_v = pt.intensity;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end_time - begin_time).count();
    ROS_INFO("[GaussianProcessNode::%s]\tTotal time consumed: %ld seconds.", __func__, elapsed );
    // if (elapsed > 60)
    //     sample_res = sample_res < 0.1 ? sample_res + 0.01 : 0.1;
}
void
GaussianProcessNode::samplePoint(const double x, const double y, const double z, visualization_msgs::Marker &samp)
{
    gp_regression::Data::Ptr qq = std::make_shared<gp_regression::Data>();
    qq->coord_x.push_back(x);
    qq->coord_y.push_back(y);
    qq->coord_z.push_back(z);
    std::vector<double> ff,vv;
    reg_->evaluate(obj_gp, qq, ff, vv);
    if (std::abs(ff.at(0)) <= 0.01) {
        const double mid_v = ( min_v + max_v ) * 0.5;
        geometry_msgs::Point pt;
        std_msgs::ColorRGBA cl;
        pcl::PointXYZI pt_pcl;
        pt.x = x;
        pt.y = y;
        pt.z = z;
        cl.a = 1.0;
        cl.b = 0.0;
        cl.r = (vv[0]<mid_v) ? 1/(mid_v - min_v) * (vv[0] - min_v) : 1.0;
        cl.g = (vv[0]>mid_v) ? -1/(max_v - mid_v) * (vv[0] - mid_v) + 1 : 1.0;
        pt_pcl.x = x;
        pt_pcl.y = y;
        pt_pcl.z = z;
        //intensity is variance
        pt_pcl.intensity = vv[0];
        //locks
        std::lock_guard<std::mutex> lock (*mtx_marks);
        std::lock_guard<std::mutex> lk (mtx_samp);
        samp.points.push_back(pt);
        samp.colors.push_back(cl);
        real_explicit_ptr->push_back(pt_pcl);
        markers->markers[markers->markers.size()-1] = samp;
    }
}

void
GaussianProcessNode::marchingSampling(const bool first_time, const float leaf_size, const float leaf_pass)
{
    auto begin_time = std::chrono::high_resolution_clock::now();
    if(!markers)
        return;

    visualization_msgs::Marker samples;
    samples.header.frame_id = anchor;
    samples.header.stamp = ros::Time();
    samples.lifetime = ros::Duration(5.0);
    samples.ns = "samples";
    samples.id = 0;
    samples.type = visualization_msgs::Marker::POINTS;
    samples.action = visualization_msgs::Marker::ADD;
    samples.scale.x = 0.025;
    samples.scale.y = 0.025;
    samples.scale.z = 0.025;

    markers->markers.push_back(samples);
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
    real_explicit_ptr->header.frame_id = proc_frame;
    oct_cent = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    s_oct->setInputCloud(oct_cent);
    marchingCubes(*start, leaf_size, leaf_pass, samples); //process will block here until everything is done
    //remove duplicate points
    pcl::VoxelGrid<pcl::PointXYZI> vg;
    vg.setInputCloud(real_explicit_ptr);
    vg.setLeafSize(leaf_pass, leaf_pass, leaf_pass);
    pcl::PointCloud<pcl::PointXYZI> tmp;
    vg.filter(tmp);
    pcl::copyPointCloud(tmp, *real_explicit_ptr);
    real_explicit_ptr->header.frame_id = proc_frame;
    ROS_INFO("[GaussianProcessNode::%s]\tFound %ld points approximately on GP surface...",__func__,
            real_explicit_ptr->size());
    //determine min e max variance
    if (first_time){
        min_v = 10.0;
        max_v = 0.0;
        //only do this if it is the first time, so assuming global variance will lower
        //with each update, you'll see points progressively turn green
        for (const auto& pt: real_explicit_ptr->points)
        {
            if (pt.intensity <= min_v)
                min_v = pt.intensity;
            if (pt.intensity >= max_v)
                max_v = pt.intensity;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end_time - begin_time).count();
    ROS_INFO("[GaussianProcessNode::%s]\tTotal time consumed: %ld seconds.", __func__, elapsed );
}

//Doesnt fully work!! looks like diagonal adjacency is also needed,
//It does work however for small leaf sizes...
void
GaussianProcessNode::marchingCubes(pcl::PointXYZ start, const float leaf, const float pass, visualization_msgs::Marker &samp)
{
    {//protected section, mark this cube as already explored
        std::lock_guard<std::mutex> lock (mtx_samp);
        s_oct->addPointToCloud(start, oct_cent);
    }//end of protected section
    const size_t steps = std::round(leaf/pass);
    const double mid_v = ( min_v + max_v ) * 0.5;
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
                    geometry_msgs::Point p;
                    std_msgs::ColorRGBA cl;
                    p.x = x;
                    p.y = y;
                    p.z = z;
                    cl.a = 1.0;
                    cl.b = 0.0;
                    cl.r = (pt.intensity<mid_v) ? 1/(mid_v - min_v) * (pt.intensity - min_v) : 1.0;
                    cl.g = (pt.intensity>mid_v) ? -1/(max_v - mid_v) * (pt.intensity - mid_v) + 1 : 1.0;
                    {//protected section
                        std::lock_guard<std::mutex> lock (mtx_samp);
                        std::lock_guard<std::mutex> lock2 (*mtx_marks);
                        real_explicit_ptr->push_back(pt);
                        samp.points.push_back(p);
                        samp.colors.push_back(cl);
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
    {
        std::lock_guard<std::mutex> lock (*mtx_marks);
        markers->markers[markers->markers.size()-1] = samp;
        //update visualization
        publishAtlas();
    }
    ros::spinOnce();
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
                threads.emplace_back(&GaussianProcessNode::marchingCubes, this, pt, leaf, pass, std::ref(samp));
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
GaussianProcessNode::computePredictedShapeMsg()
{

    pcl::PointCloud<pcl::PointXYZI> real_explicit;
    real_explicit.header = real_explicit_ptr->header;
    reMeanAndDenormalizeData(real_explicit_ptr, real_explicit);

    // Followed this tutorial to compute the mesh: http://www.pointclouds.org/assets/icra2012/surface.pdf
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    const pcl::PointCloud<pcl::PointXYZ>::Ptr my_cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ> >();
    my_cloud->width = real_explicit.width;
    my_cloud->height = real_explicit.height;
    for( auto pt: real_explicit.points)
    {
        pcl::PointXYZ p;
        p.x = pt.x;
        p.y = pt.y;
        p.z = pt.z;

        my_cloud->points.push_back( p );
    }
    tree->setInputCloud( my_cloud );
    n.setInputCloud( my_cloud );
    n.setSearchMethod (tree);
    n.setKSearch (20);
    n.useSensorOriginAsViewPoint();
    n.compute(*normals);

    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields ( *my_cloud, *normals, *cloud_with_normals);

    pcl::search::KdTree<pcl::PointNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointNormal>);
    tree2->setInputCloud (cloud_with_normals);

    pcl::PolygonMesh triangles;

    // pcl::Poisson<pcl::PointNormal> poisson;
    // poisson.setDepth(9);
    // poisson.setInputCloud(cloud_with_normals);
    // poisson.reconstruct (triangles);
    // pcl::MarchingCubesHoppe<pcl::PointNormal> mc;
    // mc.setInputCloud(cloud_with_normals);
    // mc.setGridResolution(0.001, 0.001, 0.001);
    // mc.setIsoLevel(0.9);
    // mc.setPercentageExtendGrid(0.8);
    // mc.reconstruct(triangles);
    pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
    gp3.setInputCloud(cloud_with_normals);
    gp3.setMaximumNearestNeighbors(100);
    gp3.setSearchRadius(0.05);
    gp3.setMu(2.5);
    gp3.setMaximumSurfaceAngle(M_PI/4); // 45 degrees
    gp3.setMinimumAngle(M_PI/18); // 10 degrees
    gp3.setMaximumAngle(2*M_PI/3); // 120 degrees
    if (gp3.getNormalConsistency())
        gp3.setNormalConsistency(false);
    gp3.reconstruct(triangles);

    // pcl::io::savePLYFile("/home/tabjones/Desktop/prova.ply", triangles);

    // convert from pcl::PolygonMesh  into shape_msgs::Mesh
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromPCLPointCloud2(triangles.cloud, cloud);
    // first the vertices
    for( auto pct: cloud.points )
    {
        geometry_msgs::Point p;
        pcl::PointXYZ pt = pct;
        p.x = pt.x;
        p.y = pt.y;
        p.z = pt.z;
        predicted_shape_.vertices.push_back(p);
    }
    // and then the faces
    // ToDO: check that the normal is pointing outwards
    for(int i = 0; i < triangles.polygons.size(); ++i)
    {
        pcl::Vertices v = triangles.polygons.at(i);
        shape_msgs::MeshTriangle t;
        t.vertex_indices.at(0) = v.vertices.at( 0 );
        t.vertex_indices.at(1) = v.vertices.at( 1 );
        t.vertex_indices.at(2) = v.vertices.at( 2 );
        predicted_shape_.triangles.push_back( t );
    }
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
GaussianProcessNode::synthTouch(const gp_regression::Path &sol)
{
    if (synth_type == 0){
        //touching at random
        int id = getRandIn(0, real_explicit_ptr->size()-1);
        Eigen::Vector3d p;
        p[0] = real_explicit_ptr->points.at(id).x;
        p[1] = real_explicit_ptr->points.at(id).y;
        p[2] = real_explicit_ptr->points.at(id).z;
        Eigen::Vector3d n;
        gp_regression::Data::Ptr qq = std::make_shared<gp_regression::Data>();
        qq->coord_x.push_back(p[0]);
        qq->coord_y.push_back(p[1]);
        qq->coord_z.push_back(p[2]);
        std::vector<double> ff,vv;
        Eigen::MatrixXd G;
        reg_->evaluate(obj_gp, qq, ff, vv, G);
        if (!G.row(0).isMuchSmallerThan(1e3, 1e-1) || G.row(0).isZero(1e-5)){
            ROS_WARN("[GaussianProcessNode::%s]\tGradien is wrong ignoring it.",__func__);
            n = Eigen::Vector3d::UnitX();
        }
        else{
            n = G.row(0);
            n.normalize();
        }
        gp_regression::Path::Ptr touch = boost::make_shared<gp_regression::Path>();
        raycast(p,n, *touch);
        cb_update(touch);
    }
    else if(synth_type == 1){
        //touch at solution point
        //normal is facing outside
        Eigen::Vector3d p;
        p[0] = sol.points[0].point.x;
        p[1] = sol.points[0].point.y;
        p[2] = sol.points[0].point.z;
        Eigen::Vector3d n;
        n[0] = sol.directions[0].vector.x;
        n[1] = sol.directions[0].vector.y;
        n[2] = sol.directions[0].vector.z;
        gp_regression::Path::Ptr touch = boost::make_shared<gp_regression::Path>();
        raycast(p,n, *touch);
        cb_update(touch);
    }
    else if(synth_type ==2){
        //sliding path
        gp_regression::Path::Ptr touch = boost::make_shared<gp_regression::Path>();
        for (size_t i = sol.points.size()-1; i>=0; --i)
        {
            int steps = 5;
            //traverse in reversal so we start from root
            Eigen::Vector3d p;
            p[0] = sol.points[i].point.x;
            p[1] = sol.points[i].point.y;
            p[2] = sol.points[i].point.z;
            Eigen::Vector3d n;
            n[0] = sol.directions[i].vector.x;
            n[1] = sol.directions[i].vector.y;
            n[2] = sol.directions[i].vector.z;
            Eigen::Vector3d start = raycast(p,n, *touch);
            if (i==0)
                break;
            Eigen::Vector3d end; //end is next point in path
            end[0] = sol.points[i-1].point.x;
            end[1] = sol.points[i-1].point.y;
            end[2] = sol.points[i-1].point.z;
            int j(0);
            while (j<steps-1)
            {
                Eigen::Vector3d dir = end - start;
                dir.normalize();
                double dist = L2(start, end);
                double step_size = dist/(steps-j);
                start += dir*step_size;
                //update the normal by asking gp
                gp_regression::Data::Ptr qq = std::make_shared<gp_regression::Data>();
                qq->coord_x.push_back(start[0]);
                qq->coord_y.push_back(start[1]);
                qq->coord_z.push_back(start[2]);
                std::vector<double> ff,vv;
                Eigen::MatrixXd G;
                reg_->evaluate(obj_gp, qq, ff, vv, G);
                if (!G.row(0).isMuchSmallerThan(1e3, 1e-1) || G.row(0).isZero(1e-5)){
                    ROS_WARN("[GaussianProcessNode::%s]\tGradien is wrong ignoring it.",__func__);
                }
                else{
                    n = G.row(0);
                    n.normalize();
                }
                start = raycast(start, n, *touch, true);
                ++j;
            }
        }
        cb_update(touch);
    }
    else
        ROS_ERROR("[GaussianProcessNode::%s]\tTouch type (%d) not implemented",__func__, synth_type);
}
Eigen::Vector3d //return normalized point touched
GaussianProcessNode::raycast(Eigen::Vector3d &point, const Eigen::Vector3d &normal, gp_regression::Path &touched, bool no_external)
{
    //move away and start raycasting
    point += normal*0.7;
    std::vector<int> k_id;
    std::vector<float> k_dist;
    //move along direction
    size_t max_steps(50);
    size_t count(0);
    for (size_t i=0; i<max_steps; ++i)
    {
        pcl::PointXYZ pt;
        pt.x = point[0];
        pt.y = point[1];
        pt.z = point[2];
        if (kd_full.radiusSearch(pt, 0.03, k_id, k_dist, 1) > 0){
            //we intersected the object, aka touch
            Eigen::Vector3d norm_p(
                    full_object->points[k_id[0]].x,
                    full_object->points[k_id[0]].y,
                    full_object->points[k_id[0]].z
                    );
            Eigen::Vector3d unnorm_p (norm_p);
            reMeanAndDenormalizeData(unnorm_p);
            geometry_msgs::PointStamped p;
            p.point.x = unnorm_p[0];
            p.point.y = unnorm_p[1];
            p.point.z = unnorm_p[2];
            touched.points.push_back(p);
            std_msgs::Bool iof;
            iof.data = true;
            std_msgs::Float32 d;
            d.data = 0.0;
            touched.isOnSurface.push_back(iof);
            touched.distances.push_back(d);
            ROS_INFO("[GaussianProcessNode::%s]\tTouched the object!!",__func__);
            return norm_p;
        }
        else{ //no touch, external point
            if (no_external){
                point -= (normal*0.03);
                continue;
            }
            if (count==0){
                //add this point
                k_id.resize(1);
                k_dist.resize(1);
                kd_full.nearestKSearch(pt, 1, k_id, k_dist);
                float dist = std::sqrt(k_dist[0]);
                if (dist< 0.3 || dist > 1.0){
                    ++count;
                    point -= (normal*0.03);
                    continue;
                }
                Eigen::Vector3d unnorm_p(point);
                reMeanAndDenormalizeData(unnorm_p);
                geometry_msgs::PointStamped p;
                p.point.x = unnorm_p[0];
                p.point.y = unnorm_p[1];
                p.point.z = unnorm_p[2];
                touched.points.push_back(p);
                std_msgs::Bool iof;
                iof.data = false;
                touched.isOnSurface.push_back(iof);
                std_msgs::Float32 d;
                d.data = dist*current_scale_;
                touched.distances.push_back(d);
                count =0;
            }
            count = count>=20 ? 0 : ++count;
        }
        point -= (normal*0.03);
    }
    return Eigen::Vector3d::Zero();
}

void GaussianProcessNode::automatedSynthTouch()
{
    if (start && (steps>0) && simulate_touch){
        gp_regression::GetNextBestPathRequest req;
        gp_regression::GetNextBestPathResponse res;
        req.var_desired.data = synth_var_goal;
        cb_get_next_best_path(req,res);
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
        node.automatedSynthTouch();
        rate.sleep();
    }
    //someone killed us :(
    return 0;
}
