#ifndef GP_EXPLORER_HPP_
#define GP_EXPLORER_HPP_

#include <memory>
#include <thread>
#include <mutex>
#include <unordered_map>
#include <iostream>
#include <Eigen/Dense>
#include <gp_regression/gp_regression_exception.h>
#include <random_generation.hpp>

#include <ros/ros.h>
#include <ros/console.h>
#include <ros/spinner.h>
#include <ros/callback_queue.h>
#include <ros/callback_queue_interface.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <gp_regression/StopExploration.h>

#include <atlas/atlas.hpp>

namespace gp_atlas_rrt
{

class ExplorerBase
{
    public:
    ExplorerBase()=delete;
    ExplorerBase(const ros::NodeHandle n, const std::string ns):
        is_running(false), name(ns)
    {
        father_nh = std::make_shared<ros::NodeHandle>(n);
    }
    /**
     * \brief exploration step, must also callAvailable for ros
     * should also loop on is_running condition
     */
    virtual void explore()=0;

    /**
     * \brief start the exploration, asinchronously
     */
    virtual void startExploration ()
    {
        if (is_running){
            ROS_WARN("[ExplorerBase::%s]\tExplorer is already running.",__func__);
            return;
        }
        //spawn the nodehandle
        nh = std::make_shared<ros::NodeHandle>(*father_nh, name);
        cb_queue = std::make_shared<ros::CallbackQueue>();
        nh->setCallbackQueue(&(*cb_queue));
        srv_stop = nh->advertiseService("stop", &ExplorerBase::cb_stop, this);
        //start the worker thread
        is_running = true;
        worker = std::thread(&ExplorerBase::explore, this);
    }
    /**
     * \brief stop exploration (and thread)
     */
    virtual void stopExploration()
    {
        is_running = false;
        worker.join();
        cb_queue.reset();
        nh.reset();
    }
    /**
     * \brief service stop callback
     */
    virtual void cb_stop()
    {
        stopExploration();
    }
    /**
     * \brief set a marker array to update with visualization during exploration
     */
    virtual void setMarkers(const visualization_msgs::MarkerArrayPtr &mp,
                            const std::shared_ptr<std::mutex> &array_guard)
    {
        markers = mp;
        mtx_ptr = array_guard;
    }

    /**
     * \brief create a marker from a chart and stores it into markers array
     */
    virtual void createNodeMarker(const Chart &c)
    {
        if (!markers){
            ROS_WARN_THROTTLE(30,"[ExplorerBase::%s]\tNo marker array to update is provided, visualization is disabled.",__func__);
            return;
        }
        //frame id is not set, cause we can't access it here. rosNode should update
        //all markers frame id before publishing them.
        visualization_msgs::Marker disc;
        disc.header.frame_id = "update_me_before_publishing!";
        disc.header.stamp = ros::Time::now();
        disc.lifetime = ros::Duration(0);
        disc.ns = "Atlas Nodes";
        disc.id = c.getId();
        disc.type = visualization_msgs::Marker::CYLINDER;
        disc.action = visualization_msgs::Marker::ADD;
        disc.scale.x = c.getRadius();
        disc.scale.y = c.getRadius();
        disc.scale.z = 0.001;
        disc.color.a = 0.4;
        disc.color.r = 0.545;
        disc.color.b = 0.843;
        disc.color.g = 0.392;
        Eigen::Matrix3d rot;
        rot.col(0) =  c.getTanBasisOne();
        rot.col(1) =  c.getTanBasisTwo();
        rot.col(2) =  c.getNormal();
        Eigen::Quaterniond q(rot);
        q.normalize();
        disc.pose.orientation.x = q.x();
        disc.pose.orientation.y = q.y();
        disc.pose.orientation.z = q.z();
        disc.pose.orientation.w = q.w();
        disc.pose.position.x = c.getCenter()[0];
        disc.pose.position.y = c.getCenter()[1];
        disc.pose.position.z = c.getCenter()[2];
        std::lock_guard<std::mutex> guard(*mtx_ptr);
        markers->markers.push_back(disc);
    }
    /**
     * \brief create a branch marker between two nodes and store it into marker array
     */
    virtual void createBranchMarker(const Chart &child, const Chart &parent)
    {
        if (!markers){
            ROS_WARN_THROTTLE(30,"[ExplorerBase::%s]\tNo marker array to update is provided, visualization is disabled.",__func__);
            return;
        }
        geometry_msgs::Point e;
        geometry_msgs::Point s;

        visualization_msgs::Marker branch;
        branch.header.frame_id = "update_me_before_publishing!";
        branch.header.stamp = ros::Time();
        branch.lifetime = ros::Duration(0);
        branch.ns = "Atlas Branches";
        //need to know the branch id, too bad branches don't have it.
        //Lets use Cantor pairing function: 0.5(a+b)(a+b+1)+b
        branch.id = 0.5*(child.getId() + parent.getId())*(child.getId()+parent.getId()+1) + parent.getId();
        branch.type = visualization_msgs::Marker::LINE_STRIP;
        branch.action = visualization_msgs::Marker::ADD;
        s.x = child.getCenter()[0];
        s.y = child.getCenter()[1];
        s.z = child.getCenter()[2];
        branch.points.push_back(s);
        e.x = parent.getCenter()[0];
        e.y = parent.getCenter()[1];
        e.z = parent.getCenter()[2];
        branch.points.push_back(e);
        branch.scale.x = 0.002;
        branch.color.a = 0.4;
        branch.color.r = 0.129;
        branch.color.g = 0.929;
        branch.color.b = 0.6;
        std::lock_guard<std::mutex> guard(*mtx_ptr);
        markers->markers.push_back(branch);
    }
    /**
     * \brief highlight solution path
     */
    virtual void highlightSolution(const std::vector<std::size_t> &path)
    {
        if (!markers){
            ROS_WARN_THROTTLE(30,"[ExplorerBase::%s]\tNo marker array to update is provided, visualization is disabled.",__func__);
            return;
        }
        //ids of branches to color
        std::vector<std::size_t> cons;
        cons.resize(path.size() -1);
        for (std::size_t i=0; i<cons.size(); ++i)
            cons.at(i) = 0.5*(path.at(i) + path.at(i+1))*(path.at(i) + path.at(i+1) + 1) + path.at(i+1);
        std::lock_guard<std::mutex> guard(*mtx_ptr);
        for (std::size_t i =0; i< markers->markers.size(); ++i)
        {
            if (markers->markers.at(i).ns.compare("Atlas Nodes")==0)
                for (const auto& id: path)
                    if (markers->markers.at(i).id == id){
                        markers->markers.at(i).color.a = 1.0;
                        markers->markers.at(i).color.r = 0.9;
                        markers->markers.at(i).color.b = 0.3;
                        markers->markers.at(i).color.g = 0.0;
                    }
            if (markers->markers.at(i).ns.compare("Atlas Branches")==0)
                for (const auto& id: cons)
                    if (markers->markers.at(i).id == id){
                        markers->markers.at(i).color.a = 1.0;
                        markers->markers.at(i).color.r = 0.8;
                        markers->markers.at(i).color.b = 0.2;
                        markers->markers.at(i).color.g = 0.0;
                    }
        }
    }
    /**
     * \brief Recurse connections from give child node id to root (root is always 0 id)
     * \return ids of traversed nodes, including starting one and root.
     */
    virtual std::vector<std::size_t> getPathToRoot (const std::size_t &id) const
    {
        std::vector<std::size_t> path;
        path.push_back(id);
        std::size_t parent = id;
        while (parent != 0)
        {
            if (branches.count(parent) == 1){
                auto search = branches.find(parent);
                if (search == branches.end())
                    throw gp_regression::GPRegressionException("Unconnected Child Node");
                parent = search->second;
                path.push_back(parent);
            }
            else{
                ROS_ERROR("multiple parents are not supported yet!!");
            }
        }
        return path;
    }

    /**
     * \brief Connect two nodes identified by id, first passed id is child node
     */
    virtual inline void connect(const std::size_t child, const std::size_t parent)
    {
        branches.emplace(std::make_pair(child,parent));
    }

    protected:
    ///Connection map  key=childNodes, value=oneOfItsParents,
    //multiple equivalent keys can be present, thus givin a child more than one parent
    //this is not yet supported
    std::unordered_multimap<std::size_t, std::size_t> branches;
    ///explorer is running?
    bool is_running;
    ///father and child nodehandle
    std::shared_ptr<ros::NodeHandle> father_nh;
    std::shared_ptr<ros::NodeHandle> nh;
    ///namespace of child
    std::string name;
    ///child custom callback queue
    std::shared_ptr<ros::CallbackQueue> cb_queue;
    ///stop service server
    ros::ServiceServer srv_stop;
    ///worker thread
    std::thread worker;
    ///marker array pointer to update during exploration
    visualization_msgs::MarkerArrayPtr markers;
    //and its relative mutex for thread safety
    std::shared_ptr<std::mutex> mtx_ptr;

};
}

#endif
