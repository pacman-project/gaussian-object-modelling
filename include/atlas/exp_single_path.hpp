#ifndef GP_EXPLORER_SINGLE_PATH_HPP_
#define GP_EXPLORER_SINGLE_PATH_HPP_

#include <atlas/atlas_variance.hpp>
#include <atlas/explorer.hpp>

namespace gp_atlas_rrt
{

class ExplorerSinglePath : public ExplorerBase
{
    public:
    typedef std::shared_ptr<ExplorerSinglePath> Ptr;
    typedef std::shared_ptr<const ExplorerSinglePath> ConstPtr;

    ExplorerSinglePath()=delete;
    ExplorerSinglePath(const ros::NodeHandle n, const std::string ns):
        ExplorerBase(n,ns)
    {
    }

    /*
     * \brief set Atlas model
     */
    virtual void setAtlas(const AtlasVariance::Ptr &a)
    {
        atlas = a;
    }
    /*
     * \brief Set start point (mandatory call)
     */
    virtual void setStart(const Eigen::Vector3d &sp)
    {
        start_point = sp;
    }
    virtual void setMaxNodes(const std::size_t &m)
    {
        max_nodes = m;
    }
    /**
     * \brief exploration step, must also callAvailable for ros
     * should also loop on is_running condition.
     */
    virtual void explore()
    {
        if (!atlas){
            ROS_ERROR("[ExplorerSinglePath::%s]\tAtlas not set, set it first",__func__);
            return;
        }
        std::size_t parent = atlas->createNode(start_point);
        createNodeMarker(atlas->getNode(parent));
        ros::Rate rate(50);
        while (atlas->countNodes() < max_nodes && !hasSolution())
        {
            if (atlas->isSolution(parent))
            {
                solution = getPathToRoot(parent);
                highlightSolution(solution);
                std::lock_guard<std::mutex> lock(*mtx_ptr);
                is_running = false;
                ROS_INFO("[ExplorerSinglePath::%s]\tSolution Found!",__func__);
                return;
            }
            Eigen::Vector3d next_point = atlas->getNextState(parent);
            createSamplesMarker(atlas->getNode(parent), next_point);
            std::size_t child = atlas->createNode(next_point);
            connect(child, parent);
            createNodeMarker(atlas->getNode(child));
            createBranchMarker(atlas->getNode(child), atlas->getNode(parent));
            parent = child;
            //ros stuff
            rate.sleep();
            cb_queue->callAvailable();
        }
        if (atlas->countNodes() >= max_nodes){
            ROS_INFO("[ExplorerSinglePath::%s]\tMax number of nodes reached, returning last created node as solution",__func__);
            solution = getPathToRoot(parent);
            highlightSolution(solution);
        }
        std::lock_guard<std::mutex> lock(*mtx_ptr);
        is_running = false;
    }

    /**
     * \brief check if we have a solution, this works if explorer was started
     */
    virtual inline bool hasSolution() const
    {
        //this func should lock cause also main thread is checking it
        std::lock_guard<std::mutex> lock (*mtx_ptr);
        return (!is_running);
    }
    /**
     * \brief get the solution
     */
    virtual inline std::vector<std::size_t> getSolution() const
    {
        return solution;
    }

    virtual void createSamplesMarker(const Chart &c, const Eigen::Vector3d &projected)
    {
        for (size_t i=0; i<c.samples.rows(); ++i)
        {
            visualization_msgs::Marker samp;
            samp.header.frame_id = "update_me_before_publishing!";
            samp.header.stamp = ros::Time::now();
            samp.lifetime = ros::Duration(0.5);
            samp.ns = "Atlas Node(" + std::to_string(c.getId()) + ") Samples";
            samp.id = i;
            samp.type = visualization_msgs::Marker::SPHERE;
            samp.action = visualization_msgs::Marker::ADD;
            samp.scale.x = 0.01;
            samp.scale.y = 0.01;
            samp.scale.z = 0.01;
            samp.color.a = 0.5;
            if (i==0){
                //is winner
                samp.color.r = 0.9;
                samp.color.b = 0.0;
                samp.color.g = 0.9;
            }
            else{
                samp.color.r = 0.0;
                samp.color.b = 0.8;
                samp.color.g = 0.9;
            }
            samp.pose.position.x = c.samples(i,0);
            samp.pose.position.y = c.samples(i,1);
            samp.pose.position.z = c.samples(i,2);
            std::lock_guard<std::mutex> guard(*mtx_ptr);
            markers->markers.push_back(samp);
        }
        geometry_msgs::Point e;
        geometry_msgs::Point s;

        visualization_msgs::Marker proj;
        proj.header.frame_id = "update_me_before_publishing!";
        proj.header.stamp = ros::Time();
        proj.lifetime = ros::Duration(0.5);
        proj.ns = "Atlas Node(" + std::to_string(c.getId()) + ") Samples";
        proj.id = c.samples.rows();
        proj.type = visualization_msgs::Marker::ARROW;
        proj.action = visualization_msgs::Marker::ADD;
        s.x = c.samples(0,0);
        s.y = c.samples(0,1);
        s.z = c.samples(0,2);
        proj.points.push_back(s);
        e.x = projected[0];
        e.y = projected[1];
        e.z = projected[2];
        proj.points.push_back(e);
        proj.scale.x = 0.001;
        proj.scale.y = 0.01;
        proj.scale.z = 0.01;
        proj.color.a = 0.4;
        proj.color.r = 0.0;
        proj.color.g = 0.2;
        proj.color.b = 0.8;
        std::lock_guard<std::mutex> guard(*mtx_ptr);
        markers->markers.push_back(proj);
    }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    protected:
    //starting point
    Eigen::Vector3d start_point;
    //termination on max num nodes
    std::size_t max_nodes;
    //solution path
    std::vector<std::size_t> solution;
};
}

#endif
