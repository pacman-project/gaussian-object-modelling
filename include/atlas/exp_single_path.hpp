#ifndef GP_EXPLORER_SINGLE_PATH_HPP_
#define GP_EXPLORER_SINGLE_PATH_HPP_

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
    virtual ~ExplorerSinglePath(){}

    /*
     * \brief set Atlas model
     */
    virtual void setAtlas(const std::shared_ptr<AtlasVariance> &a)
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
     * \brief exploration loop, must also callAvailable for ros
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
            exp_step(parent);
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
     * \brief single exploration step
     */
    virtual void exp_step(std::size_t &node)
    {
        Eigen::Vector3d next_point = atlas->getNextState(node);
        createSamplesMarker(atlas->getNode(node), next_point);
        std::size_t child = atlas->createNode(next_point);
        connect(child, node);
        createNodeMarker(atlas->getNode(child));
        createBranchMarker(atlas->getNode(child), atlas->getNode(node));
        node = child;
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

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    protected:
    //atlas pointer
    std::shared_ptr<AtlasVariance> atlas;
    //starting point
    Eigen::Vector3d start_point;
    //termination on max num nodes
    std::size_t max_nodes;
    //solution path
    std::vector<std::size_t> solution;
};
}

#endif
