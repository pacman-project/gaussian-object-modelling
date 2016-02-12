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
        while (atlas->countNodes() < max_nodes && is_running)
        {
            if (atlas->isSolution(parent))
            {
                solution = getPathToRoot(parent);
                highlightSolution(solution);
                is_running = false;
                ROS_INFO("[ExplorerSinglePath::%s]\tSolution Found!",__func__);
                return;
            }
            Eigen::Vector3d next_point = atlas->getNextState(parent);
            std::cout<<"nextstate\n";
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
            is_running = false;
    }

    /**
     * \brief check if we have a solution, this works if explorer was started
     */
    virtual inline bool hasSolution() const
    {
        return (!is_running);
    }
    /**
     * \brief get the solution
     */
    virtual inline std::vector<std::size_t> getSolution() const
    {
        return solution;
    }

    protected:
    AtlasVariance::Ptr atlas;
    //starting point
    Eigen::Vector3d start_point;
    //termination on max num nodes
    std::size_t max_nodes;
    //solution path
    std::vector<std::size_t> solution;
};
}

#endif
