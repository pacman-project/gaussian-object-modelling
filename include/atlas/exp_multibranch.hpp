#ifndef GP_EXPLORER_MULTIBRANCH_HPP_
#define GP_EXPLORER_MULTIBRANCH_HPP_

#include <atlas/exp_single_path.hpp>

namespace gp_atlas_rrt
{

class ExplorerMultiBranch : public ExplorerSinglePath
{
    public:
    typedef std::shared_ptr<ExplorerMultiBranch> Ptr;
    typedef std::shared_ptr<const ExplorerMultiBranch> ConstPtr;

    ExplorerMultiBranch()=delete;
    ExplorerMultiBranch(const ros::NodeHandle n, const std::string ns):
        ExplorerSinglePath(n,ns), bias(0.4)
    {
    }
    virtual ~ExplorerMultiBranch(){}

    virtual void setAtlas(const std::shared_ptr<AtlasCollision> &a)
    {
        atlas = a;
    }

    virtual inline void setBias(double b)
    {
        if (b>=0.0 && b<=1.0)
            bias = b;
    }
    /**
     * \brief exploration step, must also callAvailable for ros
     * should also loop on is_running condition.
     */
    virtual void explore()
    {
        if (!atlas){
            ROS_ERROR("[ExplorerMultibranch::%s]\tAtlas not set, set it first",__func__);
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
                ROS_INFO("[ExplorerMultibranch::%s]\tSolution Found!",__func__);
                return;
            }
            const double dice = getRandIn(0.0, 1.0, true);
            if (dice < bias){
                //we extend from a random node which is not the current
                int id(parent);
                if (atlas->countNodes() > 1)
                    while (id == parent)
                        id = getRandIn(0, atlas->countNodes()-1);
                parent = id;
            }
            exp_step(parent);
            //ros stuff
            rate.sleep();
            cb_queue->callAvailable();
        }
        if (atlas->countNodes() >= max_nodes){
            ROS_INFO("[ExplorerMultibranch::%s]\tMax number of nodes reached, returning last created node as solution",__func__);
            solution = getPathToRoot(parent);
            highlightSolution(solution);
        }
        std::lock_guard<std::mutex> lock(*mtx_ptr);
        is_running = false;
    }
    virtual void exp_step(std::size_t &node)
    {
        Eigen::Vector3d next_point = atlas->getNextState(node);
        if (next_point.isZero())
            return;
        createSamplesMarker(atlas->getNode(node), next_point);
        std::size_t child = atlas->createNode(next_point);
        connect(child, node);
        createNodeMarker(atlas->getNode(child));
        createBranchMarker(atlas->getNode(child), atlas->getNode(node));
        node = child;
    }


    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    protected:
    //atlas pointer
    std::shared_ptr<AtlasCollision> atlas;
    double bias; //probability to chose a random node
                 //to extend instead of last one. (should be in [0,1])
};
}

#endif
