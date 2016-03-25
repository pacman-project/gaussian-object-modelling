#ifndef GP_ATLAS_COLLISION_HPP_
#define GP_ATLAS_COLLISION_HPP_

#include <atlas/atlas_variance.hpp>

namespace gp_atlas_rrt
{

class AtlasCollision : public AtlasVariance
{
    public:
    typedef std::shared_ptr<AtlasCollision> Ptr;
    typedef std::shared_ptr<const AtlasCollision> ConstPtr;

    AtlasCollision()=delete;
    AtlasCollision(const gp_regression::Model::Ptr &gp, const gp_regression::ThinPlateRegressor::Ptr &reg):
        AtlasVariance(gp,reg)
    {}
    virtual ~AtlasCollision(){}


    virtual Eigen::Vector3d getNextState(const std::size_t& id)
    {
        if (!gp_reg)
            throw gp_regression::GPRegressionException("Empty Regressor pointer");
        if (id >= nodes.size())
            throw gp_regression::GPRegressionException("Out of Range node id");
        if (!nodes[id].expandable)
            return Eigen::Vector3d::Zero();
        if (nodes.at(id).samp_chosen < 0)
            sampleOnChart(nodes.at(id));
        Eigen::Vector3d chosen;
        chosen.setZero();
        nodes[id].expandable = false;
        for (size_t i=0; i<nodes.at(id).vars_ids.size(); ++i)
        {
            std::size_t s_id = nodes.at(id).vars_ids[i].second;
            if (nodes.at(id).samp_chosen == s_id)
                continue;
            if (!isInCollision(nodes.at(id).samples.row(s_id), id)){
                nodes.at(id).samp_chosen = s_id;
                chosen = nodes.at(id).samples.row(s_id);
                nodes[id].expandable = true;
                break;
            }
        }
        if (!nodes[id].expandable){
            std::cout<<"[Atlas::getNextState] No viable extending direction found, cannot extend the node"<<std::endl;
            --num_expandables;
            return Eigen::Vector3d::Zero();
        }
        Eigen::Vector3d nextState;
        const Eigen::Vector3d G = nodes.at(id).getGradient();
        //project the chosen
        project(chosen, nextState, G);
        //and done
        return nextState;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    protected:
    //true if collision found
    virtual bool isInCollision(const Eigen::Vector3d &pt, const std::size_t &self)
    {
        for (std::size_t i=0; i<nodes.size(); ++i)
        {
            if (i == self)
                continue;
            const double dist = (pt[0] - nodes[i].getCenter()[0])*(pt[0] - nodes[i].getCenter()[0]) +
                                (pt[1] - nodes[i].getCenter()[1])*(pt[1] - nodes[i].getCenter()[1]) +
                                (pt[2] - nodes[i].getCenter()[2])*(pt[2] - nodes[i].getCenter()[2]);
            if (dist <= std::pow(nodes[i].getRadius(),2))
                return true;
        }
        return false;
    }
};
}

#endif
