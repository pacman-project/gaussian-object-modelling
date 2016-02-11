#ifndef GP_ATLAS_VARIANCE_HPP_
#define GP_ATLAS_VARIANCE_HPP_

#include <atlas/atlas.hpp>

namespace gp_atlas_rrt
{

class AtlasVariance : public AtlasBase
{
    public:
    typedef std::shared_ptr<AtlasVariance> Ptr;
    typedef std::shared_ptr<const AtlasVariance> ConstPtr;

    AtlasVariance()=delete;
    AtlasVariance(const gp_regression::Model::Ptr &gp, const gp_regression::ThinPlateRegressor::Ptr &reg):
        AtlasBase(gp,reg), var_factor(1.96), disc_samples_factor(2000)
    {
        var_tol = 0.5; //this should be give by user
                       //whoever uses this class will take care of it, by calling
                       //setVarianceTolGoal()
    }

    ///Set the variance tolerance to surpass, for a node to be considered solution
    virtual inline void setVarianceTolGoal(const double vt)
    {
        var_tol = vt;
    }

    // ///reset Atlas with new parameters and then recieve a new starting point (root)
    // virtual void init(const double var_tolerance, const gp_regression::Model::Ptr &gpm, const gp_regression::ThinPlateRegressor::Ptr &gpr)
    // {
    //     clear();
    //     var_tol = var_tolerance;
    //     setGPModel(gpm);
    //     setGPRegressor(gpr);
    // }

    virtual std::size_t createNode(const Eigen::Vector3d& center)
    {
        if (!gp_reg)
            throw gp_regression::GPRegressionException("Empty Regressor pointer");
        gp_regression::Data::Ptr c = std::make_shared<gp_regression::Data>();
        c->coord_x.push_back(center(0));
        c->coord_y.push_back(center(1));
        c->coord_z.push_back(center(2));
        std::vector<double> f,v;
        Eigen::MatrixXd gg;
        gp_reg->evaluate(gp_model, c, f, v, gg);
        Eigen::Vector3d g = gg.row(0);
        if (std::abs(f.at(0)) > 1e-2)
            std::cout<<"[Atlas::createNode] Chart center is not on GP surface!"<<std::endl;
        Chart node (center, nodes.size(), g, std::sqrt(v.at(0))*var_factor, v.at(0));
        nodes.push_back(node);
        return node.getId();
    }

    virtual Eigen::Vector3d getNextState(const std::size_t& id)
    {
        if (!gp_reg)
            throw gp_regression::GPRegressionException("Empty Regressor pointer");
        if (id >= nodes.size())
            throw gp_regression::GPRegressionException("Out of Range node id");
        //get some useful constants from the disc
        const Eigen::Vector3d N = nodes.at(id).getNormal();
        const Eigen::Vector3d Tx = nodes.at(id).getTanBasisOne();
        const Eigen::Vector3d Ty = nodes.at(id).getTanBasisTwo();
        const Eigen::Vector3d C = nodes.at(id).getCenter();
        const Eigen::Vector3d G = nodes.at(id).getGradient();
        const double R = nodes.at(id).getRadius();
        //prepare the samples storage
        const std::size_t tot_samples = std::round(disc_samples_factor * R);
        nodes.at(id).samples.resize(tot_samples, 3);
        std::vector<double> f,v;
        //transformation into the kinect frame from local
        Eigen::Matrix4d Tkl;
        Tkl <<  Tx(0), Ty(0), N(0), C(0),
                Tx(1), Ty(1), N(1), C(1),
                Tx(2), Ty(2), N(2), C(2),
                0,     0,     0,    1;
        //keep the max variance found
        double max_v(0.0);
        //and which sample it was
        std::size_t s_idx(0);
        //uniform annulus sampling from R/5 to R
        for (std::size_t i=0; i<tot_samples; ++i)
        {
            const double r = getRandIn(R/5, R, true);
            const double th = getRandIn(0.0, 2*M_PI);
            //point in local frame, uniformely sampled
            Eigen::Vector4d pL;
            pL <<   std::sqrt(r) * std::cos(th),
                    std::sqrt(r) * std::sin(th),
                    0,
                    1;
            //the same point in kinect frame
            Eigen::Vector4d pK = Tkl * pL;
            gp_regression::Data::Ptr query = std::make_shared<gp_regression::Data>();
            query->coord_x.push_back(pK(0));
            query->coord_y.push_back(pK(1));
            query->coord_z.push_back(pK(2));
            //store the sample for future use (even plotting)
            nodes.at(id).samples(i,0) = pK(0);
            nodes.at(id).samples(i,1) = pK(1);
            nodes.at(id).samples(i,2) = pK(2);
            //evaluate the sample
            gp_reg->evaluate(gp_model, query, f, v);
            if (v.at(0) > max_v){
                max_v = v.at(0);
                s_idx = i;
            }
        }
        //the winner is:
        Eigen::Vector3d chosen = nodes.at(id).samples.row(s_idx);
        Eigen::Vector3d nextState;
        //project the chosen
        project(chosen, nextState, G);
        //and done
        return nextState;
    }

    virtual inline bool isSolution(const std::size_t &id) const
    {
        return (getNode(id).getVariance() > var_tol);
    }

    protected:
    //radius is proportional to sqrt variance of its center times 95% confidence
    //this factor is hardcoded to 1.96
    double var_factor;
    //how many disc samples multiplier (total samples are proportional to radius)
    //which in turn is proportional to variance
    std::size_t disc_samples_factor;
    //variance threshold for solution
    double var_tol;
};
}

#endif
