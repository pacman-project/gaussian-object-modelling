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
        AtlasBase(gp,reg), var_factor(0.7), disc_samples_factor(500)
    {
        var_tol = 0.5; //this should be give by user
                       //whoever uses this class will take care of it, by calling
                       //setVarianceTolGoal()
    }
    virtual ~AtlasVariance(){}

    ///Set the variance tolerance to surpass, for a node to be considered solution
    virtual inline void setVarianceTolGoal(const double vt)
    {
        var_tol = vt;
    }

    virtual inline void setVarRadiusFactor(const double vf)
    {
        var_factor = vf;
    }

    virtual inline void setDiscSampleFactor(const double dsf)
    {
        disc_samples_factor = dsf;
    }

    // ///reset Atlas with new parameters and then recieve a new starting point (root)
    // virtual void init(const double var_tolerance, const gp_regression::Model::Ptr &gpm, const gp_regression::ThinPlateRegressor::Ptr &gpr)
    // {
    //     clear();
    //     var_tol = var_tolerance;
    //     setGPModel(gpm);
    //     setGPRegressor(gpr);
    // }

    virtual double computeRadiusFromVariance(const double v) const
    {
        if (v > 0.5){
            std::cout<<"Variance is too big "<<v<<std::endl;
            return 0.01;
        }
        if (v < 0){
            std::cout<<"Variance is negative "<<v<<std::endl;
            return 0.3;
        }
        if (std::isnan(v) || std::isinf(v)){
            std::cout<<"Variance is NAN / Inf "<<v<<std::endl;
            return 0.05;
        }
        return ( -var_factor*v + 0.4);
    }

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
        if (g.isZero(1e-3) || !g.isMuchSmallerThan(1e3,1e-1)){
            std::cout<<"[Atlas::createNode] Chart gradient is wrong, trying to perturbate test point.\n";
            std::cout<<"[Atlas::createNode] Gradient is\n"<<g<<std::endl;
            std::cout<<"[Atlas::createNode] Chart center is\n"<<center<<std::endl;
            c->clear();
            c->coord_x.push_back(center(0) + getRandIn(1e-3, 1e-2));
            c->coord_y.push_back(center(1) + getRandIn(1e-3, 1e-2));
            c->coord_z.push_back(center(2) + getRandIn(1e-3, 1e-2));
            gp_reg->evaluate(gp_model, c, f, v, gg);
            g=gg.row(0);
            // throw gp_regression::GPRegressionException("Gradient is zero");
        }
        if (std::abs(f.at(0)) > 0.01 || std::isnan(f.at(0)) || std::isinf(f.at(0)))
            std::cout<<"[Atlas::createNode] Chart center is not on GP surface! f(x) = "<<f.at(0)<<std::endl;
        if (g.isZero(1e-3) || !g.isMuchSmallerThan(1e3, 1e-1)){
            std::cout<<"[Atlas::createNode] Gradient is still Zero Or too big! Resetting to Xaxis\n";
            std::cout<<"[Atlas::createNode] Gradient was\n"<<g<<std::endl;
            g = Eigen::Vector3d::UnitX();
        }
        Chart node (center, nodes.size(), g, v.at(0));
        node.setRadius(computeRadiusFromVariance(v.at(0)));
        nodes.push_back(node);
        std::cout<<"[Atlas::createNode] Created node "<<node.getId()<<std::endl;
        return node.getId();
    }


    virtual Eigen::Vector3d getNextState(const std::size_t& id)
    {
        if (!gp_reg)
            throw gp_regression::GPRegressionException("Empty Regressor pointer");
        if (id >= nodes.size())
            throw gp_regression::GPRegressionException("Out of Range node id");
        //the winner is:
        sampleOnChart(nodes.at(id));
        nodes.at(id).samp_chosen = nodes.at(id).vars_ids.at(0).second;
        Eigen::Vector3d chosen = nodes.at(id).samples.row(nodes.at(id).samp_chosen);
        // std::cout<<"chosen "<<chosen<<" s_idx "<<s_idx<<std::endl;
        // std::cout<<"samples dim: "<<nodes.at(id).samples.rows()<<" x "<<nodes.at(id).samples.cols()<<std::endl;
        // std::cout<<"samples "<<nodes.at(id).samples<<std::endl;
        Eigen::Vector3d nextState;
        const Eigen::Vector3d G = nodes.at(id).getGradient();
        //project the chosen
        project(chosen, nextState, G);
        //and done
        return nextState;
    }

    virtual inline bool isSolution(const std::size_t &id)
    {
        if (id >= nodes.size())
            throw gp_regression::GPRegressionException("Out of Range node id");
        return (getNode(id).getVariance() > var_tol);
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    protected:
    //radius is inversely proportional to variance
    double var_factor;
    //how many disc samples multiplier (total samples are proportional to radius)
    //which in turn is proportional to variance
    std::size_t disc_samples_factor;
    //variance threshold for solution
    double var_tol;

    virtual void sampleOnChart(Chart& c)
    {
        //get some useful constants from the disc
        const Eigen::Vector3d N = c.getNormal();
        const Eigen::Vector3d Tx = c.getTanBasisOne();
        const Eigen::Vector3d Ty = c.getTanBasisTwo();
        const Eigen::Vector3d C = c.getCenter();
        const double R = c.getRadius();
        // std::cout<<"N\n"<<N <<std::endl;
        // std::cout<<"Tx\n"<<Tx <<std::endl;
        // std::cout<<"Ty\n"<<Ty <<std::endl;
        // std::cout<<"C\n"<<C <<std::endl;
        // std::cout<<"G\n"<<G <<std::endl;
        // std::cout<<"R "<<R <<std::endl;
        //prepare the samples storage
        const std::size_t tot_samples = std::ceil(std::abs(disc_samples_factor) * R);
        // std::cout<<"total samples "<<tot_samples<<std::endl;
        c.samples.resize(tot_samples, 3);
        std::vector<double> f,v;
        //transformation into the kinect frame from local
        Eigen::Matrix4d Tkl;
        Tkl <<  Tx(0), Ty(0), N(0), C(0),
                Tx(1), Ty(1), N(1), C(1),
                Tx(2), Ty(2), N(2), C(2),
                0,     0,     0,    1;
        // std::cout<<"Tkl "<<Tkl<<std::endl;
        //uniform annulus sampling
        for (std::size_t i=0; i<tot_samples; ++i)
        {
            double r = getRandIn(0.8, 1.0, true);
            double th = getRandIn(0.0, 2*M_PI);
            if (std::isnan(r) || std::isinf(r) || std::isnan(th) || std::isinf(th)){
                std::cout<<"r: "<<r <<"th: "<<th <<std::endl;
                r = getRandIn(0.8, 1.0, true);
                th = getRandIn(0.0, 2*M_PI);
            }
            //point in local frame, uniformely sampled
            Eigen::Vector4d pL;
            pL <<   R * std::sqrt(r) * std::cos(th),
                    R * std::sqrt(r) * std::sin(th),
                    0,
                    1;
            //the same point in kinect frame
            Eigen::Vector4d pK = Tkl * pL;
            gp_regression::Data::Ptr query = std::make_shared<gp_regression::Data>();
            query->coord_x.push_back(pK(0));
            query->coord_y.push_back(pK(1));
            query->coord_z.push_back(pK(2));
            //store the sample for future use (even plotting)
            c.samples(i,0) = pK(0);
            c.samples(i,1) = pK(1);
            c.samples(i,2) = pK(2);
            //evaluate the sample
            gp_reg->evaluate(gp_model, query, f, v);
            if (std::isnan(v.at(0)) || std::isinf(v.at(0)) ||
                    std::isnan(f.at(0)) || std::isinf(f.at(0))){
                std::cout << "[Atlas::getNextState] Found NAN. Fatal. f=" <<f.at(0)<<" v=" <<v.at(0)<<std::endl;
                std::cout<<"point: "<<pK <<std::endl;
                std::cout<<"point Local: "<<pL <<std::endl;
                std::cout<<"r: "<<r <<"th: "<<th <<std::endl;
                std::cout<<"Tkl:\n"<<Tkl <<std::endl;
                throw gp_regression::GPRegressionException("v is nan or inf");
            }
            //keep variances and ids
            c.vars_ids.push_back(std::make_pair(v.at(0), i));
        }
        std::sort(c.vars_ids.begin(), c.vars_ids.end(),
                [](const std::pair<double,std::size_t> &a, const std::pair<double,std::size_t> &b)
                {
                    return (a.first > b.first);
                });
    }
};
}

#endif
