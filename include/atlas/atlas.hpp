#ifndef GP_ATLAS_HPP_
#define GP_ATLAS_HPP_

#include <memory>
#include <unordered_map>
#include <iostream>
#include <Eigen/Dense>
#include <gp_regression/gp_regression_exception.h>
#include <gp_regression/gp_modelling.h>
#include <random_generation.hpp>

namespace gp_atlas_rrt
{
    /**
     * \brief Container for a chart
     */
    struct Chart
    {
        //no accidental empty construction
        Chart()=delete;

        //only way to construct a Chart! (also prevents implicit conversions)
        explicit Chart(const Eigen::Vector3d &c, const std::size_t i, const Eigen::Vector3d &g
                ,const double r):
            id(i), C(c), G(g), R(r)
        {
            gp_regression::computeTangentBasis(G, N,Tx,Ty);
        }
        ~Chart() {}

        typedef std::shared_ptr<Chart> Ptr;
        typedef std::shared_ptr<const Chart> ConstPtr;

        inline Eigen::Vector3d getNormal() const
        {
            return N;
        }

        inline Eigen::Vector3d getGradient() const
        {
            return G;
        }

        void setGradient(const Eigen::Vector3d &g)
        {
            G = g;
            gp_regression::computeTangentBasis(G, N,Tx,Ty);
        }

        void setRadius(const double r)
        {
            R = r;
        }

        inline double getRadius() const
        {
            return R;
        }

        inline Eigen::Vector3d getCenter() const
        {
            return C;
        }

        inline std::size_t getId() const
        {
            return id;
        }

        inline Eigen::Vector3d getTanBasisOne() const
        {
            return Tx;
        }

        inline Eigen::Vector3d getTanBasisTwo() const
        {
            return Ty;
        }

        Eigen::MatrixXd samples; //collection of uniform disc samples (nx3)
        private:
        std::size_t id;       // unique identifier
        Eigen::Vector3d C;     // origin point
        Eigen::Vector3d G;     // Gradient
        Eigen::Vector3d N;     // normal pointing outside surface
        Eigen::Vector3d Tx;    // tangent basis 1
        Eigen::Vector3d Ty;    // tangent basis 2
        double R;              // chart radius
    };
/**
 * \brief Base Atlas class
 *
 * Charts are nodes of the Atlas
 */
class AtlasBase
{
    public:
    AtlasBase(const gp_regression::Model::Ptr &gp, const gp_regression::ThinPlateRegressor::Ptr &reg):
        gp_model(gp), gp_reg(reg) {}
    virtual ~AtlasBase(){}

    ///Initial state initialization  is left to implement on  child classes with
    ///no strictly required signature

    /**
     * \brief get a copy of a node
     */
    virtual Chart getNode(const std::size_t &id) const
    {
        if (id < nodes.size())
            return nodes.at(id);
        else
            throw gp_regression::GPRegressionException("Out of Range node id");
    }
    /**
     * \brief get all node ids the given node is connected to
     * TODO this should go to the planner
     */
    virtual std::vector<std::size_t> getConnections (const std::size_t &id) const
    {
        if (id < nodes.size())
        {
            auto branch = branches.find(id);
            if (branch != branches.end())
                return branch->second;
            else
                return std::vector<std::size_t>();
        }
        else
            throw gp_regression::GPRegressionException("Out of Range node id");
    }

    /**
     * \brief Get a new node center to explore from a given node id.
     */
    virtual Eigen::Vector3d getNextState(const std::size_t& )=0;

    /**
     * \brief Tell if passed node is global solution
     */
    virtual bool isSolution(const Chart&)=0;
    virtual inline bool isSolution(const std::size_t& id)
    {
        return isSolution(getNode(id));
    }

    /**
     * \brief Contruct a Node from the given center and stores it
     * \return its id
     */
    virtual std::size_t createNode(const Eigen::Vector3d&)=0;

    /**
     * \brief Connect two nodes
     * TODO this should go to the planner
     */
    virtual void connect(const std::size_t, const std::size_t)=0;

    protected:
    ///Pointer to gp_model
    gp_regression::Model::Ptr gp_model;
    ///Pointer to regressor
    gp_regression::ThinPlateRegressor::Ptr gp_reg;
    ///Node storage
    std::vector<Chart> nodes;
    ///Connection map
    //TODO this should be moved to the planner
    std::unordered_map<std::size_t, std::vector<std::size_t>> branches;

    /**
     * \brief project a point on gp surface
     *
     * \param[in] in Input point to project
     * \param[out] out projected point on gp, if successful, otherwise untouched.
     * \paran[in] normal Unnormalized gradient along which projection takes place.
     * \param[in] step_mul multiplier to step lenght.
     * \param[in] f_tol tolerance on f(x). First convergence criteria.
     * \param[in] max_iter total iterations to try before converging. Second convergence criteria.
     * \param[in] improve_tol tolerance on f(x) improvement. Thired convergence criteria.
     */
    virtual void project(const Eigen::Vector3d &in, Eigen::Vector3d &out, const Eigen::Vector3d &normal,
            const double f_tol= 1e-2, const double improve_tol= 1e-6, const unsigned int max_iter=5000, const double step_mul = 1.0)
    {
        if (!gp_reg)
            throw gp_regression::GPRegressionException("Empty regressor pointer");
        Eigen::Vector3d current = in;
        std::vector<double> current_f;
        gp_regression::Data::Ptr currentP = std::make_shared<gp_regression::Data>();
        unsigned int iter = 0;
        while(iter < max_iter)
        {
            // clear vectors of current values
            currentP->clear();

            // and fill with current values
            currentP->coord_x.push_back( current(0) );
            currentP->coord_y.push_back( current(1) );
            currentP->coord_z.push_back( current(2) );

            // evaluate the current result
            gp_reg->evaluate(gp_model, currentP, current_f);

            // check tolerances
            if( std::abs(current_f.at(0)) < f_tol )
            {
                std::cout << "[Atlas::project] CONVERGENCE: Function evaluation reached tolerance." << std::endl;
                out = current;
                return;
            }

            // perform the step using the gradient descent method
            // put minus in front, cause normals are all pointing outwards
            current -= step_mul*current_f.at(0)*normal;

            // cehck improvment tolerance
            gp_regression::Data::Ptr outP = std::make_shared<gp_regression::Data>();
            outP->coord_x.push_back( current(0) );
            outP->coord_y.push_back( current(1) );
            outP->coord_z.push_back( current(2) );
            std::vector<double> out_f;
            gp_reg->evaluate(gp_model, outP, out_f);
            if( std::abs(out_f.at(0) - current_f.at(0)) < improve_tol )
            {
                std::cout << "[Atlas::project] CONVERGENCE: Function improvement reached tolerance." << std::endl;
                out = current;
                return;
            }
            ++iter;
        }
        std::cout << "[Atlas::project] CONVERGENCE: Reached maximum number of iterations." << std::endl;
        out = current;
    }

};

class AtlasVariance : public AtlasBase
{
    public:
    typedef std::shared_ptr<AtlasVariance> Ptr;
    typedef std::shared_ptr<const AtlasVariance> ConstPtr;

    AtlasVariance()=delete;
    AtlasVariance(const gp_regression::Model::Ptr &gp, const gp_regression::ThinPlateRegressor::Ptr &reg):
        AtlasBase(gp,reg), var_factor(1.96), disc_samples(200)
    {
    }

    /// recieve a starting point
    virtual void init(const Eigen::Vector3d &root)
    {
        //TODO
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
        if (std::abs(f.at(0)) > 1e-2)
            std::cout<<"[Atlas::createNode] Chart center is not on GP surface!"<<std::endl;
        Chart node (center, nodes.size(), g, std::sqrt(v.at(0))*var_factor);
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
        nodes.at(id).samples.resize(disc_samples, 3);
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
        //uniform disc sampling
        for (std::size_t i=0; i<disc_samples; ++i)
        {
            const double r = getRandIn(0.0, R, true);
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

    virtual bool isSolution(const Chart&)
    {
        //TODO
    }

    protected:
    //radius is proportional to sqrt variance of its center times 95% confidence
    double var_factor;
    //how many disc samples to try
    std::size_t disc_samples;
};
}

#endif
