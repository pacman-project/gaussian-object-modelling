#ifndef GP_ATLAS_HPP_
#define GP_ATLAS_HPP_

#include <memory>
#include <unordered_map>
#include <iostream>
#include <Eigen/Dense>
#include <gp_regression/gp_regression_exception.h>
#include <gp_regression/gp_regressors.h>
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
                ,const double v):
            id(i), C(c), G(g), V(v), samp_chosen(-1)
        {
            gp_regression::computeTangentBasis(G, N,Tx,Ty);
        }
        virtual ~Chart() {}

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

        inline double getVariance() const
        {
            return V;
        }

        inline void resetSamples()
        {
            samples.resize(0,0);
        }

        //these can be public, they dont affect the disc functionalites
        Eigen::MatrixXd samples; //collection of uniform disc samples (nx3)
        int samp_chosen; //the chosen sample id, if sampling was not done it is -1
        std::vector<std::pair<double,std::size_t>> vars_ids; //vector of sample variances with their index

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        protected:
        std::size_t id;       // unique identifier
        Eigen::Vector3d C;     // origin point
        Eigen::Vector3d G;     // Gradient
        Eigen::Vector3d N;     // normal pointing outside surface
        Eigen::Vector3d Tx;    // tangent basis 1
        Eigen::Vector3d Ty;    // tangent basis 2
        double R;              // chart radius
        double V;              // variance of chart (of its origin)
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
     * \brief
     * count how many nodes the atlas currently have
     */
    virtual inline std::size_t countNodes() const
    {
        return nodes.size();
    }
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
     * \brief reset Atlas, clearing contents
     */
    virtual void clear()
    {
        nodes.clear();
        gp_model.reset();
        gp_reg.reset();
    }
    /**
     * \brief set GP model to use
     */
    virtual void setGPModel(const gp_regression::Model::Ptr &gpm)
    {
        gp_model = gpm;
    }
    /**
     * \brief set GP regressor to use
     */
    virtual void setGPRegressor(const gp_regression::ThinPlateRegressor::Ptr &gpr)
    {
        gp_reg = gpr;
    }

    /**
     * \brief Get a new node center to explore from a given node id.
     */
    virtual Eigen::Vector3d getNextState(const std::size_t& )=0;

    /**
     * \brief Tell if passed node is global solution
     */
    virtual inline bool isSolution(const std::size_t&)=0;

    /**
     * \brief Contruct a Node from the given center and stores it
     * \return its id
     */
    virtual std::size_t createNode(const Eigen::Vector3d&)=0;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    protected:
    ///Pointer to gp_model
    gp_regression::Model::Ptr gp_model;
    ///Pointer to regressor
    gp_regression::ThinPlateRegressor::Ptr gp_reg;
    ///Node storage
    std::vector<Chart> nodes;

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
            const double f_tol= 1e-2, const double improve_tol= 1e-7, const unsigned int max_iter=500, const double step_mul = 0.001)
    {
        if (!gp_reg)
            throw gp_regression::GPRegressionException("Empty regressor pointer");
        Eigen::Vector3d current = in;
        std::vector<double> current_f, v;
        unsigned int iter = 0;
        Eigen::MatrixXd N;
        Eigen::Vector3d g = normal;
        while(iter < max_iter)
        {
            // std::cout<<"iter "<<iter<<std::endl;
            gp_regression::Data::Ptr currentP = std::make_shared<gp_regression::Data>();
            // clear vectors of current values
            current_f.clear();
            v.clear();

            // and fill with current values
            currentP->coord_x.push_back( current(0) );
            currentP->coord_y.push_back( current(1) );
            currentP->coord_z.push_back( current(2) );

            // evaluate the current result
            gp_reg->evaluate(gp_model, currentP, current_f);

            if (std::isnan(current_f.at(0)) || std::isinf(current_f.at(0))){
                std::cout << "[Atlas::project] Found NAN function evaluation. Fatal" << std::endl;
                std::cout<<"current "<<current<<std::endl;
                throw gp_regression::GPRegressionException("f is nan or inf");
            }

            // std::cout << "current f" << current_f.at(0) << std::endl;

            // check tolerances
            if( std::abs(current_f.at(0)) < f_tol )
            {
                std::cout << "[Atlas::project] CONVERGENCE: Function evaluation reached tolerance." << std::endl;
                out = current;
                return;
            }

            // perform the step using the gradient descent method
            // put minus in front, cause normals are all pointing outwards
            Eigen::Vector3d cur_step = step_mul*current_f.at(0)*g;
            if (!cur_step.isMuchSmallerThan(1e3, 1e-1) || cur_step.isZero(1e-6)){
                std::cout<<"[Atlas::project] Current step is wrong! Ignoring\n";
                std::cout<<"[Atlas::project] It was\n"<<cur_step<<std::endl;
            }
            else
                current -= cur_step;

            // check improvment tolerance
            gp_regression::Data::Ptr outP = std::make_shared<gp_regression::Data>();
            outP->coord_x.push_back( current(0) );
            outP->coord_y.push_back( current(1) );
            outP->coord_z.push_back( current(2) );
            std::vector<double> out_f;
            gp_reg->evaluate(gp_model, outP, out_f, v, N);
            if (!N.row(0).isMuchSmallerThan(1e3, 1e-1) || N.row(0).isZero(1e-5)){
                std::cout<<"[Atlas::project] Gradient is wrong! Resetting to previous\n";
                std::cout<<"[Atlas::project] Gradient was\n"<<N.row(0)<<std::endl;
            }
            else
                g = N.row(0);
            if( std::abs(out_f.at(0) - current_f.at(0)) < improve_tol )
            {
                std::cout << "[Atlas::project] CONVERGENCE: Function improvement reached tolerance." << std::endl;
                out = current;
                return;
            }
            ++iter;
        }
        std::cout << "[Atlas::project] CONVERGENCE: Reached maximum number of iterations. F: "<<current_f.at(0)<<" Var:"<<v.at(0)<< std::endl;
        out = current;
    }

};
}

#endif
