#ifndef GP_REGRESSION___ATLAS_HPP
#define GP_REGRESSION___ATLAS_HPP

#include <memory>
#include <iostream>
#include <Eigen/Dense>
#include <gp_regression/gp_regression_exception.h>
#include <gp_regression/gp_regressor.hpp>
#include <gp_regression/gp_modelling.h>

namespace gp_regression
{
/**
 * \brief Base Atlas class
 */
class AtlasBase
{
    public:
    /**
     * \brief Container for a chart
     */
    struct Chart
    {
        //no accidental empty construction
        Chart()=delete;

        //only way to construct a Chart! (also prevents implicit conversions)
        explicit Chart(const Eigen::Vector3d &c, const Eigen::Vector3d &n= Eigen::Vector3d::UnitZ(), const double r=0.03):
            center(c), N(n), radius(r)
        {
            computeTangentBasis(N,Tx,Ty);
        }
        ~Chart() {}

        typedef std::shared_ptr<Chart> Ptr;
        typedef std::shared_ptr<const Chart> ConstPtr;

        inline Eigen::Vector3d getNormal() const
        {
            return N;
        }

        void setNormal(const Eigen::Vector3d &n)
        {
            N = n;
            computeTangentBasis(N,Tx,Ty);
        }

        void setRadius(const double r)
        {
            radius = r;
        }

        inline double getRadius() const
        {
            return radius;
        }

        inline Eigen::Vector3d getCenter() const
        {
            return center;
        }

        private:
        Eigen::Vector3d center;     // origin point
        Eigen::Vector3d N;          // normal pointing outside surface
        Eigen::Vector3d Tx;         // tangent basis 1
        Eigen::Vector3d Ty;         // tangent basis 2
        double radius;              // chart radius
    };

    /**
     * \brief Get a new intermediate point to reach, given the current point.
     *
     * I.E. set a new direction for RRT to explore
     */
    virtual Eigen::Vector3d getNextState(const Eigen::Vector3d& )=0;

    /**
     * \brief Tell if a point satisfies the Goal provided
     */
    virtual bool gapSatisfied(const Eigen::Vector3d&, const Eigen::Vector3d& )=0;

    /**
     * \brief Get a cost between two points
     */
    virtual double cost(const Eigen::Vector3d&, const Eigen::Vector3d&)=0;

    protected:
    ///Pointer to gp_model
    Model::Ptr gp_model;
};

class AtlasVarianceBased : public Atlas
{
    public:
    typedef std::shared_ptr<AtlasVarianceBased> Ptr;
    typedef std::shared_ptr<const AtlasVarianceBased> ConstPtr;

    AtlasVarianceBased()=delete;
    AtlasVarianceBased(Model::ConstPtr gp): gp_model(gp), var_factor()
    {
    }

    protected:
    //radius is inversely proportional to variance of its center, by this factor
    double var_factor;
};
}

#endif
