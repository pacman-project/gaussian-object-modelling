#ifndef GP_REGRESSION___GP_PROJECTOR_H
#define GP_REGRESSION___GP_PROJECTOR_H

#include <gp_regression/gp_regressor.hpp>
#include <unsupported/Eigen/NonLinearOptimization>

namespace gp_regression
{

/*
 * \brief Container for a chart
 */
struct Chart
{
        Eigen::Vector3d C; // points
        Eigen::Vector3d Tx; // tangent basis 1
        Eigen::Vector3d Ty; // tangent basis 2
        double R;  // size
};

struct Atlas
{
        std::vector<Chart> charts;
        Eigen::MatrixXd adjency;
};

template <typename CovType>
class GPProjector
{
public:
        virtual ~GPProjector() {}

        // pointer to the covariance function type
        CovType* kernel_;

        /**
        * \brief Solves the projection of a point onto f(x)=0, where f(x)~GP(m(x),k(x,x')).
        * \param[in] GP
        * \param[in] Chart (x_i, Tx_i, Ty_i, r_i) where x_i is the cart center,
        *            [Tx_i, Ty_i] is a basis of the tangent space, and r_i is the chart size.
        * \param[in] Initial point on chart x_i'
        * \param[out] Point x_j such that f(x_j)~GP(m(x_j),k(x_j,x'))=0.
        */
        void project(const Model &gp, const Chart &chart, const Eigen::Vector3d &in, Eigen::Vector3d &out)
        {
                return;
        }

        void generateChart(const Model &gp, const Eigen::Vector3d &C, const double &R, Chart &chart)
        {
                chart.C = C;
                chart.R = R;
                Data q;

                q.coord_x.push_back( C(0) );
                q.coord_y.push_back( C(1) );
                q.coord_z.push_back( C(2) );
                Eigen::MatrixXd N, Tx, Ty;
                std::vector<double> f, v;
                gp_regression::GPRegressor<CovType> regressor;
                regressor.evaluate(gp, q, f, v, N, Tx, Ty);
                chart.Tx = Tx.row(0);
                chart.Ty = Ty.row(0);
                return;
        }

        void updateAtlas(const Chart &chart, Atlas &atlas, int parent_index)
        {
                atlas.charts.push_back(chart);
                atlas.adjency.resize(atlas.adjency.rows() + 1, atlas.adjency.cols() + 1);
                atlas.adjency(atlas.adjency.rows(), parent_index) = 1.0;
                atlas.adjency(parent_index, atlas.adjency.cols()) = 1.0;
                return;
        }
private:

};
}

#endif
