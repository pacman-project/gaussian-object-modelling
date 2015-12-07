#ifndef GP_REGRESSION___GP_PROJECTOR_H
#define GP_REGRESSION___GP_PROJECTOR_H

#include <gp_regression/gp_regressor.hpp>
#include <unsupported/Eigen/NonLinearOptimization>

namespace gp_regression
{

//struct ProjectorFunctor
//{
//        int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
//        {
//                gp_regression::GPRegressor<Gaussian> regressor;
//                std::vector<double> current_f, current_v;
//                Data currentP;
//                currentP.coord_x.push_back( x(0) );
//                currentP.coord_y.push_back( x(1) );
//                currentP.coord_z.push_back( x(2) );

//                regressor.evaluate(gp_, currentP, current_f, current_v);
//                fvec(0) = current_f.at(0);
//                return 0;
//        }

//        int df(const Eigen::VectorXd &x, Eigen::MatrixXd &fjac) const
//        {
//                gp_regression::GPRegressor<Gaussian> regressor;
//                std::vector<double> current_f, current_v;
//                Data currentP;
//                currentP.coord_x.push_back( x(0) );
//                currentP.coord_y.push_back( x(1) );
//                currentP.coord_z.push_back( x(2) );

//                regressor.evaluate(gp_, currentP, current_f, current_v, fjac);

//                return 0;
//        }

//        Model gp_;

//        int inputs() const { return 3; }// inputs is the dimension of x.
//        int values() const { return 1; } // "values" is the number of f_i and
//};

/*
 * \brief Container for a chart
 */
struct Chart
{
        Eigen::Vector3d C; // points
        Eigen::Vector3d N; // (inward) normal t chart
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
        //CovType* kernel_;

        /**
        * \brief Solves the projection of a point x_j onto f(x)=0, where f(x) ~ GP(m(x),k(x,x')),
        * using a gradient-descent like method: x^{k+1} = x^{k} - h*[f(x^{k})/f'(x^{k})],
        * initialized at a point on the chart.
        *
        * \param[in] GP
        * \param[in] Chart (x_i, Tx_i, Ty_i, r_i) where x_i is the cart center,
        *            [Tx_i, Ty_i] is a basis of the tangent space, and r_i is the chart size.
        * \param[in] Initial point on chart x_i'
        * \param[out] Point x_j such that f(x_j)~GP(m(x_j),k(x_j,x'))=0.
        */
        /**
         * @brief project Uses a gradient-descent method to project a point on a surface.
         * @param gp The gp model such that f(x) ~ gp(m(x),k(x,x_i))
         * @param regressor The regressor was used to generate the gp model.
         * @param chart Initial chart
         * @param in Initial point assumed on chart x_i', but not necessarily
         * @param out Projected point on surface f(x_j) ~ 0
         * @param step_size Scale of the gradient-descent step f(x)/f'(x).
         * @param eps_f_eval Tolerance for function evaluation. If less than this, projection converged.
         * @param max_iter Maximum number of iteration (function is evaluated at every iteration).
         * If greater than this, projection didn't converge.
         * @param eps_x Tolerance for the (scaled) step. If less than this, projection didn't converge.
         * @return
         */
        bool project(const Model &gp, GPRegressor<CovType> &regressor, const Chart &chart, const Eigen::Vector3d &in,
                    Eigen::Vector3d &out, double step_size = 1.0,
                    double eps_f_eval = 1e-10, int max_iter = 5000, double eps_x = 1e-15)
        {
                Eigen::Vector3d current = in;
                std::vector<double> current_f, current_v;
                Data currentP;
                int iter = 0;
                int impr_counter = 0;
                while(true)
                {
                        // clear vectors of current values
                        currentP.coord_x.clear();
                        currentP.coord_y.clear();
                        currentP.coord_z.clear();
                        current_f.clear();
                        current_v.clear();

                        // and fill with current values
                        currentP.coord_x.push_back( current(0) );
                        currentP.coord_y.push_back( current(1) );
                        currentP.coord_z.push_back( current(2) );

                        // evaluate the current result
                        regressor.evaluate(gp, currentP, current_f, current_v);

                        // print stats at current
                        // std::cout << "iter: " << iter << std::endl;
                        // std::cout << "current x: " << std::endl << current << std::endl;
                        // std::cout << "current dx: " << std::endl << (step_size*current_f.at(0)*chart.N).norm() << std::endl;
                        // std::cout << "current f(x): " << current_f.at(0) << std::endl << std::endl;

                        // check tolerances
                        if( std::abs(current_f.at(0)) < eps_f_eval )
                        {
                                std::cout << "[Converged] Function evaluation reached tolerance." << std::endl;
                                out = current;
                                return true;
                        }
                        if( (step_size*current_f.at(0)*chart.N).norm() < eps_x )
                        {
                                std::cout << "[NotConverged] Exited by step size tolerance violation." << std::endl;
                                out = current;
                                return false;
                        }

                        // perform the step using the gradient descent method
                        // x_j = x_i - a*f'(x_0)
                        // that is, f'(x_i) = gradient
                        // a = c*f(x_i) is the signed scale factor
                        current += step_size*current_f.at(0)*chart.N;

                        // cehck improvment tolerance
                        Data outP;
                        outP.coord_x.push_back( current(0) );
                        outP.coord_y.push_back( current(1) );
                        outP.coord_z.push_back( current(2) );
                        std::vector<double> out_f, out_v;
                        regressor.evaluate(gp, outP, out_f, out_v);
                        if( std::abs(out_f.at(0) - current_f.at(0)) == 0 )
                        {
                                impr_counter++;
                                if( impr_counter == 15 )
                                {
                                        std::cout << "[NotConverged] Function is not improving." << std::endl;
                                        out = current;
                                        return false;
                                }
                        }

                        // update iterator, and check tolerance
                        iter++;
                        if( iter == max_iter - 1 )
                        {
                                std::cout << "[NotConverged] Maximum number of iteration reached." << std::endl;
                                out = current;
                                return false;
                        }

                }

//                Eigen::VectorXd x(3);
//                x(0) = in(0);
//                x(1) = in(1);
//                x(2) = in(2);
//                std::cout << "x0: " << x << std::endl;

//                ProjectorFunctor functor;
//                functor.gp_ = gp;
//                Eigen::LevenbergMarquardt<ProjectorFunctor, double> lm(functor);
//                lm.minimize(x);
//                out(0) = x(0);
//                out(1) = x(1);
//                out(2) = x(2);

                return 0;
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
                chart.N = N.row(0);
                chart.Tx = Tx.row(0);
                chart.Ty = Ty.row(0);
                return;
        }

        void addChartToAtlas(const Chart &chart, Atlas &atlas, int parent_index)
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
