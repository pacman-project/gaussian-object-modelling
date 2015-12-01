#ifndef GP_REGRESSION___GP_REGRESSOR_H
#define GP_REGRESSION___GP_REGRESSOR_H

#include <vector>
#include <functional>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>

#include <gp_regression/cov_functions.h>
#include <gp_regression/gp_regression_exception.h>

namespace gp_regression
{

/*
 * \brief Helper function to compute a basis in the tangnet plane defined
 * by a normal vector
 */
void computeTangentBasis(const Eigen::Vector3d &N, Eigen::Vector3d &Tx, Eigen::Vector3d &Ty)
{
    Eigen::Vector3d NN = N.normalized();
    Eigen::Matrix3d TProj = Eigen::Matrix3d::Identity() - NN*NN.transpose();
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(TProj, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Tx = svd.matrixU().col(0);
    Ty = svd.matrixU().col(1);
}

/*
 * \brief Container for input data representing the raw 3D points
 */
struct Data
{
        std::vector<double> coord_x;
        std::vector<double> coord_y;
        std::vector<double> coord_z;
        std::vector<double> label;
};

/*
 * \brief Container for model parameters that represent an object
 */
struct Model
{
        Eigen::MatrixXd P; // points
        Eigen::MatrixXd N; // (inward) normal at points
        Eigen::MatrixXd Tx; // tangent basis 1
        Eigen::MatrixXd Ty; // tangent basis 2
        Eigen::VectorXd Y;  // labels
        Eigen::MatrixXd Kpp; // covariance with selected kernel
        Eigen::MatrixXd InvKpp; // inverse of covariance with selected kernel
        Eigen::VectorXd InvKppY; // weights
        Eigen::MatrixXd Kppdiff; // differential of covariance with selected kernel
};

/*
 * \brief Handle for propagating a single GP map from 3D points to paramters,
 * from-to parameters of 3D points, sample points, etc.
 *
 */
template <typename CovType>
class GPRegressor
{
public:
        // pointer to the covariance function type
        CovType* kernel_;

        virtual ~GPRegressor() {}

            /**
             * \brief Solves the regression problem, computes the model parameters.
             * \param[in]  data 3D points.
             * \param[out] gp Gaussian Process parameters.
             * \pre All non-empty vectors must contain valid data and their size
             * should be equal among them. Data vectors not used in this function can
             * remain empty, so they can be used for 3D or 2D. For now, only 3D is
             * assumed.
             */
            void create(const Data &data, Model &gp)
            {
                // validate data
                assertData(data);

                // configure gp
                convertToEigen(data.coord_x, data.coord_y, data.coord_z, gp.P);
                convertToEigen(data.label, gp.Y);
                gp.N.resize(gp.P.rows(), gp.P.cols());
                gp.Tx.resize(gp.P.rows(), gp.P.cols());
                gp.Ty.resize(gp.P.rows(), gp.P.cols());

                // go! // ToDo: avoid the for loops.
                buildEuclideanDistanceMatrix(gp.P, gp.P, gp.Kpp);
                gp.Kppdiff = gp.Kpp;

                for(int i = 0; i < gp.Kpp.rows(); ++i)
                {
                        for(int j = 0; j < gp.Kpp.cols(); ++j)
                        {
                               gp.Kpp(i,j) = kernel_->compute(gp.Kpp(i,j));
                               gp.Kppdiff(i,j) = kernel_->computediff(gp.Kpp(i,j));
                        }
                }

                gp.InvKpp = gp.Kpp.inverse();
                gp.InvKppY = gp.InvKpp*gp.Y;
                // normal and tangent computation, could be done potentially in the
                // loop above, but just to keep it slightly separated for now
                for(int i = 0; i < gp.Kpp.rows(); ++i)
                {
                        for(int j = 0; j < gp.Kpp.cols(); ++j)
                        {
                               gp.N.row(i) += gp.InvKppY(j)*gp.Kppdiff(i,j)*(gp.P.row(i) - gp.P.row(j));
                        }
                        gp.N.row(i).normalize();
                        Eigen::Vector3d N = gp.N.row(i);
                        Eigen::Vector3d Tx, Ty;
                        computeTangentBasis(N, Tx, Ty);
                        gp.Tx.row(i) = Tx;
                        gp.Ty.row(i) = Ty;
                }
        }

            /**
             * \brief Generates data using the GP
             * \param[in]  data 3D points.
             */
            void evaluate(const Model &gp, Data &query, std::vector<double> &f, std::vector<double> &v)
            {

                // validate data
                assertData(query);

                if (!query.label.empty())
                    throw GPRegressionException("Query is already labeled!");

                // go!
                Eigen::MatrixXd Q;
                Eigen::MatrixXd Kqp, Kpq;
                Eigen::MatrixXd Kqq;
                //function evaluation and associated variance
                Eigen::VectorXd F, V_diagonal;
                Eigen::MatrixXd V;
                convertToEigen(query.coord_x, query.coord_y, query.coord_z, Q);
                buildEuclideanDistanceMatrix(Q, gp.P, Kqp);

                for(int i = 0; i < Kqp.rows(); ++i)
                {
                    for(int j = 0; j < Kqp.cols(); ++j)
                    {
                        Kqp(i,j) = kernel_->compute(Kqp(i,j));
                    }
                }
                Kpq = Kqp.transpose();
                buildEuclideanDistanceMatrix(Q, Q, Kqq);
                for(int i = 0; i < Kqq.rows(); ++i)
                {
                    for(int j = 0; j < Kqq.cols(); ++j)
                    {
                        Kqq(i,j) = kernel_->compute(Kqq(i,j));
                    }
                }
                F = Kqp*gp.InvKppY;
                V = Kqq - Kqp*gp.InvKpp*Kpq;
                V_diagonal = V.diagonal();
                convertToSTD(F, f);
                convertToSTD(V_diagonal, v);
            }

            /**
             * \brief Updates the GP with the new data
             * \param[in]  data 3D points.
             */
            void update(const Data &new_data, Model &gp)
            {
            }

            GPRegressor()
            {
                kernel_ = new CovType();
            }

        private:
            /*
             * Conversion functions
             */
            void convertToEigen(const std::vector<double> &a,
                    const std::vector<double> &b,
                    const std::vector<double> &c,
                    Eigen::MatrixXd &M) const
            {
                M.resize(a.size(), 3);
                M.col(0) = Eigen::Map<Eigen::VectorXd>((double *)a.data(), a.size());
                M.col(1) = Eigen::Map<Eigen::VectorXd>((double *)b.data(), b.size());
                M.col(2) = Eigen::Map<Eigen::VectorXd>((double *)c.data(), c.size());
            }

            void convertToEigen(const std::vector<double> &a, Eigen::VectorXd &M) const
            {
                M = Eigen::Map<Eigen::VectorXd>((double *)a.data(), a.size());
            }

            void convertToSTD(const Eigen::VectorXd &M, std::vector<double> &a) const
            {
                a = std::vector<double>(M.data(), M.data() + M.size());
            }
            /*
             * Utility functions
             */
            void buildEuclideanDistanceMatrix(const Eigen::MatrixXd &A,
                    const Eigen::MatrixXd &B,
                    Eigen::MatrixXd &D) const
            {
                D = -2*A*B.transpose();
                D.colwise() += A.cwiseProduct(A).rowwise().sum();
                D.rowwise() += B.cwiseProduct(B).rowwise().sum().transpose();
            }

            void assertData(const Data &data) const
            {
                if (data.coord_x.empty() && data.coord_y.empty()
                        && data.coord_z.empty() && data.std_dev_x.empty()
                        && data.std_dev_y.empty() && data.std_dev_z.empty()
                        && data.label.empty())
                {
                    throw GPRegressionException("All input data is empty!");
                }
        }
};



}

#endif
