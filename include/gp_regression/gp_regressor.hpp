#ifndef GP_REGRESSION___GP_REGRESSOR_H
#define GP_REGRESSION___GP_REGRESSOR_H

#include <vector>
#include <functional>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>
#include<Eigen/StdVector>

#include <gp_regression/cov_functions.h>
#include <gp_regression/gp_regression_exception.h>

namespace gp_regression
{

/**
 * @brief The Data struct Container for input and query data.
 */
struct Data
{
        std::vector<double> coord_x;
        std::vector<double> coord_y;
        std::vector<double> coord_z;
        std::vector<double> label;
};

/**
 * @brief The Model struct Container for a Gaussian Process model.
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

/**
 * @brief The GPRegressor class
 */
template <typename CovType>
class GPRegressor
{
public:
        // pointer to the covariance function type
        CovType* kernel_;

        virtual ~GPRegressor() {}

        /**
         * @brief create Solves the regression problem given some input data.
         * @param data Input data.
         * @param gp Gaussian process parameters.
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
         * @brief evaluate The f''(x) version of evaluate.
         * @param gp The gaussian process, f(x) ~ gp[m(x), v(x)].
         * @param query The query value, x.
         * @param f The function value, m(x).
         * @param v The variance of the function value, v(x).
         * @param N The normal at the query value, f'(x) = N(f(x))
         * @param Tx First basis of the tangent plane at the query value.
         * @param Ty Second basis of the tangent plane at the query value.
         */
        void evaluate(const Model &gp, Data &query, std::vector<double> &f, std::vector<double> &v,
                      Eigen::MatrixXd &N, Eigen::MatrixXd &Tx, Eigen::MatrixXd &Ty)
        {
                evaluate(gp, query, f, v, N);
                Tx.resizeLike(N);
                Ty.resizeLike(N);
                for(int i = 0; i < N.rows(); ++i)
                {
                        Eigen::Vector3d tempTx, tempTy, tempN;
                        tempN = N.row(i);
                        computeTangentBasis(tempN, tempTx, tempTy);
                        Tx.row(i) = tempTx;
                        Ty.row(i) = tempTy;
                }
        }

        /**
         * @brief evaluate The f'(x) version of evaluate.
         * @param gp The gaussian process, f(x) ~ gp[m(x), v(x)].
         * @param query The query value, x.
         * @param f The function value, m(x).
         * @param v The variance of the function value, v(x).
         * @param N The normal at the query value, f'(x) = N(f(x))
         */
        void evaluate(const Model &gp, Data &query, std::vector<double> &f, std::vector<double> &v, Eigen::MatrixXd &N)
        {
                // validate data
                assertData(query);

                if (!query.label.empty())
                throw GPRegressionException("Query is already labeled!");

                // go!
                Eigen::MatrixXd Q;
                Eigen::MatrixXd Kqp, Kpq, Kqpdiff;
                Eigen::MatrixXd Kqq;
                //function evaluation and associated variance
                Eigen::VectorXd F, V_diagonal;
                Eigen::MatrixXd V;
                convertToEigen(query.coord_x, query.coord_y, query.coord_z, Q);
                buildEuclideanDistanceMatrix(Q, gp.P, Kqp);
                N.resizeLike(Q);

                for(int i = 0; i < Kqp.rows(); ++i)
                {
                        for(int j = 0; j < Kqp.cols(); ++j)
                        {
                                Kqp(i,j) = kernel_->compute(Kqp(i,j));
                                N.row(i) += gp.InvKppY(j)*kernel_->computediff(Kqp(i,j))*(Q.row(i) - gp.P.row(j));
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

                // normalize, or return real gradient value
                /*for(int i = 0; i < N.rows(); ++i)
                {
                        N.row(i).normalize();
                }*/
        }

        /**
         * @brief evaluate The f(x) version of evaluate.
         * @param gp The gaussian process, f(x) ~ gp[m(x), v(x)].
         * @param query The query value, x.
         * @param f The function value, m(x).
         * @param v The variance of the function value, v(x).
         */
        void evaluate(const Model &gp, Data &query, std::vector<double> &f, std::vector<double> &v)
        {
                Eigen::MatrixXd dummyN;
                dummyN.resize(query.coord_x.size(), 3);
                evaluate(gp, query, f, v, dummyN);
        }

        /**
         * @brief update Updates the gaussian process with new_data.
         * @param new_data This is the new data added to the model.
         * @param gp The gaussian process to be updated
         */
        void update(const Data &new_data, Model &gp)
        {
        }

        /**
         * @brief setCovFunction
         * @param kernel It requires the same type of kernel the regressor was
         * created with, but it can have different parameters. You need to use this function
         * if you want to change the default parameters the regressor/cov. function
         * are created with.
         */
        void setCovFunction(CovType &kernel)
        {
                kernel_ = &kernel;
        }

        /**
         * @brief GPRegressor Default constructor, it uses the default constructor of
         * the covariance function.
         */
        GPRegressor()
        {
                kernel_ = new CovType();
        }

private:

        /**
         * @brief convertToEigen
         * @param a
         * @param b
         * @param c
         * @param M
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

        /**
         * @brief convertToEigen
         * @param a
         * @param M
         */
        void convertToEigen(const std::vector<double> &a, Eigen::VectorXd &M) const
        {
                M = Eigen::Map<Eigen::VectorXd>((double *)a.data(), a.size());
        }

        /**
         * @brief convertToSTD
         * @param M
         * @param a
         */
        void convertToSTD(const Eigen::VectorXd &M, std::vector<double> &a) const
        {
                a = std::vector<double>(M.data(), M.data() + M.size());
        }

        /**
         * @brief buildEuclideanDistanceMatrix
         * @param A
         * @param B
         * @param D
         */
        void buildEuclideanDistanceMatrix(const Eigen::MatrixXd &A,
            const Eigen::MatrixXd &B,
            Eigen::MatrixXd &D) const
        {
                D = -2*A*B.transpose();
                D.colwise() += A.cwiseProduct(A).rowwise().sum();
                D.rowwise() += B.cwiseProduct(B).rowwise().sum().transpose();
        }

        /**
         * @brief assertData
         * @param data
         */
        void assertData(const Data &data) const
        {
                if (data.coord_x.empty() && data.coord_y.empty()
                    && data.coord_z.empty() && data.label.empty())
                {
                        throw GPRegressionException("All input data is empty!");
                }
        }

        /**
         * @brief computeTangentBasis
         * @param N
         * @param Tx
         * @param Ty
         */
        void computeTangentBasis(const Eigen::Vector3d &N, Eigen::Vector3d &Tx, Eigen::Vector3d &Ty)
        {
                Eigen::Vector3d NN = N.normalized();
                Eigen::Matrix3d TProj = Eigen::Matrix3d::Identity() - NN*NN.transpose();
                Eigen::JacobiSVD<Eigen::Matrix3d> svd(TProj, Eigen::ComputeFullU | Eigen::ComputeFullV);
                Tx = svd.matrixU().col(0);
                Ty = svd.matrixU().col(1);
        }
};



}

#endif
