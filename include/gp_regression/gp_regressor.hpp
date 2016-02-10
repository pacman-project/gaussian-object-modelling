#ifndef GP_REGRESSION___GP_REGRESSOR_H
#define GP_REGRESSION___GP_REGRESSOR_H

#include <vector>
#include <functional>
#include <memory>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <Eigen/StdVector>

#include <gp_regression/cov_functions.h>
#include <gp_regression/gp_regression_exception.h>

namespace gp_regression
{

/**
 * @brief computeTangentBasis
 * @param[in] grad  The gradient (Unnormalized)
 * @param[out] N The Normal (normalized)
 * @param[out] Tx First tangent unit vector
 * @param[out] Ty Second tangent unit vector
 */
//Moved outside of class, this is more an utility than an active part of
//regression. It is more convenient as a global function. - Tabjones
void computeTangentBasis(const Eigen::Vector3d &grad, Eigen::Vector3d &N, Eigen::Vector3d &Tx, Eigen::Vector3d &Ty)
{
    N = grad.normalized();
    Tx = Eigen::Vector3d::UnitX() - (N*(N.dot(Eigen::Vector3d::UnitX())));
    Tx.normalize();
    Ty = N.cross(Tx);
    Ty.normalize();
}

/**
 * @brief The Data struct Container for input and query data.
 */
struct Data
{
        std::vector<double> coord_x;
        std::vector<double> coord_y;
        std::vector<double> coord_z;
        std::vector<double> label;
        std::vector<double> sigma2;
        typedef std::shared_ptr<Data> Ptr;
        typedef std::shared_ptr<const Data> ConstPtr;
        void clear()
        {
            coord_x.clear();
            coord_y.clear();
            coord_z.clear();
            label.clear();
            sigma2.clear();
        }
};

/**
 * @brief The Model struct Container for a Gaussian Process model.
 */
struct Model
{
        double R;          // larger pairwise distance in training (includes internal/external)
        Eigen::MatrixXd P; // points
        Eigen::VectorXd Y;  // labels
        Eigen::VectorXd S2;  // noise
        Eigen::MatrixXd N; // (inward) normal at points [not computed by default]
        Eigen::MatrixXd Tx; // tangent basis 1 [not computed by default]
        Eigen::MatrixXd Ty; // tangent basis 2 [not computed by default]
        Eigen::MatrixXd Kpp; // the covariance matrix
        Eigen::LDLT<Eigen::MatrixXd> cholesker; // the robust cholesky-based solver
        Eigen::VectorXd alpha; // weights, alpha, this is the only required thing to keep
        Eigen::MatrixXd Kppdiff; // differential of covariance with selected kernel [not computed by default]
        Eigen::MatrixXd Kppdiffdiff; // twice differential of covariance with selected kernel [not computed by default]
        typedef std::shared_ptr<Model> Ptr;
        typedef std::shared_ptr<const Model> ConstPtr;
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
         * @param[in] data Input data.
         * @param[out] gp Gaussian process parameters.
         *
         * \Note: the templatation of >bool withNormals> is not friendly readable,
         *        however it makes it efficient by solving the branching at
         *        compilation time (assuming a modern and good compiler)
         */
        template <bool withNormals>
        void create(Data::ConstPtr data, Model::Ptr &gp)
        {
                // validate data
                assertData(data);

                // reset output, we dont care what there was there... yeah, we are badasses
                gp = std::make_shared<Model>();

                // configure gp matrices
                convertToEigen(data->coord_x, data->coord_y, data->coord_z, gp->P);
                convertToEigen(data->label, gp->Y);
                convertToEigen(data->sigma2, gp->S2);

                if(withNormals)
                {
                        gp->N.resize(gp->P.rows(), gp->P.cols());
                        // gp->Tx.resize(gp->P.rows(), gp->P.cols());
                        // gp->Ty.resize(gp->P.rows(), gp->P.cols());
                }

                // compute pairwise distance matrix / covariance
                buildEuclideanDistanceMatrix(gp->P, gp->P, gp->Kpp);

                // find larger pairwise distance and normalize pairwise distance matrix
                gp->R = gp->Kpp.maxCoeff();

                // copy euclidean matrix
                if(withNormals)
                {
                        gp->Kppdiff.resizeLike(gp->Kpp);
                        // gp->Kppdiffdiff.resizeLike(Kpp);
                }

                for(int i = 0; i < gp->Kpp.rows(); ++i)
                {
                        for(int j = 0; j < gp->Kpp.cols(); ++j)
                        {
                                // do it in this order, so you can make the most of the same matrix
                                if(withNormals)
                                {
                                        // gp->Kppdiffdiff(i,j) = kernel_->computediffdiff(gp->Kpp(i,j));
                                        gp->Kppdiff(i,j) = kernel_->computediff(gp->Kpp(i,j));
                                }
                                if (!(data->sigma2.empty()) && i==j)
                                        gp->Kpp(i,j) = kernel_->compute(gp->Kpp(i,j)) + data->sigma2.at(i);
                                else
                                        gp->Kpp(i,j) = kernel_->compute(gp->Kpp(i,j));
                        }
                }

                gp->cholesker.setZero();
                gp->cholesker.compute(gp->Kpp);
                gp->alpha = gp->cholesker.solve(gp->Y);

                // normal and tangent computation
                if(withNormals)
                {
                        for(int i = 0; i < gp->Kpp.rows(); ++i)
                        {
                                for(int j = 0; j < gp->Kpp.cols(); ++j)
                                {
                                        gp->N.row(i) += gp->alpha(j)*gp->Kppdiff(i,j)*(gp->P.row(i) - gp->P.row(j));
                                }
                                gp->N.row(i).normalize();
                                Eigen::Vector3d N = gp->N.row(i);
                                // Eigen::Vector3d Tx, Ty;
                                // computeTangentBasis(N, Tx, Ty);
                                // gp->Tx.row(i) = Tx;
                                // gp->Ty.row(i) = Ty;
                        }
                }
        }

        /**
         * @brief evaluate The f''(x) version of evaluate.
         * @param[in] gp The gaussian process, f(x) ~ gp[m(x), v(x)].
         * @param[in] query The query value, x.
         * @param[out] f The function value, m(x).
         * @param[out] v The variance of the function value, v(x).
         * @param[out] N The normal at the query value, f'(x) = N(f(x))
         * @param[out] Tx First basis of the tangent plane at the query value.
         * @param[out] Ty Second basis of the tangent plane at the query value.
         */
        void evaluate(Model::ConstPtr gp, Data::ConstPtr query, std::vector<double> &f, std::vector<double> &v,
                      Eigen::MatrixXd &N, Eigen::MatrixXd &Tx, Eigen::MatrixXd &Ty)
        {
                if(!gp)
                        throw GPRegressionException("Empty Model pointer");

                evaluate(gp, query, f, v, N);
                Tx.resizeLike(N);
                Ty.resizeLike(N);

                for(int i = 0; i < N.rows(); ++i)
                {
                        Eigen::Vector3d tempTx, tempTy, tempN, tempG;
                        tempG = N.row(i);
                        computeTangentBasis(tempG, tempN, tempTx, tempTy);
                        Tx.row(i) = tempTx;
                        Ty.row(i) = tempTy;
                }
        }

        /**
         * @brief evaluate The f'(x) version of evaluate.
         * @param[in] gp The gaussian process, f(x) ~ gp[m(x), v(x)].
         * @param[in] query The query value, x.
         * @param[out] f The function value, m(x).
         * @param[out] v The variance of the function value, v(x).
         * @param[out] N The normal (un-normalized) at the query value, f'(x) = N(f(x))
         */
        void evaluate(Model::ConstPtr gp, Data::ConstPtr query, std::vector<double> &f, std::vector<double> &v, Eigen::MatrixXd &N)
        {
                if(!gp)
                        throw GPRegressionException("Empty Model pointer");

                // validate data
                assertData(query);

                if (!query->label.empty())
                        throw GPRegressionException("Query is already labeled!");

                // go!
                Eigen::MatrixXd Q;
                Eigen::MatrixXd Kqp, Kpq;
                Eigen::MatrixXd Kqq;
                Eigen::VectorXd F, V_diagonal;
                Eigen::MatrixXd V, Vt;
                convertToEigen(query->coord_x, query->coord_y, query->coord_z, Q);
                buildEuclideanDistanceMatrix(Q, gp->P, Kqp);
                N.resizeLike(Q);

                for(int i = 0; i < Kqp.rows(); ++i)
                {
                        for(int j = 0; j < Kqp.cols(); ++j)
                        {
                                N.row(i) += gp->alpha(j)*kernel_->computediff(Kqp(i,j))*(Q.row(i) - gp->P.row(j));
                                Kqp(i,j) = kernel_->compute(Kqp(i,j));
                        }
                        // N.row(i).normalize(); // to return the gradient properly
                }
                F = Kqp*gp->alpha;

                // needed for the variance
                Kpq = Kqp.transpose();
                buildEuclideanDistanceMatrix(Q, Q, Kqq);

                for(int i = 0; i < Kqq.array().size(); ++i)
                        Kqq.array()(i) = kernel_->compute(Kqq.array()(i));

                // V = gp->cholesker.matrixL().solve(Kpq); // this is giving negative and large values
                                                           // perhaps it is not the correct function
                V = gp->cholesker.solve(Kpq);
                // Vt = V.transpose();
                V = Kqq - Kqp*V;
                V_diagonal = V.diagonal();

                // conversions
                convertToSTD(F, f);
                convertToSTD(V_diagonal, v);
        }

        /**
         * @brief evaluate The f(x) version of evaluate.
         * @param gp The gaussian process, f(x) ~ gp[m(x), v(x)].
         * @param query The query value, x.
         * @param f The function value, m(x).
         * @param v The variance of the function value, v(x).
         */
        void evaluate(Model::ConstPtr gp, Data::ConstPtr query, std::vector<double> &f, std::vector<double> &v)
        {
                if(!gp)
                        throw GPRegressionException("Empty Model pointer");

                // validate data
                assertData(query);

                if (!query->label.empty())
                        throw GPRegressionException("Query is already labeled!");

                // go!
                Eigen::MatrixXd Q;
                Eigen::MatrixXd Kqp, Kpq;
                Eigen::MatrixXd Kqq;
                Eigen::VectorXd F, V_diagonal;
                Eigen::MatrixXd V, Vt;
                convertToEigen(query->coord_x, query->coord_y, query->coord_z, Q);
                buildEuclideanDistanceMatrix(Q, gp->P, Kqp);

                for(int i = 0; i < Kqp.array().size(); ++i)
                        Kqp.array()(i) = kernel_->compute(Kqp.array()(i));

                F = Kqp*gp->alpha;

                // needed for the variance
                Kpq = Kqp.transpose();
                buildEuclideanDistanceMatrix(Q, Q, Kqq);

                for(int i = 0; i < Kqq.array().size(); ++i)
                        Kqq.array()(i) = kernel_->compute(Kqq.array()(i));

                // V = gp->cholesker.matrixL().solve(Kpq); // this is giving negative and large values
                                                           // perhaps it is not the correct function
                V = gp->cholesker.solve(Kpq);
                // Vt = V.transpose();
                V = Kqq - Kqp*V;
                V_diagonal = V.diagonal();

                // conversions
                convertToSTD(F, f);
                convertToSTD(V_diagonal, v);
        }

        /**
         * @brief evaluate The f'(x) version of evaluate.
         * @param[in] gp The gaussian process, f(x) ~ gp[m(x), v(x)].
         * @param[in] query The query value, x.
         * @param[out] f The function value, m(x).
         */
        void evaluate(Model::ConstPtr gp, Data::ConstPtr query, std::vector<double> &f)
        {
                if(!gp)
                        throw GPRegressionException("Empty Model pointer");

                // validate data
                assertData(query);

                if (!query->label.empty())
                        throw GPRegressionException("Query is already labeled!");

                // go!
                Eigen::MatrixXd Q;
                Eigen::MatrixXd Kqp;
                Eigen::VectorXd F;
                convertToEigen(query->coord_x, query->coord_y, query->coord_z, Q);
                buildEuclideanDistanceMatrix(Q, gp->P, Kqp);

                for(int i = 0; i < Kqp.array().size(); ++i)
                        Kqp.array()(i) = kernel_->compute(Kqp.array()(i));

                F = Kqp*gp->alpha;

                // conversions
                convertToSTD(F, f);
        }

        /**
         * @brief update Updates the gaussian process with new_data.
         * @param new_data This is the new data added to the model.
         * @param gp The gaussian process to be updated
         *
         *  \Note: The template could be avoided with an argument,
         *         just keeping it for consistency
         */
        template <bool withNormals>
        void update(Data::ConstPtr new_data, Model::Ptr gp)
        {
                // validate new data
                assertData(new_data);

                if(!gp)
                        throw GPRegressionException("Empty model pointer");

                // configure gp matrices
                Eigen::MatrixXd new_P;
                Eigen::VectorXd new_Y, new_S2;
                convertToEigen(new_data->coord_x, new_data->coord_y, new_data->coord_z, new_P);
                convertToEigen(new_data->label, new_Y);
                convertToEigen(new_data->sigma2, new_S2);

                int n = new_data->label.size();
                int p = gp->Y.rows();

                // compute extra values
                if(withNormals)
                {
                        // ToDO
                        // gp->N.resize(gp->P.rows(), gp->P.cols());
                        // gp->Tx.resize(gp->P.rows(), gp->P.cols());
                        // gp->Ty.resize(gp->P.rows(), gp->P.cols());
                }

                // compute pairwise distance matrix / covariance
                Eigen::MatrixXd Kpn, Knn, Knp;
                buildEuclideanDistanceMatrix(new_P, new_P, Knn);
                buildEuclideanDistanceMatrix(gp->P, new_P, Kpn);

                // copy euclidean matrix
                if(withNormals)
                {
                        // ToDO
                        // gp->Kppdiff.resizeLike(gp->Kpp);
                        // gp->Kppdiffdiff.resizeLike(Kpp);
                }

                for(int i = 0; i < p; ++i)
                {
                        for(int j = 0; j < n; ++j)
                        {
                                // do it in this order, so you can make the most of the same matrix
                                if(withNormals)
                                {
                                        // ToDO
                                        // gp->Kppdiffdiff(i,j) = kernel_->computediffdiff(gp->Kpp(i,j));
                                        // gp->Kppdiff(i,j) = kernel_->computediff(gp->Kpp(i,j));
                                }
                                Kpn(i,j) = kernel_->compute(Kpn(i,j));
                        }
                }
                Knp = Kpn.transpose();

                for(int i = 0; i < n; ++i)
                {
                        for(int j = 0; j < n; ++j)
                        {
                                // do it in this order, so you can make the most of the same matrix
                                if(withNormals)
                                {
                                        // ToDO
                                        // gp->Kppdiffdiff(i,j) = kernel_->computediffdiff(gp->Kpp(i,j));
                                        // gp->Kppdiff(i,j) = kernel_->computediff(gp->Kpp(i,j));
                                }
                                if (!(new_data->sigma2.empty()) && i==j)
                                        Knn(i,j) = kernel_->compute(Knn(i,j)) + new_data->sigma2.at(i);
                                else
                                        Knn(i,j) = kernel_->compute(Knn(i,j));
                        }
                }

                gp->Kpp.conservativeResize(gp->Kpp.rows() + n, gp->Kpp.cols() + n);
                gp->Kpp.block(p, p, n, n) = Knn;
                gp->Kpp.block(0, p, p, n) = Kpn;
                gp->Kpp.block(p, 0, n, p) = Knp;

                gp->Y.conservativeResize(p + n);
                gp->Y.block(p, 0, n, 1) = new_Y;
                gp->S2.conservativeResize(p + n);
                gp->S2.block(p, 0, n, 1) = new_S2;
                gp->P.conservativeResize(p + n, 3);
                gp->P.block(p, 0, n, 3) = new_P;

                // ToDO: find new larger pairwise distance
                // gp->R = gp->Kpp.maxCoeff();

                gp->cholesker.setZero();
                gp->cholesker.compute(gp->Kpp);
                gp->alpha = gp->cholesker.solve(gp->Y);

                // normal and tangent computation
                if(withNormals)
                {       // ToDO
                        /*for(int i = 0; i < Kpp.rows(); ++i)
                        {
                                for(int j = 0; j < Kpp.cols(); ++j)
                                {
                                        gp->N.row(i) += gp->alpha(j)*gp->Kppdiff(i,j)*(gp->P.row(i) - gp->P.row(j));
                                }
                                gp->N.row(i).normalize();
                                Eigen::Vector3d N = gp->N.row(i);
                                // Eigen::Vector3d Tx, Ty;
                                // computeTangentBasis(N, Tx, Ty);
                                // gp->Tx.row(i) = Tx;
                                // gp->Ty.row(i) = Ty;
                        }*/
                }
                return;
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
                for(int i = 0; i < D.array().size(); ++i)
                        D.array()(i) = std::sqrt(D.array()(i));
        }

        /**
         * @brief assertData
         * @param data
         */
        void assertData(Data::ConstPtr data) const
        {
                if (!data)
                        throw GPRegressionException("Empty data pointer");
                if (data->coord_x.empty() && data->coord_y.empty()
                    && data->coord_z.empty() && data->label.empty())
                {
                        throw GPRegressionException("All input data is empty!");
                }
        }

};
}

#endif
