#ifndef GP_REGRESSION___GP_REGRESSOR_H
#define GP_REGRESSION___GP_REGRESSOR_H

#include <vector>
#include <functional>

#include <Eigen/Core>
#include <Eigen/LU>

#include <gp_regression/cov_functions.h>
#include <gp_regression/gp_regression_exception.h>

namespace gp_regression
{

/*
 * \brief Container for input data representing the raw 3D points
 */
struct Data
{
        std::vector<double> coord_x;
        std::vector<double> coord_y;
        std::vector<double> coord_z;
        std::vector<double> std_dev_x;
        std::vector<double> std_dev_y;
        std::vector<double> std_dev_z;
        std::vector<double> label;
};

/*
 * \brief Container for model parameters that represent an object
 */
struct Model
{
        Eigen::MatrixXd P; // points 
        Eigen::MatrixXd E; // noise
        Eigen::VectorXd Y;  // labels
        Eigen::MatrixXd Kpp; // covariance with selected kernel
        Eigen::MatrixXd InvKpp; // inverse of covariance with selected kernel
        Eigen::VectorXd InvKppY; // weights
};

/*
 * \brief Handle for propagating a single GP map from 3D points to paramters, 
 * from-to parameters of 3D points, sample points, etc. 
 *
 */
template <class CovType>
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
                convertToEigen(data.std_dev_x, data.std_dev_y, data.std_dev_z, gp.E);
                convertToEigen(data.label, gp.Y);

                // go! // ToDo: avoid the for loops.
                buildEuclideanDistanceMatrix(gp.P, gp.P, gp.Kpp);

                for(int i = 0; i < gp.Kpp.rows(); ++i)
                {
                        for(int j = 0; j < gp.Kpp.cols(); ++j)
                        {
                               gp.Kpp(i,j) = kernel_->compute(gp.Kpp(i,j)); 
                        }
                }

                gp.InvKpp = gp.Kpp.inverse();
                gp.InvKppY = gp.InvKpp*gp.Y;
        }

        /** 
         * \brief Generates data using the GP
         * \param[in]  data 3D points.
         */
        void estimate(const Model &gp, Data &query)
        {
                // validate gp

                // validate data
                assertData(query);

                if (!query.label.empty())
                        throw GPRegressionException("Query is already labeled!");

                // go!
                Eigen::MatrixXd Q;
                Eigen::MatrixXd Kqp;
                Eigen::VectorXd YPost;
                convertToEigen(query.coord_x, query.coord_y, query.coord_z, Q);
                buildEuclideanDistanceMatrix(Q, gp.P, Kqp);
                
                for(int i = 0; i < Kqp.rows(); ++i)
                {
                        for(int j = 0; j < Kqp.cols(); ++j)
                        {
                               Kqp(i,j) = kernel_->compute(Kqp(i,j)); 
                        }
                }
                
                YPost = Kqp*gp.InvKppY;

                convertToSTD(YPost, query.label);
        }

        /** 
         * \brief Updates the GP with the new data
         * \param[in]  data 3D points.
         */
        void update(const Data &new_data, Model &gp)
        {
        	create(new_data, gp);                
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
        	double rowSize = M.rows() + a.size();
                M.resize(rowSize, 3);
                M.col(0) << Eigen::Map<Eigen::VectorXd>((double *)a.data(), a.size());
                M.col(1) << Eigen::Map<Eigen::VectorXd>((double *)b.data(), b.size());
                M.col(2) << Eigen::Map<Eigen::VectorXd>((double *)c.data(), c.size());
        }

        void convertToEigen(const std::vector<double> &a, Eigen::VectorXd &M) const
        {
        	M.resize(M.size() + a.size());
                M << Eigen::Map<Eigen::VectorXd>((double *)a.data(), a.size());
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

// Convenience typedefs. Note that these will use the default constructors!

class GaussianRegressor : public GPRegressor<gp_regression::Gaussian> {};
// class LaplaceRegressor : public GPRegressor<gp_regression::Laplace> {};
// class InvMultiQuadRegressor : public GPRegressor<gp_regression::InvMultiQuad> {};

}

#endif
