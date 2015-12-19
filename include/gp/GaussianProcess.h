/** @file GaussianProcess.h
 *
 *
 * @author	Claudio Zito
 *
 * @copyright  Copyright (C) 2015 Claudio, University of Birmingham, UK
 *
 * @license  This file copy is licensed to you under the terms described in
 *           the License.txt file included in this distribution.
 *
 * Refer to Gaussian process library for Machine Learning.
 *
 */
#ifndef __GP_GAUSSIANPROCESS_H__
#define __GP_GAUSSIANPROCESS_H__
#define _USE_MATH_DEFINES

//------------------------------------------------------------------------------

#include <cstdio>
#include <cmath>
#include "gp/SampleSet.h"
#include "gp/CovLaplace.h"
#include "gp/CovThinPlate.h"
// #include <omp.h>

//------------------------------------------------------------------------------

namespace gp {

//------------------------------------------------------------------------------

/** Utily constants */
static const double log2pi = std::log(numeric_const<double>::TWO_PI);

//------------------------------------------------------------------------------

/*
    * Gaussian Process.
    *
    */
template <class CovTypePtr, class CovTypeDesc> class GaussianProcess {
public:
    /** Needed for good alignment for the Eigen's' data structures */
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    /** Pointer to the class */
    typedef boost::shared_ptr<GaussianProcess<CovTypePtr, CovTypeDesc> > Ptr;

    /** Descriptor file */
    class Desc {
        public:
            /** Initial size of the kernel matrix */
			size_t initialLSize;
            /** Noise use to compute K(x,x) */
			double noise;
            /** Covariance description file */
            CovTypeDesc covTypeDesc;

			/** Enable atlas */
			bool atlas;

            /** C'tor */
            Desc() {
                setToDefault();
            }

            /** Set to default */
            void setToDefault() {
                initialLSize = 1500;
                noise = numeric_const<double>::ZERO;
                covTypeDesc.setToDefault();

				atlas = false;
            }

            /** Creates the object from the description. */
            Ptr create() const {
		Ptr pointer(new GaussianProcess<CovTypePtr, CovTypeDesc>);
                pointer->create(*this);
                return pointer;
            }

            /** Assert valid descriptor files */
            bool isValid(){
                if (!std::isfinite(initialLSize))
                    return false;
                if (noise < numeric_const<double>::ZERO)
                    return false;
                return true;
            }
    };

    /** Predict f_* ~ GP(x_*) */
    virtual double f(const Vec3& xStar) {
        if (sampleset->empty()) return 0;

        //Eigen::Map<const Eigen::VectorXd> x_star(x.v, input_dim);
        compute();
        update_alpha();
        update_k_star(xStar);
        //std::cout << "size alpha=" << alpha->size() << " k_star=" << k_star->size() << std::endl;
        return (double)k_star->dot(*alpha);
    }

    /** Predict variance v[f_*] ~ var(x_*)  */
	virtual double var(const Vec3& xStar) {
        if (sampleset->empty()) return .0;

        //Eigen::Map<const Eigen::VectorXd> x_star(x.v, input_dim);
        compute();
        update_alpha();
        update_k_star(xStar); //update_k_star(x_star);
        size_t n = sampleset->rows();
        Eigen::VectorXd v = L->topLeftCorner(n, n).triangularView<Eigen::Lower>().solve(*k_star);
		return (double)cf->get(xStar, xStar) - v.dot(v); //cf->get(x_star, x_star) - v.dot(v);
    }

    	/** Predict f, var, N, Tx and Ty */
	virtual void evaluate(const Vec3& x, Real& fx, Real& varx, Eigen::Vector3d& normal, Eigen::Vector3d& tx, Eigen::Vector3d& ty) {
		clock_t t = clock();

		fx = f(x);
		varx = var(x);

		const size_t n = sampleset->rows(), nnew = 1;
		// resize L if necessary
		if (sampleset->rows() + nnew > static_cast<std::size_t>(L->rows())) {
			L->conservativeResize(nnew + initialLSize, nnew + initialLSize);
		}
//#pragma omp parallel for
//		for (size_t j = n; j < nnew; ++j) {
			Eigen::VectorXd k(n+nnew), kdiff(n+nnew);
			for (size_t i = 0; i<n; ++i) {
				k(i) = cf->get(sampleset->x(i), sampleset->x(n));
				kdiff(i) = cf->getDiff(sampleset->x(i), sampleset->x(n));
				//context.write("GP::add_patters(): Computing k(%lu, %lu)\r", i, j);
			}
			k(n) = cf->get(x, x);
			kdiff(n) = cf->getDiff(x, x);

            std::cout<<"before invkstar\n"<<std::flush;
			const Eigen::VectorXd invKstar = k.inverse();
            std::cout<<"invkstar\n"<<std::flush;
			Vec yy = sampleset->y();
			yy.push_back(fx);
			const Eigen::VectorXd invKstarY = invKstar * convertToEigen(yy);
			for (size_t i = 0; i < nnew; ++i)
				normal += invKstarY(n)*kdiff(i)*(convertToEigen(sampleset->x(i) - x));
			normal.normalize();
			computeTangentBasis(normal, tx, ty);
//		}
		printf("GP::evaluate(): Elapsed time: %.4fs\n", (float)(clock() - t) / CLOCKS_PER_SEC);
	}


    /** Set training data */
    void set(SampleSet::Ptr trainingData) {
        sampleset = trainingData;
    }

    /** Get name of the covariance function */
    std::string getName() const {
        return cf->getName();
    }

    /** Add input-output pairs to sample set. */
    void add_patterns(const Vec3Seq& newInputs, const Vec& newTargets) {
        assert(newInputs.size() == newTargets.size());

        // the size of the inputs before adding new samples
        const size_t n = sampleset->rows();
        sampleset->add(newInputs, newTargets);

        // create kernel matrix if sampleset is empty
        if (n == 0) {
            cf->setLogHyper(true);
            compute();
        }
        else {
            clock_t t = clock();
            const size_t nnew = sampleset->rows();
            // resize L if necessary
            if (sampleset->rows() > static_cast<std::size_t>(L->rows())) {
                L->conservativeResize(nnew + initialLSize, nnew + initialLSize);
            }
//#pragma omp parallel for
            for (size_t j = n; j < nnew; ++j) {
                Eigen::VectorXd k(j);
                for (size_t i = 0; i<j; ++i) {
                    k(i) = cf->get(sampleset->x(i), sampleset->x(j));
                    //printf("GP::add_patters(): Computing k(%lu, %lu)\r", i, j);
                }
                const double kappa = cf->get(sampleset->x(j), sampleset->x(j));
                L->topLeftCorner(j, j).triangularView<Eigen::Lower>().solveInPlace(k);
                L->block(j,0,1,j) = k.transpose();
                (*L)(j,j) = sqrt(kappa - k.dot(k));
            }
            printf("GP::add_patterns(): Elapsed time: %.4fs\n", (float)(clock() - t)/CLOCKS_PER_SEC);
        }
        alpha_needs_update = true;
    }

    /** Compute loglikelihood */
    double logLikelihood() {
        compute();
        update_alpha();
        int n = sampleset->rows();
		const std::vector<double>& targets = sampleset->y();
        Eigen::Map<const Eigen::VectorXd> y(&targets[0], sampleset->rows());
		double det = (double)(2 * L->diagonal().head(n).array().log().sum());
		return (double)(- 0.5*y.dot(*alpha) - 0.5*det - 0.5*n*log2pi);
    }

    /** D'tor */
    virtual ~GaussianProcess() {};

protected:

    /** Desriptor file */
    Desc desc;

    /** pointer to the covariance function type */
    CovTypePtr cf;
    /** The training sample set. */
    boost::shared_ptr<SampleSet> sampleset;
    /** Alpha is cached for performance. */
    boost::shared_ptr<Eigen::VectorXd> alpha;
    /** Last test kernel vector. */
    boost::shared_ptr<Eigen::VectorXd> k_star;
    /** Linear solver used to invert the covariance matrix. */
    // Eigen::LLT<Eigen::MatrixXd> solver;
    boost::shared_ptr<Eigen::MatrixXd> L;
    /** Input vector dimensionality. */
    size_t input_dim;
    /** Noise parameter */
	Real delta_n;
    /** Enable/disable to update the alpha vector */
    bool alpha_needs_update;
    //initial L size
	size_t initialLSize;

	/** (inward) normal sequence */
	Eigen::MatrixXd N;
	/** Tangent basis in x direction */
	Eigen::MatrixXd Tx;
	/** Tangen basis in y direction */
	Eigen::MatrixXd Ty;
	/** Weights */
	Eigen::VectorXd InvKppY;

    /** Compute k_* = K(x_*, x) */
    void update_k_star(const Vec3&x_star) {
        k_star->resize(sampleset->rows());
        for(size_t i = 0; i < sampleset->rows(); ++i) {
            (*k_star)(i) = cf->get(x_star, sampleset->x(i));
        }
    }

    /** Update alpha vector (mean) */
    void update_alpha() {
        // can previously computed values be used?
        if (!alpha_needs_update) return;
        alpha_needs_update = false;
        alpha->resize(sampleset->rows());
        // Map target values to VectorXd
        const std::vector<double>& targets = sampleset->y();
        Eigen::Map<const Eigen::VectorXd> y(&targets[0], sampleset->rows());
        size_t n = sampleset->rows();
        *alpha = L->topLeftCorner(n, n).triangularView<Eigen::Lower>().solve(y);
        L->topLeftCorner(n, n).triangularView<Eigen::Lower>().adjoint().solveInPlace(*alpha);
    }

    /** Compute covariance matrix and perform cholesky decomposition. */
    virtual void compute() {
        // can previously computed values be used?
        if (!cf->getLogHyper()) return;
        clock_t t = clock();
        cf->setLogHyper(false);
        // input size
        const size_t n = sampleset->rows();
		const size_t dim = sampleset->cols();
		// resize L if necessary
        if (n > L->rows()) L->resize(n + initialLSize, n + initialLSize);
        // compute kernel matrix (lower triangle)
        size_t counter = 0;
		// differential of covariance with selected kernel
		Eigen::MatrixXd Kppdiff;
		Kppdiff.resize(n, n);
		N.resize(n, dim);
		Tx.resize(n, dim);
		Ty.resize(n, dim);
        //#pragma omp parallel for
        for(size_t i = 0; i < n; ++i) {
            for(size_t j = 0; j <= i; ++j) {
                const double k_ij = cf->get(sampleset->x(i), sampleset->x(j));
                // add noise on the diagonal of the kernel
                (*L)(i, j) = i == j ? k_ij + delta_n : k_ij;
                //printf("GP::compute(): Computing k(%lu, %lu)\r", i, j);
				if (desc.atlas)
					Kppdiff(i, j) = cf->getDiff(sampleset->x(i), sampleset->x(j));
            }
		}
		// atlas computation
		if (desc.atlas) {
			InvKppY = L->inverse() * convertToEigen(sampleset->y());

//#pragma omp parallel for
			for (size_t i = 0; i < L->rows(); ++i) {
				for (size_t j = 0; j < L->cols(); ++j)
					N.row(i) += InvKppY(j)*Kppdiff(i, j)*(convertToEigen(sampleset->x(i) - sampleset->x(j)));

				N.row(i).normalize();
				Eigen::Vector3d Ni = N.row(i);
				Eigen::Vector3d Txi, Tyi;
				computeTangentBasis(Ni, Txi, Tyi);
				Tx.row(i) = Txi;
				Ty.row(i) = Tyi;
			}
		}

		// perform cholesky factorization
        //solver.compute(K.selfadjointView<Eigen::Lower>());
        L->topLeftCorner(n, n) = L->topLeftCorner(n, n).selfadjointView<Eigen::Lower>().llt().matrixL();
        alpha_needs_update = true;
        printf("GP::Compute(): Elapsed time: %.4fs\n", (float)(clock() - t)/CLOCKS_PER_SEC);
    }

	/** Compute tangent basis. N must be normalised */
	void computeTangentBasis(const Eigen::Vector3d &N, Eigen::Vector3d &Tx, Eigen::Vector3d &Ty)
	{
		Eigen::Matrix3d TProj = Eigen::Matrix3d::Identity() - N*N.transpose();
		Eigen::JacobiSVD<Eigen::Matrix3d> svd(TProj, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Tx = svd.matrixU().col(0);
		Ty = svd.matrixU().col(1);
	}

    /** Create from description file */
    void create(const Desc& desc) {
		this->desc = desc;

        cf = desc.covTypeDesc.create();
        //sampleset = desc.trainingData;
        input_dim = 3;
        delta_n = desc.noise;
        initialLSize = desc.initialLSize;
        alpha.reset(new Eigen::VectorXd);
        k_star.reset(new Eigen::VectorXd);
        L.reset(new Eigen::MatrixXd);
        L->resize(initialLSize, initialLSize);
        alpha_needs_update = true;
    }

    /** Default C'tor */
	GaussianProcess() {}
};

//------------------------------------------------------------------------------

/** List of legal types */
typedef GaussianProcess<gp::BaseCovFunc::Ptr, gp::Laplace::Desc> LaplaceRegressor;
//typedef GaussianProcess<gp::Laplace::Ptr, gp::Laplace::Desc> LaplaceRegressor;
//typedef GaussianProcess<gp::ThinPlate> ThinPlateRegressor;
//class LaplaceRegressor : public GaussianProcess<gp::Laplace> {};

//------------------------------------------------------------------------------

}
#endif /* __GP_GAUSSIANPROCESS_H__ */

