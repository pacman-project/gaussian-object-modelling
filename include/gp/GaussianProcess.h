#ifndef __GP_GAUSSIANPROCESS_H__
#define __GP_GAUSSIANPROCESS_H__
#define _USE_MATH_DEFINES

//------------------------------------------------------------------------------

#include <cstdio>
#include <cmath>
#include <Eigen/Dense>
#include "gp/SampleSet.h"
#include "gp/CovLaplace.h"
#include "gp/CovThinPlate.h"
#include <omp.h>

//------------------------------------------------------------------------------

namespace gp {

//------------------------------------------------------------------------------

static const double log2pi = std::log(numeric_const<double>::TWO_PI);
static const double initial_L_size = 15000;
/*
 * \brief Handle for propagating a single GP map from 3D points to paramters, 
 * from-to parameters of 3D points, sample points, etc. 
 *
 */

template <class CovTypePtr, class CovTypeDesc> class GaussianProcess {
public: 
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	/** POinter to the class */
	typedef boost::shared_ptr<GaussianProcess<CovTypePtr, CovTypeDesc> > Ptr;
	
	/** Descriptor file */
	class Desc {
	public:
		double initialLSize;
		double noise;
		CovTypeDesc covTypeDesc;
		
		/** C'tor */
		Desc() {
			setToDefault();
		}
		
		/** Set to default */
		void setToDefault() {
			initialLSize = 10000;
			noise = numeric_const<double>::ZERO;
			covTypeDesc.setToDefault();
		}
		
		/** Creates the object from the description. */
		//CREATE_FROM_OBJECT_TEMPLATE_DESC_0(GaussianProcess, CovType, GaussianProcess::Ptr, SampleSet::Ptr)
		Ptr create() const {
//			GaussianProcess<CovTypePtr, CovTypeDesc> *pObject = new GaussianProcess<CovTypePtr, CovTypeDesc>();
//			Ptr pointer(pObject);
//			pObject->create(*this);
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
	
	/** Predict target value for given input.
	* @param x input vector
	* @return predicted value */
	virtual double f(const Vec3& xStar) {
		if (sampleset->empty()) return 0;

		//Eigen::Map<const Eigen::VectorXd> x_star(x.v, input_dim);
		compute();
		update_alpha();
		update_k_star(xStar);
		//std::cout << "size alpha=" << alpha->size() << " k_star=" << k_star->size() << std::endl;
		return k_star->dot(*alpha);
	}
	
	/** Predict variance of prediction for given input.
	* @param x input vector
	* @return predicted variance */
	virtual double var(const Vec3& xStar) {
		if (sampleset->empty()) return 0;
		
		//Eigen::Map<const Eigen::VectorXd> x_star(x.v, input_dim);
		compute();
		update_alpha();
		update_k_star(xStar); //update_k_star(x_star);
		int n = sampleset->size();
		Eigen::VectorXd v = L->topLeftCorner(n, n).triangularView<Eigen::Lower>().solve(*k_star);
		return cf->get(xStar, xStar) - v.dot(v); //cf->get(x_star, x_star) - v.dot(v);
	}
	
	void set(SampleSet::Ptr trainingData) {
		sampleset = trainingData;
	}
	
	std::string getName() const {
		return cf->getName();
	}
	
	/** Add input-output-pair to sample set.
	* Add a copy of the given input-output-pair to sample set.
	* @param x input array
	* @param y output value
	*/
	void add_patterns(const Vec3Seq& newInputs, const Vec& newTargets) {
		assert(newInputs.size() == newTargets.size());
		
		// the size of the inputs before adding new samples
		const size_t n = sampleset->size();
		sampleset->add(newInputs, newTargets);
		
		// create kernel matrix if sampleset is empty
		if (n == 0) {
			cf->setLogHyper(true);
			compute();
		}
		else {
			clock_t t = clock();
			const size_t nnew = sampleset->size();
			// resize L if necessary
			if (sampleset->size() > static_cast<std::size_t>(L->rows())) {
				L->conservativeResize(nnew + initial_L_size, nnew + initial_L_size);
			}
			#pragma omp parallel for
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
	
	double log_likelihood() {
		compute();
		update_alpha();
		int n = sampleset->size();
		const std::vector<double>& targets = sampleset->y();
		Eigen::Map<const Eigen::VectorXd> y(&targets[0], sampleset->size());
		double det = 2 * L->diagonal().head(n).array().log().sum();
		return -0.5*y.dot(*alpha) - 0.5*det - 0.5*n*log2pi;
	}
	
	virtual ~GaussianProcess() {};
	
protected:
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
	double delta_n;
	bool alpha_needs_update;
	
	
	void update_k_star(const Vec3&x_star) {
		k_star->resize(sampleset->size());
		for(size_t i = 0; i < sampleset->size(); ++i) {
			(*k_star)(i) = cf->get(x_star, sampleset->x(i));
		}
	}
	void update_alpha() {
		// can previously computed values be used?
		if (!alpha_needs_update) return;
		alpha_needs_update = false;
		alpha->resize(sampleset->size());
		// Map target values to VectorXd
		const std::vector<double>& targets = sampleset->y();
		Eigen::Map<const Eigen::VectorXd> y(&targets[0], sampleset->size());
		int n = sampleset->size();
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
		const size_t n = sampleset->size();
		// resize L if necessary
		if (n > L->rows()) L->resize(n + initial_L_size, n + initial_L_size);
		// compute kernel matrix (lower triangle)
		size_t counter = 0;
		//#pragma omp parallel for
		for(size_t i = 0; i < n; ++i) {
			for(size_t j = 0; j <= i; ++j) {
				const double k_ij = cf->get(sampleset->x(i), sampleset->x(j));
				// add noise on the diagonal of the kernel
				(*L)(i, j) = i == j ? k_ij + delta_n : k_ij;
				//printf("GP::compute(): Computing k(%lu, %lu)\r", i, j);
			}
		}
		// perform cholesky factorization
		//solver.compute(K.selfadjointView<Eigen::Lower>());
		L->topLeftCorner(n, n) = L->topLeftCorner(n, n).selfadjointView<Eigen::Lower>().llt().matrixL();
		alpha_needs_update = true;
		printf("GP::Compute(): Elapsed time: %.4fs\n", (float)(clock() - t)/CLOCKS_PER_SEC);
	}
	
	/** Create from description file */
	void create(const Desc& desc) {
		cf = desc.covTypeDesc.create();
		//sampleset = desc.trainingData;
		input_dim = 3;
		delta_n = desc.noise;
		alpha.reset(new Eigen::VectorXd);
		k_star.reset(new Eigen::VectorXd);
		L.reset(new Eigen::MatrixXd);
		L->resize(desc.initialLSize, desc.initialLSize);
		alpha_needs_update = true;
	}
	
	/** C'tor */
	GaussianProcess() {}
};

//------------------------------------------------------------------------------

typedef GaussianProcess<gp::BaseCovFunc::Ptr, gp::Laplace::Desc> LaplaceRegressor;
//typedef GaussianProcess<gp::Laplace::Ptr, gp::Laplace::Desc> LaplaceRegressor;
//typedef GaussianProcess<gp::ThinPlate> ThinPlateRegressor;
//class LaplaceRegressor : public GaussianProcess<gp::Laplace> {};

//------------------------------------------------------------------------------

}
#endif /* __GP_H__ */
