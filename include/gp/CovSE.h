/** @file CovCovSE.h
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
#ifndef __GP__COVSE_H__
#define __GP__COVSE_H__

//------------------------------------------------------------------------------

#include "gp/Covs.h"
#include <cmath>

//------------------------------------------------------------------------------

namespace gp {

//------------------------------------------------------------------------------

class CovSE : public BaseCovFunc {
public:
	typedef boost::shared_ptr<CovSE> Ptr;

	/** Descriptor file */
	class Desc : public BaseCovFunc::Desc {
	public:
		typedef boost::shared_ptr<CovSE::Desc> Ptr;
		double sigma;
        double length;
		
		/** Default C'tor */
		Desc() {
			setToDefault();
		}
		
		/** Set values to default */
		void setToDefault() {
			BaseCovFunc::Desc::setToDefault();
			inputDim = 0;
			paramDim = 2;
			sigma = 1.0;
			length = 1.0;
		}
		
		/** Creates the object from the description. */
		CREATE_FROM_OBJECT_DESC_0(CovSE, BaseCovFunc::Ptr)
		
		/** Assert valid descriptor files */
		bool isValid(){ 
			if (!std::isfinite(sigma) || !std::isfinite(length))
				return false;
			return true;
		}
	};
	
	/** Get name of the covariance functions */
	virtual std::string getName() const {
		return "CovSE";
	}
        
	/** Compute the kernel */
	virtual double get(const Eigen::VectorXd& xi, const Eigen::VectorXd& xj, const bool dirac = false) const {
		const double z = ((xi - xj) / ell).squaredNorm();
		const double noise = dirac ? sn2 : .0;
		return sf2*exp(-0.5*z) + noise;
	}

	/** SE derivate = -1*invLenght*[sigma_f^2*exp(sqrt((p_1-p_2)'*(p_1-p_2))/(-leng))] */
	virtual inline double getDiff(const Eigen::VectorXd& xi, const Eigen::VectorXd& xj, const size_t dx, const bool dirac = false) const {
		const double k = get(xi, xj, dirac); //sqrt(DD);
		// if dx < 0 then I compute the sum of the partial derivative
		return dx < 0 ? ((xj - xi) / ell).sum() * k : -sf2 * (1 / ell) * (xi(dx) - xj(dx)) * k;
		//-golem::REAL_ONE * ell * k
	}

	virtual inline double getDiff2(const Eigen::VectorXd& xi, const Eigen::VectorXd& xj, const size_t dx1, const size_t dx2, const bool dirac = false) const {
		const double k = get(xi, xj, dirac); //sqrt(DD);
		const double noise = dirac ? sn2 : .0;
//		printf("sf2=%f ell=%f 1/ell=%f noise=%f xi-xj=%f xi-xj=%f, p1=%f p2=%f\n", sf2, ell, 1/ell, sn2, xi[dx1] - xj[dx1], xi[dx2] - xj[dx2], sf2 * ell * noise, sf2 * ell * (xi[dx1] - xj[dx1]) * (xi[dx2] - xj[dx2]) * k);
		return sf2 * (1 / ell) * (noise - (1 / ell) * (xi[dx1] - xj[dx1]) * (xi[dx2] - xj[dx2])) * k;
		//		return noise + (1/ell * k) + (((xj(dx1) - xi(dx1)) * (xj(dx2) - xi(dx2))) / (ell*ell)) * k;
	}

	/** Covariance gradient of two input vectors with respect to the hyperparameters.
	*  @param x1 first input vector
	*  @param x2 second input vector
	*  @param grad covariance gradient */
	virtual void grad(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2, Eigen::VectorXd& g) const {
		double z = ((x1 - x2) / ell).squaredNorm();
		double k = sf2*exp(-0.5*z);
		g << k*z, 2 * k;
		//double z = loghyper(0) != .0 ? -(x1 - x2).lpNorm<2>()*get(x1, x2) / pow(loghyper(0), 2.0) : -(x1 - x2).lpNorm<2>()*get(x1, x2);//((x1 - x2)*sqrt3 / invLength).norm();
		//const double EE = (double)(x1 - x2).lpNorm<2>();//sqrt(DD);
		//const double power = -1 * EE*invLength;
		//double k = 2 * loghyper(1)*exp(power);
		//g << z, k;
	};

	/** Update parameter vector.
	*  @param p new parameter vector */
	virtual void setLogHyper(const Eigen::VectorXd &p) {
		BaseCovFunc::setLogHyper(p);
		ell = exp(loghyper(0));
		sf2 = exp(2 * loghyper(1)); //2 * loghyper(1) * loghyper(1);//
	}

	~CovSE(){}

private:
	/** Hyper-parameters */
	double ell; // second element of loghyper
	
	double sf2; // first element of loghyper
        
    /** Create from descriptor */
	void create(const Desc& desc) {
		BaseCovFunc::create(desc);
		loghyper.resize(paramDim);
		loghyper(0) = desc.length;
		loghyper(1) = desc.sigma;
		ell = exp(loghyper(0));
		sf2 = exp(2 * loghyper(1)); //2 * loghyper(1) * loghyper(1);//
	}
        
    CovSE() : BaseCovFunc() {}
};

//------------------------------------------------------------------------------

class CovSEArd : public BaseCovFunc {
public:
	typedef boost::shared_ptr<CovSEArd> Ptr;

	/** Descriptor file */
	class Desc : public BaseCovFunc::Desc {
	public:
		typedef boost::shared_ptr<CovSEArd::Desc> Ptr;
		double sigma;
		double length;

		/** Default C'tor */
		Desc() {
			setToDefault();
		}

		/** Set values to default */
		void setToDefault() {
			BaseCovFunc::Desc::setToDefault();
			inputDim = 0;
			paramDim = 1;
			sigma = 1.0;
			length = 1.0;
		}

		/** Creates the object from the description. */
		CREATE_FROM_OBJECT_DESC_0(CovSEArd, BaseCovFunc::Ptr)
			/** Assert valid descriptor files */
			bool isValid(){
				if (!std::isfinite(sigma) || !std::isfinite(length))
					return false;
				return true;
			}
	};

	/** Get name of the covariance functions */
	virtual std::string getName() const {
		return "CovSEArd";
	}

	/** Compute the kernel */
	virtual double get(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2, const bool dirac = false) const {
		const double z = (x1 - x2).cwiseQuotient(ell).squaredNorm();
		const double noise = dirac ? sn2 : .0;
		return sf2*exp(-0.5*z) + noise;
	}
	/** SE derivate = -1*invLenght*[sigma_f^2*exp(sqrt((p_1-p_2)'*(p_1-p_2))/(-leng))] */
	virtual inline double getDiff(const Eigen::VectorXd& xi, const Eigen::VectorXd& xj, const size_t dx, const bool dirac = false) const {
		const double k = get(xi, xj, dirac);
		return -sf2 * ell[dx] * (xi(dx) - xj(dx)) * k;
	}

	virtual inline double getDiff2(const Eigen::VectorXd& xi, const Eigen::VectorXd& xj, const size_t dx1, const size_t dx2, const bool dirac = false) const {
		const double k = get(xi, xj, dirac); //sqrt(DD);
		const double noise = dirac ? sn2 : .0;
		return sf2 * ell[dx1] * (noise - ell[dx2] * (xi[dx1] - xj[dx1]) * (xi[dx2] - xj[dx2])) * k;
	}

	/** Covariance gradient of two input vectors with respect to the hyperparameters.
	*  @param x1 first input vector
	*  @param x2 second input vector
	*  @param grad covariance gradient */
	virtual void grad(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2, Eigen::VectorXd& g) const {
		Eigen::VectorXd z = (x1 - x2).cwiseQuotient(ell).array().square();
		const double k = sf2*exp(-0.5*z.sum());
		g.head(inputDim) = z * k;
		g(inputDim) = 2.0 * k;		
	}

	/** Update parameter vector.
	*  @param p new parameter vector */
	virtual void setLogHyper(const Eigen::VectorXd &p) {
		BaseCovFunc::setLogHyper(p);
		for (size_t i = 0; i < inputDim; ++i) 
			ell(i) = exp(loghyper(i)); // loghyper(i) * loghyper(i);
		sf2 = exp(2 * loghyper(inputDim)); // loghyper(inputDim) * loghyper(inputDim);
//		sn2 = loghyper(inputDim + 1) * loghyper(inputDim + 1);
	}

	~CovSEArd(){}

private:
	/** Hyper-parameters */
	Eigen::VectorXd ell; // second element of loghyper
	double sf2; // first element of loghyper

	/** Create from descriptor */
	void create(const Desc& desc) {
		BaseCovFunc::create(desc);
		inputDim = desc.inputDim;
		paramDim = desc.inputDim + 1;
		loghyper.resize(paramDim);
		ell.resize(inputDim);
		for (size_t i = 0; i < inputDim; ++i) {
			loghyper(i) = desc.length;
			ell(i) = exp(loghyper(i));  // loghyper(i) * loghyper(i);
		}
		loghyper(inputDim) = desc.sigma;
		sf2 = exp(2 * loghyper(inputDim)); //loghyper(inputDim) * loghyper(inputDim);
		sn2 = desc.noise * desc.noise;//loghyper(2) * loghyper(2); //2 * loghyper(1) * loghyper(1);//
	}

	CovSEArd() : BaseCovFunc() {}
};


}

//------------------------------------------------------------------------------

#endif

